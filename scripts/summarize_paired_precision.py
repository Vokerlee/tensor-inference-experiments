#!/usr/bin/env python3
"""Export complete-run paired precision summaries without discarding raw evidence."""

import argparse
import gzip
import json
from pathlib import Path

from summarize_primitive_pilots import digest, fraction, verify_sources, write_csv


COUNTERS = (
    "useful_products", "fp8_products", "reference_vectors", "selected_vectors",
    "mode_groups", "fp8_mode_groups", "within_pair_mode_transitions",
    "reference_gemm_calls", "candidate_activation_rows",
)
SCHEDULE_COUNTERS = (
    "resident_phases", "windows", "mixed_windows", "natural_mode_changes", "packed_mode_changes",
    "bf16_rows", "fp8_rows", "reference_weight_capacity_bytes", "selected_weight_capacity_bytes",
    "selected_weight_capacity_bytes_without_residency", "peak_argument_payload_bytes",
    "peak_result_payload_bytes", "peak_row_index_bytes",
)


def summarize(report: dict, *, record_sha256: str | None = None) -> tuple[list[dict], list[dict]]:
    verify_sources(report)
    if [case["case"] for case in report["cases"]] != report["case_order"]:
        raise ValueError("incomplete paired precision case sequence")
    if "prepared_weight_statistics" not in report:
        raise ValueError("missing final prepared-weight statistics")
    expected = set(report["target_layers"])
    context = {
        "model": report["model"],
        "record_sha256": record_sha256 or digest(json.dumps(report, sort_keys=True).encode()),
        "experiments_commit": report.get("experiments", {}).get("commit"),
        "minimum_fp8_rows": report.get("settings", {}).get("minimum_fp8_rows", 1),
        "cohort": report.get("settings", {}).get("cohort"),
        "row_buffer": report.get("settings", {}).get("row_buffer", 0),
        "resident_rows": report.get("settings", {}).get("resident_rows"),
        "execution": report.get("settings", {}).get("execution", "oracle"),
    }
    rows, layers = [], []
    for case in report["cases"]:
        if any(set(case[key]) != expected for key in (
                "layers", "activation_summaries", "skipped_dtype_calls")):
            raise ValueError("inconsistent layer coverage")
        totals = {key: sum(layer[key] for layer in case["layers"].values()) for key in COUNTERS}
        removed = totals["reference_vectors"] - totals["selected_vectors"]
        if removed < 0 or totals["fp8_products"] > totals["useful_products"]:
            raise ValueError("invalid logical work counts")
        row = {**context, "case": case["case"],
               "inputs": len(report["dataset"].get("prompts", report["dataset"].get("manifest", []))),
               **totals,
               "fp8_product_fraction": fraction(totals["fp8_products"], totals["useful_products"]),
               "output_vector_reduction_fraction": fraction(removed, totals["reference_vectors"]),
               "skipped_dtype_calls": sum(case["skipped_dtype_calls"].values())}
        for key in ("correct", "changed_predictions", "nearest_neighbor_agreement", "trajectory_rmse",
                    "trajectory_rmse_vs_bf16_split"):
            if key in case:
                row[key] = case[key]
        for comparison in ("vs_original", "vs_bf16_split"):
            row.update({f"{comparison}_{key}": value for key, value in case.get(comparison, {}).items()})
        for key in ("per_image_psnr", "per_image_psnr_vs_bf16_split"):
            if key in case:
                values = case[key]
                row[f"{key}_mean"] = sum(values) / len(values) if all(x is not None for x in values) else None
        rows.append(row)
        for key in SCHEDULE_COUNTERS:
            values = [layer.get("schedule", {}).get(key, 0) for layer in case["layers"].values()]
            row[f"schedule_{key}"] = ((max(values, default=0) if key.startswith("peak_") else sum(values))
                                      if context["row_buffer"] else None)
        for name, counters in case["layers"].items():
            traces = case["activation_summaries"][name]
            layers.append({
                **context, "case": case["case"], "layer": name,
                **{key: counters[key] for key in COUNTERS},
                "activation_scale_exponents": json.dumps(counters["activation_scale_exponents"], sort_keys=True),
                "normalized_bound_histogram": json.dumps(counters["normalized_bound_histogram"], sort_keys=True),
                "normalized_moment_score_histogram": json.dumps(counters.get("normalized_moment_score_histogram", {}), sort_keys=True),
                "activation_calls": len(traces),
                "observed_activation_elements": sum(trace["elements"] for trace in traces),
                "observed_zero_elements": sum(trace["zero_elements"] for trace in traces),
                "maximum_observed_amax": max((trace["amax"] for trace in traces), default=None),
                "skipped_dtype_calls": case["skipped_dtype_calls"][name],
                **{f"schedule_{key}": counters.get("schedule", {}).get(key, 0)
                   if context["row_buffer"] else None for key in SCHEDULE_COUNTERS},
            })
    return rows, layers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    rows, layers = [], []
    for path in args.input:
        raw = path.read_bytes()
        if path.suffix == ".gz":
            raw = gzip.decompress(raw)
        current_rows, current_layers = summarize(json.loads(raw), record_sha256=digest(raw))
        rows.extend(current_rows)
        layers.extend(current_layers)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_prefix.with_suffix(".csv"), rows)
    write_csv(args.output_prefix.with_suffix(".layers.csv"), layers)
    print(f"Exported {len(rows)} cases and {len(layers)} layer/case records")


if __name__ == "__main__":
    main()
