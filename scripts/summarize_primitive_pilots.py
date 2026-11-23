#!/usr/bin/env python3
"""Export paired arithmetic and convolution-geometry reports with source provenance."""

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_sources(report: dict) -> None:
    if report["experiments"]["dirty"] or report["engine"]["dirty"]:
        raise ValueError("arithmetic exports require a clean executed source checkout")
    commit = report["experiments"]["commit"]
    sources = dict(report["source_sha256"])
    if "convolution_reference_sha256" in report:
        sources["convolution_accumulation_reference.py"] = report["convolution_reference_sha256"]
    for name, expected in sources.items():
        data = subprocess.check_output(["git", "show", f"{commit}:scripts/{name}"], cwd=ROOT)
        if digest(data) != expected:
            raise ValueError(f"executed source hash mismatch: {commit} {name}")
    recorded_engine = subprocess.check_output(
        ["git", "rev-parse", f"{commit}:deps/tensor-inference-engine"], cwd=ROOT, text=True).strip()
    if recorded_engine != report["engine"]["commit"]:
        raise ValueError("executed engine does not match the recorded gitlink")


def export_run(path: Path, provenance: dict, layer_rows: list) -> list[dict]:
    raw = path.read_bytes()
    report = json.loads(raw)
    verify_sources(report)
    if not report["cases"] or report["cases"][-1]["name"] != f"guarded-{report['settings']['widths'][-1]}":
        raise ValueError(f"incomplete arithmetic report: {path}")
    manifest = report["manifest"]
    manifest_key = digest(json.dumps(manifest, sort_keys=True).encode())
    provenance["image_manifests"][manifest_key] = manifest
    provenance["runs"][path.stem] = {
        "raw_sha256": digest(raw),
        "image_manifest": manifest_key,
        **{key: report[key] for key in ("model", "settings", "experiments", "engine",
                                       "source_sha256", "checkpoint", "environment", "preprocessing")},
        "convolution_reference_sha256": report.get("convolution_reference_sha256"),
        "target_layers": report.get("target_layers"),
        "predictions": {case["name"]: case["predictions"] for case in report["cases"]},
    }
    provenance["runs"][path.stem]["settings"] = {
        key: value for key, value in report["settings"].items() if key != "output"}
    rows = []
    for case in report["cases"]:
        totals = {}
        for name, counters in case["layers"].items():
            layer_rows.append({"run": path.stem, "case": case["name"], "layer": name, **counters})
            for key, value in counters.items():
                if key == "max_output_difference":
                    totals[key] = max(totals.get(key, 0), value)
                else:
                    totals[key] = totals.get(key, 0) + value
        has_guard = case["name"].startswith("guarded-")
        comparison = case.get("vs_group_fp32", {})
        row = {
            "run": path.stem, "model": report["model"], "case": case["name"],
            "k_slice": report["settings"]["k_slice"],
            "strategy": report["settings"].get("strategy", "whole"),
            "bound_kind": report["settings"].get("bound_kind", "min"),
            "width": int(case["name"].rsplit("-", 1)[1]) if has_guard else None,
            "samples": case["samples"], "correct": case["correct"],
            "changed_predictions_vs_group_fp32": comparison.get("changed_predictions"),
            "changed_logits_vs_group_fp32": comparison.get("changed_logits"),
            "logit_relative_mse_vs_group_fp32": comparison.get("logit_relative_mse"),
            "logit_max_abs_vs_group_fp32": comparison.get("logit_max_abs_difference"),
            "whole_guard_product_fraction": fraction(totals.get("accepted_products", 0),
                                                      totals.get("useful_products", 0)) if has_guard else None,
            "raw_merge_replaced_fraction": fraction(totals.get("replaced_group_merges", 0),
                                                     totals.get("reference_group_merges", 0)) if has_guard else None,
            "zero_gated_merge_replaced_fraction": fraction(
                totals.get("zero_gated_reference_merges", 0) - totals.get("zero_gated_selected_merges", 0),
                totals.get("zero_gated_reference_merges", 0)) if has_guard else None,
            "same_input_whole_merge_replaced_fraction": fraction(
                totals.get("zero_gated_reference_merges", 0) - totals.get("zero_gated_whole_slice_merges", 0),
                totals.get("zero_gated_reference_merges", 0)) if has_guard else None,
            "valid_source_reuse": fraction(totals.get("valid_lowered_group_uses", 0),
                                           totals.get("source_groups_quantized", 0)),
            "primitive_relative_mse": fraction(totals.get("squared_output_difference", 0),
                                               totals.get("squared_reference_output", 0)) if has_guard else None,
            **totals,
        }
        rows.append(row)
    return rows


def export_geometry(path: Path, provenance: dict) -> dict:
    raw = path.read_bytes()
    report = json.loads(raw)
    commit = report["revision"]
    source = subprocess.check_output(["git", "show", f"{commit}:scripts/convolution_geometry.py"], cwd=ROOT)
    if digest(source) != report["script_sha256"]:
        raise ValueError(f"geometry source hash mismatch: {path}")
    provenance["geometry"][path.stem] = {
        key: report[key] for key in ("revision", "script_sha256", "model", "image_size",
                                     "torch", "torchvision", "unet_config_sha256")}
    provenance["geometry"][path.stem]["raw_sha256"] = digest(raw)
    row = {key: report[key] for key in (
        "model", "image_size", "matrix_and_conv_macs", "conv_macs", "linear_macs",
        "attention_macs", "other_explicit_matrix_macs", "spatial_conv_macs",
        "depthwise_conv_macs", "short_full_reduction_macs_k_le_25")}
    for k in (128, 256):
        for mapping in ("concatenated-channels", "separate-channel-windows"):
            schedules = [next(s for s in layer["schedules"] if s["reduction_width"] == k
                              and s["mapping"] == mapping) for layer in report["layers"]]
            row[f"k{k}_{mapping}_vectors"] = sum(s["vector_issue_groups"] for s in schedules)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--geometry", type=Path, nargs="*", default=[])
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--unet-source", type=Path)
    args = parser.parse_args()
    if len({p.stem for p in args.input}) != len(args.input):
        parser.error("input report stems must be unique")
    provenance = {"schema": 1, "runs": {}, "geometry": {}, "image_manifests": {}}
    provenance["exporter"] = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "sha256": digest(Path(__file__).read_bytes()),
    }
    if args.unet_source is not None:
        provenance["unet_config_source"] = json.loads(args.unet_source.read_text())
    rows, layers = [], []
    for path in sorted(args.input):
        rows.extend(export_run(path, provenance, layers))
    geometry = [export_geometry(path, provenance) for path in sorted(args.geometry)]
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_prefix.with_suffix(".csv"), rows)
    write_csv(args.output_prefix.with_suffix(".layers.csv"), layers)
    write_csv(args.output_prefix.with_suffix(".geometry.csv"), geometry)
    args.output_prefix.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    print(f"Exported {len(rows)} cases, {len(layers)} layer rows and {len(geometry)} geometries")


if __name__ == "__main__":
    main()
