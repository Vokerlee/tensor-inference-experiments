#!/usr/bin/env python3
"""Regenerate selective-reference tables from complete, clean-source evidence."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


MODELS = ("convnextv2", "dinov3")
METHODS = ("full-direct", "full-tiled", "hybrid-chains4", "hybrid-tensorcore")


def read_report(root: Path, relative: str, sources: dict) -> dict:
    path = root / relative
    raw = gzip.decompress(path.with_suffix(path.suffix + ".gz").read_bytes()) if not path.exists() else path.read_bytes()
    report = json.loads(raw)
    if report.get("status") != "complete" or report.get("invalid_cases"):
        raise ValueError(f"incomplete or invalid evidence: {relative}")
    if any(report[key]["dirty"] for key in ("experiments", "engine")):
        raise ValueError(f"dirty canonical source: {relative}")
    sources[relative] = hashlib.sha256(raw).hexdigest()
    return report


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def timing_fields(result: dict) -> dict:
    if len(result["samples"]) < 3:
        raise ValueError("insufficient timing repetitions")
    return {key: result[key] for key in ("median_wall_ms", "median_cuda_elapsed_ms", "speedup_over_best_full")}


def summarize(root: Path) -> tuple[dict, dict[str, list[dict]]]:
    sources, operators, models, fidelity, stages = {}, [], [], [], []
    for model in MODELS:
        parent = read_report(root, f"selective-operators-{model}-20261124/report.json", sources)
        for operator in parent["operators"]:
            if set(operator["methods"]) != set(METHODS):
                raise ValueError("operator method coverage changed")
            for method in METHODS[2:]:
                result = operator["methods"][method]
                if any(result[key] for key in ("bf16_mismatches", "false_acceptances", "error_bound_violations", "invalid_indices")):
                    raise ValueError("operator correctness failure")
                operators.append({"model": parent["model"], "layer": operator["layer"], "method": method,
                    "M": operator["shape_mnk"][0], "N": operator["shape_mnk"][1], "K": operator["shape_mnk"][2],
                    "outputs": result["outputs"], "fallback_outputs": result["fallback_outputs"],
                    "fallback_fraction": result["fallback_outputs"] / result["outputs"],
                    "unrepaired_bf16_mismatches": result["unrepaired_bf16_mismatches"], **timing_fields(result)})
        audit_path = f"selective-model-{model}-20261124/report.json"
        audit = read_report(root, audit_path, sources)
        identities = set()
        for row in audit["runs"]:
            identity = (row["image_id"], row["policy"], row["method"])
            if identity in identities:
                raise ValueError("duplicate model audit")
            identities.add(identity)
            intermediate = sum(call["output_mismatches"] or 0 for call in row["calls"])
            if (not row["finite"] or intermediate or row["final_output_mismatches"] or row["control_mismatches"]
                    or row["invalid_events"] or row["timed_invalid_events"] or not row["timed_output_identical"]):
                raise ValueError("model fidelity failure")
            fidelity.append({"model": audit["model"], "image_id": row["image_id"], "policy": row["policy"],
                "method": row["method"], "calls": len(row["calls"]),
                "compared_intermediates": sum(call["output_mismatches"] is not None for call in row["calls"]),
                "intermediate_outputs": sum(call["outputs"] for call in row["calls"]),
                "fallback_outputs": sum(call["fallback_outputs"] or 0 for call in row["calls"]),
                "controller_events": len(row["controller_comparison"]),
                "intermediate_mismatches": intermediate, "final_mismatches": row["final_output_mismatches"],
                "controller_mismatches": row["control_mismatches"]})
        timing = read_report(root, f"selective-model-timing-{model}-20261124/report.json", sources)
        if timing["audit_sha256"] != sources[audit_path]:
            raise ValueError("timing audit hash changed")
        for row in timing["cases"]:
            for method, result in row["methods"].items():
                if result["output_mismatches"] or result["invalid_events"]:
                    raise ValueError("timed fidelity failure")
                models.append({"model": timing["model"], "image_id": row["image_id"], "policy": row["policy"],
                               "method": method, **timing_fields(result)})
        component = read_report(root, f"selective-stages-{model}-20261124.json", sources)
        for row in component["cases"]:
            for name, result in row["components"].items():
                stages.append({"model": component["model"], "layer": row["layer"], "backend": row["backend"],
                    "component": name, "unresolved": row["unresolved"], "outputs": row["outputs"],
                    "median_wall_ms": result["median_wall_ms"], "median_cuda_elapsed_ms": result["median_cuda_elapsed_ms"]})
    cast = read_report(root, "precision-cast-audit-20261124.json", sources)
    if cast["compiled"]["True"]["differences_from_eager"]:
        raise ValueError("configured cast-preserving guard disagrees")
    summary = {"schema": 1, "sources_sha256": sources,
        "operator_cases": len(operators) // 2, "model_audits": len(fidelity),
        "compared_intermediate_events": sum(row["compared_intermediates"] for row in fidelity),
        "model_timing_cases": len(models) // 4,
        "hybrid_operator_speedup_range": [min(row["speedup_over_best_full"] for row in operators),
                                          max(row["speedup_over_best_full"] for row in operators)],
        "hybrid_model_speedup_range": [min(row["speedup_over_best_full"] for row in models if row["method"].startswith("hybrid")),
                                       max(row["speedup_over_best_full"] for row in models if row["method"].startswith("hybrid"))],
        "hybrid_cases_reaching_1_25": sum(row["speedup_over_best_full"] >= 1.25 for row in operators + models
                                         if row["method"].startswith("hybrid")),
        "cast_audit": cast["compiled"], "aggregate_model_timings": []}
    for model in sorted({row["model"] for row in models}):
        for policy in ("bf16", "dynamic8", "live4"):
            rows = [row for row in models if row["model"] == model and row["policy"] == policy]
            images = sorted({row["image_id"] for row in rows})
            mean_times = {method: sum(row["median_wall_ms"] for row in rows if row["method"] == method) / len(images)
                          for method in METHODS}
            best_full = sum(min(row["median_wall_ms"] for row in rows if row["image_id"] == image and
                                row["method"] in METHODS[:2]) for image in images) / len(images)
            summary["aggregate_model_timings"].append({"model": model, "policy": policy,
                "inputs": len(images), "mean_of_median_wall_ms": mean_times, "best_full_wall_ms": best_full,
                "speedup_over_best_full": {method: best_full / mean_times[method] for method in METHODS[2:]}})
    return summary, {"operators": operators, "models": models, "fidelity": fidelity, "stages": stages}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True, help="output filename prefix")
    args = parser.parse_args()
    summary, tables = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for name, rows in tables.items():
        write_csv(args.output.with_suffix(f".{name}.csv"), rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
