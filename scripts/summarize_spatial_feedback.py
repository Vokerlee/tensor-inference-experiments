#!/usr/bin/env python3
"""Export paired spatial-feedback errors, model results and exact provenance."""

import argparse
import json
import subprocess
from pathlib import Path

from summarize_primitive_pilots import ROOT, digest, fraction, verify_sources, write_csv


def export(path: Path, provenance: dict, layer_rows: list) -> list[dict]:
    raw = path.read_bytes()
    report = json.loads(raw)
    verify_sources(report)
    expected = ["baseline", *report["policies"]]
    if [case["policy"] for case in report["cases"]] != expected:
        raise ValueError("incomplete spatial-feedback case sequence")
    if "local" not in report:
        raise ValueError("this export requires paired local diagnostics")
    manifest_key = digest(json.dumps(report["manifest"], sort_keys=True).encode())
    provenance["image_manifests"][manifest_key] = report["manifest"]
    record = {key: report[key] for key in (
        "model", "settings", "policies", "experiments", "engine", "source_sha256",
        "checkpoint", "preprocessing", "environment", "layers", "limitations")}
    record["settings"] = {key: value for key, value in record["settings"].items() if key != "output"}
    provenance["runs"][path.stem] = {
        **record, "raw_sha256": digest(raw), "image_manifest": manifest_key,
        "predictions": {case["policy"]: case["predictions"] for case in report["cases"]},
    }
    totals = {policy: {} for policy in report["policies"]}
    for layer, by_policy in report["local"].items():
        for policy, values in by_policy.items():
            layer_rows.append({"run": path.stem, "layer": layer, "policy": policy,
                               **report["layers"][layer], **values})
            for key, value in values.items():
                totals[policy][key] = totals[policy].get(key, 0) + value
    rows = []
    baseline_error = totals["none"]
    for case in report["cases"]:
        policy = case["policy"]
        current = totals.get(policy, {})
        spatial = [by_policy for layer, by_policy in report["local"].items()
                   if report["layers"][layer]["kernel_size"][1] > 1]
        rows.append({
            "run": path.stem, "model": report["model"], "bits": report["settings"]["bits"],
            "policy": policy, "samples": case["samples"], "correct": case["correct"],
            **case["vs_original"],
            "changed_predictions_vs_no_feedback": case.get("vs_no_feedback", {}).get("changed_predictions"),
            "input_error_ratio_vs_no_feedback": fraction(current.get("input_error_squared", 0),
                                                          baseline_error["input_error_squared"]) if current else None,
            "output_error_ratio_vs_no_feedback": fraction(current.get("output_error_squared", 0),
                                                           baseline_error["output_error_squared"]) if current else None,
            "spatial_layers": len(spatial),
            "spatial_layers_with_lower_error": sum(
                values[policy]["output_error_squared"] < values["none"]["output_error_squared"]
                for values in spatial) if current else None,
            **current,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    if len({path.stem for path in args.input}) != len(args.input):
        parser.error("input report names must be unique")
    provenance = {"schema": 1, "runs": {}, "image_manifests": {}, "exporter": {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "sha256": digest(Path(__file__).read_bytes()),
        "verification_helper_sha256": digest((ROOT / "scripts/summarize_primitive_pilots.py").read_bytes()),
    }}
    rows, layers = [], []
    for path in sorted(args.input):
        rows.extend(export(path, provenance, layers))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_prefix.with_suffix(".csv"), rows)
    write_csv(args.output_prefix.with_suffix(".layers.csv"), layers)
    args.output_prefix.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    print(f"Exported {len(rows)} cases and {len(layers)} layer/policy records")


if __name__ == "__main__":
    main()
