#!/usr/bin/env python3
"""Collect compact, source-hashed classifier and paired diffusion results."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def provenance(path: Path, report: dict, root: Path) -> dict:
    keys = ["experiments", "engine", "script_sha256", "reference_sha256",
            "policy_sha256", "library_sha256", "environment", "checkpoint",
            "checkpoint_files", "revision", "settings", "reference"]
    result = {key: report[key] for key in keys if key in report}
    result.update(source=str(path.relative_to(root)), source_sha256=digest(path))
    if "dataset" in report:
        dataset = report["dataset"]
        manifest = json.dumps(dataset["manifest"], sort_keys=True).encode()
        result["dataset"] = {key: value for key, value in dataset.items() if key != "manifest"}
        result["dataset"]["manifest_sha256"] = hashlib.sha256(manifest).hexdigest()
    if "settings" in result:
        result["settings"] = {key: value for key, value in result["settings"].items()
                               if key not in {"output", "library", "reference_run"}}
    return result


def classifier(path: Path, root: Path) -> dict:
    report = load(path)
    baseline = next(case for case in report["quality"] if case["case"] == "baseline")
    cases = []
    for case in report["quality"]:
        result = {key: value for key, value in case.items()
                  if key not in {"predictions", "targets", "correct"}}
        result["prediction_sha256"] = hashlib.sha256(json.dumps(case["predictions"]).encode()).hexdigest()
        result["baseline_only_correct"] = sum(a and not b for a, b in zip(
            baseline["correct"], case["correct"], strict=True))
        result["case_only_correct"] = sum(b and not a for a, b in zip(
            baseline["correct"], case["correct"], strict=True))
        cases.append(result)
    return {"provenance": provenance(path, report, root), "geometry": report["layers"],
            "cases": cases}


def diffusion(path: Path, root: Path) -> dict:
    report = load(path)
    cases = []
    for case in report["cases"]:
        result = {key: value for key, value in case.items()
                  if key not in {"trajectory_relative_mse_by_step", "pixel_mse_vs_baseline",
                                 "pixel_psnr_vs_baseline"}}
        for key in ["pixel_mse_vs_baseline", "pixel_psnr_vs_baseline"]:
            if key in case:
                values = case[key]
                result[key] = {"mean": mean(values), "median": median(values),
                               "min": min(values), "max": max(values)}
        if "trajectory_relative_mse_by_step" in case:
            result["last_observed_latent_relative_mse"] = case["trajectory_relative_mse_by_step"][-1]
        cases.append(result)
    return {"provenance": provenance(path, report, root), "cases": cases}


def operator_summary(operator: dict) -> dict:
    result = {key: value for key, value in operator.items() if key != "cases"}
    result["cases"] = [{key: value for key, value in case.items()
                        if not key.startswith("row_")} for case in operator["cases"]]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"schema": 1, "classifiers": {}, "diffusion": {}, "class_consistency": {},
              "matrix_work": {}, "streams": {}, "operators": {}, "timing": {}, "systolic": {},
              "limitations": ["Exploratory pilots, not confirmatory tests or target ASIC energy",
                              "Recorded execution revisions/dirty flags are preserved verbatim",
                              "Current scripts can be newer than an individual exploratory run",
                              "PSNR measures same-seed fidelity, not generation distribution quality"]}
    for path in sorted(args.input.glob("*.json")):
        if path.resolve() == args.output.resolve() or path.name == "summary.json":
            continue
        source = load(path)
        if "quality" in source:
            report["classifiers"][path.stem] = classifier(path, args.input)
        elif "operators" in source:
            report["operators"][path.stem] = {
                "provenance": provenance(path, source, args.input), "geometry": source["layers"],
                "operators": [operator_summary(operator) for operator in source["operators"]]}
        elif source.get("layers") and "prediction" in source["layers"][0]:
            report["streams"][path.stem] = {
                "provenance": provenance(path, source, args.input), "layers": source["layers"]}
        elif "classifiers" in source:
            report["class_consistency"][path.stem] = {
                "source_sha256": digest(path), **source}
        elif "target_mlp_flops" in source:
            report["matrix_work"][path.stem] = {"source_sha256": digest(path), **source}
        elif path.name == "final-selection-timing.json":
            report["timing"] = {"source_sha256": digest(path), **source}
        elif path.name == "systolic-actions.json":
            report["systolic"] = {"source_sha256": digest(path), **source}
    for path in sorted(args.input.glob("*/report.json")):
        report["diffusion"][path.parent.name] = diffusion(path, args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
