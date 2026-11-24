#!/usr/bin/env python3
"""Repeat complete-model software timings against independently audited outputs."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from activation_pilot import ENGINE, ROOT, digest, load_data, revision, write_json
from modern_models import vision
from paired_precision_pilot import receiving_layers
from selective_reference_models import instrumented, prepare_layers
from selective_reference_operator_pilot import METHODS, time_call


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--trials", type=int, default=11)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    if audit["status"] != "complete" or audit["invalid_cases"] or args.trials < 3:
        parser.error("complete valid audit and at least three trials required")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    model, data, checkpoint = vision(audit["model"])
    layers = receiving_layers(model)
    weights = prepare_layers(layers)
    loader, _, manifest = load_data(data, 1, "quality", 1, 250)
    if checkpoint != audit["checkpoint"] or manifest != audit["dataset"]:
        raise ValueError("audit model or input provenance changed")
    images = [image.cuda().bfloat16() for image, _ in loader]
    report = {"schema": 1, "status": "running", "model": audit["model"],
        "audit": str(args.audit), "audit_sha256": digest(args.audit),
        "experiments": revision(ROOT), "engine": revision(ENGINE),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "source_sha256": {name: digest(ROOT / "scripts" / name) for name in (
            "selective_reference_model_timing.py", "selective_reference_models.py",
            "selective_bf16_reference.py", "selective_reference_operator_pilot.py")},
        "native_sha256": digest(ENGINE / "build/libtensor_inference_engine_torch.so"),
        "trials": args.trials, "cases": [], "invalid_cases": [],
        "limitations": ["Software emulator latency, not target-chip latency or energy",
            "Each method is warmed; trial order rotates across methods and cases",
            "Runtime source control, bounds, compaction, repair and output conversion included",
            "Immutable weight preparation and post-timing output/invalid checks excluded",
            "Same three existing deterministic inputs per model, not independent accuracy trials"]}
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "report.json"
    references = [row for row in audit["runs"] if row["method"] == "full-direct"]
    for case_index, row in enumerate(references):
        image = images[row["image_id"]]
        expected = torch.tensor(row["output"], device="cuda", dtype=torch.float32)
        results = {method: {"samples": [], "output_mismatches": 0, "invalid_events": 0} for method in METHODS}

        def run(method):
            with instrumented(layers, weights, method, row["policy"]) as current:
                output, timing = time_call(lambda: model(image))
            result = results[method]
            result["output_mismatches"] += int((output.float() != expected).sum())
            result["invalid_events"] += sum(int(value.item()) for value in current["invalid"])
            return timing

        for method in METHODS:
            run(method)
        for trial in range(args.trials):
            rotation = (trial + case_index) % len(METHODS)
            order = METHODS[rotation:] + METHODS[:rotation]
            for method in order:
                timing = run(method)
                timing["trial"] = trial
                results[method]["samples"].append(timing)
        for method, result in results.items():
            for metric in ("wall_ms", "cuda_elapsed_ms"):
                result["median_" + metric] = statistics.median(sample[metric] for sample in result["samples"])
            if result["output_mismatches"] or result["invalid_events"]:
                report["invalid_cases"].append({"image_id": row["image_id"], "policy": row["policy"], "method": method})
        best = min(results[method]["median_wall_ms"] for method in METHODS[:2])
        for result in results.values():
            result["speedup_over_best_full"] = best / result["median_wall_ms"]
        report["cases"].append({"image_id": row["image_id"], "policy": row["policy"], "methods": results})
        write_json(destination, report)
        print(audit["model"], row["image_id"], row["policy"],
              {key: round(value["speedup_over_best_full"], 3) for key, value in results.items()}, flush=True)
    report["status"] = "invalid" if report["invalid_cases"] else "complete"
    write_json(destination, report)


if __name__ == "__main__":
    main()
