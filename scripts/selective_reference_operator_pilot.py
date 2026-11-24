#!/usr/bin/env python3
"""Validate/time selective BF16 observations on pinned modern-model operands."""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from activation_pilot import ENGINE, ROOT, digest, load_data, revision, write_json
from modern_models import vision
from paired_precision_pilot import receiving_layers
from selective_bf16_reference import (candidate_product, full_reference, hybrid,
                                      prepare_native)


METHODS = ("full-direct", "full-tiled", "hybrid-chains4", "hybrid-tensorcore")


def encoded(values):
    return values.detach().cpu().contiguous().view(torch.uint16).numpy()


def capture(model_name, row_limit):
    model, preprocessing, checkpoint = vision(model_name)
    model.eval().requires_grad_(False)
    loader, _, manifest = load_data(preprocessing, 1, "quality", 1, 250)
    image, _ = next(iter(loader))
    operands, shapes, handles = [], set(), []

    def hook(name):
        def observe(layer, incoming):
            weight = layer.weight.reshape(layer.weight.shape[0], -1)
            signature = tuple(weight.shape)
            if signature in shapes:
                return
            shapes.add(signature)
            values = incoming[0]
            if isinstance(layer, torch.nn.Conv2d):
                matrix = values.permute(0, 2, 3, 1).reshape(-1, weight.shape[1])
            else:
                matrix = values.reshape(-1, weight.shape[1])
            count = min(row_limit, matrix.shape[0])
            indices = torch.linspace(0, matrix.shape[0] - 1, count, device=matrix.device).long()
            operands.append((name, matrix[indices].contiguous().cpu(),
                             weight.t().contiguous().cpu(), layer.bias.detach().cpu() if layer.bias is not None else None,
                             {"full_rows": matrix.shape[0], "sample_rows": count,
                              "row_selection": "inclusive linspace rounded toward zero", "weight_shape": list(weight.shape)}))
        return observe

    for name, layer in receiving_layers(model).items():
        handles.append(layer.register_forward_pre_hook(hook(name)))
    try:
        original = model(image.cuda().bfloat16()).float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    del model
    torch.cuda.empty_cache()
    return operands, {"model": model_name, "checkpoint": checkpoint, "dataset": manifest,
                      "used_loader_batch": 0, "original_output": original.tolist()}


def time_call(function):
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    start_event.record()
    result = function()
    end_event.record()
    end_event.synchronize()
    wall = (time.perf_counter() - start) * 1000
    return result, {"wall_ms": wall, "cuda_elapsed_ms": start_event.elapsed_time(end_event)}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("convnextv2-tiny", "dinov3-small"), required=True)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rows, args.trials) < 1:
        parser.error("positive rows/trials required")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    args.output.mkdir(parents=True, exist_ok=False)
    operands, model_record = capture(args.model, args.rows)
    report = {"schema": 1, "status": "running", **model_record,
        "experiments": revision(ROOT), "engine": revision(ENGINE), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(), "trials": args.trials,
        "source_sha256": {name: digest(ROOT / "scripts" / name) for name in (
            "selective_reference_operator_pilot.py", "selective_bf16_reference.py", "modern_models.py")},
        "native_sha256": digest(ENGINE / "build/libtensor_inference_engine_torch.so"), "operators": [], "invalid_cases": [],
        "limitations": [
            "First unique receiving-weight shape per model, deterministic rows from the first declared loader batch",
            "Declared sequential RN FP32 primitive128 reference; not the actual unspecified target accelerator",
            "Only an explicit post-bias BF16 observation is protected, not arbitrary FP32 consumers",
            "chains4 has a source-defined RN graph; Tensor Core bounds assume the documented-minimum FP32 rounding model",
            "Timing includes runtime norm/filter/compaction/fallback/conversion, excludes immutable weight preparation",
            "GPU and wall elapsed times are software emulation performance, never modeled-hardware latency or energy",
            "All methods and unfavorable operators are retained; corpus is not a random population sample",
        ]}
    destination = args.output / "report.json"
    write_json(destination, report)
    for index, (name, source, weight, bias, selection) in enumerate(operands):
        arrays = args.output / f"operator-{index:02d}-operands.npz"
        np.savez_compressed(arrays, source_bf16=encoded(source), weight_bf16=encoded(weight),
                            bias_bf16=encoded(bias) if bias is not None else np.empty(0, dtype=np.uint16))
        a, b = source.cuda(), weight.cuda()
        shift = bias.cuda() if bias is not None else None
        start = time.perf_counter()
        prepared = prepare_native(b)
        torch.cuda.synchronize()
        preparation_ms = (time.perf_counter() - start) * 1000
        reference = full_reference(a, b, shift)
        reference_fp32 = torch.ops.tensor_inference_engine.ordered_gemm(a, b)[0]
        if shift is not None:
            reference_fp32 += shift.float()
        functions = {
            "full-direct": lambda: full_reference(a, b, shift),
            "full-tiled": lambda: full_reference(a, b, shift, tiled=True),
            "hybrid-chains4": lambda: hybrid(a, prepared, shift, backend="chains4"),
            "hybrid-tensorcore": lambda: hybrid(a, prepared, shift, backend="tensorcore"),
        }
        results = {}
        for method, function in functions.items():
            result = function()
            actual = result if method.startswith("full") else result.output
            mismatch = int((actual.view(torch.int16) != reference.view(torch.int16)).sum())
            row = {"bf16_mismatches": mismatch, "samples": []}
            if not method.startswith("full"):
                candidate = candidate_product(a, b, 128, method.removeprefix("hybrid-"))
                if shift is not None:
                    candidate += shift.float()
                candidate_bf16 = candidate.bfloat16()
                false_accept = int((result.accepted & (candidate_bf16.view(torch.int16) != reference.view(torch.int16))).sum())
                difference = (candidate.double() - reference_fp32.double()).abs()
                bound_violations = int((difference > result.error_bound.double()).sum())
                audit = args.output / f"operator-{index:02d}-{method}-audit.npz"
                np.savez_compressed(audit, accepted=result.accepted.cpu().numpy(),
                    error_bound=result.error_bound.cpu().numpy(), candidate_fp32=candidate.cpu().numpy(),
                    reference_fp32=reference_fp32.cpu().numpy(), output_bf16=encoded(result.output))
                row.update(fallback_outputs=result.fallback_outputs, outputs=actual.numel(),
                           false_acceptances=false_accept, error_bound_violations=bound_violations,
                           invalid_indices=int(result.invalid_indices.item()),
                           unrepaired_bf16_mismatches=int((candidate_bf16.view(torch.int16) != reference.view(torch.int16)).sum()),
                           audit=str(audit), audit_sha256=digest(audit))
                if false_accept or bound_violations or row["invalid_indices"]:
                    report["invalid_cases"].append({"layer": name, "method": method, "reason": "invalid selective result"})
            if mismatch:
                report["invalid_cases"].append({"layer": name, "method": method, "reason": "output mismatch"})
            results[method] = row
        for function in functions.values():
            function()
        for trial in range(args.trials):
            order = list(METHODS)
            rotation = (trial + index) % len(order)
            for method in order[rotation:] + order[:rotation]:
                _, timing = time_call(functions[method])
                results[method]["samples"].append(timing)
        for row in results.values():
            row["median_wall_ms"] = statistics.median(value["wall_ms"] for value in row["samples"])
            row["median_cuda_elapsed_ms"] = statistics.median(value["cuda_elapsed_ms"] for value in row["samples"])
        best = min(results["full-direct"]["median_wall_ms"], results["full-tiled"]["median_wall_ms"])
        for row in results.values():
            row["speedup_over_best_full"] = best / row["median_wall_ms"]
        report["operators"].append({"index": index, "layer": name, **selection,
            "shape_mnk": [a.shape[0], b.shape[1], a.shape[1]], "operands": str(arrays),
            "operands_sha256": digest(arrays), "weight_norm_preparation_ms": preparation_ms, "methods": results})
        write_json(destination, report)
        print(args.model, name, {key: (round(value["speedup_over_best_full"], 3), value.get("fallback_outputs"))
                                for key, value in results.items()}, flush=True)
    report["status"] = "invalid" if report["invalid_cases"] else "complete"
    write_json(destination, report)


if __name__ == "__main__":
    main()
