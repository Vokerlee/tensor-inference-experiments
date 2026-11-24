#!/usr/bin/env python3
"""Separate component timings on the exact previously captured operator inputs."""

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch

from activation_pilot import ENGINE, ROOT, digest, revision, write_json
from selective_bf16_reference import (ETA32, candidate_product, gamma, prepare_native)
from selective_reference_operator_pilot import time_call


def bf16(array):
    return torch.from_numpy(array.copy()).view(torch.bfloat16).cuda().contiguous()


def component_functions(source, weight, bias, backend):
    reduction = source.shape[1]
    pieces = math.ceil(reduction / 128)
    width = min(128, reduction)
    reference_gamma = gamma(width + pieces)
    fast_depth = math.ceil(width / 4) + 2 + pieces if backend == "chains4" else width + pieces
    fast_gamma = gamma(fast_depth, 24 if backend == "chains4" else 23)
    coefficient = math.nextafter(reference_gamma + fast_gamma, math.inf)
    denominator = math.nextafter(1 - max(reference_gamma, fast_gamma), 0.0)
    underflow = math.nextafter(4 * (reduction + 4 * pieces) * ETA32 / denominator, math.inf)
    if backend == "tensorcore":
        underflow *= 2.0 ** 23
    candidate = candidate_product(source, weight.values, 128, backend)
    norms = torch.ops.tensor_inference_engine.bf16_norms(source)

    def filter_only():
        return torch.ops.tensor_inference_engine.bf16_observe(
            candidate, norms, weight.norms, coefficient, underflow, bias)

    projected, accepted, _ = filter_only()
    indices = (~accepted).flatten().nonzero().flatten()

    def repair_only():
        if indices.numel() == 0:
            return candidate
        return torch.ops.tensor_inference_engine.ordered_gemm(
            source, weight.values, 128, 1, indices, candidate, False)

    return {
        "candidate_only": lambda: candidate_product(source, weight.values, 128, backend),
        "source_norm_only": lambda: torch.ops.tensor_inference_engine.bf16_norms(source),
        "filter_only_precomputed_norms": filter_only,
        "compaction_only": lambda: (~accepted).flatten().nonzero().flatten(),
        "repair_only_precomputed_indices": repair_only,
    }, indices.numel(), projected.numel()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operators", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=11)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent = json.loads(args.operators.read_text())
    if parent["status"] != "complete" or parent["invalid_cases"] or args.trials < 3:
        parser.error("complete valid operator report and >=3 trials required")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    report = {"schema": 1, "status": "running", "model": parent["model"],
        "experiments": revision(ROOT), "engine": revision(ENGINE),
        "operator_report": str(args.operators), "operator_report_sha256": digest(args.operators),
        "source_sha256": {name: digest(ROOT / "scripts" / name) for name in (
            "selective_reference_stage_pilot.py", "selective_bf16_reference.py",
            "selective_reference_operator_pilot.py")},
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "native_sha256": digest(ENGINE / "build/libtensor_inference_engine_torch.so"),
        "cases": [], "limitations": [
            "Components use warm cached inputs, with prerequisite tensors precomputed",
            "Isolated component medians are not an additive causal decomposition of complete latency",
            "Candidate-only timing returns FP32, omits bias/observation, and is not an exact replacement",
            "No-fallback repair timing measures a Python no-op, not a native kernel",
            "These are software emulator timings, not target-device latency or energy"]}
    for case_index, operator in enumerate(parent["operators"]):
        operands = Path(operator["operands"])
        if digest(operands) != operator["operands_sha256"]:
            raise ValueError("captured operand hash changed")
        with np.load(operands, allow_pickle=False) as arrays:
            source = bf16(arrays["source_bf16"])
            weight = prepare_native(bf16(arrays["weight_bf16"]))
            bias = bf16(arrays["bias_bf16"]).float() if arrays["bias_bf16"].size else None
        for backend in ("chains4", "tensorcore"):
            functions, unresolved, outputs = component_functions(source, weight, bias, backend)
            samples = {name: [] for name in functions}
            for function in functions.values():
                function()
            names = tuple(functions)
            for trial in range(args.trials):
                rotation = (case_index + trial) % len(names)
                for name in names[rotation:] + names[:rotation]:
                    _, timing = time_call(functions[name])
                    samples[name].append(timing)
            report["cases"].append({"layer": operator["layer"], "backend": backend,
                "shape_mnk": operator["shape_mnk"], "unresolved": unresolved, "outputs": outputs,
                "components": {name: {"samples": values,
                    "median_wall_ms": statistics.median(value["wall_ms"] for value in values),
                    "median_cuda_elapsed_ms": statistics.median(value["cuda_elapsed_ms"] for value in values)}
                    for name, values in samples.items()}})
        write_json(args.output, report)
    report["status"] = "complete"
    write_json(args.output, report)


if __name__ == "__main__":
    main()
