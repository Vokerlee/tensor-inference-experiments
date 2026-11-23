#!/usr/bin/env python3
"""Measure native pre-issue planning on shape-matched synthetic operands."""

import argparse
import json
import platform
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch

from accumulation_scale_diagnostics import encode
from benchmark_gemm import digest, revision
from integer_segment_backend import ENGINE, load_native, plan, weight_statistics


ROOT = Path(__file__).resolve().parents[1]
# (name, streamed rows M, reduction K, output columns N); not model executions.
SHAPES = (
    ("vit-up", 197, 768, 3072),
    ("vit-down", 197, 3072, 768),
    ("swin-up", 3136, 96, 384),
    ("swin-down", 3136, 384, 96),
    ("resnet-conv", 3136, 576, 64),
    ("unet-conv", 4096, 2880, 320),
)


def timed_graphs(functions: dict, *, batch: int, iterations: int, trials: int,
                 seed: int) -> dict:
    """Keep all graph outputs alive and interleave trial order deterministically."""
    graphs, outputs = {}, {}
    for name, function in functions.items():
        for _ in range(5):
            function()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs[name] = [function() for _ in range(batch)]
        graphs[name] = graph
    samples = {name: [] for name in functions}
    generator = random.Random(seed)
    for _ in range(trials):
        order = list(functions)
        generator.shuffle(order)
        for name in order:
            graph = graphs[name]
            graph.replay()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                graph.replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / (batch * iterations))
    return {name: {"median_us": statistics.median(values), "samples_us": values}
            for name, values in samples.items()}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph-batch", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--seed", type=int, default=731)
    args = parser.parse_args()
    if min(args.graph_batch, args.iterations, args.trials) < 1:
        parser.error("graph-batch, iterations and trials must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    load_native()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    library = ENGINE / "build/libtensor_inference_engine_torch.so"
    report = {
        "schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiments": revision(ROOT), "engine": revision(ENGINE),
        "source_sha256": digest(Path(__file__)),
        "backend_sha256": digest(ROOT / "scripts/integer_segment_backend.py"),
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "native_build": json.loads((library.parent / "tensor_inference_engine_build.json").read_text()),
            "library_sha256": digest(library),
        },
        "settings": {key: value for key, value in vars(args).items() if key != "output"},
        "limitations": [
            "Synthetic inputs at model-like shapes, not inference execution",
            "Cached code metadata; source quantization and statistics are excluded",
            "No matrix multiplication executes the selected partitions",
            "CUDA time is not ASIC energy, throughput or end-to-end speedup",
            "GEMM comparator: BF16 inputs, FP32 output and accumulation, no reduced reduction",
            "Metadata uses int32 software storage, not a compact hardware packet",
        ],
        "cases": [],
    }
    for name, rows, reduction, columns in SHAPES:
        a = torch.randn(rows, reduction, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(columns, reduction, device="cuda", dtype=torch.bfloat16)
        codes, exponent = encode(a, "int8")
        weight_codes, weight_exponent = encode(w, "int8")
        weights = weight_statistics(weight_codes, weight_exponent)
        maximum = codes.abs().amax(-1).int().contiguous()
        l1 = codes.abs().sum(-1).int().contiguous()
        functions = {"gemm_bf16_fp32": lambda: torch.ops.tensor_inference_engine.gemm(a, w.T, True)}
        for groups in (4, 8):
            for bound in ("max", "min"):
                functions[f"plan_k{groups * 32}_{bound}"] = (
                    lambda groups=groups, bound=bound: plan(
                        maximum, exponent, weights, groups_per_slice=groups,
                        bits=24, l1=l1 if bound == "min" else None))
        # The GEMM is only a time comparator; its BF16 arithmetic differs from
        # the represented INT8 operands used to generate the plans.
        actual = functions["gemm_bf16_fp32"]()
        torch.testing.assert_close(actual, a.float() @ w.float().T, rtol=0.0001, atol=0.001)
        timings = timed_graphs(functions, batch=args.graph_batch, iterations=args.iterations,
                              trials=args.trials, seed=args.seed)
        for value in timings.values():
            value["relative_to_gemm"] = value["median_us"] / timings["gemm_bf16_fp32"]["median_us"]
        report["cases"].append({"name": name, "M": rows, "K": reduction, "N": columns,
                                "groups": reduction // 32, "windows": (columns + 127) // 128,
                                "timings": timings})
        print(name, {key: round(value["median_us"], 3) for key, value in timings.items()}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
