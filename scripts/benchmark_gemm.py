#!/usr/bin/env python3
"""Compare the custom BF16 CUDA GEMM with Torch under matched arithmetic."""

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "deps/tensor-inference-engine"
DEFAULT_SHAPES = [(128, 128, 128), (197, 2304, 768), (197, 768, 3072),
                  (784, 256, 128), (1024, 1024, 1024), (2048, 2048, 2048)]


def revision(directory):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def shape(text):
    try:
        values = tuple(int(x) for x in text.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must be MxNxK") from error
    if len(values) != 3 or any(x <= 0 for x in values):
        raise argparse.ArgumentTypeError("shape must contain three positive dimensions MxNxK")
    return values


def measure(function, args):
    for _ in range(args.warmup):
        function()
    torch.cuda.synchronize()
    graph = None
    outputs = None
    if args.mode == "graph":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = [function() for _ in range(args.graph_batch)]
        function = graph.replay
    operations = args.iterations * (args.graph_batch if graph else 1)
    samples = []
    for _ in range(args.trials):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / operations)
    # Keep graph outputs alive until timing and replay are complete.
    del outputs, graph
    return {"median_us": statistics.median(samples), "samples_us": samples}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--library", type=Path, default=ENGINE / "build/libtensor_inference_engine_torch.so")
    result.add_argument("--shape", action="append", type=shape, help="repeatable MxNxK")
    result.add_argument("--rhs-layout", choices=["row", "column", "both"], default="both")
    result.add_argument("--mode", choices=["graph", "eager"], default="graph")
    result.add_argument("--warmup", type=int, default=10)
    result.add_argument("--iterations", type=int, default=50)
    result.add_argument("--trials", type=int, default=5)
    result.add_argument("--graph-batch", type=int, default=10)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--output", type=Path, default=ROOT / "results/gemm-baseline.json")
    return result


@torch.inference_mode()
def main():
    cli = parser()
    args = cli.parse_args()
    if min(args.iterations, args.trials, args.graph_batch) <= 0 or args.warmup < 0:
        cli.error("iterations, trials and graph-batch must be positive; warmup must be nonnegative")
    if not torch.cuda.is_available():
        cli.error("a CUDA device is required")
    if not args.library.is_file():
        cli.error("build the engine Torch adapter before benchmarking")
    torch.ops.load_library(str(args.library.resolve()))
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    build_info = args.library.parent / "tensor_inference_engine_build.json"
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=10).strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        driver = None
    report = {
        "schema": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiments": revision(ROOT), "engine": revision(ENGINE),
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "driver_versions": driver,
            "native_build": json.loads(build_info.read_text()) if build_info.is_file() else None,
            "torch_git": torch.version.git_version,
            "library_sha256": digest(args.library),
            "engine_requirements_sha256": digest(ENGINE / "requirements.txt"),
            "experiment_requirements_sha256": digest(ROOT / "requirements.txt"),
        },
        "settings": {
            "mode": args.mode, "warmup": args.warmup, "iterations": args.iterations,
            "trials": args.trials, "graph_batch": args.graph_batch, "seed": args.seed,
            "input_dtype": "bfloat16", "output_dtype": "bfloat16",
            "accumulation": "float32", "tf32": False,
            "bf16_reduced_precision_reduction": False,
        },
        "cases": [],
    }
    layouts = ["row", "column"] if args.rhs_layout == "both" else [args.rhs_layout]
    for m, n, k in args.shape or DEFAULT_SHAPES:
        for layout in layouts:
            a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            b = (torch.randn(n, k, device="cuda", dtype=torch.bfloat16).t() if layout == "column"
                 else torch.randn(k, n, device="cuda", dtype=torch.bfloat16))
            custom = lambda: torch.ops.tensor_inference_engine.gemm(a, b)
            reference = lambda: torch.mm(a, b)
            actual, expected = custom(), reference()
            torch.testing.assert_close(actual, expected, rtol=.008, atol=.02)
            fp32 = a.float() @ b.float()
            record = {
                "m": m, "n": n, "k": k, "rhs_layout": layout,
                "max_abs_vs_torch": (actual.float() - expected.float()).abs().max().item(),
                "max_abs_vs_fp32": (actual.float() - fp32).abs().max().item(),
                "custom": measure(custom, args), "torch": measure(reference, args),
            }
            for backend in ["custom", "torch"]:
                record[backend]["tflops"] = 2 * m * n * k / (record[backend]["median_us"] * 1e6)
            record["speedup"] = record["torch"]["median_us"] / record["custom"]["median_us"]
            report["cases"].append(record)
            print(f"{m}x{n}x{k} {layout:6s}: custom {record['custom']['median_us']:.2f} us, "
                  f"Torch {record['torch']['median_us']:.2f} us, ratio {record['speedup']:.3f}x", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {len(report['cases'])} cases to {args.output}")


if __name__ == "__main__":
    main()
