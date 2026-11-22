#!/usr/bin/env python3
"""Time native mask selection and representation preparation separately from GEMM."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from activation_pilot import ENGINE, ROOT, digest, packed_gram, revision, selection, write_json
from benchmark_gemm import measure, shape
from activation_policies import adaptive_representation


@torch.inference_mode()
def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--shape", action="append", type=shape, help="MxNxK")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    library = ENGINE / "build/libtensor_inference_engine_torch.so"
    torch.ops.load_library(str(library))
    torch.manual_seed(20261122)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    settings = argparse.Namespace(warmup=10, iterations=50, trials=5, mode="graph", graph_batch=10)
    report = {"schema": 1, "experiments": revision(ROOT), "engine": revision(ENGINE),
              "script_sha256": digest(Path(__file__)), "library_sha256": digest(library),
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "settings": vars(settings), "cases": [],
              "limitations": ["Synthetic BF16 inputs; no sparse/INT execution",
                              "CUDA latency, not ASIC latency or energy",
                              "Preparation retains dense reconstructed values",
                              "Does not include the joint sparse policy or sparse packing",
                              "Metadata regeneration is a setup cost, not necessarily per inference"]}
    for m, n, k in args.shape or [(128, 3072, 768), (128, 768, 3072), (128, 128, 1024)]:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        floating = x.float()
        q = selection.quantize_symmetric(x, 4).bfloat16().float()
        full_gram = selection.group_gram(weight)
        gram = packed_gram(full_gram)
        policy_args = SimpleNamespace(group_size=32, selector_backend="cuda", low_format="e2m1", mode_rows=128)
        native_args = SimpleNamespace(group_size=32, selector_backend="cuda-tile", low_format="e2m1", mode_rows=128)

        def select(diagonal):
            return torch.ops.tensor_inference_engine.select_2of4(floating, q, gram, diagonal)[0]

        def prepare(diagonal):
            quantized = selection.quantize_symmetric(x, 4).bfloat16().float()
            return torch.ops.tensor_inference_engine.select_2of4(
                x.float(), quantized, gram, diagonal)[0].bfloat16()

        functions = {
            "metadata_regeneration": lambda: packed_gram(selection.group_gram(weight)),
            "bf16_gemm": lambda: torch.ops.tensor_inference_engine.gemm(x, weight),
            "diagonal_selector": lambda: select(True),
            "gram_selector": lambda: select(False),
            "int4_dense_preparation": lambda: selection.quantize_symmetric(x, 4).bfloat16(),
            "diagonal_preparation": lambda: prepare(True),
            "gram_preparation": lambda: prepare(False),
            "gram_prepare_then_bf16_gemm": lambda: torch.ops.tensor_inference_engine.gemm(
                prepare(False), weight),
            "fused_int4_dense_preparation": lambda: torch.ops.tensor_inference_engine.prepare_activations(x),
            "fused_diagonal_preparation": lambda: torch.ops.tensor_inference_engine.prepare_activations(
                x, gram, 4, 1),
            "fused_gram_preparation": lambda: torch.ops.tensor_inference_engine.prepare_activations(
                x, gram, 4, 2),
            "reference_precision_policy": lambda: adaptive_representation(
                x, full_gram, gram, "precision", "gram", .01, policy_args),
            "native_precision_policy_with_counts": lambda: adaptive_representation(
                x, full_gram, gram, "precision", "gram", .01, native_args),
            "native_diagonal_precision_kernel": lambda: torch.ops.tensor_inference_engine.adaptive_precision(
                x, gram, .01, True, True),
            "native_gram_precision_kernel": lambda: torch.ops.tensor_inference_engine.adaptive_precision(x, gram),
            "native_low_error_precision_kernel": lambda: torch.ops.tensor_inference_engine.adaptive_precision_geometry(
                x, gram, .01, True, 2),
        }
        timings = {name: measure(function, settings) for name, function in functions.items()}
        record = {"M": m, "N": n, "K": k, "timings": timings,
                  "bf16_weight_bytes": 2 * k * n, "packed_gram_bytes": 10 * k,
                  "metadata_regeneration_multiplications": (k // 4) * 10 * n}
        report["cases"].append(record)
        print({"shape": [m, n, k], **{name: value["median_us"] for name, value in timings.items()}},
              flush=True)
    write_json(args.output, report)


if __name__ == "__main__":
    main()
