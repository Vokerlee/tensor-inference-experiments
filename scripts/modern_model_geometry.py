#!/usr/bin/env python3
"""Inventory pinned current-model shapes and fixed-width matrix interface work."""

import argparse
import math
from pathlib import Path

import torch

from activation_pilot import ENGINE, ROOT, digest, revision, write_json
from convolution_geometry import inventory
from modern_models import VISION, sana_transformer_meta, vision


def matrix_interface(rows: int, reduction: int, columns: int) -> dict:
    """Interface capacities for two declared modes; no SRAM or timing inference."""
    windows = math.ceil(columns / 128)
    result = {}
    for name, width in (("bf16", 128), ("fp8", 256)):
        vectors = rows * windows * math.ceil(reduction / width)
        result[name] = {"k_slice": width, "output_columns": 128,
                        "vector_count": vectors, "lhs_capacity_bytes": vectors * 256,
                        "fp32_output_capacity_bytes": vectors * 512,
                        "product_slots": vectors * width * 128}
    return result


def profile_matrices(model, execute):
    """Record executed dense projection shapes alongside complete operator MACs."""
    matrices = []

    def capture(name):
        def hook(module, args, output):
            if isinstance(module, torch.nn.Linear):
                reduction, columns = module.in_features, module.out_features
                rows = output.numel() // columns
            elif module.kernel_size == (1, 1) and module.groups == 1:
                reduction, columns = module.in_channels, module.out_channels
                rows = output.numel() // columns
            else:
                return
            matrices.append({"name": name, "kind": type(module).__name__, "M": rows,
                             "K": reduction, "N": columns, "macs": rows * reduction * columns,
                             "interface": matrix_interface(rows, reduction, columns)})
        return hook

    handles = [module.register_forward_hook(capture(name))
               for name, module in model.named_modules()
               if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d))]
    try:
        geometry = inventory(model, execute)
    finally:
        for handle in handles:
            handle.remove()
    return geometry, matrices


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*VISION, "sana-sprint"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int)
    args = parser.parse_args()
    if args.model == "sana-sprint":
        model, checkpoint = sana_transformer_meta()
        image_size = args.image_size or 1024
        if image_size <= 0 or image_size % 32:
            parser.error("SANA image size must be positive and divisible by32")
        latent = image_size // 32
        inputs = torch.empty(1, 32, latent, latent, device="meta")

        def execute():
            return model(inputs, encoder_hidden_states=torch.empty(1, 300, 2304, device="meta"),
                         timestep=torch.empty(1, device="meta"), guidance=torch.empty(1, device="meta"))
    else:
        model, data, checkpoint = vision(args.model, device="meta", pretrained=False)
        image_size = args.image_size or data.configuration["input_size"][-1]
        if image_size <= 0:
            parser.error("image size must be positive")
        inputs = torch.empty(1, 3, image_size, image_size, device="meta", dtype=torch.bfloat16)

        def execute():
            return model(inputs)

    geometry, matrices = profile_matrices(model, execute)
    report = {
        "schema": 1, "model": args.model, "checkpoint": checkpoint,
        "experiments": revision(ROOT), "engine": revision(ENGINE), "torch": torch.__version__,
        "source_sha256": {name: digest(ROOT / "scripts" / name) for name in (
            "modern_model_geometry.py", "modern_models.py", "convolution_geometry.py")},
        "image_size": image_size, "scope": "denoiser only" if args.model == "sana-sprint" else "vision model",
        "parameter_elements": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_bytes_if_bf16": sum(parameter.numel() * 2 for parameter in model.parameters()),
        "geometry": geometry, "matrix_shapes": matrices,
        "target_matrix_macs": sum(row["macs"] for row in matrices),
        "limitations": [
            "Meta execution: no pretrained weights or numerical quality evaluation",
            "MAC inventory excludes elementwise work and is not an energy breakdown",
            "BF16 parameter bytes exclude buffers, activations, workspace and quantized copies",
            "BF16 K128 / FP8 K256 both have128outputs and256-byte LHS vectors",
            "FP8 counts are a hypothetical all-FP8 contract, not validated format selection",
            "FP32 output capacity is not an SRAM transaction count or per-clock cadence",
            "Mode switches, weight delivery, buffering and array filling/draining are unpriced",
        ],
    }
    write_json(args.output, report)
    denominator = geometry["matrix_and_conv_macs"]
    print(args.model, "target matrix share", report["target_matrix_macs"] / denominator,
          "spatial conv share", geometry["spatial_conv_macs"] / denominator,
          "depthwise share", geometry["depthwise_conv_macs"] / denominator, flush=True)


if __name__ == "__main__":
    main()
