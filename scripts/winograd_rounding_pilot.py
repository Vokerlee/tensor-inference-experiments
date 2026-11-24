#!/usr/bin/env python3
"""Measure current-tile plane rounding at real decoder receiver boundaries."""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from activation_pilot import ENGINE, ROOT, digest, revision, write_json
from modern_models import SANA, provenance, snapshot
from winograd_decoder_pilot import eligible
from winograd_reference import inverse_outputs, transform_inputs, transform_weights, transforms
from winograd_rounding import prepare, round_planes
from winograd_streaming import encode


def selected_tiles(values, tile, padding, count):
    height, width = values.shape[-2] + 2 * padding - 2, values.shape[-1] + 2 * padding - 2
    rows, columns = math.ceil(height / tile), math.ceil(width / tile)
    total = values.shape[0] * rows * columns
    indices = torch.linspace(0, total - 1, min(count, total), device=values.device).round().long().unique()
    padded = functional.pad(values, (padding, padding + (-width) % tile,
                                     padding, padding + (-height) % tile))
    batch = indices // (rows * columns)
    top, left = (indices // columns) % rows * tile, indices % columns * tile
    channels = torch.arange(values.shape[1], device=values.device)
    offsets = torch.arange(tile + 2, device=values.device)
    result = padded[batch[:, None, None, None], channels[None, :, None, None],
                    top[:, None, None, None] + offsets[None, None, :, None],
                    left[:, None, None, None] + offsets[None, None, None, :]]
    return result, indices


def compare(actual, reference):
    error = actual.double() - reference.double()
    squared = float(error.square().sum())
    energy = float(reference.double().square().sum())
    return {"squared_error": squared, "relative_mse": squared / energy if energy else None,
            "maximum_absolute_error": float(error.abs().max())}


@torch.inference_mode()
def main():
    import json
    from diffusers import AutoencoderDC

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--input-index", type=int, default=0)
    parser.add_argument("--tiles", type=int, default=32)
    parser.add_argument("--tile", type=int, choices=(2, 4), default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tiles < 1 or args.input_index < 0:
        parser.error("positive tile count and nonnegative input index required")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    input_path = args.inputs / "inputs.npz"
    manifest = json.loads((args.inputs / "inputs.json").read_text())
    if manifest["status"] != "complete" or digest(input_path) != manifest["arrays_sha256"]:
        raise ValueError("incomplete or modified decoder inputs")
    with np.load(input_path, allow_pickle=False) as arrays:
        values = arrays["inputs"][args.input_index:args.input_index + 1].copy()
        expected = arrays["raw"][args.input_index:args.input_index + 1].copy()
    if len(values) != 1:
        raise ValueError("input index outside capture")
    directory = snapshot(*SANA)
    model = AutoencoderDC.from_pretrained(directory, subfolder="vae", torch_dtype=torch.bfloat16)
    model.eval().requires_grad_(False).to("cuda")
    layers = eligible(model.decoder)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": 1, "status": "running", "model": "dc-ae-sana-decoder",
              "experiments": revision(ROOT), "engine": revision(ENGINE), "torch": torch.__version__,
              "checkpoint": provenance(*SANA, directory), "inputs": manifest,
              "input_sha256": digest(input_path),
              "settings": {"input_index": args.input_index, "tiles": args.tiles, "tile": args.tile,
                           "ranks": [1, 4, 16], "bits": [4, 8], "group": 32},
              "source_sha256": {name: digest(ROOT / "scripts" / name) for name in (
                  "winograd_rounding_pilot.py", "winograd_rounding.py", "winograd_reference.py",
                  "winograd_streaming.py", "winograd_decoder_pilot.py", "activation_pilot.py")},
              "operators": [], "limitations": [
                  "Local probes reuse one actual decoder input; they do not establish full-model quantized quality",
                  "Reference uses FP32 transformed input and BF16-reconstructed transformed weights",
                  "Actual receiving products and inverse transforms are diagnostic measurements, not deployed controller work",
                  "Isolated-channel surrogate omits cross-channel terms present in measured complete outputs",
                  "Rank16 is an expensive diagnostic comparator, not the proposed bounded-rank controller",
                  "No native packed execution, target-chip cycles or energy are measured"]}
    path = args.output / "report.json"
    write_json(path, report)

    def observe(name):
        def hook(layer, inputs, output):
            tiles, indices = selected_tiles(inputs[0], args.tile, layer.padding[0], args.tiles)
            scheme = transforms(args.tile, dtype=torch.float32, device="cuda")
            source = transform_inputs(tiles, scheme)
            coefficients = transform_weights(layer.weight, scheme).bfloat16().float()
            reference = inverse_outputs(source @ coefficients, scheme)
            number = len(report["operators"])
            arrays = args.output / f"operator-{number:02d}-source.npz"
            np.savez_compressed(arrays, source=source.cpu().numpy(),
                                selected_indices=indices.cpu().numpy(), reference=reference.cpu().numpy())
            row = {"layer": name, "input_shape": list(inputs[0].shape), "output_shape": list(output.shape),
                   "source_shape": list(source.shape), "arrays": str(arrays), "arrays_sha256": digest(arrays),
                   "weight_sha256": __import__("hashlib").sha256(coefficients.cpu().numpy().tobytes()).hexdigest(),
                   "variants": [], "geometry": []}
            geometries = {}
            for rank in (1, 4, 16):
                geometry = prepare(coefficients, args.tile, rank).to("cuda")
                geometries[rank] = geometry
                target = args.output / f"operator-{number:02d}-rank{rank}-geometry.npz"
                np.savez_compressed(target, **{key: getattr(geometry, key).cpu().numpy() for key in (
                    "factor", "diagonal", "normalizer", "eigenvalues", "metric")})
                row["geometry"].append({"rank": rank, "geometry": str(target), "geometry_sha256": digest(target),
                                        "mean_top_rank_energy": float((geometry.eigenvalues[:, -rank:].sum(-1)
                                            / geometry.eigenvalues.sum(-1).clamp_min(1e-30)).mean())})
            for bits in (4, 8):
                for rank in (0, 1, 4, 16):
                    reconstructed, stats = (encode(source, bits) if rank == 0
                                            else round_planes(source, geometries[rank], bits))
                    reconstructed = reconstructed.bfloat16().float()
                    result = inverse_outputs(reconstructed @ coefficients, scheme)
                    record = {"bits": bits, "rank": rank, "statistics": stats, **compare(result, reference)}
                    target = args.output / f"operator-{number:02d}-a{bits}-r{rank}.npz"
                    np.savez_compressed(target, reconstructed=reconstructed.cpu().numpy(),
                                        output=result.cpu().numpy())
                    record.update({"arrays": str(target), "arrays_sha256": digest(target)})
                    row["variants"].append(record)
            report["operators"].append(row)
            write_json(path, report)
            print(f"{len(report['operators'])}/{len(layers)} {name}", flush=True)
        return hook

    handles = [layer.register_forward_hook(observe(name)) for name, layer in layers.items()]
    try:
        result = model.decode(torch.from_numpy(values).to("cuda", torch.bfloat16), return_dict=False)[0]
        if not np.array_equal(result.float().cpu().numpy(), expected):
            raise ValueError("unmodified decoder replay differs from original capture")
    finally:
        for handle in handles:
            handle.remove()
    report["status"] = "complete"
    write_json(path, report)


if __name__ == "__main__":
    main()
