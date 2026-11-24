#!/usr/bin/env python3
"""Compare known direct/one-axis/two-axis lowerings on identical decoder tiles."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from activation_pilot import ENGINE, ROOT, digest, revision, write_json
from integer_winograd import multiply, prepare
from modern_models import SANA, provenance, snapshot
from winograd_decoder_pilot import eligible
from winograd_mapping import FP8, Convolution, work
from winograd_one_axis import inverse_outputs as inverse_one
from winograd_one_axis import transform_inputs as inputs_one
from winograd_one_axis import transform_weights as weights_one
from winograd_one_axis import work as work_one
from winograd_reference import inverse_outputs, transform_inputs, transform_weights, transforms
from winograd_rounding_pilot import compare, selected_tiles
from winograd_shift_rounding import ShiftGeometry


def representations(tiles, weight):
    """All three forms produce the same 2x2 spatial output tile in real arithmetic."""
    scheme = transforms(dtype=torch.float32, device=tiles.device)
    count = tiles.shape[0]
    direct_source = functional.unfold(tiles, 3).transpose(1, 2).reshape(1, count * 4, -1)
    direct_weight = weight.reshape(weight.shape[0], -1).t()[None].contiguous()
    yield "direct", direct_source, direct_weight, lambda values: values.reshape(count, 4, -1).transpose(1, 2).reshape(count, -1, 2, 2)
    strips = torch.cat([tiles[..., :3], tiles[..., 1:]], dim=0)

    def inverse_strips(values):
        spatial = inverse_one(values)[..., 0]
        return torch.stack((spatial[:count], spatial[count:]), dim=-1)

    yield "one-axis", inputs_one(strips), weights_one(weight), inverse_strips
    yield "two-axis", transform_inputs(tiles, scheme), transform_weights(weight, scheme), lambda values: inverse_outputs(values, scheme)


@torch.inference_mode()
def main():
    from diffusers import AutoencoderDC

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--tiles", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tiles < 1:
        parser.error("positive tile count required")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    path = args.inputs / "inputs.npz"
    manifest = json.loads((args.inputs / "inputs.json").read_text())
    if manifest["status"] != "complete" or digest(path) != manifest["arrays_sha256"]:
        raise ValueError("input capture changed")
    with np.load(path, allow_pickle=False) as arrays:
        values, expected = arrays["inputs"][:1].copy(), arrays["raw"][:1].copy()
    directory = snapshot(*SANA)
    model = AutoencoderDC.from_pretrained(directory, subfolder="vae", torch_dtype=torch.bfloat16)
    model.eval().requires_grad_(False).to("cuda")
    layers = eligible(model.decoder)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": 1, "status": "running", "model": "dc-ae-sana-decoder",
              "experiments": revision(ROOT), "engine": revision(ENGINE),
              "checkpoint": provenance(*SANA, directory), "inputs": manifest,
              "input_sha256": digest(path), "torch": torch.__version__,
              "native_sha256": digest(ENGINE / "build/libtensor_inference_engine_torch.so"),
              "settings": {"tiles": args.tiles, "input_index": 0, "K": 256, "N": 128,
                           "bits": [4, 8], "weights": 8},
              "source_sha256": {name: digest(ROOT / "scripts" / name) for name in (
                  "winograd_axis_pilot.py", "winograd_one_axis.py", "integer_winograd.py",
                  "winograd_reference.py", "winograd_rounding_pilot.py", "winograd_shift_rounding.py",
                  "plane_encoding.py", "activation_pilot.py")},
              "operators": [], "limitations": [
                  "Known lowerings, not a claimed new convolution identity",
                  "One unchanged decoder forward; isolated local probes, not full-model adapted quality",
                  "All forms compare to the same spatial FP32 convolution; prepared W8 representations differ",
                  "Current K256 native scales and exact integer partial oracle; INT8 mode is proposed, not native FP8",
                  "Issued vectors are interface work; useful products, transforms and gating differ",
                  "No target-chip cycles, SRAM transactions or energy are measured"]}
    target = args.output / "report.json"
    write_json(target, report)

    def observe(name):
        def hook(layer, inputs, output):
            tiles, indices = selected_tiles(inputs[0], 2, layer.padding[0], args.tiles)
            tiles, weight = tiles.float(), layer.weight.float()
            reference = functional.conv2d(tiles, weight)
            number = len(report["operators"])
            source_path = args.output / f"operator-{number:02d}-source.npz"
            np.savez_compressed(source_path, tiles=tiles.cpu().numpy(), reference=reference.cpu().numpy(),
                                selected_indices=indices.cpu().numpy())
            shape = Convolution(inputs[0].shape[0], output.shape[-2], output.shape[-1],
                                layer.in_channels, layer.out_channels)
            row = {"layer": name, "arrays": str(source_path), "arrays_sha256": digest(source_path),
                   "input_shape": list(inputs[0].shape), "output_shape": list(output.shape),
                   "variants": [], "work": [work(shape, FP8), work_one(shape, FP8), work(shape, FP8, 2)]}
            for method, source, coefficients, inverse in representations(tiles, weight):
                weights = prepare(coefficients)
                packed = torch.zeros(weights.codes.shape[1], source.shape[0], dtype=torch.int16, device="cuda")
                geometry = ShiftGeometry(packed, "cuda", rank=0)
                for bits in (4, 8):
                    products, stats = multiply(source, weights, bits=bits, geometry=geometry)
                    result = inverse(products)
                    array = args.output / f"operator-{number:02d}-{method}-a{bits}.npz"
                    np.savez_compressed(array, output=result.cpu().numpy())
                    row["variants"].append({"method": method, "bits": bits, **compare(result, reference),
                                            "statistics": stats, "arrays": str(array),
                                            "arrays_sha256": digest(array),
                                            "weight_code_sha256": hashlib.sha256(weights.codes.cpu().numpy().tobytes()).hexdigest()})
            report["operators"].append(row)
            write_json(target, report)
            print(f"{len(report['operators'])}/{len(layers)} {name}", flush=True)
        return hook

    handles = [layer.register_forward_hook(observe(name)) for name, layer in layers.items()]
    try:
        actual = model.decode(torch.from_numpy(values).to("cuda", torch.bfloat16), return_dict=False)[0]
        if not np.array_equal(actual.float().cpu().numpy(), expected):
            raise ValueError("original decoder replay changed")
    finally:
        for handle in handles:
            handle.remove()
    report["status"] = "complete"
    write_json(target, report)


if __name__ == "__main__":
    main()
