#!/usr/bin/env python3
"""Probe fixed versus recomputed quantizer control on two decoder trajectories.

Sampled local interventions use identical unchanged receiving weights. They
are not a changed full-network quantization policy or natural failure rates.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from activation_pilot import ENGINE, ROOT, digest, revision, write_json
from modern_models import SANA, provenance, snapshot
from sana_resize_pilot import changed, prepared, upsampling_blocks
from spatial_feedback_reference import quantize


LAYER = "up_blocks.0.1.conv1"


def reconstruct_at_exponents(values, exponents, bits):
    source = values.float().permute(0, 2, 3, 1)
    grouped = source.reshape(*source.shape[:-1], source.shape[-1] // 32, 32)
    scales = torch.ldexp(torch.ones_like(exponents, dtype=torch.float32), exponents)[..., None]
    limit = (1 << (bits - 1)) - 1
    codes = (grouped / scales).round()
    restored = (codes.clamp(-limit, limit) * scales).reshape_as(source)
    return restored.permute(0, 3, 1, 2).contiguous().bfloat16(), int((codes.abs() > limit).sum())


def local_comparison(original, candidate, weight, bias, bits):
    coefficient = torch.zeros(original.shape[1])
    baseline = quantize(original, coefficient, bits=bits)
    live = quantize(candidate, coefficient, bits=bits)
    fixed, clipped = reconstruct_at_exponents(candidate, baseline.exponents, bits)

    def receive(values):
        return F.conv2d(values.cuda().bfloat16(), weight, bias).float().cpu()

    base, frozen, current = map(receive, (baseline.reconstructed, fixed, live.reconstructed))
    direct, control, total = frozen - base, current - frozen, current - base
    identity = total.double() - direct.double() - control.double()
    if torch.count_nonzero(identity):
        raise ValueError("diagnostic decomposition failed in exact FP64 difference arithmetic")
    original_unquantized, candidate_unquantized = map(receive, (original, candidate))
    return {
        "bits": bits, "source_shape": list(original.shape), "output_shape": list(base.shape),
        "source_squared_difference": float((candidate.float().double() - original.float()).square().sum()),
        "source_squared_reference": float(original.float().double().square().sum()),
        "sampled_groups": baseline.exponents.numel(),
        "changed_scale_groups": int((baseline.exponents != live.exponents).sum()),
        "fixed_control_clipped_scalars": clipped,
        "unquantized_receiver_squared_difference": float((candidate_unquantized.double() - original_unquantized).square().sum()),
        "data_path_squared_difference": float(direct.double().square().sum()),
        "control_path_squared_difference": float(control.double().square().sum()),
        "total_squared_difference": float(total.double().square().sum()),
        "twice_path_inner_product": float(2 * (direct.double() * control.double()).sum()),
        "decomposition_max_absolute_residual": float(identity.abs().max()),
    }


@torch.inference_mode()
def main():
    from diffusers import AutoencoderDC

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    directory = snapshot(*SANA)
    vae = AutoencoderDC.from_pretrained(directory, subfolder="vae", torch_dtype=torch.bfloat16)
    vae.eval().requires_grad_(False).to("cuda")
    layer = dict(vae.decoder.named_modules())[LAYER]
    blocks = upsampling_blocks(vae.decoder)
    metadata = prepared(blocks, "fold-bf16")
    report = {
        "schema": 1, "status": "running", "model": "dc-ae-sana-decoder",
        "experiments": revision(ROOT), "engine": revision(ENGINE),
        "source_sha256": {name: digest(ROOT / "scripts" / name) for name in (
            "decoder_control_path_probe.py", "sana_resize_pilot.py", "resize_convolution_reference.py",
            "spatial_feedback_reference.py")},
        "checkpoint": provenance(*SANA, directory), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(), "layer": LAYER, "input_bundles": [], "records": [],
        "limitations": [
            "Three predeclared 8x66 source patches per input; not whole-layer or failure-rate sampling",
            "Original versus folded-BF16 decoder trajectories; the receiving convolution weights are identical",
            "Memoryless source-group32 POT quantizer, no residual feedback state in this diagnostic",
            "Fixed controller is a local counterfactual, not an evaluated deployable policy",
            "The receiver uses BF16 Torch convolution; no target arithmetic or energy inferred",
            "No full-network quantized-policy quality metric is measured here",
        ],
    }
    for bundle in args.inputs:
        manifest = json.loads((bundle / "inputs.json").read_text())
        path = bundle / "inputs.npz"
        if manifest["status"] != "complete" or digest(path) != manifest["arrays_sha256"]:
            raise ValueError("changed or incomplete input bundle")
        report["input_bundles"].append({"manifest": manifest, "sha256": digest(path)})
        with np.load(path) as data:
            inputs, expected = data["inputs"].copy(), data["raw"].copy()
        for index, latent in enumerate(inputs):
            captures = {}

            def capture(name):
                def hook(module, incoming):
                    values = incoming[0]
                    height, width = values.shape[-2:]
                    windows = []
                    for row in (height // 4, height // 2, 3 * height // 4):
                        top, left = row - 4, width // 2 - 33
                        windows.append({"top": top, "left": left,
                                        "values": values[..., top:top + 8, left:left + 66].cpu().contiguous()})
                    captures[name] = windows
                return hook

            source = torch.from_numpy(latent[None]).to(device="cuda", dtype=torch.bfloat16)
            handle = layer.register_forward_pre_hook(capture("original"))
            try:
                result = vae.decode(source, return_dict=False)[0].float().cpu().numpy()
                if not np.array_equal(result, expected[index:index + 1]):
                    raise ValueError("original decoder replay mismatch")
            finally:
                handle.remove()
            handle = layer.register_forward_pre_hook(capture("folded"))
            try:
                with changed(blocks, metadata, "fold-bf16", []):
                    vae.decode(source, return_dict=False)
            finally:
                handle.remove()
            for patch, (original, candidate) in enumerate(zip(captures["original"], captures["folded"])):
                for bits in (4, 8):
                    report["records"].append({"bundle": str(bundle), "input_index": index,
                                              "patch_index": patch, "top": original["top"],
                                              "left": original["left"],
                                              **local_comparison(original["values"], candidate["values"],
                                                                 layer.weight, layer.bias, bits)})
            write_json(args.output, report)
            print(f"{bundle.name} input {index}: {len(report['records'])} records", flush=True)
    report["status"] = "complete"
    write_json(args.output, report)


if __name__ == "__main__":
    main()
