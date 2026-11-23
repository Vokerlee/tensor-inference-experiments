#!/usr/bin/env python3
"""Test known resize folding and online source quantization in the SANA decoder."""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from activation_pilot import ROOT, digest, write_json
from modern_models import SANA, provenance, snapshot
from resize_convolution_reference import arithmetic_counts, convolve_phases, phase_bank
from sana_decoder_feedback import environment, image_values, metrics, save_outputs
from source_feedback_backend import quantize_source, quantize_weights
from spatial_feedback_reference import kernel_coefficient


CASES = ("original", "fold-fp32", "fold-bf16", "expanded-a8", "source-a8",
         "fold-w8", "fold-w8-a8", "fold-w8-a8-feedback",
         "expanded-a4", "source-a4", "fold-w8-a4", "fold-w8-a4-feedback")


def upsampling_blocks(decoder) -> dict:
    return {
        name: layer for name, layer in decoder.named_modules()
        if getattr(layer, "interpolate", False) and getattr(layer, "factor", None) == 2
        and getattr(layer, "interpolation_mode", None) == "nearest"
        and isinstance(getattr(layer, "conv", None), torch.nn.Conv2d)
        and layer.conv.kernel_size == (3, 3) and layer.conv.padding == (1, 1)
        and layer.conv.stride == (1, 1) and layer.conv.dilation == (1, 1)
        and layer.conv.groups == 1 and layer.conv.in_channels % 32 == 0
    }


def prepared(blocks: dict, case: str) -> dict:
    result = {}
    for name, block in blocks.items():
        bank = phase_bank(block.conv.weight)
        if case != "fold-fp32":
            bank = bank.bfloat16()
        if case.startswith("fold-w8"):
            bank = quantize_weights(bank)
        result[name] = {
            "bank": bank,
            "coefficient": (kernel_coefficient(bank, coarse=True) if case.endswith("-feedback")
                            else torch.zeros(block.conv.in_channels, device=bank.device)),
        }
    return result


@contextmanager
def changed(blocks: dict, metadata: dict, case: str, records: list):
    originals = {name: block.forward for name, block in blocks.items()}

    def replacement(name, block):
        def forward(values):
            channels, height, width = values.shape[1:]
            record = {"layer": name, "input_shape": list(values.shape),
                      **arithmetic_counts(values.shape[0], channels, block.conv.out_channels,
                                          height, width)}
            source = values
            if case.startswith("expanded"):
                source = F.interpolate(source, scale_factor=2, mode="nearest")
            if "-a" in case:
                bits = 4 if "-a4" in case else 8
                quantized = quantize_source(source, metadata[name]["coefficient"], bits=bits)
                record["quantizer"] = quantized.statistics()
                source = quantized.reconstructed.contiguous()
            if case.startswith("fold"):
                bank = metadata[name]["bank"]
                bias = block.conv.bias.to(bank.dtype) if block.conv.bias is not None else None
                main = convolve_phases(source.to(bank.dtype), bank, bias).to(values.dtype)
                record["executed_reference_macs"] = record["reference_folded_macs"]
                record["interpolated_tensor_materialized"] = False
            else:
                if not case.startswith("expanded"):
                    source = F.interpolate(source, scale_factor=2, mode="nearest")
                main = block.conv(source)
                record["executed_reference_macs"] = record["expanded_macs"]
                record["interpolated_tensor_materialized"] = True
            if block.shortcut:
                shortcut = values.repeat_interleave(block.repeats, dim=1,
                                                    output_size=values.shape[1] * block.repeats)
                main = main + F.pixel_shuffle(shortcut, block.factor)
            records.append(record)
            return main
        return forward

    try:
        for name, block in blocks.items():
            block.forward = replacement(name, block)
        yield
    finally:
        for name, block in blocks.items():
            block.forward = originals[name]


@torch.inference_mode()
def main():
    from diffusers import AutoencoderDC

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--case", choices=CASES, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    input_path = args.inputs / "inputs.npz"
    manifest = json.loads((args.inputs / "inputs.json").read_text())
    if manifest["status"] != "complete" or digest(input_path) != manifest["arrays_sha256"]:
        raise ValueError("incomplete or changed decoder input bundle")
    with np.load(input_path) as arrays:
        inputs, original = arrays["inputs"].copy(), arrays["raw"].copy()
    directory = snapshot(*SANA)
    vae = AutoencoderDC.from_pretrained(directory, subfolder="vae", torch_dtype=torch.bfloat16)
    vae.eval().requires_grad_(False).to("cuda")
    blocks = upsampling_blocks(vae.decoder)
    report = {
        "schema": 1, "status": "running", "model": "dc-ae-sana-decoder",
        "environment": environment(), "checkpoint": provenance(*SANA, directory),
        "inputs": manifest, "input_sha256": digest(input_path),
        "source_sha256": {name: digest(ROOT / "scripts" / name) for name in (
            "sana_resize_pilot.py", "resize_convolution_reference.py", "sana_decoder_feedback.py")},
        "blocks": list(blocks), "cases": [],
        "limitations": [
            "Nearest-neighbor coefficient folding is established prior art, not a new decomposition",
            "Real-arithmetic equivalence is not BF16 bit equivalence; folded weights are rounded",
            "Folded weights grow16/9; reference computes an extra border row/column",
            "Only selected upsampling branches change; shortcut and other decoder operators remain",
            "Reconstructed BF16 operands, not packed integer or target hardware execution",
            "MAC counts and source scalars are not energy, latency or memory traffic measurements",
            "Pixel fidelity to original generated images does not establish semantic or perceptual quality",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "report.json"
    write_json(path, report)
    for case in dict.fromkeys(("original", *(args.case or CASES))):
        metadata = prepared(blocks, case)
        record = {"case": case, "calls": [], "coefficients": {
            name: values["coefficient"].cpu().tolist() for name, values in metadata.items()}}
        outputs = []
        context = changed(blocks, metadata, case, record["calls"])
        with context:
            for index, source in enumerate(inputs):
                before = len(record["calls"])
                values = torch.from_numpy(source[None]).to(device="cuda", dtype=torch.bfloat16)
                raw = vae.decode(values, return_dict=False)[0].float().cpu().numpy()
                if not np.isfinite(raw).all():
                    raise ValueError(f"nonfinite output: {case}")
                if case == "original" and not np.array_equal(raw, original[index:index + 1]):
                    raise ValueError("unmodified branch replay does not match captured original")
                for row in record["calls"][before:]:
                    row["input_index"] = index
                outputs.append(raw)
                print(f"{case}: decoded {index + 1}/{len(inputs)}", flush=True)
        values = np.concatenate(outputs)
        record["raw_vs_original"] = metrics(values, original)
        record["image_vs_original"] = metrics(image_values(values), image_values(original))
        record["per_input"] = [{"index": i, "raw_vs_original": metrics(values[i], original[i]),
                                "image_vs_original": metrics(image_values(values[i:i + 1]),
                                                             image_values(original[i:i + 1]))}
                               for i in range(len(inputs))]
        record.update(save_outputs(args.output / case, values))
        report["cases"].append(record)
        write_json(path, report)
        print(case, record["image_vs_original"], flush=True)
        del metadata
        torch.cuda.empty_cache()
    report["status"] = "complete"
    write_json(path, report)


if __name__ == "__main__":
    main()
