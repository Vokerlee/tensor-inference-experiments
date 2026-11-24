#!/usr/bin/env python3
"""Replay complete SANA decoder outputs with bounded Winograd convolution."""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from activation_pilot import ENGINE, ROOT, digest, revision, write_json
from modern_models import SANA, provenance, snapshot
from sana_decoder_feedback import image_values, metrics, save_outputs
from winograd_streaming import convolution, prepare
from winograd_integer_models import POLICIES as INTEGER_POLICIES, modified as integer_modified


POLICIES = {
    "f2-fp32": (2, torch.float32, None, False),
    "f2-bf16": (2, torch.bfloat16, None, False),
    "f2-a8": (2, torch.bfloat16, 8, False),
    "f2-a4": (2, torch.bfloat16, 4, False),
    "f4-bf16": (4, torch.bfloat16, None, False),
    "f4-a8": (4, torch.bfloat16, 8, False),
    "f2-bf16-stream": (2, torch.bfloat16, None, True),
    "f2-a8-stream": (2, torch.bfloat16, 8, True),
}


def eligible(decoder):
    return {name: layer for name, layer in decoder.named_modules()
            if isinstance(layer, torch.nn.Conv2d) and layer.groups == 1
            and layer.kernel_size == (3, 3) and layer.stride == (1, 1)
            and layer.dilation == (1, 1) and layer.padding in ((0, 0), (1, 1))
            and layer.padding_mode == "zeros" and layer.in_channels % 32 == 0}


@contextmanager
def modified(layers, case, records, tile_batch, evidence=None):
    if case in INTEGER_POLICIES:
        with integer_modified(layers, case, records, tile_batch, evidence) as metadata:
            yield metadata
        return
    tile, operand_dtype, bits, streaming = POLICIES[case]
    originals = {name: layer.forward for name, layer in layers.items()}
    coefficients = {name: prepare(layer.weight, tile, operand_dtype) for name, layer in layers.items()}

    def replace(name, layer):
        def forward(values):
            result, stats = convolution(values, coefficients[name], layer.bias,
                                        padding=layer.padding[0], tile_batch=tile_batch, bits=bits,
                                        reduction_block=128, streaming_inverse=streaming)
            records.append({"layer": name, "input_shape": list(values.shape),
                            "output_shape": list(result.shape), **stats})
            return result
        return forward

    try:
        for name, layer in layers.items():
            layer.forward = replace(name, layer)
        yield
    finally:
        for name, layer in layers.items():
            layer.forward = originals[name]


@torch.inference_mode()
def main():
    from diffusers import AutoencoderDC

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--tile-batch", type=int, default=64)
    parser.add_argument("--minimum-outputs", type=int, default=0)
    parser.add_argument("--case", action="append", choices=(*POLICIES, *INTEGER_POLICIES))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1 or args.tile_batch < 1 or args.minimum_outputs < 0:
        parser.error("positive input count and tile batch required")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    source = args.inputs / "inputs.npz"
    manifest = json.loads((args.inputs / "inputs.json").read_text())
    if manifest["status"] != "complete" or digest(source) != manifest["arrays_sha256"]:
        raise ValueError("incomplete or modified decoder inputs")
    with np.load(source, allow_pickle=False) as arrays:
        inputs, expected = arrays["inputs"][:args.count].copy(), arrays["raw"][:args.count].copy()
    if len(inputs) != args.count:
        raise ValueError("requested more inputs than captured")
    directory = snapshot(*SANA)
    model = AutoencoderDC.from_pretrained(directory, subfolder="vae", torch_dtype=torch.bfloat16)
    model.eval().requires_grad_(False).to("cuda")
    layers = {name: layer for name, layer in eligible(model.decoder).items()
              if layer.out_channels >= args.minimum_outputs}
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": 1, "status": "running", "model": "dc-ae-sana-decoder",
        "experiments": revision(ROOT), "engine": revision(ENGINE), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(), "checkpoint": provenance(*SANA, directory),
        "inputs": manifest, "input_sha256": digest(source),
        "settings": {"count": args.count, "tile_batch": args.tile_batch,
                     "floating_K": 128, "proposed_integer_K": 256,
                     "cases": args.case or list(POLICIES), "floating_activation_group": 32,
                     "integer_activation_group": 256, "minimum_outputs": args.minimum_outputs},
        "source_sha256": {name: digest(ROOT / "scripts" / name) for name in (
            "winograd_decoder_pilot.py", "winograd_streaming.py", "winograd_reference.py",
            "sana_decoder_feedback.py", "modern_models.py", "activation_pilot.py",
            "integer_winograd.py", "winograd_rounding.py", "winograd_integer_models.py",
            "winograd_shift_rounding.py", "plane_rounding_backend.py", "integer_segment_backend.py",
            "plane_encoding.py")},
        "eligible_layers": list(layers), "cases": [],
        "limitations": [
            "Known Winograd transforms are numerical baselines, not a novel convolution algorithm",
            "All policies use transformed weights; none learns activation calibration parameters",
            "Live INT codes reconstruct BF16 operands, not native packed INT4/8 or FP8 hardware",
            "K128 segmented Torch products do not claim bit equality with a specific array graph",
            "Output pixel fidelity is not FID, GenEval or perceptual/task quality",
            "Tile-batch working sets are bounded, but transformed weights are cached on the GPU",
            "GPU runtime and allocator memory are not target-chip latency or energy",
            "Input images are reused captured decoder states, not new independent prompts",
            "Integer cases propose INT8 K256, not native FP8; their INT4 codes do not imply extra throughput",
            "Integer cases use one current activation scale per K256 block and prepared per-column weight scales",
        ],
    }
    native = ENGINE / "build/libtensor_inference_engine_torch.so"
    if any("native" in case or "shift" in case for case in (args.case or [])):
        report["native_sha256"] = digest(native)
        build = ENGINE / "build/tensor_inference_engine_build.json"
        report["native_build"] = json.loads(build.read_text()) if build.exists() else None
    path = args.output / "report.json"
    write_json(path, report)
    original = None
    for case in ("original", *(args.case or POLICIES)):
        record = {"case": case, "calls": []}
        outputs = []
        if case == "original":
            for index, values in enumerate(inputs):
                result = model.decode(torch.from_numpy(values[None]).to("cuda", torch.bfloat16),
                                      return_dict=False)[0].float().cpu().numpy()
                if not np.array_equal(result, expected[index:index + 1]):
                    raise ValueError("original replay differs from captured decoder output")
                outputs.append(result)
        else:
            with modified(layers, case, record["calls"], args.tile_batch,
                          args.output / case / "prepared.npz") as metadata:
                record.update(metadata or {})
                for index, values in enumerate(inputs):
                    before = len(record["calls"])
                    result = model.decode(torch.from_numpy(values[None]).to("cuda", torch.bfloat16),
                                          return_dict=False)[0].float().cpu().numpy()
                    outputs.append(result)
                    for call in record["calls"][before:]:
                        call["input_index"] = index
                    print(f"{case}: decoded {index + 1}/{len(inputs)}", flush=True)
        raw = np.concatenate(outputs)
        if not np.isfinite(raw).all():
            raise ValueError("nonfinite final decoder output")
        if case == "original":
            original = raw
        record["raw_vs_original"] = metrics(raw, original)
        record["image_vs_original"] = metrics(image_values(raw), image_values(original))
        record["per_input"] = [
            {"index": i, "raw_vs_original": metrics(value, original[i]),
             "image_vs_original": metrics(image_values(value[None]), image_values(original[i:i + 1]))}
            for i, value in enumerate(raw)]
        record.update(save_outputs(args.output / case, raw))
        report["cases"].append(record)
        write_json(path, report)
        print(case, record["image_vs_original"], flush=True)
    report["status"] = "complete"
    write_json(path, report)


if __name__ == "__main__":
    main()
