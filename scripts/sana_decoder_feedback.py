#!/usr/bin/env python3
"""Replay exact SANA decoder inputs under source-group quantization policies."""

import argparse
import json
import math
import platform
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from activation_pilot import ENGINE, ROOT, digest, revision, write_json
from modern_models import SANA, provenance, sana_pipeline, snapshot
from live_source_feedback import quantize_live_rows, weight_metadata
from paired_precision_pilot import PROMPTS
from source_feedback_backend import quantize_source, quantize_weights
from spatial_feedback_reference import kernel_coefficient


CASES = ("original", "none8", "coarse8", "kernel8", "none4", "coarse4", "kernel4")
SOURCES = ("sana_decoder_feedback.py", "source_feedback_backend.py",
           "spatial_feedback_reference.py", "modern_models.py", "activation_pilot.py",
           "live_source_feedback.py")


def environment() -> dict:
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "tf32": False, "bf16_reduced_precision_reduction": False,
        "experiments": revision(ROOT), "engine": revision(ENGINE),
        "source_sha256": {name: digest(ROOT / "scripts" / name) for name in SOURCES},
        "native_sha256": digest(ENGINE / "build/libtensor_inference_engine_torch.so"),
    }


def image_values(raw: np.ndarray) -> np.ndarray:
    # Diffusers denormalizes before converting its BF16 output to NumPy.
    values = torch.from_numpy(raw).bfloat16()
    return (values * 0.5 + 0.5).clamp(0, 1).float().numpy().transpose(0, 2, 3, 1)


def metrics(values: np.ndarray, reference: np.ndarray) -> dict:
    error = values.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(error ** 2))
    energy = float(np.mean(reference.astype(np.float64) ** 2))
    return {
        "rmse": math.sqrt(mse), "mean_error": float(error.mean()),
        "maximum_absolute_error": float(np.abs(error).max()),
        "relative_mse": mse / energy if energy else None,
        "psnr_unit_range": -10 * math.log10(mse) if mse else None,
        "identical": bool(np.array_equal(values, reference)),
    }


def eligible_layers(decoder, scope: str) -> dict:
    """Shape-only rule; no per-module tuning or activation calibration."""
    return {
        name: layer for name, layer in decoder.named_modules()
        if isinstance(layer, torch.nn.Conv2d) and layer.groups == 1
        and layer.in_channels % 32 == 0 and layer.out_channels >= 128
        and layer.padding_mode == "zeros"
        and (scope == "all" or layer.kernel_size != (1, 1))
    }


def coefficient(layer, case: str) -> torch.Tensor:
    if case.startswith("live"):
        return weight_metadata(layer.weight)
    if case.startswith("none"):
        return torch.zeros(layer.in_channels, device=layer.weight.device)
    return kernel_coefficient(layer.weight, coarse=case.startswith("coarse"))


@contextmanager
def modified(layers: dict, coefficients: dict, case: str, records: list):
    originals = {name: layer.forward for name, layer in layers.items()}

    def replace(name, layer):
        def forward(values):
            if case.startswith("live"):
                result, dynamic = quantize_live_rows(values, coefficients[name], bits=int(case[-1]))
            else:
                result = quantize_source(values, coefficients[name], bits=int(case[-1]))
                dynamic = None
            stats = result.statistics()  # Reject invalid events before the consumer runs.
            layout = (torch.channels_last if values.is_contiguous(memory_format=torch.channels_last)
                      else torch.contiguous_format)
            reconstructed = result.reconstructed.contiguous(memory_format=layout)
            output = layer._conv_forward(reconstructed, layer.weight, layer.bias)
            records.append({"layer": name, "input_shape": list(values.shape),
                            "output_shape": list(output.shape), "dynamic": dynamic, **stats})
            return output
        return forward

    try:
        for name, layer in layers.items():
            layer.forward = replace(name, layer)
        yield
    finally:
        for name, layer in layers.items():
            layer.forward = originals[name]


@contextmanager
def convolution_inventory(decoder, records: list):
    def observe(name):
        def hook(layer, inputs, output):
            records.append({
                "layer": name, "input_shape": list(inputs[0].shape),
                "output_shape": list(output.shape), "kernel": list(layer.kernel_size),
                "groups": layer.groups,
                "macs": output.numel() * (layer.in_channels // layer.groups)
                * math.prod(layer.kernel_size),
            })
        return hook
    handles = [layer.register_forward_hook(observe(name))
               for name, layer in decoder.named_modules() if isinstance(layer, torch.nn.Conv2d)]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def save_outputs(directory: Path, raw: np.ndarray) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    images = image_values(raw)
    path = directory / "arrays.npz"
    np.savez_compressed(path, raw=raw, images=images)
    for index, image in enumerate(images):
        Image.fromarray((image * 255).round().clip(0, 255).astype(np.uint8)).save(
            directory / f"{index:03d}.png")
    return {"arrays": str(path), "arrays_sha256": digest(path)}


def prepare(args):
    """Capture the actual scaled decoder input, not scheduler callback latents."""
    pipe, checkpoint = sana_pipeline()
    captured, decoded, images = [], [], []
    original_decode = pipe.vae.decode

    def capture(values, *decode_args, **decode_kwargs):
        if values.dtype != torch.bfloat16:
            raise ValueError("expected BF16 decoder input")
        captured.append(values.detach().float().cpu().numpy())
        result = original_decode(values, *decode_args, **decode_kwargs)
        if result[0].dtype != torch.bfloat16:
            raise ValueError("expected BF16 raw decoder output")
        decoded.append(result[0].detach().float().cpu().numpy())
        return result

    pipe.vae.decode = capture
    prompts = (args.prompt or PROMPTS)[:args.count]
    report = {"schema": 1, "status": "running", "environment": environment(),
              "checkpoint": checkpoint, "prompts": list(prompts), "seed": args.seed,
              "steps": args.steps, "guidance_scale": 4.5, "resolution": [1024, 1024],
              "input_dtype": "bfloat16", "input_capture": "argument of vae.decode after scaling"}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "inputs.json", report)
    for index, prompt in enumerate(prompts):
        generated = pipe(prompt=prompt, num_inference_steps=args.steps, guidance_scale=4.5,
                         height=1024, width=1024, output_type="np",
                         generator=torch.Generator("cuda").manual_seed(args.seed + index)).images
        images.append(generated)
        print(f"captured decoder input {index + 1}/{len(prompts)}", flush=True)
    if len(captured) != len(prompts) or len(decoded) != len(prompts):
        raise ValueError("expected exactly one decoder call per request; tiling is unsupported")
    inputs, raw, images = map(np.concatenate, (captured, decoded, images))
    if not all(np.isfinite(values).all() for values in (inputs, raw, images)):
        raise ValueError("nonfinite original pipeline output")
    if not np.array_equal(image_values(raw), images):
        raise ValueError("postprocessing replay does not match original pipeline")
    path = args.output / "inputs.npz"
    np.savez_compressed(path, inputs=inputs, raw=raw, images=images)
    report.update({"status": "complete", "arrays_sha256": digest(path),
                   "inputs_shape": list(inputs.shape), "raw_shape": list(raw.shape)})
    write_json(args.output / "inputs.json", report)


def run(args):
    from diffusers import AutoencoderDC

    input_path = args.inputs / "inputs.npz"
    input_record = json.loads((args.inputs / "inputs.json").read_text())
    if input_record["status"] != "complete" or digest(input_path) != input_record["arrays_sha256"]:
        raise ValueError("incomplete or modified decoder inputs")
    with np.load(input_path) as arrays:
        inputs, expected = arrays["inputs"].copy(), arrays["raw"].copy()
    directory = snapshot(*SANA)
    vae = AutoencoderDC.from_pretrained(directory, subfolder="vae", torch_dtype=torch.bfloat16)
    vae.eval().requires_grad_(False).to("cuda")
    layers = eligible_layers(vae.decoder, args.scope)
    report = {
        "schema": 1, "status": "running", "model": "dc-ae-sana-decoder",
        "environment": environment(), "checkpoint": provenance(*SANA, directory),
        "inputs": input_record, "input_sha256": digest(input_path),
        "settings": {"scope": args.scope, "weight_bits": args.weight_bits,
                     "activation_group": 32, "zero_preservation": False},
        "eligibility": "dense, C divisible32, output channels>=128, zero padding; spatial/all",
        "limitations": [
            "BF16 reconstructed operands and Torch convolution, not packed integer hardware",
            "No hardware latency, energy, traffic or packed-lowbit throughput measurement",
            "Fixed final decoder inputs are valid because the decoder does not feed denoising",
            "Pixel fidelity to the original is not FID, GenEval or perceptual task quality",
            "Kernel feedback uses a white independent residual approximation, not a guarantee",
            "No activation calibration, retraining or label-fitted coefficients",
        ], "cases": [],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    write_json(report_path, report)
    original = None
    cases = list(dict.fromkeys(("original", *(args.case or CASES))))
    if args.weight_bits and "weights-only" not in cases:
        cases.insert(1, "weights-only")
    for case in cases:
        if case == "weights-only":
            for layer in layers.values():
                layer.weight.copy_(quantize_weights(layer.weight, bits=args.weight_bits))
        coeff = {name: coefficient(layer, case) for name, layer in layers.items()}
        record = {"case": case, "calls": [], "per_input": []}
        outputs, inventory = [], []
        context = (convolution_inventory(vae.decoder, inventory) if case == "original"
                   else modified(layers, coeff, case, record["calls"]) if case != "weights-only"
                   else convolution_inventory(vae.decoder, []))
        with context:
            for index, values in enumerate(inputs):
                before = len(record["calls"])
                source = torch.from_numpy(values[None]).to(device="cuda", dtype=torch.bfloat16)
                result = vae.decode(source, return_dict=False)[0].float().cpu().numpy()
                if not np.isfinite(result).all():
                    raise ValueError(f"nonfinite decoder output: {case}, input {index}")
                if case == "original" and not np.array_equal(result, expected[index:index + 1]):
                    raise ValueError("original decoder replay is not bit-exact to captured output")
                for call in record["calls"][before:]:
                    call["input_index"] = index
                outputs.append(result)
                print(f"{case}: decoded {index + 1}/{len(inputs)}", flush=True)
        raw = np.concatenate(outputs)
        if case == "original":
            original = raw
            report["convolutions"] = inventory
            report["eligible_layers"] = list(layers)
            report["eligible_macs"] = sum(row["macs"] for row in inventory if row["layer"] in layers)
            report["convolution_macs"] = sum(row["macs"] for row in inventory)
        record["raw_vs_original"] = metrics(raw, original)
        record["image_vs_original"] = metrics(image_values(raw), image_values(original))
        record["per_input"] = [
            {"index": i, "raw_vs_original": metrics(value, original[i]),
             "image_vs_original": metrics(image_values(value[None]), image_values(original[i:i + 1]))}
            for i, value in enumerate(raw)]
        record.update(save_outputs(args.output / case, raw))
        record["coefficients"] = {
            name: ({key: tensor.cpu().tolist() for key, tensor in value.items()}
                   if isinstance(value, dict) else value.cpu().tolist())
            for name, value in coeff.items()
        } if case not in ("original", "weights-only") else {}
        report["cases"].append(record)
        write_json(report_path, report)
        print(case, record["image_vs_original"], flush=True)
    report["status"] = "complete"
    write_json(report_path, report)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    capture = subparsers.add_parser("prepare")
    capture.add_argument("--count", type=int, choices=range(1, 5), default=4)
    capture.add_argument("--prompt", action="append")
    capture.add_argument("--steps", type=int, default=2)
    capture.add_argument("--seed", type=int, default=20261123)
    capture.add_argument("--output", type=Path, required=True)
    replay = subparsers.add_parser("run")
    replay.add_argument("--inputs", type=Path, required=True)
    replay.add_argument("--scope", choices=("spatial", "all"), default="spatial")
    replay.add_argument("--weight-bits", type=int, choices=(4, 8))
    replay.add_argument("--case", action="append", choices=(*CASES, "live4", "live8"))
    replay.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA required")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    (prepare if args.mode == "prepare" else run)(args)


if __name__ == "__main__":
    main()
