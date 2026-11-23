#!/usr/bin/env python3
"""Pair spatial feedback policies on fixed inputs and modified classifier DAGs."""

import argparse
import platform
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torchvision
from torch.nn import functional as F

from activation_pilot import ENGINE, ROOT, digest, load_data, revision, write_json
from primitive_accumulation_pilot import differences
from spatial_feedback_reference import kernel_coefficient, quantize


POLICIES = ("none", "unit", "half", "kernel", "kernel-coarse", "kernel-coarse-zero")


@dataclass
class LocalError:
    input_scalars: int = 0
    input_error_squared: float = 0.0
    input_reference_squared: float = 0.0
    output_error_squared: float = 0.0
    output_reference_squared: float = 0.0
    clipped_values: int = 0
    introduced_nonzeros: int = 0

    def update(self, values, reconstructed, output, reference, result):
        self.input_scalars += values.numel()
        self.input_error_squared += float((reconstructed.double() - values.double()).square().sum())
        self.input_reference_squared += float(values.double().square().sum())
        self.output_error_squared += float((output.double() - reference.double()).square().sum())
        self.output_reference_squared += float(reference.double().square().sum())
        self.clipped_values += result.clipped_values
        self.introduced_nonzeros += result.introduced_nonzeros


def convolve(values, layer):
    return F.conv2d(values, layer.weight, layer.bias, layer.stride,
                    layer.padding, layer.dilation, layer.groups)


def coefficients(layer, policy):
    if policy in ("none", "unit", "half"):
        return torch.full((layer.in_channels,), {"none": 0, "unit": 1, "half": 0.5}[policy],
                          device=layer.weight.device, dtype=torch.float32)
    return kernel_coefficient(layer.weight, coarse=policy.startswith("kernel-coarse"))


def reconstruct(values, coefficient, policy, bits):
    return quantize(values, coefficient, bits=bits, preserve_zeros=policy.endswith("-zero"))


@contextmanager
def modified(layers, metadata, policy, bits):
    originals = {name: layer.forward for name, layer in layers.items()}

    def replacement(name, layer):
        def forward(values):
            result = reconstruct(values, metadata[name][policy], policy, bits)
            return convolve(result.reconstructed.to(values.dtype), layer)
        return forward

    try:
        for name, layer in layers.items():
            layer.forward = replacement(name, layer)
        yield
    finally:
        for name, layer in layers.items():
            layer.forward = originals[name]


@contextmanager
def local_diagnostics(layers, metadata, policies, bits):
    """Compare receiving products on identical original-trajectory inputs."""
    records = {name: {policy: LocalError() for policy in policies} for name in layers}

    def hook(name):
        def evaluate(layer, inputs, reference):
            values = inputs[0]
            for policy in policies:
                result = reconstruct(values, metadata[name][policy], policy, bits)
                reconstructed = result.reconstructed.to(values.dtype)
                records[name][policy].update(values, reconstructed, convolve(reconstructed, layer),
                                             reference, result)
        return evaluate

    handles = [layer.register_forward_hook(hook(name)) for name, layer in layers.items()]
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("resnet18", "resnet50"), default="resnet18")
    parser.add_argument("--bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--policy", action="append", choices=POLICIES)
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--holdout-offset", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--skip-local", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.samples_per_class, args.batch_size) < 1 or args.holdout_offset < 0:
        parser.error("positive sample/batch sizes and nonnegative offset required")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    policies = list(dict.fromkeys(["none", *(args.policy or POLICIES)]))
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if args.model == "resnet18":
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet18(weights=weights)
    else:
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        model = torchvision.models.resnet50(weights=weights)
    model = model.eval().to(device="cuda", dtype=torch.bfloat16)
    layers = {name: layer for name, layer in model.named_modules()
              if isinstance(layer, torch.nn.Conv2d) and layer.groups == 1
              and layer.in_channels % 32 == 0 and layer.padding_mode == "zeros"}
    metadata = {name: {policy: coefficients(layer, policy) for policy in policies}
                for name, layer in layers.items()}
    loader, classes, manifest = load_data(weights, args.samples_per_class, "quality",
                                         args.batch_size, args.holdout_offset)
    batches = [(images.cuda().bfloat16(), torch.tensor([classes[int(i)] for i in labels],
                                                     device="cuda")) for images, labels in loader]
    targets = torch.cat([labels for _, labels in batches])
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
    report = {
        "schema": 1, "model": args.model,
        "settings": {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()},
        "policies": policies, "experiments": revision(ROOT), "engine": revision(ENGINE),
        "source_sha256": {name: digest(ROOT / "scripts" / name) for name in (
            "spatial_feedback_pilot.py", "spatial_feedback_reference.py", "activation_pilot.py",
            "primitive_accumulation_pilot.py")},
        "checkpoint": {"url": weights.url, "sha256": digest(checkpoint)},
        "manifest": manifest, "preprocessing": str(weights.transforms()),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(), "tf32": False,
                        "bf16_reduced_precision_reduction": False},
        "limitations": [
            "Reconstructed activation codes with unchanged BF16 weights and Torch convolutions",
            "No integer or systolic execution, hardware timing or energy measurement",
            "Kernel autocorrelation criterion assumes white independent equal-variance residuals",
            "Actual residuals need not satisfy that approximation",
            "Small exploratory Imagenette subset, not full ImageNet accuracy",
            "Unaligned and grouped convolutions are unchanged",
            "Static weight statistics and live residuals; no activation calibration or retraining",
        ],
        "layers": {
            name: {"kernel_size": list(layer.kernel_size), "input_channels": layer.in_channels,
                   "coefficient_min": float(metadata[name].get("kernel", metadata[name]["none"]).min()),
                   "coefficient_max": float(metadata[name].get("kernel", metadata[name]["none"]).max()),
                   "coarse_nonzero_fraction": float((kernel_coefficient(layer.weight, coarse=True) != 0).float().mean())}
            for name, layer in layers.items()},
        "cases": [],
    }
    if args.skip_local:
        baseline = torch.cat([model(images).float().cpu() for images, _ in batches])
    else:
        with local_diagnostics(layers, metadata, policies, args.bits) as records:
            baseline = torch.cat([model(images).float().cpu() for images, _ in batches])
        report["local"] = {name: {policy: asdict(record) for policy, record in by_policy.items()}
                           for name, by_policy in records.items()}
    quantized_reference = None
    for policy in ("baseline", *policies):
        if policy == "baseline":
            logits = baseline
        else:
            with modified(layers, metadata, policy, args.bits):
                logits = torch.cat([model(images).float().cpu() for images, _ in batches])
        if policy == "none":
            quantized_reference = logits
        case = {"policy": policy, "samples": len(targets),
                "correct": int((logits.argmax(-1) == targets.cpu()).sum()),
                "predictions": logits.argmax(-1).tolist(),
                "vs_original": differences(logits, baseline)}
        if quantized_reference is not None:
            case["vs_no_feedback"] = differences(logits, quantized_reference)
        report["cases"].append(case)
        write_json(args.output, report)
        print(policy, case["correct"], case["vs_original"], flush=True)


if __name__ == "__main__":
    main()
