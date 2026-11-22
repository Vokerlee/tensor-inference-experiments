#!/usr/bin/env python3
"""Compare primitive-output accumulation contracts inside pretrained classifiers."""

import argparse
import platform
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import torch
import torchvision

from accumulation_scale_diagnostics import encode
from activation_pilot import (ENGINE, ROOT, digest, load_data, load_model,
                              revision, write_json)
from primitive_accumulation_reference import Counters, multiply


@contextmanager
def arithmetic(layers, prepared, activation_format, reduction_width, width):
    """Replace only receiving MLP linears, retaining every other graph edge."""
    original = {name: layer.forward for name, layer in layers.items()}
    counters = {name: Counters() for name in layers}

    def replacement(name, layer):
        codes, exponents = prepared[name]

        def forward(values):
            shape = values.shape
            activation, exponent = encode(values.reshape(-1, shape[-1]), activation_format)
            result = multiply(activation, exponent, codes, exponents,
                              reduction_width=reduction_width,
                              accumulator_width=width, counters=counters[name])
            if layer.bias is not None:
                result += layer.bias.float()
            return result.to(values.dtype).reshape(*shape[:-1], layer.out_features)

        return forward

    try:
        for name, layer in layers.items():
            layer.forward = replacement(name, layer)
        yield counters
    finally:
        for name, layer in layers.items():
            layer.forward = original[name]


def differences(values: torch.Tensor, reference: torch.Tensor) -> dict:
    delta = values.double() - reference.double()
    energy = reference.double().square().sum().item()
    return {
        "logit_rmse": delta.square().mean().sqrt().item(),
        "logit_relative_mse": delta.square().sum().item() / energy if energy else None,
        "logit_max_abs_difference": delta.abs().amax().item(),
        "changed_logits": int(torch.count_nonzero(delta)),
        "changed_predictions": int((values.argmax(-1) != reference.argmax(-1)).sum()),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("vit_b_16", "swin_t"), required=True)
    parser.add_argument("--activation-format", choices=("int8", "e2m1"), default="int8")
    parser.add_argument("--k-slice", type=int, choices=(128, 256), default=128)
    parser.add_argument("--width", type=int, action="append", choices=(20, 24, 32, 48))
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--holdout-offset", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples_per_class < 1 or args.batch_size < 1 or args.holdout_offset < 0:
        parser.error("positive sample/batch sizes and a nonnegative holdout offset are required")
    widths = list(dict.fromkeys(args.width or (24, 32)))
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    model, weights, layers = load_model(args.model)
    loader, class_indices, manifest = load_data(
        weights, args.samples_per_class, "quality", args.batch_size, args.holdout_offset)
    batches = [(images.to(device="cuda", dtype=torch.bfloat16),
                torch.tensor([class_indices[label] for label in labels], device="cuda"))
               for images, labels in loader]
    targets = torch.cat([labels for _, labels in batches])
    prepared = {name: encode(layer.weight.detach(), "int8") for name, layer in layers.items()}
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
    report = {
        "schema": 1, "model": args.model,
        "settings": {**{key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()}, "widths": widths},
        "experiments": revision(ROOT), "engine": revision(ENGINE),
        "source_sha256": {name: digest(ROOT / "scripts" / name) for name in (
            "primitive_accumulation_pilot.py", "primitive_accumulation_reference.py",
            "accumulation_scale_diagnostics.py", "activation_pilot.py", "activation_policies.py")},
        "checkpoint": {"url": weights.url, "sha256": digest(checkpoint)},
        "manifest": manifest, "preprocessing": str(weights.transforms()),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(), "tf32": False,
                        "bf16_reduced_precision_reduction": False},
        "limitations": [
            "MLP edges only; fixed group32 INT8 weight encoding; current-input power-of-two activation scales",
            "No calibration, label-based policy selection or retraining",
            "Explicit integer design contracts, not current native FP8/INT8 hardware support",
            "Fixed FP32 primitive outputs on both paths; no retained integer state across K slices",
            "FP64 represents bounded integers in the oracle, not a deployed arithmetic proposal",
            "Local changes use identical represented inputs; whole-model changes include propagation",
            "Classifier uses all 1000 ImageNet classes on an Imagenette subset",
            "No target latency, energy or novelty conclusion from this numerical pilot",
        ],
        "cases": [],
    }
    baseline = None
    group_reference = None
    for name, width in [("baseline", None), ("group-fp32", None)] + [
            (f"guarded-{width}", width) for width in widths]:
        if name == "baseline":
            outputs = torch.cat([model(images).float() for images, _ in batches])
            counters = {}
        else:
            with arithmetic(layers, prepared, args.activation_format, args.k_slice, width) as counts:
                outputs = []
                for index, (images, _) in enumerate(batches):
                    outputs.append(model(images).float())
                    print(f"{args.model} {name}: batch {index + 1}/{len(batches)}", flush=True)
                outputs = torch.cat(outputs)
                counters = {key: asdict(value) for key, value in counts.items()}
        if not torch.isfinite(outputs).all():
            raise RuntimeError(f"nonfinite logits in {name}")
        if baseline is None:
            baseline = outputs
        if name == "group-fp32":
            group_reference = outputs
        record = {
            "name": name, "samples": len(targets),
            "correct": int((outputs.argmax(-1) == targets).sum()),
            "predictions": outputs.argmax(-1).cpu().tolist(),
            "vs_original": differences(outputs, baseline),
            "layers": counters,
        }
        if group_reference is not None:
            record["vs_group_fp32"] = differences(outputs, group_reference)
        report["cases"].append(record)
        write_json(args.output, report)
        print({key: value for key, value in record.items() if key not in {"layers", "predictions"}}, flush=True)


if __name__ == "__main__":
    main()
