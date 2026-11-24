#!/usr/bin/env python3
"""Compare attention roles with matched projection splits and full-model outputs."""

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from activation_pilot import ENGINE, ROOT, digest, load_data, revision, write_json
from activation_rounding_model_pilot import compare
from activation_rounding_models import modified
from attention_rounding import POLICIES, prepare, split_attention
from modern_models import vision
from paired_precision_pilot import receiving_layers


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--offset", type=int, default=250)
    parser.add_argument("--policy", action="append", choices=POLICIES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1 or args.offset < 0:
        parser.error("positive count and nonnegative allocation offset required")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    policies = args.policy or list(POLICIES)
    args.output.mkdir(parents=True, exist_ok=False)
    model, data, checkpoint = vision("dinov3-small")
    layers = receiving_layers(model)
    geometry = prepare(layers)
    arrays = {}
    for name, variants in geometry.items():
        for (metric, rank), values in variants.items():
            for field in ("factor", "diagonal", "residual_norm", "normalizer", "eigenvalues"):
                arrays[f"{name}-{metric}-{rank}-{field}"] = getattr(values, field).cpu().numpy()
    geometry_path = args.output / "geometry.npz"
    np.savez_compressed(geometry_path, **arrays)
    loader, classes, manifest = load_data(data, math.ceil(args.count / 10), "quality", 1, args.offset)
    report = {"schema": 1, "status": "running", "model": "dinov3-small",
        "experiments": revision(ROOT), "engine": revision(ENGINE), "torch": torch.__version__,
        "checkpoint": checkpoint, "dataset": manifest, "count": args.count, "offset": args.offset,
        "gpu": torch.cuda.get_device_name(), "policies": policies,
        "geometry": str(geometry_path), "geometry_sha256": digest(geometry_path),
        "source_sha256": {name: digest(ROOT / "scripts" / name) for name in (
            "attention_rounding_model_pilot.py", "attention_rounding.py", "activation_rounding.py",
            "activation_rounding_models.py", "activation_rounding_model_pilot.py", "modern_models.py")},
        "runs": [], "invalid_cases": [], "limitations": [
            "Every policy uses identical three-way Q/K/V output-column splitting, including the BF16 control",
            "Split BF16 may differ from the original fused Torch projection; both final references are retained",
            "QKV roles sharing settings reuse one source encoding; other selected projections use nearest INT8",
            "Balanced geometry changes only immutable weight-derived rounding metadata, not model weights",
            "All policies use current group scales and live model trajectories without activation calibration",
            "Attention role counts describe the same original matrix MACs, not guaranteed energy savings",
            "BF16 reconstructed activations feed Torch; no native packed integer or hardware timing claim",
            "Descriptor MSE/cosine are fidelity diagnostics on this fixed allocation, not population task accuracy"]}
    destination = args.output / "report.json"
    for image_id, (images, labels) in enumerate(loader):
        if image_id >= args.count:
            break
        image = images.cuda().bfloat16()
        original = model(image).float().cpu()
        with modified(layers, {}, "bf16", record=False), split_attention(layers, geometry, "bf16", record=False):
            reference = model(image).float().cpu()
        reference_macs = None
        for policy in policies:
            other_policy = "bf16" if policy == "bf16" else "uniform8"
            with modified(layers, {}, other_policy) as others, split_attention(layers, geometry, policy) as attention:
                output = model(image).float().cpu()
            finite = bool(torch.isfinite(output).all())
            total_macs = sum(row["macs"] for row in others + attention)
            if reference_macs is None:
                reference_macs = total_macs
            if total_macs != reference_macs:
                raise ValueError("policy changed original matrix MAC coverage")
            row = {"image_id": image_id, "target": classes[int(labels[0])], "policy": policy,
                "finite": finite, "output": output.tolist(), "split_reference": reference.tolist(),
                "original_reference": original.tolist(), "matched_error": compare(output, reference) if finite else None,
                "original_error": compare(output, original) if finite else None,
                "reference_split_error": compare(reference, original), "selected_macs": total_macs,
                "other_calls": others, "attention_calls": attention}
            if not finite or (policy == "bf16" and not row["matched_error"]["identical"]):
                report["invalid_cases"].append({"image_id": image_id, "policy": policy})
            report["runs"].append(row)
            write_json(destination, report)
            print(image_id, policy, row["matched_error"]["relative_mse"] if finite else "nonfinite", flush=True)
    report["status"] = "invalid" if report["invalid_cases"] else "complete"
    write_json(destination, report)


if __name__ == "__main__":
    main()
