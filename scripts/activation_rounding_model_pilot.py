#!/usr/bin/env python3
"""Evaluate current-input rounding through complete pretrained vision graphs."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from activation_pilot import ENGINE, ROOT, digest, load_data, revision, write_json
from activation_rounding_models import POLICIES, modified, prepare_layers
from modern_models import vision
from paired_precision_pilot import receiving_layers


def compare(output: torch.Tensor, reference: torch.Tensor) -> dict:
    current, target = output.double(), reference.double()
    error = current - target
    error_energy, target_energy = float(error.square().sum()), float(target.square().sum())
    cosine = float(torch.nn.functional.cosine_similarity(current, target, dim=-1)[0])
    return {"squared_error": error_energy,
            "relative_mse": error_energy / target_energy if target_energy else None,
            "cosine_distance": max(0.0, min(2.0, 1 - cosine)),
            "maximum_error": float(error.abs().amax()), "identical": torch.equal(output, reference)}


def save_geometry(path: Path, prepared: dict) -> None:
    arrays = {}
    for name, geometries in prepared.items():
        for rank, geometry in geometries.items():
            for field in ("factor", "diagonal", "residual_norm", "normalizer", "eigenvalues"):
                arrays[f"{name}-rank{rank}-{field}"] = getattr(geometry, field).cpu().numpy()
    np.savez_compressed(path, **arrays)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("dinov3-small", "convnextv2-tiny"), default="dinov3-small")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--offset", type=int, default=250)
    parser.add_argument("--policy", choices=POLICIES, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1 or args.offset < 0:
        parser.error("positive count and nonnegative allocation offset required")
    policies = args.policy or list(POLICIES)
    if args.model != "dinov3-small" and any(policy.startswith("qkv") for policy in policies):
        parser.error("QKV role policies require the transformer model")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    args.output.mkdir(parents=True, exist_ok=False)
    model, data, checkpoint = vision(args.model)
    layers = receiving_layers(model)
    if any(not isinstance(layer, torch.nn.Linear) for layer in layers.values()):
        raise ValueError("this model protocol requires Linear receivers")
    prepared = prepare_layers(layers)
    loader, classes, manifest = load_data(data, math.ceil(args.count / 10), "quality", 1, args.offset)
    geometry_path = args.output / "geometry.npz"
    save_geometry(geometry_path, prepared)
    report = {"schema": 1, "status": "running", "model": args.model,
        "experiments": revision(ROOT), "engine": revision(ENGINE), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(), "checkpoint": checkpoint, "dataset": manifest,
        "count": args.count, "allocation_offset": args.offset, "policies": policies,
        "selected_layers": list(layers), "geometry": str(geometry_path), "geometry_sha256": digest(geometry_path),
        "source_sha256": {name: digest(ROOT / "scripts" / name) for name in (
            "activation_rounding_model_pilot.py", "activation_rounding_models.py",
            "activation_rounding.py", "modern_models.py", "activation_pilot.py")},
        "runs": [], "invalid_cases": [], "limitations": [
            "Paired fixed dataset allocation, not population accuracy or an independent model-training experiment",
            "Model role selects policy shape; no activation calibration, tuning labels or future inputs select rounding",
            "QKV policies use lower precision in every selected QKV projection and nearest INT8 elsewhere",
            "Weights remain BF16 and reconstructed activation values feed actual Torch consumers",
            "Other operators retain Torch arithmetic, not a complete device-arithmetic reference",
            "DINOv3 output is a descriptor; cosine/MSE are fidelity diagnostics, not classification accuracy",
            "Per-call bits and MACs describe transformed work; they are not energy savings or native INT throughput",
            "No software timing or target hardware energy/latency claim"]}
    destination = args.output / "report.json"
    for image_id, (images, labels) in enumerate(loader):
        if image_id >= args.count:
            break
        image, target = images.cuda().bfloat16(), classes[int(labels[0])]
        reference = model(image).float().cpu()
        for policy in policies:
            with modified(layers, prepared, policy) as calls:
                output = model(image).float().cpu()
            finite = bool(torch.isfinite(output).all())
            row = {"image_id": image_id, "target": target, "policy": policy, "finite": finite,
                   "output": output.tolist(), "reference": reference.tolist(), "calls": calls}
            if finite:
                row.update(compare(output, reference))
                if args.model == "convnextv2-tiny":
                    row.update(prediction=int(output.argmax()), reference_prediction=int(reference.argmax()),
                               correct=int(output.argmax()) == target, reference_correct=int(reference.argmax()) == target)
            if not finite or (policy == "bf16" and not row["identical"]):
                report["invalid_cases"].append({"image_id": image_id, "policy": policy})
            report["runs"].append(row)
            write_json(destination, report)
            print(args.model, image_id, policy, row.get("relative_mse"), row.get("cosine_distance"), flush=True)
    report["status"] = "invalid" if report["invalid_cases"] else "complete"
    write_json(destination, report)


if __name__ == "__main__":
    main()
