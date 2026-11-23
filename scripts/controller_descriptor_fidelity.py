#!/usr/bin/env python3
"""Describe replay outputs with a pinned DINOv3 model; this is not generation quality."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from activation_pilot import ENGINE, ROOT, digest, revision, write_json
from modern_models import vision


def image_from_values(values):
    pixels = (np.asarray(values) * 255).round().clip(0, 255).astype(np.uint8)
    if pixels.ndim != 3 or pixels.shape[-1] != 3:
        raise ValueError("expected RGB HWC image")
    return Image.fromarray(pixels)


def fidelity(value, reference):
    value, reference = np.asarray(value, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if value.ndim != 1 or value.shape != reference.shape or not np.isfinite(value).all():
        raise ValueError("finite matching descriptor vectors required")
    norms = float(np.linalg.norm(value) * np.linalg.norm(reference))
    if not norms or not np.isfinite(reference).all():
        raise ValueError("nonzero finite reference and descriptor required")
    return {"cosine_distance": max(0.0, 1 - float(np.dot(value, reference)) / norms),
            "relative_mse": float(np.sum((value - reference) ** 2) / np.sum(reference ** 2))}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, preprocessing, checkpoint = vision("dinov3-small", dtype=torch.float32)
    report = {"schema": 1, "status": "running", "experiments": revision(ROOT), "engine": revision(ENGINE),
        "script_sha256": digest(Path(__file__)), "model_adapter_sha256": digest(ROOT / "scripts/modern_models.py"),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "dtype": "float32",
        "tf32": False, "descriptor_checkpoint": checkpoint, "source_reports": [], "records": [],
        "references": [], "limitations": [
            "Descriptor consistency to the original image is not GenEval, FID, human preference or a semantic-quality guarantee",
            "Official pinned DINOv3 preprocessing consumes rounded/clipped 8-bit RGB for both reference and changed images",
            "All original inputs and repeated control variants retain identity; repetitions are not independent samples",
            "FP32 descriptor extraction is a diagnostic observer, not a deployed controller or target hardware cost",
        ]}
    cache, references = {}, {}

    def describe(image):
        tensor = preprocessing.transform(image).unsqueeze(0).to(device="cuda", dtype=torch.float32)
        result = model(tensor).float().cpu().numpy()[0]
        if result.ndim != 1 or not np.isfinite(result).all():
            raise ValueError("invalid DINO descriptor")
        return result

    for path in args.input:
        source = json.loads(path.read_text())
        if source.get("status") != "complete":
            raise ValueError("complete replay report required")
        report["source_reports"].append({"path": str(path), "sha256": digest(path), "bits": source["bits"]})
        for row in source["runs"]:
            key = (row["bundle"], row["bundle_input_index"])
            if key not in references:
                bundle = Path(row["bundle"])
                manifest = json.loads((bundle / "inputs.json").read_text())
                if manifest["status"] != "complete" or digest(bundle / "inputs.npz") != manifest["arrays_sha256"]:
                    raise ValueError("invalid original input bundle")
                with np.load(bundle / "inputs.npz", allow_pickle=False) as values:
                    image = values["images"][row["bundle_input_index"]]
                references[key] = describe(image_from_values(image))
                report["references"].append({"bundle": key[0], "bundle_input_index": key[1],
                    "arrays_sha256": manifest["arrays_sha256"], "descriptor": references[key].tolist()})
            arrays = Path(row["arrays"])
            if digest(arrays) != row["arrays_sha256"]:
                raise ValueError("changed replay output arrays")
            identity = row["arrays_sha256"]
            if identity not in cache:
                with np.load(arrays, allow_pickle=False) as values:
                    cache[identity] = describe(image_from_values(values["images"][0]))
            descriptor = cache[identity]
            report["records"].append({"source_report": str(path), "bits": source["bits"],
                **{field: row[field] for field in ("image_id", "bundle", "bundle_input_index", "policy", "backend", "control")},
                "arrays_sha256": identity, "descriptor": descriptor.tolist(),
                **fidelity(descriptor, references[key])})
            write_json(args.output, report)
        print(path, len(report["records"]), flush=True)
    report.update(status="complete", unique_described_outputs=len(cache))
    write_json(args.output, report)


if __name__ == "__main__":
    main()
