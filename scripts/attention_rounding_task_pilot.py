#!/usr/bin/env python3
"""Class-balanced frozen-feature task test of live attention rounding policies."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from activation_pilot import ENGINE, ROOT, digest, revision, write_json
from activation_rounding_model_pilot import compare
from activation_rounding_models import modified
from attention_rounding import POLICIES, prepare, split_attention
from feature_benchmark import dataset, nearest
from modern_models import vision
from paired_precision_pilot import receiving_layers


DEFAULT_POLICIES = ("bf16", "uniform8", "uniform4-all", "qkv4-nearest", "qkv4-raw-r1",
                    "qkv4-balanced-r4", "qk4-v8", "q4-k8-v8", "q8-k4-v8")


def modes(policy: str) -> tuple[str, str]:
    if policy == "uniform4-all":
        return "uniform4", "qkv4-nearest"
    return ("bf16" if policy == "bf16" else "uniform8"), policy


def compressed_record(path: Path, record: dict) -> str:
    payload = (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode()
    path.write_bytes(gzip.compress(payload, compresslevel=6, mtime=0))
    return hashlib.sha256(payload).hexdigest()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery-per-class", type=int, default=20)
    parser.add_argument("--test-per-class", type=int, default=1)
    parser.add_argument("--test-offset", type=int, default=0)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--policy", action="append", choices=(*POLICIES, "uniform4-all"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.gallery_per_class, args.test_per_class, args.neighbors) < 1 or args.test_offset < 0:
        parser.error("positive class/neighbor counts and nonnegative offset required")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    policies = args.policy or list(DEFAULT_POLICIES)
    if len(set(policies)) != len(policies):
        parser.error("duplicate policies")
    settings = {"gallery_per_class": args.gallery_per_class, "test_per_class": args.test_per_class,
                "test_offset": args.test_offset, "neighbors": args.neighbors, "policies": policies}
    args.output.mkdir(parents=True, exist_ok=args.resume)
    (args.output / "queries").mkdir(exist_ok=args.resume)
    model, data, checkpoint = vision("dinov3-small")
    layers = receiving_layers(model)
    geometries = prepare(layers)
    gallery_loader, gallery_manifest = dataset(data, train=True, per_class=args.gallery_per_class, batch_size=16)
    test_loader, test_manifest = dataset(data, train=False, per_class=args.test_per_class, offset=args.test_offset)
    hashes = {name: digest(ROOT / "scripts" / name) for name in (
        "attention_rounding_task_pilot.py", "attention_rounding.py", "activation_rounding.py",
        "activation_rounding_models.py", "feature_benchmark.py", "modern_models.py",
        "activation_rounding_model_pilot.py", "paired_precision_pilot.py", "activation_pilot.py")}
    destination = args.output / "report.json"
    if args.resume:
        report = json.loads(destination.read_text())
        if (report["settings"] != settings or report["checkpoint"] != checkpoint
                or report["source_sha256"] != hashes or report["gallery_dataset"] != gallery_manifest
                or report["test_dataset"] != test_manifest):
            raise ValueError("resumed source, settings or dataset changed")
        gallery_path = Path(report["gallery"])
        if digest(gallery_path) != report["gallery_sha256"]:
            raise ValueError("gallery changed")
        with np.load(gallery_path, allow_pickle=False) as arrays:
            gallery = torch.from_numpy(arrays["features"].copy())
            gallery_labels = torch.from_numpy(arrays["labels"].copy())
        for row in report["queries"]:
            path = args.output / row["file"]
            if digest(path) != row["archive_sha256"]:
                raise ValueError("completed query changed")
    else:
        gallery_features, gallery_targets = [], []
        with modified(layers, {}, "bf16", record=False), split_attention(layers, {}, "bf16", record=False):
            for images, labels in gallery_loader:
                gallery_features.append(model(images.cuda().bfloat16()).float().cpu())
                gallery_targets.append(labels)
        gallery, gallery_labels = torch.cat(gallery_features), torch.cat(gallery_targets)
        if not torch.isfinite(gallery).all():
            raise ValueError("nonfinite reference gallery")
        gallery_path = args.output / "gallery.npz"
        np.savez_compressed(gallery_path, features=gallery.numpy(), labels=gallery_labels.numpy())
        report = {"schema": 1, "status": "running", "model": "dinov3-small",
            "experiments": revision(ROOT), "engine": revision(ENGINE), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(), "settings": settings, "checkpoint": checkpoint,
            "source_sha256": hashes, "gallery_dataset": gallery_manifest, "test_dataset": test_manifest,
            "gallery": str(gallery_path), "gallery_sha256": digest(gallery_path),
            "queries": [], "invalid_cases": [], "limitations": [
                "Frozen-feature CIFAR100 kNN task, not official ImageNet/DINOv3 benchmark accuracy",
                "Labeled train gallery is used only by the evaluator, never by quantizer preparation or decisions",
                "Evaluation labels do not select scales, ranks, policies or component precision",
                "Current input activations feed every actual modified graph; weights remain BF16",
                "All cases use matched Q/K/V output splits; original and split references are retained",
                "Reference gallery uses batch16, query graphs batch1; same gallery is fixed for every policy",
                "Declared precision and operation counts are not native packed throughput or device energy",
                "Exploratory policies and test allocations are explicit; a separate replication is needed for a selected winner"]}
        write_json(destination, report)
    completed = len(report["queries"])
    for image_id, (images, labels) in enumerate(test_loader):
        if image_id < completed:
            continue
        image, target = images.cuda().bfloat16(), int(labels[0])
        original = model(image).float().cpu()
        with modified(layers, {}, "bf16", record=False), split_attention(layers, {}, "bf16", record=False):
            reference = model(image).float().cpu()
        predictions, reference_neighbors, reference_similarity = nearest(reference, gallery, gallery_labels, args.neighbors)
        reference_prediction = int(predictions[0])
        query = {"image_id": image_id, "target": target, "dataset_index": test_manifest["indices"][image_id],
                 "split_reference": reference.tolist(), "original_reference": original.tolist(),
                 "reference_prediction": reference_prediction, "reference_neighbors": reference_neighbors.tolist(),
                 "reference_similarity": reference_similarity.tolist(), "reference_split_error": compare(reference, original),
                 "runs": []}
        compact = {"image_id": image_id, "target": target, "reference_prediction": reference_prediction,
                   "predictions": {}, "relative_mse": {}}
        for policy in policies:
            other, attention = modes(policy)
            with modified(layers, {}, other) as other_calls, split_attention(layers, geometries, attention) as attention_calls:
                output = model(image).float().cpu()
            finite = bool(torch.isfinite(output).all())
            row = {"policy": policy, "finite": finite, "output": output.tolist(),
                   "other_calls": other_calls, "attention_calls": attention_calls}
            if finite:
                predicted, neighbors, similarities = nearest(output, gallery, gallery_labels, args.neighbors)
                row.update(prediction=int(predicted[0]), correct=int(predicted[0]) == target,
                           neighbors=neighbors.tolist(), similarity=similarities.tolist(),
                           matched_error=compare(output, reference), original_error=compare(output, original))
                compact["predictions"][policy] = row["prediction"]
                compact["relative_mse"][policy] = row["matched_error"]["relative_mse"]
            if not finite or (policy == "bf16" and not row["matched_error"]["identical"]):
                report["invalid_cases"].append({"image_id": image_id, "policy": policy})
            query["runs"].append(row)
        relative = Path("queries") / f"{image_id:05d}.json.gz"
        path = args.output / relative
        compact.update(file=relative.as_posix(), payload_sha256=compressed_record(path, query), archive_sha256=digest(path))
        report["queries"].append(compact)
        write_json(destination, report)
        if (image_id + 1) % 10 == 0:
            print(image_id + 1, {policy: sum(row["predictions"].get(policy) == row["target"] for row in report["queries"])
                               for policy in policies}, flush=True)
    expected = 100 * args.test_per_class
    if len(report["queries"]) != expected:
        raise ValueError("incomplete test allocation")
    report["status"] = "invalid" if report["invalid_cases"] else "complete"
    write_json(destination, report)


if __name__ == "__main__":
    main()
