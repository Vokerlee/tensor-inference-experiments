#!/usr/bin/env python3
"""Measure grouped activation-selection rules on pretrained vision MLPs."""

import argparse
import hashlib
import json
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import Imagenette
from torchvision.models import Swin_T_Weights, ViT_B_16_Weights, swin_t, vit_b_16


from activation_policies import (ENGINE, ROOT, adaptive_representation, common_representation,
                                 packed_gram, selection, sparse_selection)


MODELS = {
    "vit_b_16": (vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1),
    "swin_t": (swin_t, Swin_T_Weights.IMAGENET1K_V1),
}
POLICIES = ("magnitude", "norm", "diagonal", "gram", "gated")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def revision(directory: Path) -> dict:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def quantiles(values: torch.Tensor) -> dict:
    fractions = torch.tensor([0., .25, .5, .75, .95, 1.], device=values.device)
    return dict(zip(("min", "q25", "median", "q75", "q95", "max"),
                    values.float().quantile(fractions).cpu().tolist()))


def load_model(name: str):
    factory, weights = MODELS[name]
    model = factory(weights=weights).eval().to(device="cuda", dtype=torch.bfloat16)
    layers = {name: module for name, module in model.named_modules()
              if isinstance(module, torch.nn.Linear) and ".mlp." in name}
    return model, weights, layers


def load_data(weights, samples_per_class: int, split: str, batch_size: int, holdout_offset: int = 0):
    cache = Path.home() / ".cache/tensor-inference-experiments/datasets"
    dataset = Imagenette(cache, split="val", size="160px", download=True,
                         transform=weights.transforms())
    generator = random.Random(20261122)
    indices = []
    # Reserve eight images/class for operator diagnostics, disjoint from task evaluation.
    offset = 0 if split == "operator" else 8 + holdout_offset
    for category in range(10):
        candidates = [index for index, (_, label) in enumerate(dataset._samples) if label == category]
        generator.shuffle(candidates)
        chosen = candidates[offset:offset + samples_per_class]
        if len(chosen) != samples_per_class or (split == "operator" and samples_per_class > 8):
            raise ValueError("requested split exceeds the declared disjoint sample allocation")
        indices.extend(chosen)
    generator.shuffle(indices)
    categories = weights.meta["categories"]
    class_indices = [categories.index(names[0]) for names in dataset.classes]
    manifest = [{"path": str(Path(dataset._samples[index][0]).relative_to(cache)),
                 "target": class_indices[dataset._samples[index][1]]} for index in indices]
    return (DataLoader(Subset(dataset, indices), batch_size=batch_size, num_workers=2,
                       pin_memory=True), class_indices, manifest)


def make_report(args, weights) -> dict:
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
    return {
        "schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()},
        "experiments": revision(ROOT), "engine": revision(ENGINE),
        "script_sha256": digest(Path(__file__)),
        "reference_sha256": digest(ENGINE / "src/selection_reference.py"),
        "policy_sha256": digest(ROOT / "scripts/activation_policies.py"),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(), "dtype": "bfloat16",
                        "tf32": False, "bf16_reduced_precision_reduction": False},
        "checkpoint": {"url": weights.url, "sha256": digest(checkpoint)},
        "dataset": {"name": "Imagenette2-160 validation", "source_resolution": 160,
                    "archive_md5": Imagenette._ARCHIVES["160px"][1],
                    "preprocessing": str(weights.transforms())},
        "limitations": ["Reconstructed INT values: not packed low-bit execution",
                        "FP32 scalar reference timing is not hardware-controller cost",
                        "No target-accelerator energy measurement",
                        "Only MLP linear layers are transformed; weights stay fixed BF16"],
        "layers": [],
    }


def geometry(layers: dict) -> list:
    results = []
    for name, layer in layers.items():
        weight = layer.weight.detach().t().contiguous()
        gram = selection.group_gram(weight)
        rho = selection.coherence(gram)
        record = {"name": name, "K": weight.shape[0], "N": weight.shape[1],
                  "coherence": quantiles(rho),
                  "rich_fraction_delta_0_1": (rho > .1 / 2.1).float().mean().item(),
                  "full_metadata_bytes": (weight.shape[0] // 4) * 10 * 4,
                  "bf16_weight_bytes": weight.numel() * 2}
        results.append(record)
    return results


def capture_inputs(model, layers, loader, rows_per_image: int | None) -> dict:
    captured = {name: [] for name in layers}

    def hook(name):
        def capture(module, args):
            values = args[0].detach().reshape(args[0].shape[0], -1, args[0].shape[-1])
            indices = (torch.arange(values.shape[1], device=values.device) if rows_per_image is None
                       else torch.linspace(0, values.shape[1] - 1, rows_per_image,
                                           device=values.device).long())
            captured[name].append(values[:, indices].reshape(-1, values.shape[-1]).cpu())
        return capture

    handles = [layer.register_forward_pre_hook(hook(name)) for name, layer in layers.items()]
    try:
        for images, _ in loader:
            model(images.to(device="cuda", dtype=torch.bfloat16))
    finally:
        for handle in handles:
            handle.remove()
    return {name: torch.cat(chunks).to("cuda") for name, chunks in captured.items()}


def operator_measurements(layers, inputs, args):
    records = []
    for name, layer in layers.items():
        weight = layer.weight.detach().t().contiguous().float()
        gram = selection.group_gram(weight)
        packed = packed_gram(gram)
        x = inputs[name].float()
        target_energy = (x @ weight).square().sum().item()
        record = {"name": name, "rows": x.shape[0], "K": weight.shape[0],
                  "N": weight.shape[1], "cases": []}
        for bits in args.bits:
            q = x if bits == 16 else selection.quantize_symmetric(x, bits, args.group_size)
            # Match the actual model-input cast, including scale reconstruction rounding.
            q = q.bfloat16().float()
            reference_indices = None
            for policy in POLICIES:
                altered, indices = sparse_selection(x, q, gram, packed, policy, args.selector_backend)
                error = altered - x
                output_error = (error @ weight).square().sum(dim=-1)
                group_error = selection.group_loss(error, gram).sum(dim=-1)
                if policy == "diagonal":
                    reference_indices = indices
                case = {"bits": bits, "policy": policy,
                        "output_relative_mse": output_error.sum().item() / target_energy,
                        "sum_group_relative_mse": group_error.sum().item() / target_energy,
                        "row_output_error": output_error.cpu().tolist()}
                if reference_indices is not None:
                    case["changed_vs_diagonal"] = (indices != reference_indices).float().mean().item()
                record["cases"].append(case)
        records.append(record)
        print(json.dumps({"operator": name, "cases": [
            {k: v for k, v in case.items() if k != "row_output_error"}
            for case in record["cases"]]}), flush=True)
    return records


def quality_measurements(model, layers, loader, class_indices, args):
    """Paired whole-model evaluations; no thresholds fitted to these labels."""
    metadata = {name: selection.group_gram(layer.weight.detach().t().contiguous())
                for name, layer in layers.items()}
    packed = {name: packed_gram(gram) for name, gram in metadata.items()}
    batches = [(images.to(device="cuda", dtype=torch.bfloat16),
                torch.tensor([class_indices[label] for label in labels], device="cuda"))
               for images, labels in loader]
    records = []
    baseline_predictions = None
    for case in args.case:
        handles = []
        mode_counts = torch.zeros(6, device="cuda", dtype=torch.int64)
        mode_macs = torch.zeros_like(mode_counts)
        common_calls = torch.zeros(2, device="cuda", dtype=torch.int64)
        if case != "baseline":
            policy, bits_text = case.split(":")
            adaptive = policy.startswith(("adaptive-", "precision-", "sparsity-", "common-"))
            bits = 16 if adaptive else int(bits_text)
            if not adaptive and (policy not in (*POLICIES, "dense", "affine", "fp4") or bits not in (4, 8, 16)):
                raise ValueError(f"unsupported case {case}")

            def hook(name):
                def alter(module, inputs):
                    x = inputs[0]
                    if policy.startswith("common-"):
                        conditional = policy.startswith("common-adaptive-")
                        if not conditional and bits_text != "16":
                            raise ValueError("fixed common masks currently support BF16 only")
                        score = policy.rsplit("-", 1)[1]
                        q, counts, elements = common_representation(
                            x, metadata[name], score, float(bits_text) if conditional else None)
                        mode_counts.add_(counts)
                        mode_macs.add_(elements * module.out_features)
                        dense_k = (x.shape[-1] + 127) // 128
                        sparse_k = (x.shape[-1] // 2 + 127) // 128
                        n_tiles = (module.out_features + 127) // 128
                        dense_windows, sparse_windows = counts[0] // dense_k, counts[5] // dense_k
                        common_calls[0].add_((dense_windows + sparse_windows) * dense_k * n_tiles)
                        common_calls[1].add_((dense_windows * dense_k + sparse_windows * sparse_k) * n_tiles)
                        return (q.to(x.dtype), *inputs[1:])
                    if adaptive:
                        family, score = policy.split("-", 1)
                        q, counts, elements = adaptive_representation(
                            x, metadata[name], packed[name], family, score,
                            float(bits_text), args)
                        mode_counts.add_(counts)
                        mode_macs.add_(elements * module.out_features)
                        return (q.to(x.dtype), *inputs[1:])
                    if (args.selector_backend in {"cuda", "cuda-tile"} and bits in (4, 8)
                            and policy in {"dense", "diagonal", "gram"}):
                        mode = {"dense": 0, "diagonal": 1, "gram": 2}[policy]
                        q, _ = torch.ops.tensor_inference_engine.prepare_activations(
                            x.reshape(-1, x.shape[-1]).contiguous(),
                            None if mode == 0 else packed[name], bits, mode)
                        return (q.reshape_as(x), *inputs[1:])
                    q = (x.float() if bits == 16 else
                         selection.quantize_symmetric(x, bits, args.group_size))
                    if policy == "affine":
                        q = selection.quantize_affine(x, bits, args.group_size)
                    elif policy == "fp4":
                        q = selection.quantize_e2m1(x, args.group_size)
                    q = q.bfloat16().float()
                    if policy not in {"dense", "affine", "fp4"}:
                        q, _ = sparse_selection(x, q, metadata[name], packed[name],
                                                policy, args.selector_backend)
                    return (q.to(x.dtype), *inputs[1:])
                return alter

            handles = [layer.register_forward_pre_hook(hook(name)) for name, layer in layers.items()
                       if args.scope == "all" or name.endswith(".3")]
        predictions, targets = [], []
        restricted_predictions = []
        try:
            for images, labels in batches:
                logits = model(images).float()
                predictions.extend(logits.argmax(-1).cpu().tolist())
                restricted_predictions.extend(
                    logits[:, class_indices].argmax(-1).cpu().tolist())
                targets.extend(labels.cpu().tolist())
        finally:
            for handle in handles:
                handle.remove()
        if case == "baseline":
            baseline_predictions = predictions
        correct = [prediction == target for prediction, target in zip(predictions, targets)]
        restricted_correct = [class_indices[prediction] == target
                              for prediction, target in zip(restricted_predictions, targets)]
        record = {"case": case, "scope": args.scope, "count": len(targets),
                  "top1_imagenet1000": sum(correct) / len(correct),
                  "top1_restricted10": sum(restricted_correct) / len(correct),
                  "predictions": predictions, "targets": targets,
                  "correct": correct}
        if mode_counts.sum() > 0:
            names = ["bf16_dense", "int8_dense", "int8_sparse",
                     f"{args.low_format}_dense", f"{args.low_format}_sparse",
                     "bf16_sparse"]
            record["mode_tile_counts"] = dict(zip(names, mode_counts.cpu().tolist()))
            record["mode_dense_equivalent_macs"] = dict(zip(names, mode_macs.cpu().tolist()))
        if common_calls[0] > 0:
            record["common_array_calls_dense_vs_selected"] = common_calls.cpu().tolist()
        if baseline_predictions is not None:
            record["agreement_with_baseline"] = sum(
                a == b for a, b in zip(predictions, baseline_predictions)) / len(targets)
        records.append(record)
        print(json.dumps({key: value for key, value in record.items()
                          if key not in ("predictions", "targets", "correct")}), flush=True)
    return records


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", choices=MODELS, required=True)
    result.add_argument("--phase", choices=["geometry", "operator", "quality"], required=True)
    result.add_argument("--samples-per-class", type=int, default=4)
    result.add_argument("--rows-per-image", type=int, default=8)
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument("--bits", type=int, nargs="+", default=[16, 8, 4], choices=[4, 8, 16])
    result.add_argument("--group-size", type=int, default=32)
    result.add_argument("--scope", choices=["all", "down"], default="all")
    result.add_argument("--case", action="append", default=None,
                        help="repeat baseline or POLICY:BITS; includes dense:4/8")
    result.add_argument("--selector-backend", choices=["reference", "cuda", "cuda-tile"], default="reference")
    result.add_argument("--holdout-offset", type=int, default=0,
                        help="skip additional images/class after the eight diagnostic images")
    result.add_argument("--mode-rows", type=int, choices=[16, 32, 64, 128], default=128,
                        help="precision granularity; below128 requires different format support")
    result.add_argument("--low-format", choices=["int4", "affine4", "e2m1"], default="int4")
    result.add_argument("--library", type=Path,
                        default=ENGINE / "build/libtensor_inference_engine_torch.so")
    result.add_argument("--output", type=Path, required=True)
    return result


@torch.inference_mode()
def main():
    args = parser().parse_args()
    if min(args.samples_per_class, args.rows_per_image, args.batch_size) <= 0:
        raise ValueError("sample, row and batch counts must be positive")
    if args.holdout_offset < 0:
        raise ValueError("holdout offset must be nonnegative")
    torch.manual_seed(20261122)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if args.selector_backend in {"cuda", "cuda-tile"}:
        torch.ops.load_library(str(args.library.resolve()))
    if args.phase == "quality":
        args.case = args.case or ["baseline", "dense:4", "diagonal:16", "gram:16",
                                  "diagonal:4", "gram:4"]
    model, weights, layers = load_model(args.model)
    report = make_report(args, weights)
    report["layers"] = geometry(layers)
    if args.phase == "operator":
        loader, _, manifest = load_data(weights, args.samples_per_class, "operator", args.batch_size)
        report["dataset"]["manifest"] = manifest
        inputs = capture_inputs(model, layers, loader, args.rows_per_image)
        report["operators"] = operator_measurements(layers, inputs, args)
    elif args.phase == "quality":
        loader, class_indices, manifest = load_data(
            weights, args.samples_per_class, "quality", args.batch_size, args.holdout_offset)
        report["dataset"]["manifest"] = manifest
        report["quality"] = quality_measurements(model, layers, loader, class_indices, args)
    write_json(args.output, report)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
