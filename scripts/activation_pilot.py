#!/usr/bin/env python3
"""Measure grouped activation-selection rules on pretrained vision MLPs."""

import argparse
import hashlib
import importlib.util
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


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "deps/tensor-inference-engine"
SPEC = importlib.util.spec_from_file_location("selection", ENGINE / "src/selection_reference.py")
selection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selection)
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


def load_data(weights, samples_per_class: int, split: str, batch_size: int):
    cache = Path.home() / ".cache/tensor-inference-experiments/datasets"
    dataset = Imagenette(cache, split="val", size="160px", download=True,
                         transform=weights.transforms())
    generator = random.Random(20261122)
    indices = []
    # Reserve eight images/class for operator diagnostics, disjoint from task evaluation.
    offset = 0 if split == "operator" else 8
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
        "settings": vars(args) | {"output": str(args.output)},
        "experiments": revision(ROOT), "engine": revision(ENGINE),
        "script_sha256": digest(Path(__file__)),
        "reference_sha256": digest(ENGINE / "src/selection_reference.py"),
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


def capture_inputs(model, layers, loader, rows_per_image: int) -> dict:
    captured = {name: [] for name in layers}

    def hook(name):
        def capture(module, args):
            values = args[0].detach().reshape(args[0].shape[0], -1, args[0].shape[-1])
            indices = torch.linspace(0, values.shape[1] - 1, rows_per_image,
                                     device=values.device).long()
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
                altered, indices = selection.select_2of4(x, q, gram, policy)
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
    batches = [(images.to(device="cuda", dtype=torch.bfloat16),
                torch.tensor([class_indices[label] for label in labels], device="cuda"))
               for images, labels in loader]
    records = []
    baseline_predictions = None
    for case in args.case:
        handles = []
        if case != "baseline":
            policy, bits_text = case.split(":")
            bits = int(bits_text)
            if policy not in (*POLICIES, "dense") or bits not in (4, 8, 16):
                raise ValueError(f"unsupported case {case}")

            def hook(name):
                def alter(module, inputs):
                    x = inputs[0]
                    q = (x.float() if bits == 16 else
                         selection.quantize_symmetric(x, bits, args.group_size))
                    q = q.bfloat16().float()
                    if policy != "dense":
                        q, _ = selection.select_2of4(x, q, metadata[name], policy)
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
    result.add_argument("--output", type=Path, required=True)
    return result


@torch.inference_mode()
def main():
    args = parser().parse_args()
    if min(args.samples_per_class, args.rows_per_image, args.batch_size) <= 0:
        raise ValueError("sample, row and batch counts must be positive")
    torch.manual_seed(20261122)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    model, weights, layers = load_model(args.model)
    report = make_report(args, weights)
    report["layers"] = geometry(layers)
    if args.phase == "operator":
        loader, _, manifest = load_data(weights, args.samples_per_class, "operator", args.batch_size)
        report["dataset"]["manifest"] = manifest
        inputs = capture_inputs(model, layers, loader, args.rows_per_image)
        report["operators"] = operator_measurements(layers, inputs, args)
    elif args.phase == "quality":
        args.case = args.case or ["baseline", "dense:4", "diagonal:16", "gram:16",
                                  "diagonal:4", "gram:4"]
        loader, class_indices, manifest = load_data(
            weights, args.samples_per_class, "quality", args.batch_size)
        report["dataset"]["manifest"] = manifest
        report["quality"] = quality_measurements(model, layers, loader, class_indices, args)
    write_json(args.output, report)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
