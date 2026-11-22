#!/usr/bin/env python3
"""Paired class-conditional DiT trajectories with runtime MLP transformations."""

import argparse
import json
import shutil
from pathlib import Path

import diffusers
import numpy as np
import torch
from diffusers import DiTPipeline
from huggingface_hub import snapshot_download
from PIL import Image

from activation_pilot import (ENGINE, ROOT, adaptive_representation, digest,
                              packed_gram, revision, selection, sparse_selection, write_json)


REVISION = "eab87f77abd5aef071a632f08807fbaab0b704d0"


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--classes", type=int, nargs="+", default=[0, 217, 482, 491, 497, 566, 569, 571])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--case", action="append", default=None)
    parser.add_argument("--low-format", choices=["int4", "affine4", "e2m1"], default="int4")
    parser.add_argument("--selector-backend", choices=["cuda", "cuda-tile"], default="cuda")
    parser.add_argument("--reference-run", type=Path,
                        help="reuse an identically configured baseline, only for paired diagnostics")
    args = parser.parse_args()
    args.case = args.case or ["baseline", "dense:4", "diagonal:4", "gram:4",
                              "adaptive-diagonal:0.01", "adaptive-gram:0.01"]
    args.group_size = 32
    if (not args.reference_run and args.case[0] != "baseline") or min(args.steps, args.batch_size) <= 0:
        parser.error("baseline must be first without --reference-run; steps and batch size must be positive")
    reference = None
    if args.reference_run:
        reference = json.loads((args.reference_run / "report.json").read_text())
        expected = {"steps": args.steps, "seed": args.seed, "classes": args.classes,
                    "batch_size": args.batch_size}
        if reference["revision"] != REVISION or any(
                reference["settings"][key] != value for key, value in expected.items()):
            parser.error("reference checkpoint, steps, seed, classes and batch size must match")
        if "baseline" in args.case:
            parser.error("omit baseline from cases when reusing --reference-run")
    if any(label < 0 or label > 999 for label in args.classes):
        parser.error("class indices must be in [0,999]")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    library = ENGINE / "build/libtensor_inference_engine_torch.so"
    torch.ops.load_library(str(library))
    snapshot = Path(snapshot_download("facebook/DiT-XL-2-256", revision=REVISION,
                                      allow_patterns=["*.json", "*.bin", "*.safetensors"]))
    pipe = DiTPipeline.from_pretrained(snapshot, torch_dtype=torch.bfloat16).to("cuda")
    pipe.transformer.eval()
    pipe.vae.eval()
    pipe.set_progress_bar_config(disable=True)
    layers = {name: layer for name, layer in pipe.transformer.named_modules()
              if isinstance(layer, torch.nn.Linear) and ".ff." in name}
    metadata = {name: selection.group_gram(layer.weight.t().contiguous()) for name, layer in layers.items()}
    packed = {name: packed_gram(gram) for name, gram in metadata.items()}
    report = {"schema": 1, "model": "facebook/DiT-XL-2-256", "revision": REVISION,
              "experiments": revision(ROOT), "engine": revision(ENGINE),
              "script_sha256": digest(Path(__file__)), "library_sha256": digest(library),
              "policy_sha256": digest(ROOT / "scripts/activation_policies.py"),
              "checkpoint_files": {str(path.relative_to(snapshot)): digest(path)
                                   for path in snapshot.rglob("*") if path.is_file()},
              "settings": {key: str(value) if isinstance(value, Path) else value
                           for key, value in vars(args).items()},
              "environment": {"torch": torch.__version__, "diffusers": diffusers.__version__,
                              "gpu": torch.cuda.get_device_name(), "dtype": "bfloat16"},
              "scheduler": dict(pipe.scheduler.config), "mlp_layers": list(layers), "cases": [],
              "mode_names": ["bf16_dense", "int8_dense", "int8_sparse",
                             f"{args.low_format}_dense", f"{args.low_format}_sparse", "bf16_sparse"],
              "limitations": ["Small paired trajectory diagnostic, not FID or a distribution-quality ranking",
                              "Input reconstruction followed by ordinary BF16 computation",
                              "No ASIC energy, throughput or calibration claim"]}
    baseline_images, baseline_states = None, None
    if reference is not None:
        scheduler = {key: value for key, value in report["scheduler"].items()
                     if not key.startswith("_")}
        reference_scheduler = {key: value for key, value in reference["scheduler"].items()
                               if not key.startswith("_")}
        if (reference["checkpoint_files"] != report["checkpoint_files"]
                or reference["environment"] != report["environment"]
                or reference_scheduler != scheduler):
            raise ValueError("reference checkpoint bytes, environment and scheduler must match")
        source = args.reference_run / "baseline.npz"
        with np.load(source) as data:
            baseline_images, baseline_states = data["images"], data["states"]
        if baseline_images.shape[0] != len(args.classes) or baseline_states.shape[:2] != (
                args.steps, len(args.classes)):
            raise ValueError("reference image/trajectory counts do not match")
        shutil.copyfile(source, args.output / "baseline.npz")
        report["reference"] = {"report_sha256": digest(args.reference_run / "report.json"),
                               "array_sha256": digest(source)}
        report["cases"].append(next(case.copy() for case in reference["cases"]
                                    if case["case"] == "baseline"))
    for case in args.case:
        handles = []
        counts = torch.zeros(6, device="cuda", dtype=torch.int64)
        macs = torch.zeros_like(counts)
        if case != "baseline":
            policy, parameter = case.split(":")

            def hook(name):
                def alter(module, inputs):
                    x = inputs[0]
                    if policy.startswith(("adaptive-", "precision-", "sparsity-")):
                        family, score = policy.split("-", 1)
                        output, selected, elements = adaptive_representation(
                            x, metadata[name], packed[name], family, score,
                            float(parameter), args)
                        counts.add_(selected)
                        macs.add_(elements * module.out_features)
                    else:
                        bits = int(parameter)
                        if policy == "affine":
                            output = selection.quantize_affine(x, bits)
                        elif policy == "fp4":
                            output = selection.quantize_e2m1(x)
                        elif policy in {"sparse-residual", "double-sparse", "double-sparse-p2"}:
                            if bits != 4:
                                raise ValueError("residual baselines require four-bit streams")
                            parts = selection.sparse_residual_e2m1(
                                x, both_sparse=policy != "sparse-residual",
                                power_two=policy == "double-sparse-p2")
                            output = parts.coarse + parts.residual
                        else:
                            mode = {"dense": 0, "diagonal": 1, "gram": 2}[policy]
                            output, _ = torch.ops.tensor_inference_engine.prepare_activations(
                                x.reshape(-1, x.shape[-1]).contiguous(),
                                None if mode == 0 else packed[name], bits, mode)
                            output = output.reshape(x.shape)
                    return (output.to(x.dtype), *inputs[1:])
                return alter

            handles = [layer.register_forward_pre_hook(hook(name)) for name, layer in layers.items()]
        image_chunks, states = [], []
        try:
            for start in range(0, len(args.classes), args.batch_size):
                labels = args.classes[start:start + args.batch_size]
                batch_states = []

                def observe(module, inputs):
                    batch_states.append(inputs[0][:len(labels)].detach().float().cpu())

                observer = pipe.transformer.register_forward_pre_hook(observe)
                try:
                    images = pipe(class_labels=labels, num_inference_steps=args.steps,
                                  guidance_scale=4.0, output_type="np",
                                  generator=torch.Generator("cuda").manual_seed(args.seed + start)).images
                finally:
                    observer.remove()
                image_chunks.append(images)
                states.append(torch.stack(batch_states).numpy())
                print(f"{case}: {start + len(labels)}/{len(args.classes)} images", flush=True)
        finally:
            for handle in handles:
                handle.remove()
        images = np.concatenate(image_chunks)
        trajectory = np.concatenate(states, axis=1)
        safe_name = case.replace(":", "-")
        np.savez_compressed(args.output / f"{safe_name}.npz", images=images, states=trajectory)
        for index, image in enumerate(images):
            Image.fromarray(np.rint(image * 255).astype(np.uint8)).save(
                args.output / f"{safe_name}-{index:02}.png")
        record = {"case": case, "image_count": len(images),
                  "mode_tile_counts": counts.cpu().tolist(),
                  "mode_dense_equivalent_macs": macs.cpu().tolist()}
        if case == "baseline":
            baseline_images, baseline_states = images, trajectory
        else:
            mse = ((images.astype(np.float64) - baseline_images) ** 2).mean(axis=(1, 2, 3))
            state_error = ((trajectory.astype(np.float64) - baseline_states) ** 2).sum(axis=(2, 3, 4))
            state_energy = (baseline_states.astype(np.float64) ** 2).sum(axis=(2, 3, 4))
            record["pixel_mse_vs_baseline"] = mse.tolist()
            record["pixel_psnr_vs_baseline"] = (-10 * np.log10(mse.clip(1e-30))).tolist()
            record["trajectory_relative_mse_by_step"] = (state_error.sum(1) / state_energy.sum(1)).tolist()
        report["cases"].append(record)
        write_json(args.output / "report.json", report)
        print(record, flush=True)


if __name__ == "__main__":
    main()
