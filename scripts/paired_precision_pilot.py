#!/usr/bin/env python3
"""Execute paired precision policies in pinned vision and diffusion models."""

import argparse
import math
import platform
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from activation_pilot import ENGINE, ROOT, digest, load_data, revision, write_json
from modern_models import VISION, sana_pipeline, vision
from paired_precision_reference import POLICIES, Counters, multiply, prepare_weight
from precision_moment_diagnostics import MomentDiagnostics


DEFAULT_CASES = ("original", "bf16", "fp8-paired", "fp8-split",
                 "adaptive-paired:0.02", "adaptive-paired:0.04", "adaptive-paired:0.06",
                 "adaptive-split:0.04")
PROMPTS = (
    "A red ceramic teapot on a wooden table, soft daylight, detailed photograph.",
    "A yellow bicycle parked beside a blue door on a quiet cobblestone street.",
    "Two small sailboats on a lake at sunrise, mountains in the distance.",
    "A small robot reading a book in a greenhouse full of plants, detailed illustration.",
)


def receiving_layers(model):
    return {name: module for name, module in model.named_modules()
            if ((isinstance(module, torch.nn.Linear) and module.in_features > 128)
                or (isinstance(module, torch.nn.Conv2d) and module.in_channels > 128
                    and module.kernel_size == (1, 1) and module.stride == (1, 1)
                    and module.padding == (0, 0) and module.groups == 1))}


def activation_summary(values):
    flat = values.detach().float().flatten()
    count = min(flat.numel(), 2048)
    step = 104729
    while math.gcd(step, flat.numel()) != 1:
        step += 2
    indices = (torch.arange(count, device=flat.device) * step + 12345) % flat.numel()
    sample = flat[indices].abs()
    quantiles = torch.quantile(sample, torch.tensor([0.5, 0.9, 0.99, 0.999], device=flat.device))
    moments = torch.stack((flat.mean(), flat.square().mean().sqrt(), flat.abs().amax())).cpu().tolist()
    return {"shape": list(values.shape), "dtype": str(values.dtype), "elements": values.numel(),
            "mean": moments[0], "rms": moments[1], "amax": moments[2],
            "zero_elements": int((flat == 0).sum()), "deterministic_index_sample_size": count,
            "sample_coprime_stride": step, "sample_offset": 12345,
            "sample_abs_quantiles_50_90_99_999": quantiles.cpu().tolist()}


@contextmanager
def applied(layers, cache, case, cohort, minimum_fp8_rows=1, diagnostics=None,
            row_buffer=0, resident_rows=1024, execution="oracle", trace_schedule=False):
    policy, _, parameter = case.partition(":")
    tolerance = float(parameter) if parameter else 0.04
    counters = {name: Counters() for name in layers}
    traces = {name: [] for name in layers}
    originals = {name: module.forward for name, module in layers.items()}
    skipped = {name: 0 for name in layers}

    def replacement(name, module):
        def forward(values):
            traces[name].append(activation_summary(values))
            if policy == "original":
                if diagnostics is not None:
                    diagnostics.observe(name, module, values)
                return originals[name](values)
            if values.dtype != torch.bfloat16 or module.weight.dtype != torch.bfloat16:
                skipped[name] += 1
                return originals[name](values)
            convolution = isinstance(module, torch.nn.Conv2d)
            if convolution:
                matrix = values.permute(0, 2, 3, 1).reshape(-1, module.in_channels).contiguous()
                weight = module.weight.reshape(module.out_channels, module.in_channels)
            else:
                matrix = values.reshape(-1, module.in_features).contiguous()
                weight = module.weight
            current_policy = "bf16" if matrix.shape[0] < minimum_fp8_rows else policy
            prepared = None
            if current_policy != "bf16":
                with_moments = current_policy.startswith(("moment", "diagonal"))
                if name not in cache or (with_moments and cache[name].moments is None):
                    cache[name] = (prepare_weight(weight.detach(), with_moments=True) if with_moments
                                   else prepare_weight(weight.detach()))
                prepared = cache[name]
            output = multiply(matrix, weight, prepared, policy=current_policy, tolerance=tolerance,
                              cohort=cohort, counters=counters[name], row_buffer=row_buffer,
                              resident_rows=resident_rows, execution=execution, trace_schedule=trace_schedule)
            if module.bias is not None:
                output += module.bias.float()
            if convolution:
                output = output.reshape(values.shape[0], values.shape[2], values.shape[3], -1)
                return output.permute(0, 3, 1, 2).contiguous().to(values.dtype)
            return output.reshape(*values.shape[:-1], module.out_features).to(values.dtype)
        return forward

    try:
        for name, module in layers.items():
            module.forward = replacement(name, module)
        yield counters, traces, skipped
    finally:
        for name, module in layers.items():
            module.forward = originals[name]


def tensor_difference(actual, reference):
    error = actual.double() - reference.double()
    energy = float(reference.double().square().sum())
    normalized = F.normalize(actual.float(), dim=-1)
    base = F.normalize(reference.float(), dim=-1)
    cosine = (normalized * base).sum(-1)
    return {"rmse": float(error.square().mean().sqrt()),
            "relative_mse": float(error.square().sum()) / energy if energy else None,
            "max_absolute_difference": float(error.abs().amax()),
            "changed_scalars": int(torch.count_nonzero(error)),
            "mean_cosine": float(cosine.mean()), "minimum_cosine": float(cosine.min())}


def vision_outputs(model, batches):
    return torch.cat([model(images).float().cpu() for images, _ in batches])


def generation_outputs(pipe, embeddings, args, directory):
    images, states = [], []
    directory.mkdir(parents=True, exist_ok=True)
    for index, (prompt_embeddings, mask) in enumerate(embeddings):
        trajectory = []

        def observe(pipeline, step, timestep, callback_kwargs):
            trajectory.append(callback_kwargs["latents"].detach().float().cpu().numpy())
            return callback_kwargs

        output = pipe(prompt_embeds=prompt_embeddings, prompt_attention_mask=mask,
                      num_inference_steps=args.steps, height=1024, width=1024,
                      guidance_scale=4.5, output_type="np",
                      generator=torch.Generator("cuda").manual_seed(args.seed + index),
                      callback_on_step_end=observe).images
        if not np.isfinite(output).all():
            raise ValueError("nonfinite decoded image")
        images.append(output)
        states.append(np.stack(trajectory))
        Image.fromarray((output[0] * 255).round().clip(0, 255).astype("uint8")).save(directory / f"{index:03d}.png")
    images, states = np.concatenate(images), np.concatenate(states, axis=1)
    np.savez_compressed(directory / "arrays.npz", images=images, states=states)
    return images, states


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*VISION, "sana-sprint"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append")
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--holdout-offset", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--cohort", type=int, default=32)
    parser.add_argument("--minimum-fp8-rows", type=int, default=1)
    parser.add_argument("--diagnostic-rows", type=int, default=0)
    parser.add_argument("--diagnostic-calls", type=int, default=1)
    parser.add_argument("--row-buffer", type=int, default=0)
    parser.add_argument("--resident-rows", type=int, default=1024)
    parser.add_argument("--execution", choices=("oracle", "packed"), default="oracle")
    parser.add_argument("--trace-schedule", action="store_true")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--prompt", action="append")
    args = parser.parse_args()
    cases = list(dict.fromkeys(["original", *(args.case or DEFAULT_CASES)]))
    if min(args.cohort, args.steps, args.samples_per_class, args.batch_size,
           args.minimum_fp8_rows, args.diagnostic_calls) < 1 or min(args.holdout_offset, args.diagnostic_rows) < 0:
        parser.error("positive sizes and nonnegative holdout offset required")
    if (args.row_buffer < 0 or args.resident_rows < 1
            or (args.row_buffer and args.resident_rows % args.row_buffer)
            or (args.execution == "packed" and not args.row_buffer)):
        parser.error("packed execution needs a positive row buffer dividing the resident row count")
    if args.trace_schedule and not args.row_buffer:
        parser.error("schedule tracing requires a positive row buffer")
    for case in cases:
        policy, _, parameter = case.partition(":")
        if policy not in (*POLICIES, "original") or (parameter and (
                not math.isfinite(float(parameter)) or float(parameter) < 0)):
            parser.error("invalid policy or tolerance")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    args.output.mkdir(parents=True, exist_ok=True)
    if args.model == "sana-sprint":
        pipe, checkpoint = sana_pipeline()
        model = pipe.transformer
        prompts = args.prompt or PROMPTS
        embeddings = [pipe.encode_prompt(prompt, device=torch.device("cuda")) for prompt in prompts]
        dataset = {"prompts": list(prompts), "seeds": [args.seed + i for i in range(len(prompts))],
                   "steps": args.steps, "guidance_scale": 4.5, "image_size": [1024, 1024],
                   "max_sequence_length": 300, "scheduler": dict(pipe.scheduler.config)}
    else:
        model, data, checkpoint = vision(args.model)
        loader, classes, manifest = load_data(data, args.samples_per_class, "quality",
                                             args.batch_size, args.holdout_offset)
        batches = [(images.cuda().bfloat16(), torch.tensor([classes[int(i)] for i in labels]))
                   for images, labels in loader]
        targets = torch.cat([labels for _, labels in batches])
        dataset = {"name": "Imagenette2-160 validation", "manifest": manifest,
                   "preprocessing": str(data.transforms())}
    layers = receiving_layers(model)
    cache = {}
    diagnostics = MomentDiagnostics(args.diagnostic_rows, args.diagnostic_calls) if args.diagnostic_rows else None
    report = {
        "schema": 1, "model": args.model, "checkpoint": checkpoint, "dataset": dataset,
        "case_order": cases,
        "experiments": revision(ROOT), "engine": revision(ENGINE),
        "settings": {key: value for key, value in vars(args).items() if key != "output"},
        "source_sha256": {name: digest(ROOT / "scripts" / name) for name in (
            "paired_precision_pilot.py", "paired_precision_reference.py", "modern_models.py",
            "activation_pilot.py", "integer_segment_backend.py", "precision_moments.py",
            "precision_moment_diagnostics.py", "row_precision_schedule.py")},
        "library_sha256": digest(ENGINE / "build/libtensor_inference_engine_torch.so"),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
                        "tf32": False, "bf16_reduced_precision_reduction": False},
        "target_layers": list(layers), "cases": [],
        "limitations": [
            "Numerical reconstruction with engine BF16/FP32 GEMM, not native packed FP8 execution",
            "Prepared weight statistics and live activations; no activation calibration or label-fitted thresholds",
            "FP32 bound implementation and operand normalization do not certify task quality",
            "Dense logical vector counts do not establish SRAM transactions, latency or energy",
            "Mixed-format scheduling, scale handling and extra weight storage require hardware evaluation",
            "Small paired subsets/images, not full ImageNet or FID/GenEval",
            "Activation quantiles use deterministic indexed samples rather than all elements",
        ],
    }
    original = reference = None
    for case in cases:
        print("Running", args.model, case, flush=True)
        with applied(layers, cache, case, args.cohort, args.minimum_fp8_rows, diagnostics,
                     args.row_buffer, args.resident_rows, args.execution, args.trace_schedule) as (counters, traces, skipped):
            if args.model == "sana-sprint":
                values = generation_outputs(pipe, embeddings, args, args.output / case.replace(":", "-"))
            else:
                values = vision_outputs(model, batches)
        if case == "original":
            original = values
        if case == "bf16":
            reference = values
        record = {"case": case, "layers": {name: asdict(count) for name, count in counters.items()},
                  "activation_summaries": traces, "skipped_dtype_calls": skipped}
        if args.model == "sana-sprint":
            images, states = values
            base_images, base_states = original
            mse = ((images.astype(np.float64) - base_images) ** 2).mean(axis=(1, 2, 3))
            record["per_image_mse"] = mse.tolist()
            record["per_image_psnr"] = [-10 * math.log10(x) if x > 0 else None for x in mse]
            record["trajectory_rmse"] = float(np.sqrt(np.mean((states.astype(np.float64) - base_states) ** 2)))
            if reference is not None:
                reference_images, reference_states = reference
                mse_reference = ((images.astype(np.float64) - reference_images) ** 2).mean(axis=(1, 2, 3))
                record["per_image_mse_vs_bf16_split"] = mse_reference.tolist()
                record["per_image_psnr_vs_bf16_split"] = [
                    -10 * math.log10(x) if x > 0 else None for x in mse_reference]
                record["trajectory_rmse_vs_bf16_split"] = float(np.sqrt(np.mean(
                    (states.astype(np.float64) - reference_states) ** 2)))
            record["array_sha256"] = digest(args.output / case.replace(":", "-") / "arrays.npz")
        else:
            record["vs_original"] = tensor_difference(values, original)
            if reference is not None:
                record["vs_bf16_split"] = tensor_difference(values, reference)
            record["output_sha256"] = digest_bytes(values.numpy().tobytes())
            if checkpoint["num_classes"]:
                record["correct"] = int((values.argmax(-1) == targets).sum())
                record["predictions"] = values.argmax(-1).tolist()
                record["changed_predictions"] = int((values.argmax(-1) != original.argmax(-1)).sum())
            else:
                similarities = F.normalize(values, dim=-1) @ F.normalize(values, dim=-1).T
                baseline = F.normalize(original, dim=-1) @ F.normalize(original, dim=-1).T
                similarities.fill_diagonal_(-torch.inf)
                baseline.fill_diagonal_(-torch.inf)
                record["nearest_neighbor_agreement"] = float((similarities.argmax(-1) == baseline.argmax(-1)).float().mean())
            np.savez_compressed(args.output / f"{case.replace(':', '-')}.npz", outputs=values.numpy())
        report["cases"].append(record)
        if diagnostics is not None:
            report["local_moment_diagnostics"] = diagnostics.records
        write_json(args.output / "report.json", report)
        print(case, {key: value for key, value in record.items() if key not in (
            "layers", "activation_summaries", "predictions", "skipped_dtype_calls")}, flush=True)
    report["prepared_weight_statistics"] = {
        name: {"shape": list(prepared.reconstructed.shape),
               "original_window_l1": prepared.l1.cpu().tolist(),
               "error_window_l1": prepared.error_l1.cpu().tolist(),
               "scale_exponents": prepared.exponent_histogram,
               "receiver_moments": prepared.moments.cpu().tolist() if prepared.moments is not None else None}
        for name, prepared in cache.items()}
    write_json(args.output / "report.json", report)


def digest_bytes(values):
    import hashlib
    return hashlib.sha256(values).hexdigest()


if __name__ == "__main__":
    main()
