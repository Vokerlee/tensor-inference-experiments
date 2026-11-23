#!/usr/bin/env python3
"""Full-decoder live/frozen control experiments under a declared backend change."""

import argparse
import json
import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from activation_pilot import ROOT, digest, write_json
from controller_replay import quantize_decisions, write_tape
from live_source_feedback import weight_metadata
from modern_models import SANA, provenance, snapshot
from sana_decoder_feedback import eligible_layers, environment, image_values, metrics, save_outputs
from sana_resize_pilot import changed, prepared, upsampling_blocks


@contextmanager
def controlled(layers, metadata, policy, bits, events, calls, reference=None,
               freeze_scales=False, freeze_modes=False):
    originals = {name: layer.forward for name, layer in layers.items()}

    def replacement(name, layer):
        def forward(values):
            position = len(events)
            if reference is not None and position >= len(reference):
                raise ValueError("more controlled calls than reference events")
            old = reference[position] if reference is not None else None
            result, decision, statistics = quantize_decisions(
                values, metadata[name], layer=name, policy=policy, bits=bits, reference=old,
                freeze_scales=freeze_scales, freeze_modes=freeze_modes)
            layout = (torch.channels_last if values.is_contiguous(memory_format=torch.channels_last)
                      else torch.contiguous_format)
            output = layer._conv_forward(result.reconstructed.contiguous(memory_format=layout),
                                         layer.weight, layer.bias)
            events.append(decision)
            calls.append({"event": position, "layer": name, "shape": list(values.shape),
                          "output_shape": list(output.shape),
                          "dense_macs": output.numel() * layer.in_channels * math.prod(layer.kernel_size),
                          **statistics})
            return output
        return forward

    try:
        for name, layer in layers.items():
            layer.forward = replacement(name, layer)
        yield
        if reference is not None and len(events) != len(reference):
            raise ValueError("fewer controlled calls than reference events")
    finally:
        for name, layer in layers.items():
            layer.forward = originals[name]


def decomposition(original, frozen, live):
    def calculate(a, b, c):
        data = b.astype(np.float64) - a.astype(np.float64)
        control = c.astype(np.float64) - b.astype(np.float64)
        total = c.astype(np.float64) - a.astype(np.float64)
        return {"data_squared": float(np.sum(data ** 2)),
                "control_squared": float(np.sum(control ** 2)),
                "twice_inner_product": float(2 * np.sum(data * control)),
                "total_squared": float(np.sum(total ** 2)),
                "maximum_identity_residual": float(np.abs(total - data - control).max())}
    return {"raw": calculate(original, frozen, live),
            "pixels": calculate(*(image_values(values) for values in (original, frozen, live)))}


@torch.inference_mode()
def main():
    from diffusers import AutoencoderDC

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--policy", choices=("none", "coarse", "live"), action="append")
    parser.add_argument("--bits", choices=(4, 8), type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    policies = list(dict.fromkeys(args.policy or ("none", "coarse", "live")))
    directory = snapshot(*SANA)
    vae = AutoencoderDC.from_pretrained(directory, subfolder="vae", torch_dtype=torch.bfloat16)
    vae.eval().requires_grad_(False).to("cuda")
    blocks = upsampling_blocks(vae.decoder)
    excluded = {f"{name}.conv" for name in blocks}
    layers = {name: layer for name, layer in eligible_layers(vae.decoder, "spatial").items()
              if name not in excluded}
    metadata = {name: weight_metadata(layer.weight) for name, layer in layers.items()}
    fold_metadata = prepared(blocks, "fold-bf16")
    report = {
        "schema": 1, "status": "running", "model": "dc-ae-sana-decoder", "environment": environment(),
        "checkpoint": provenance(*SANA, directory), "bits": args.bits, "policies": policies,
        "source_sha256": {name: digest(ROOT / "scripts" / name) for name in (
            "sana_controller_replay.py", "controller_replay.py", "source_feedback_backend.py",
            "sana_resize_pilot.py", "resize_convolution_reference.py")},
        "controlled_layers": list(layers), "excluded_upsampling_receivers": sorted(excluded),
        "input_bundles": [], "runs": [], "decompositions": [],
        "limitations": [
            "Backend perturbation is known BF16 resize folding, not a specified target-chip arithmetic",
            "Live policies use source-group POT quantization and optional causal row feedback; weights remain BF16",
            "Frozen decisions on a changed trajectory are counterfactual diagnostics, not deployable alternatives",
            "Upsampling receivers excluded structurally so both backends have identical controlled event identities",
            "Trace payloads contain decisions and sensor state, not source activations/checkpoints",
            "Pixel fidelity is not semantic generation quality, and action counts are not energy",
            "Single-image calls, 32-row causal blocks and smoothing1/4; no input calibration or label tuning",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    write_json(report_path, report)
    processed = 0
    for bundle in args.inputs:
        manifest = json.loads((bundle / "inputs.json").read_text())
        input_path = bundle / "inputs.npz"
        if manifest["status"] != "complete" or digest(input_path) != manifest["arrays_sha256"]:
            raise ValueError("invalid decoder inputs")
        report["input_bundles"].append({"sha256": digest(input_path), "manifest": manifest})
        with np.load(input_path) as arrays:
            inputs, expected = arrays["inputs"].copy(), arrays["raw"].copy()
        for local_index, latent in enumerate(inputs):
            if args.limit is not None and processed >= args.limit:
                break
            image_id = processed
            source = torch.from_numpy(latent[None]).to(device="cuda", dtype=torch.bfloat16)
            raw_original = vae.decode(source, return_dict=False)[0].float().cpu().numpy()
            if not np.array_equal(raw_original, expected[local_index:local_index + 1]):
                raise ValueError("unquantized original replay is not bit-exact")
            with changed(blocks, fold_metadata, "fold-bf16", []):
                raw_folded = vae.decode(source, return_dict=False)[0].float().cpu().numpy()
            for policy in policies:
                baseline_events, outputs = None, {}
                variants = [("original", "live", False, False), ("original", "replay", True, True),
                            ("folded", "live", False, False), ("folded", "frozen", True, True)]
                if policy == "live":
                    variants += [("folded", "fixed-scales", True, False),
                                 ("folded", "fixed-modes", False, True)]
                for backend, control, scales, modes in variants:
                    events, calls = [], []
                    with controlled(layers, metadata, policy, args.bits, events, calls,
                                    baseline_events, scales, modes):
                        if backend == "folded":
                            with changed(blocks, fold_metadata, "fold-bf16", []):
                                raw = vae.decode(source, return_dict=False)[0].float().cpu().numpy()
                        else:
                            raw = vae.decode(source, return_dict=False)[0].float().cpu().numpy()
                    if not np.isfinite(raw).all():
                        raise ValueError("nonfinite full decoder result")
                    key = f"{policy}/{backend}-{control}"
                    if backend == "original" and control == "live":
                        baseline_events = events
                    if control == "replay" and not np.array_equal(raw, outputs["original-live"]):
                        raise ValueError("same-backend control replay is not bit-exact")
                    output_dir = args.output / key / f"{image_id:03d}"
                    files = save_outputs(output_dir, raw)
                    tape_path = output_dir / "decisions.npz"
                    write_tape(tape_path, events)
                    record = {
                        "image_id": image_id, "bundle": str(bundle), "bundle_input_index": local_index,
                        "policy": policy, "backend": backend, "control": control,
                        "image_vs_original": metrics(image_values(raw), image_values(raw_original)),
                        "raw_vs_original": metrics(raw, raw_original),
                        "image_vs_unquantized_same_backend": metrics(
                            image_values(raw), image_values(raw_folded if backend == "folded" else raw_original)),
                        "trace": str(tape_path), "trace_sha256": digest(tape_path),
                        "trace_bytes": tape_path.stat().st_size, "calls": calls, **files,
                    }
                    outputs[f"{backend}-{control}"] = raw
                    report["runs"].append(record)
                    write_json(report_path, report)
                    print(image_id, key, record["image_vs_original"]["rmse"], flush=True)
                report["decompositions"].append({"image_id": image_id, "policy": policy,
                    **decomposition(outputs["original-live"], outputs["folded-frozen"], outputs["folded-live"])})
                write_json(report_path, report)
            processed += 1
    report["processed_inputs"] = processed
    report["status"] = "complete"
    write_json(report_path, report)


if __name__ == "__main__":
    main()
