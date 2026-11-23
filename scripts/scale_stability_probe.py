#!/usr/bin/env python3
"""Exhaustive native BF16 scale contracts and paired decoder interval diagnostics."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from activation_pilot import ENGINE, ROOT, digest, revision, write_json
from decision_stability import ScaleTable, grouped_sources
from decoder_control_path_probe import LAYER
from integer_segment_backend import load_native
from modern_models import SANA, provenance, snapshot
from sana_resize_pilot import changed, prepared, upsampling_blocks


def measure_tables():
    load_native()
    positive_words = torch.arange(0x7f80, dtype=torch.int32).to(torch.int16)
    words = torch.cat((positive_words, (positive_words.int() + 0x8000).to(torch.int16)))
    finite = words.view(torch.bfloat16)
    if not torch.isfinite(finite).all():
        raise ValueError("finite BF16 enumeration failed")
    source = finite.cuda().reshape(-1, 1, 1).expand(-1, 1, 32).contiguous()
    coefficient = torch.zeros(32, device="cuda")
    tables, contracts = {}, []
    for bits in (4, 8):
        output, terminal, exponents, counts = torch.ops.tensor_inference_engine.source_feedback(
            source, coefficient, bits, False, False, None)
        exponents = exponents.cpu().reshape(2, -1).numpy()
        if not np.array_equal(exponents[0], exponents[1]):
            raise ValueError("native scale selection is not sign invariant")
        if not (np.diff(exponents[0, 1:]) >= 0).all():
            raise ValueError("positive native scale contract is not monotonic")
        maximum = finite[:len(positive_words)].double().numpy()
        tables[bits] = ScaleTable(maximum, exponents[0])
        positive = finite[:len(positive_words)].float()
        reference = torch.where(positive > 0, (positive / ((1 << (bits - 1)) - 1)).log2().ceil(), 0).int().numpy()
        bad = (~torch.isfinite(output).all((1, 2)) | ~torch.isfinite(terminal).all(1)).cpu().numpy()
        contracts.append({"bits": bits, "enumerated_finite_words": len(words),
            "positive_including_zero": len(positive_words), "sign_invariant": True,
            "positive_exponents_monotonic": True, "positive_exponents": exponents[0].tolist(),
            "torch_cpu_exponent_mismatch_words": np.flatnonzero(reference != exponents[0]).tolist(),
            "native_counters": counts.cpu().tolist(),
            "nonfinite_reconstructed_or_terminal_words": words.view(torch.uint16).numpy()[bad].tolist()})
    return tables, contracts


def compare(original, candidate, tables):
    first, second = grouped_sources(original), grouped_sources(candidate)
    maximum = np.abs(first).max(-1)
    changed_maximum = np.abs(second).max(-1)
    bounds = np.abs(second - first).max(-1)
    records = []
    for bits, table in tables.items():
        before = table.exponents[np.searchsorted(table.maxima, maximum)]
        after = table.exponents[np.searchsorted(table.maxima, changed_maximum)]
        changed_scales = before != after
        local = table.stable(maximum, bounds)
        global_bound = float(bounds.max())
        global_stable = table.stable(maximum, global_bound)
        if np.any(local & changed_scales) or np.any(global_stable & changed_scales):
            raise ValueError("scale guard falsely certified an observed changed decision")
        records.append({"bits": bits, "group_shape": list(maximum.shape), "groups": maximum.size,
            "changed_scales": int(changed_scales.sum()), "global_bound": global_bound,
            "local_bound_certified": int(local.sum()), "global_bound_certified": int(global_stable.sum()),
            "maximum": maximum.flatten().tolist(), "candidate_maximum": changed_maximum.flatten().tolist(),
            "observed_group_bound": bounds.flatten().tolist(),
            "original_exponents": before.flatten().tolist(), "candidate_exponents": after.flatten().tolist(),
            "local_certified": local.flatten().tolist(), "global_certified": global_stable.flatten().tolist()})
    return records


@torch.inference_mode()
def main():
    from diffusers import AutoencoderDC

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    tables, contracts = measure_tables()
    directory = snapshot(*SANA)
    vae = AutoencoderDC.from_pretrained(directory, subfolder="vae", torch_dtype=torch.bfloat16)
    vae.eval().requires_grad_(False).to("cuda")
    layer = dict(vae.decoder.named_modules())[LAYER]
    blocks = upsampling_blocks(vae.decoder)
    metadata = prepared(blocks, "fold-bf16")
    report = {"schema": 1, "status": "running", "experiments": revision(ROOT), "engine": revision(ENGINE),
        "source_sha256": {name: digest(ROOT / "scripts" / name) for name in (
            "scale_stability_probe.py", "decision_stability.py", "decoder_control_path_probe.py",
            "sana_resize_pilot.py", "resize_convolution_reference.py")},
        "native_sha256": digest(ENGINE / "build/libtensor_inference_engine_torch.so"),
        "checkpoint": provenance(*SANA, directory), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(), "contracts": contracts, "input_bundles": [], "records": [],
        "limitations": [
            "Interval guarding is established numerical-stability analysis, not a novel theorem",
            "Complete finite BF16 scale enumeration describes this native binary/device; not unmeasured target hardware",
            "Input perturbation bounds are observed using BOTH trajectories, not certified predictions for unseen tensors",
            "Filtering concerns only amax-selected scale, not codes, causal modes/state, output accuracy or model quality",
            "Three predeclared local patches per input; no global frequency or reference-execution speedup claim",
            "Nonfinite reconstruction cases in exhaustive extremes are retained and must not be accepted as valid inference",
        ]}
    write_json(args.output, report)
    for bundle in args.inputs:
        manifest = json.loads((bundle / "inputs.json").read_text())
        path = bundle / "inputs.npz"
        if manifest["status"] != "complete" or digest(path) != manifest["arrays_sha256"]:
            raise ValueError("changed or incomplete decoder input bundle")
        report["input_bundles"].append({"manifest": manifest, "sha256": digest(path)})
        with np.load(path) as data:
            inputs, expected = data["inputs"].copy(), data["raw"].copy()
        for index, latent in enumerate(inputs):
            captures = {}

            def capture(name):
                def hook(module, incoming):
                    values = incoming[0]
                    height, width = values.shape[-2:]
                    captures[name] = [values[..., row - 4:row + 4, width // 2 - 33:width // 2 + 33]
                                      .cpu().contiguous()
                                      for row in (height // 4, height // 2, 3 * height // 4)]
                return hook

            source = torch.from_numpy(latent[None]).to(device="cuda", dtype=torch.bfloat16)
            for backend in ("original", "folded"):
                handle = layer.register_forward_pre_hook(capture(backend))
                try:
                    if backend == "original":
                        raw = vae.decode(source, return_dict=False)[0].float().cpu().numpy()
                        if not np.array_equal(raw, expected[index:index + 1]):
                            raise ValueError("original decoder replay mismatch")
                    else:
                        with changed(blocks, metadata, "fold-bf16", []):
                            vae.decode(source, return_dict=False)
                finally:
                    handle.remove()
            for patch, (first, second) in enumerate(zip(captures["original"], captures["folded"])):
                for row in compare(first, second, tables):
                    report["records"].append({"bundle": str(bundle), "input_index": index,
                                               "patch_index": patch, "layer": LAYER, **row})
            write_json(args.output, report)
            print(bundle.name, index, len(report["records"]), flush=True)
    report["status"] = "complete"
    write_json(args.output, report)


if __name__ == "__main__":
    main()
