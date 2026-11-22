#!/usr/bin/env python3
"""Check local mask certificates and exact scale-bin prediction on live streams."""

import argparse
from pathlib import Path

import torch

from activation_pilot import (capture_inputs, load_data, load_model, make_report,
                              selection, write_json)
from activation_policies import sum_mode_tiles


def certificate(values, reconstructed, gram):
    grouped = values.float().reshape(-1, gram.shape[0], 4)
    q = reconstructed.float().reshape_as(grouped)
    diagonal = gram.diagonal(dim1=-2, dim2=-1)
    base = (grouped.square() * diagonal).sum(-1)
    gains = (q.square() - 2 * q * grouped) * diagonal
    scores = torch.stack([base + gains[..., i] + gains[..., j]
                          for i, j in selection.PAIRS], -1).clamp_min(0)
    best = scores.topk(2, dim=-1, largest=False).values
    rho = selection.coherence(gram)
    # Sufficient in exact arithmetic for the same finite candidates. FP32
    # implementation is checked empirically against the full local objective.
    return ((rho < 1) & ((1 + rho) * best[..., 0] < (1 - rho) * best[..., 1]))


def prediction_statistics(values):
    maximum = values.float().reshape(values.shape[0], -1, 32).abs().amax(-1)
    bins = torch.where(maximum > 0, torch.ceil(torch.log2(maximum / 7)), -10000)
    streams = {"previous_block_row_major": (bins.flatten()[1:], bins.flatten()[:-1]),
               "previous_row_same_channels": (bins[1:], bins[:-1])}
    result = {}
    for name, (actual, predicted) in streams.items():
        hit = (actual == predicted).float().mean().item()
        result[name] = {"scale_bin_hit_fraction": hit,
                        "ideal_bytes_saved_if_extra_raw_pass_exists": hit * 64 - (1 - hit) * 16,
                        "expensive_pass_saving_against_buffered_current_bin": 0}
    return result


def precision_intervals(x, gram):
    """Check sufficient block decisions from diagonal bounds, not task bounds."""
    diagonal = gram.diagonal(dim1=-2, dim2=-1)
    rho = selection.coherence(gram)
    candidates = [selection.quantize_e2m1(x).bfloat16().float(),
                  selection.quantize_symmetric(x, 8).bfloat16().float()]

    def bounds(values):
        score = (values.reshape(-1, gram.shape[0], 4).square() * diagonal).sum(-1)
        return (sum_mode_tiles(score * (1 - rho)).clamp_min(0),
                sum_mode_tiles(score * (1 + rho)))

    lower_ref, upper_ref = bounds(x)
    lower4, upper4 = bounds(candidates[0] - x)
    lower8, upper8 = bounds(candidates[1] - x)
    actual_ref = sum_mode_tiles(selection.group_loss(x, gram)).clamp_min(1e-30)
    actual4 = sum_mode_tiles(selection.group_loss(candidates[0] - x, gram)) / actual_ref
    actual8 = sum_mode_tiles(selection.group_loss(candidates[1] - x, gram)) / actual_ref
    records = []
    for tolerance in [.001, .01, .03]:
        cheap4 = upper4 <= tolerance * lower_ref
        cheap8 = (lower4 > tolerance * upper_ref) & (upper8 <= tolerance * lower_ref)
        cheap16 = (lower4 > tolerance * upper_ref) & (lower8 > tolerance * upper_ref)
        certified = cheap4 | cheap8 | cheap16
        predicted = torch.where(cheap4, 0, torch.where(cheap8, 1, 2))
        actual = torch.where(actual4 <= tolerance, 0, torch.where(actual8 <= tolerance, 1, 2))
        records.append({"tolerance": tolerance, "tile_count": certified.numel(),
                        "certified_tiles": certified.sum().item(),
                        "certified_mode_mismatches": (certified & (predicted != actual)).sum().item()})
    return records


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["vit_b_16", "swin_t"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, weights, layers = load_model(args.model)
    loader, _, manifest = load_data(weights, 1, "operator", 2)
    inputs = capture_inputs(model, layers, loader, None)
    report = make_report(args, weights)
    report["dataset"]["manifest"] = manifest
    report["limitations"].extend([
        "Certificates are exact-arithmetic statements, not a floating-point proof",
        "Scale predictors use the declared layer row-major order, not temporal video",
        "Saved bytes are an ideal conditional accounting, not measured energy",
        "A fused current-group producer needs no separate raw-data pass"])
    for name, layer in layers.items():
        x = inputs[name].float()
        gram = selection.group_gram(layer.weight.t().contiguous())
        q = selection.quantize_symmetric(x, 4).bfloat16().float()
        certified = certificate(x, q, gram)
        diagonal, _ = selection.select_2of4(x, q, gram, "diagonal")
        rich, _ = selection.select_2of4(x, q, gram, "gram")
        simple_loss = selection.group_loss(diagonal - x, gram)
        rich_loss = selection.group_loss(rich - x, gram)
        violations = certified & (simple_loss > rich_loss * (1 + 1e-5) + 1e-6)
        record = {"name": name, "rows": x.shape[0], "K": x.shape[1],
                  "certified_diagonal_fraction": certified.float().mean().item(),
                  "certified_objective_mismatches": violations.sum().item(),
                  "precision_intervals": precision_intervals(x, gram),
                  "prediction": prediction_statistics(x)}
        report["layers"].append(record)
    write_json(args.output, report)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
