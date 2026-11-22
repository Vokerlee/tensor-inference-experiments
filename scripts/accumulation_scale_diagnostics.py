#!/usr/bin/env python3
"""Screen conservative common-exponent accumulation guards on real MLP inputs.

This diagnoses representation headroom, not energy or a new hardware design.
"""

import argparse
from pathlib import Path

import torch

from activation_pilot import (ENGINE, ROOT, capture_inputs, digest, load_data,
                              load_model, revision, selection, write_json)


def encode(values: torch.Tensor, format_name: str):
    """Explicit G32 power-of-two baseline, not the complete OCP MX recipe."""
    grouped = values.float().reshape(*values.shape[:-1], values.shape[-1] // 32, 32)
    maximum = grouped.abs().amax(-1, keepdim=True)
    limit = 127 if format_name == "int8" else 6
    exponent = torch.where(maximum > 0, (maximum / limit).log2().ceil(), 0).to(torch.int32)
    scale = torch.ldexp(torch.ones_like(maximum), exponent)
    if format_name == "int8":
        codes = torch.round(grouped / scale).clamp(-127, 127).to(torch.int64)
    elif format_name == "e2m1":
        reconstructed = selection.quantize_e2m1(values, scale=scale).reshape_as(grouped)
        codes = torch.round(2 * reconstructed / scale).to(torch.int64)
        exponent = exponent - 1
    else:
        raise ValueError("format must be int8 or e2m1")
    return codes, exponent.squeeze(-1)


def fits_bound(bound: torch.Tensor, shifts: torch.Tensor, width: int) -> torch.Tensor:
    """Exact nonnegative integer bound, avoiding intermediate shift overflow.

    At most32768 groups and widths<=48 keep the sum within signed INT64.
    A sufficient guard excludes the asymmetric most-negative signed value.
    """
    if width < 2 or width > 48 or bound.shape != shifts.shape or bound.shape[-1] > 32768:
        raise ValueError("use equal grouped shapes, width2..48 and at most32768 groups")
    if torch.any(bound < 0) or torch.any((bound > 0) & (shifts < 0)):
        raise ValueError("positive bounds need nonnegative shifts")
    limit = (1 << (width - 1)) - 1
    safe_shift = shifts.clamp(0, width - 1).long()
    term_fits = ((bound == 0) | (shifts < width - 1)) & (bound <= (limit >> safe_shift))
    terms = torch.where(term_fits, bound, 0) << safe_shift
    return term_fits.all(-1) & (terms.sum(-1) <= limit)


def guarded_rows(a_codes, a_exponents, w_codes, w_exponents, width, tile_size=128,
                 *, a_statistics=None, bound_kind="min"):
    """One guard/anchor per activation row and declared weight-column window."""
    if bound_kind not in {"min", "max"}:
        raise ValueError("bound kind must be min or max")
    rows, groups, _ = a_codes.shape
    columns = w_codes.shape[0]
    tiles = (columns + tile_size - 1) // tile_size
    if a_statistics is None:
        a_max = a_codes.abs().amax(-1)
        a_l1 = a_codes.abs().sum(-1) if bound_kind == "min" else None
    else:
        a_max, a_l1 = a_statistics
        if a_max.shape != a_codes.shape[:-1] or (
                bound_kind == "min" and (a_l1 is None or a_l1.shape != a_max.shape)):
            raise ValueError("cached statistics must match the represented activation groups")
    w_max = torch.nn.functional.pad(w_codes.abs().amax(-1), (0, 0, 0, (-columns) % tile_size))
    w_l1 = torch.nn.functional.pad(w_codes.abs().sum(-1), (0, 0, 0, (-columns) % tile_size))
    w_exp = torch.nn.functional.pad(w_exponents, (0, 0, 0, (-columns) % tile_size))
    active_w = w_max > 0
    low = torch.where(active_w, w_exp, 10000).reshape(tiles, tile_size, groups).amin(1)
    high = torch.where(active_w, w_exp, -10000).reshape(tiles, tile_size, groups).amax(1)
    w_max = w_max.reshape(tiles, tile_size, groups).amax(1)
    w_l1 = w_l1.reshape(tiles, tile_size, groups).amax(1)
    active = (a_max[:, None] > 0) & (w_max[None] > 0)
    lower = a_exponents[:, None] + low[None]
    anchor = torch.where(active, lower, 10000).amin(-1)
    anchor = torch.where(active.any(-1), anchor, 0)
    shifts = a_exponents[:, None] + high[None] - anchor[..., None]
    bound = a_max[:, None] * w_l1[None]
    if bound_kind == "min":
        bound = torch.minimum(bound, a_l1[:, None] * w_max[None])
    return fits_bound(bound, shifts, width), anchor


def summarize_guard(fits, rows, columns, reduction, tile_size=128):
    """Require one width across every valid output in the declared square tile."""
    padded = torch.nn.functional.pad(fits, (0, 0, 0, (-rows) % tile_size), value=True)
    physical = padded.reshape(-1, tile_size, fits.shape[1]).all(1)
    row_lengths = (rows - torch.arange(physical.shape[0], device=fits.device) * tile_size).clamp(0, tile_size)
    column_lengths = (columns - torch.arange(physical.shape[1], device=fits.device) * tile_size).clamp(0, tile_size)
    areas = row_lengths[:, None] * column_lengths[None]
    return {"row_column_tile_fraction": fits.float().mean().item(),
            "supported_output_tiles": physical.sum().item(), "output_tiles": physical.numel(),
            "supported_useful_products": (areas * physical).sum().item() * reduction,
            "total_useful_products": rows * columns * reduction}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["vit_b_16", "swin_t"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, weights, layers = load_model(args.model)
    loader, _, manifest = load_data(weights, 1, "operator", 10)
    inputs = capture_inputs(model, layers, loader, None)
    report = {"schema": 1, "model": args.model, "checkpoint": weights.url,
              "experiments": revision(ROOT), "engine": revision(ENGINE),
              "script_sha256": digest(Path(__file__)), "manifest": manifest,
              "settings": {"batch_size": 10, "group_size": 32, "tile_sizes": [128, 256],
                           "weight_format": "int8", "widths": [24, 32, 48]},
              "limitations": ["One complete diagnostic batch, not an accuracy evaluation",
                              "Scales/operands hypothetical; original model forward is unchanged",
                              "Common-exponent integer headroom is not a hardware energy result",
                              "Zero-point, exceptional values and arbitrary scale formats excluded",
                              "Full-K lookahead and metadata/buffer retention must be priced",
                              "Window sizes are alternative contracts, not free hardware subarrays",
                              "Same matrix calls; no cycle savings inferred from fewer conversions"],
              "layers": []}
    for name, layer in layers.items():
        weight_codes, weight_exp = encode(layer.weight.detach(), "int8")
        x = inputs[name]
        record = {"name": name, "M": x.shape[0], "N": layer.out_features,
                  "K": layer.in_features, "cases": []}
        for format_name in ["int8", "e2m1"]:
            codes, exponent = encode(x, format_name)
            for width in [24, 32, 48]:
                for tile_size in [128, 256]:
                    fits, _ = guarded_rows(codes, exponent, weight_codes, weight_exp, width, tile_size)
                    record["cases"].append({"activation_format": format_name, "width": width,
                                            "tile_size": tile_size,
                                            **summarize_guard(fits, x.shape[0], layer.out_features,
                                                              layer.in_features, tile_size)})
        report["layers"].append(record)
        write_json(args.output, report)
        print(name, record["cases"], flush=True)


if __name__ == "__main__":
    main()
