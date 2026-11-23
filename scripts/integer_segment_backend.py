"""Experiment-side metadata preparation for the engine's native segment planner."""

from functools import lru_cache
from pathlib import Path

import torch
from torch.nn import functional as F


ENGINE = Path(__file__).resolve().parents[1] / "deps/tensor-inference-engine"


@lru_cache(maxsize=1)
def load_native() -> None:
    torch.ops.load_library(str(ENGINE / "build/libtensor_inference_engine_torch.so"))


def weight_statistics(codes: torch.Tensor, exponents: torch.Tensor, window: int = 128) -> dict:
    columns, groups, _ = codes.shape
    tiles = (columns + window - 1) // window
    maximum = F.pad(codes.abs().amax(-1), (0, 0, 0, (-columns) % window))
    l1 = F.pad(codes.abs().sum(-1), (0, 0, 0, (-columns) % window))
    exponent = F.pad(exponents, (0, 0, 0, (-columns) % window))
    active = maximum > 0
    low = torch.where(active, exponent, 10000).reshape(tiles, window, groups).amin(1)
    high = torch.where(active, exponent, -10000).reshape(tiles, window, groups).amax(1)
    maximum = maximum.reshape(tiles, window, groups).amax(1)
    return {
        "maximum": maximum.int().contiguous(),
        "l1": l1.reshape(tiles, window, groups).amax(1).int().contiguous(),
        "low": torch.where(maximum > 0, low, 0).int().contiguous(),
        "high": torch.where(maximum > 0, high, 0).int().contiguous(),
    }


def plan(maximum: torch.Tensor, exponent: torch.Tensor, weights: dict, *,
         groups_per_slice: int = 4, bits: int = 24, l1: torch.Tensor | None = None):
    """Inputs are already contiguous int32; preparation costs are separate."""
    load_native()
    return torch.ops.tensor_inference_engine.integer_segment_plan(
        maximum, exponent, weights["l1"], weights["low"], weights["high"],
        groups_per_slice, bits, l1, weights["maximum"] if l1 is not None else None)


def decode_ends(masks: torch.Tensor, groups_per_slice: int, group_count: int) -> torch.Tensor:
    bits = torch.arange(groups_per_slice, device=masks.device)
    ends = ((masks[..., None].int() >> bits) & 1).bool()
    return ends.flatten(-2)[..., :group_count]
