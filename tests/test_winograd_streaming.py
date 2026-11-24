"""Bounded batches, K-partition schedules and live transform-grid semantics."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as functional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from winograd_streaming import convolution, encode, prepare


@pytest.mark.parametrize("tile", [2, 4])
@pytest.mark.parametrize("streaming", [False, True])
def test_bounded_batches_with_tails_match_direct_convolution(tile, streaming):
    torch.manual_seed(812)
    values = torch.randn(2, 5, 9, 7)
    weight = torch.randn(3, 5, 3, 3)
    bias = torch.randn(3)
    prepared = prepare(weight, tile, torch.float32)
    actual, stats = convolution(values, prepared, bias, tile_batch=3,
                                reduction_block=2, streaming_inverse=streaming)
    expected = functional.conv2d(values, weight, bias, padding=1)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    assert stats["tile_batches"] > 1


def test_encoding_preserves_zero_groups_and_exact_grid_points():
    source = torch.tensor([[[0.] * 32, [0.5, -0.5] * 16]])
    restored, stats = encode(source, 4)
    torch.testing.assert_close(restored, source, rtol=0, atol=0)
    assert stats["zero_groups"] == 1
    assert stats["groups"] == 2


def test_encoding_changes_the_actual_consumed_convolution():
    torch.manual_seed(67)
    source = torch.randn(1, 32, 5, 7)
    weight = torch.randn(2, 32, 3, 3)
    prepared = prepare(weight, 2, torch.float32)
    exact, _ = convolution(source, prepared, tile_batch=4)
    changed, stats = convolution(source, prepared, tile_batch=4, bits=4)
    assert not torch.equal(exact, changed)
    assert stats["groups"] > 0
    assert sum(stats["exponent_histogram"].values()) == stats["groups"]
