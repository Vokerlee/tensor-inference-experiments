"""Verify inverse-aware geometry and actual-unit live scale decisions."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from winograd_reference import inverse_outputs, transforms
from winograd_rounding import prepare, round_planes
from winograd_streaming import encode


def test_isolated_channel_metric_matches_the_complete_spatial_output():
    torch.manual_seed(813)
    weights = torch.randn(16, 32, 7, dtype=torch.float64)
    geometry = prepare(weights, rank=4)
    error = torch.randn(16, dtype=torch.float64)
    perturbed = error[:, None, None] * weights[:, 11:12]
    expected = inverse_outputs(perturbed, transforms()).square().sum()
    actual = error @ geometry.metric[11] @ error * geometry.normalizer[11]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_full_rank_proxy_controls_isolated_channels_under_different_live_scales():
    torch.manual_seed(724)
    weights = torch.randn(16, 32, 5)
    geometry = prepare(weights, rank=16)
    scales = torch.logspace(-3, 2, 16)[:, None, None]
    source = torch.randn(16, 3, 32) * scales
    rounded, stats = round_planes(source, geometry)
    nearest, _ = encode(source, 4)
    def isolated(result):
        errors = (result - source).permute(1, 2, 0).double()
        return torch.einsum("mcp,cpq,mcq->mc", errors, geometry.metric, errors)
    assert torch.all(isolated(rounded) <= isolated(nearest) + 1e-6)
    assert stats["selected_proxy_sum"] <= stats["nearest_proxy_sum"]


def test_exact_zeros_are_preserved_and_runtime_uses_current_not_calibration_scales():
    torch.manual_seed(32)
    geometry = prepare(torch.randn(16, 32, 3), rank=1)
    source = torch.randn(16, 2, 32)
    source[:, :, ::2] = 0
    first, stats = round_planes(source, geometry, bits=8)
    second, _ = round_planes(source * 16, geometry, bits=8)
    assert stats["source_zeros_changed"] == 0
    torch.testing.assert_close(second, first * 16, rtol=0, atol=0)
