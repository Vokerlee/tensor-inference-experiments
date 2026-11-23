"""Validate folded phases against actual resize+conv, especially borders."""

import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from resize_convolution_reference import arithmetic_counts, convolve_phases, phase_bank


@pytest.mark.parametrize("height,width", [(1, 1), (1, 5), (3, 1), (4, 5)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_every_output_phase_matches_resize_convolution(height, width, with_bias):
    torch.manual_seed(117)
    values = torch.randn(2, 3, height, width, dtype=torch.float64)
    weight = torch.randn(5, 3, 3, 3, dtype=torch.float64)
    bias = torch.randn(5, dtype=torch.float64) if with_bias else None
    expected = F.conv2d(F.interpolate(values, scale_factor=2, mode="nearest"), weight, bias, padding=1)
    actual = convolve_phases(values, phase_bank(weight), bias)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)


def test_corner_impulses_and_asymmetric_kernel_retain_boundaries():
    values = torch.zeros(1, 1, 4, 5, dtype=torch.float64)
    values[0, 0, 0, 0], values[0, 0, -1, -1] = 1, 7
    weight = torch.arange(1, 10, dtype=torch.float64).reshape(1, 1, 3, 3)
    expected = F.conv2d(F.interpolate(values, scale_factor=2), weight, padding=1)
    assert torch.equal(convolve_phases(values, phase_bank(weight)), expected)


def test_bf16_weight_sums_are_prepared_in_fp32_and_growth_is_counted():
    weight = torch.ones(2, 32, 3, 3).bfloat16()
    assert phase_bank(weight).dtype == torch.float32
    counts = arithmetic_counts(1, 32, 128, 4, 5)
    assert counts["ideal_folded_macs"] * 9 == counts["expanded_macs"] * 4
    assert counts["reference_folded_macs"] > counts["ideal_folded_macs"]
    assert counts["folded_weight_scalars"] * 9 == counts["original_weight_scalars"] * 16
