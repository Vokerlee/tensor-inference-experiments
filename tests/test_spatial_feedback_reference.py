"""Check feedback state, source grouping and coefficient semantics."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from spatial_feedback_reference import kernel_coefficient, quantize


def test_zero_feedback_matches_independent_group_quantization():
    torch.manual_seed(971)
    values = torch.randn(2, 64, 3, 9)
    result = quantize(values, torch.zeros(64))
    grouped = values.permute(0, 2, 3, 1).reshape(2, 3, 9, 2, 32)
    scale = 2 ** (grouped.abs().amax(-1, keepdim=True) / 7).log2().ceil()
    expected = ((grouped / scale).round().clamp(-7, 7) * scale).reshape(2, 3, 9, 64)
    assert torch.equal(result.reconstructed, expected.permute(0, 3, 1, 2))


def test_feedback_recurrence_and_row_reset_against_scalar_oracle():
    values = torch.arange(32 * 2 * 5).reshape(1, 32, 2, 5).remainder(23).float() / 7
    coefficient = torch.linspace(-0.5, 0.5, 32)
    result = quantize(values, coefficient)
    for row in range(2):
        for channel in range(32):
            carry = 0.0
            for column in range(5):
                scale = 2.0 ** int(result.exponents[0, row, column, 0])
                source = float(values[0, channel, row, column])
                incoming = source + float(coefficient[channel]) * carry
                expected = min(max(round(incoming / scale), -7), 7) * scale
                carry = incoming - expected
                assert float(result.reconstructed[0, channel, row, column]) == expected
                assert float(result.residual[0, channel, row, column]) == pytest.approx(carry, abs=1e-6)


def test_zero_preservation_resets_carry():
    values = torch.zeros(1, 32, 1, 3)
    values[..., 0] = 1.3
    values[:, 0, :, 0] = 32
    ordinary = quantize(values, torch.ones(32))
    preserved = quantize(values, torch.ones(32), preserve_zeros=True)
    assert ordinary.introduced_nonzeros > 0
    assert preserved.introduced_nonzeros == 0
    assert (preserved.residual[..., 1:] == 0).all()


def test_coefficients_follow_kernel_sign_and_pointwise_has_no_overlap():
    lowpass = torch.ones(2, 32, 1, 3)
    highpass = lowpass.clone()
    highpass[..., 1] = -1
    assert torch.equal(kernel_coefficient(lowpass), torch.full((32,), 0.5))
    assert torch.equal(kernel_coefficient(highpass, coarse=True), torch.full((32,), -0.5))
    assert (kernel_coefficient(lowpass[..., :1]) == 0).all()
    assert (kernel_coefficient(torch.zeros_like(lowpass)) == 0).all()


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_rejected(bad):
    values = torch.full((1, 32, 1, 1), bad)
    with pytest.raises(ValueError, match="finite"):
        quantize(values, torch.zeros(32))


def test_underflowed_scale_rejected_and_small_weight_coefficient_is_scale_invariant():
    tiny = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    with pytest.raises(ValueError, match="scale"):
        quantize(torch.full((1, 32, 1, 1), tiny), torch.zeros(32), bits=8)
    weight = torch.tensor([1.0, -0.2, 0.5]).reshape(1, 1, 1, 3).repeat(2, 32, 1, 1)
    torch.testing.assert_close(kernel_coefficient(weight * 1e-17), kernel_coefficient(weight))
