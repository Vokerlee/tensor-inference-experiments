"""Check causal mode selection and surrogate limitations independently."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from live_source_feedback import quantize_live_rows, select_coefficient, weight_metadata
from source_feedback_backend import quantize_source


def test_mode_minimizes_declared_scalar_surrogate_not_actual_convolution():
    torch.manual_seed(169)
    metadata = {"rho": torch.rand(256) - 0.5, "dc_gain": torch.rand(256) * 9}
    mean, variance = torch.randn(256), torch.rand(256)
    choice = select_coefficient(mean, variance + mean.square(), metadata)

    def objective(alpha):
        return (variance * (1 + alpha * alpha - 2 * alpha * metadata["rho"])
                + mean.square() * (alpha - 1).square() * metadata["dc_gain"])

    actual = objective(choice)
    for alpha in (-0.5, 0.0, 0.5):
        assert (actual <= objective(torch.full_like(mean, alpha)) + 1e-5).all()
    assert (select_coefficient(torch.zeros(256), torch.zeros(256), metadata)
            == (metadata["rho"] * 2).round() / 2).all()


def test_kernel_dc_gain_and_signed_correlation():
    weights = torch.ones(2, 32, 3, 3)
    metadata = weight_metadata(weights)
    torch.testing.assert_close(metadata["rho"], torch.full((32,), 2 / 3))
    torch.testing.assert_close(metadata["dc_gain"], torch.full((32,), 9.0))
    metadata = weight_metadata(torch.zeros_like(weights))
    assert (metadata["rho"] == 0).all() and (metadata["dc_gain"] == 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_future_row_blocks_cannot_change_preceding_quantized_values():
    torch.manual_seed(202)
    source = torch.randn(1, 32, 65, 9, device="cuda").bfloat16()
    metadata = weight_metadata(torch.randn(8, 32, 3, 3, device="cuda"))
    first, stats = quantize_live_rows(source, metadata)
    changed = source.clone()
    changed[:, :, 32:] *= 8
    second, _ = quantize_live_rows(changed, metadata)
    assert torch.equal(first.reconstructed[:, :, :32], second.reconstructed[:, :, :32])
    assert torch.equal(first.exponents[:, :32], second.exponents[:, :32])
    first.statistics()
    second.statistics()
    assert stats["blocks"] == 3
    assert sum(stats["mode_channel_rows"].values()) == 65 * 32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_zero_dc_and_correlation_reduce_to_no_feedback_and_batch_state_resets():
    torch.manual_seed(203)
    source = torch.randn(2, 32, 33, 7, device="cuda").bfloat16()
    zeros = torch.zeros(32, device="cuda")
    actual, stats = quantize_live_rows(source, {"rho": zeros, "dc_gain": zeros})
    expected = quantize_source(source, zeros)
    assert torch.equal(actual.reconstructed, expected.reconstructed)
    assert torch.equal(actual.terminal, expected.terminal)
    assert torch.equal(actual.exponents, expected.exponents)
    assert stats["coefficient_switches"] == 0
    for item in range(2):
        separate, _ = quantize_live_rows(source[item:item + 1], {"rho": zeros, "dc_gain": zeros})
        assert torch.equal(actual.reconstructed[item:item + 1], separate.reconstructed)
