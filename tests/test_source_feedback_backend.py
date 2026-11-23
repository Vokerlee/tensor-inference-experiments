"""Verify model-layout and kernel-group adapters against independent references."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from source_feedback_backend import SourceFeedback, quantize_source, quantize_weights
from spatial_feedback_reference import quantize


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("preserve_zeros", [False, True])
def test_nchw_scan_matches_cpu_reference_without_mutating_source(bits, preserve_zeros):
    torch.manual_seed(204)
    source = torch.randn(2, 64, 3, 11).bfloat16()
    source[..., 2] = 0
    original = source.clone()
    coefficient = torch.linspace(-0.5, 0.5, 64)
    expected = quantize(source, coefficient, bits=bits, preserve_zeros=preserve_zeros)
    with torch.inference_mode():
        actual = quantize_source(source.cuda(), coefficient.cuda(), bits=bits,
                                 preserve_zeros=preserve_zeros)
    assert torch.equal(source, original)
    assert torch.equal(actual.reconstructed.cpu(), expected.reconstructed.bfloat16())
    assert torch.equal(actual.terminal.cpu(), expected.residual[..., -1].permute(0, 2, 1))
    assert torch.equal(actual.exponents.cpu(), expected.exponents)
    stats = actual.statistics()
    assert stats["clipped_values"] == expected.clipped_values
    assert stats["introduced_nonzeros"] == expected.introduced_nonzeros
    assert sum(stats["exponent_histogram"].values()) == source.numel() // 32


def test_weight_groups_do_not_cross_spatial_positions():
    weight = torch.ones(3, 64, 2, 3).bfloat16()
    weight[:, :, 0, 0] = 100
    weight[:, 32:, 1, 1] = 0.01
    result = quantize_weights(weight, bits=4)
    expected = weight.clone()
    for output in range(3):
        for row in range(2):
            for column in range(3):
                for group in range(2):
                    values = weight[output, group * 32:(group + 1) * 32, row, column].float()
                    scale = 2 ** torch.ceil(torch.log2(values.abs().max() / 7))
                    expected[output, group * 32:(group + 1) * 32, row, column] = (
                        (values / scale).round().clamp(-7, 7) * scale).bfloat16()
    assert torch.equal(result, expected)


def test_deferred_invalid_events_cannot_be_reported_as_valid_results():
    result = SourceFeedback(torch.zeros(1), torch.zeros(1), torch.zeros(1, dtype=torch.int32),
                            torch.tensor([0, 0, 1]))
    with pytest.raises(ValueError, match="invalid events"):
        result.statistics()
