"""Independent block-scaled products and malformed-source guards for FP8 control."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fp8_winograd import multiply, prepare


def test_fp8_block_scales_and_tail_match_an_independent_decoded_product():
    torch.manual_seed(92)
    source, original = torch.randn(4, 3, 300), torch.randn(4, 300, 5)
    weights = prepare(original)
    actual, stats = multiply(source, weights)
    padded = torch.nn.functional.pad(source, (0, 212)).reshape(4, 3, 2, 256)
    exponents = (padded.abs().amax(-1) / 448).log2().ceil().int()
    scales = torch.ldexp(torch.ones_like(exponents, dtype=torch.float32), exponents)
    decoded = (padded / scales[..., None]).to(torch.float8_e4m3fn).float()
    reconstructed_weights = weights.reconstructed()
    expected = sum((decoded[:, :, group] @ reconstructed_weights[:, group * 256:(group + 1) * 256])
                   * scales[:, :, group, None] for group in range(2))
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-5)
    assert torch.count_nonzero(weights.codes[:, 300:]) == 0
    assert stats["fp8_partial_vectors"] == 24


def test_zero_source_preserves_exact_zero_output_and_reports_zero_groups():
    weights = prepare(torch.ones(4, 32, 3))
    actual, stats = multiply(torch.zeros(4, 2, 32), weights)
    assert torch.count_nonzero(actual) == 0
    assert stats["zero_groups"] == 8
    assert stats["exponent_histogram"] == {"0": 8}


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_input_never_reaches_a_floating_product(bad):
    weights = prepare(torch.ones(4, 32, 3))
    source = torch.zeros(4, 1, 32)
    source[0, 0, 0] = bad
    with pytest.raises(ValueError, match="finite source"):
        multiply(source, weights)
