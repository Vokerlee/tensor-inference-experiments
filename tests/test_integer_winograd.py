"""Independent integer accumulation, K tails and group-scale boundaries."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from integer_winograd import IntegerWeights, multiply, prepare
from winograd_rounding import prepare as prepare_geometry


def test_worst_case_integer_partial_is_exact_with_bf16_operands_fp32_sum():
    source = torch.full((2, 3, 256), 127.)
    weights = IntegerWeights(torch.full((2, 256, 7), 127, dtype=torch.int8),
                             torch.zeros((2, 1, 7), dtype=torch.int32), 256, 256)
    actual, _ = multiply(source, weights)
    expected = source.long() @ weights.codes.long()
    torch.testing.assert_close(actual.double(), expected.double(), rtol=0, atol=0)


def test_each_k_block_has_its_own_postproduct_scale_and_tail_mask():
    torch.manual_seed(771)
    source = torch.randn(2, 3, 300)
    coefficients = torch.randn(2, 300, 5)
    weights = prepare(coefficients)
    actual, stats = multiply(source, weights)
    grouped = torch.nn.functional.pad(source, (0, 212)).reshape(2, 3, 2, 256)
    maximum = grouped.abs().amax(-1)
    exponents = (maximum / 127).log2().ceil().int()
    scales = torch.ldexp(torch.ones_like(maximum), exponents)
    codes = (grouped / scales[..., None]).round().clamp(-127, 127).long()
    expected = torch.zeros(2, 3, 5)
    weight_scales = torch.ldexp(torch.ones_like(weights.exponents, dtype=torch.float32), weights.exponents)
    for block in range(2):
        integer = codes[:, :, block] @ weights.codes[:, block * 256:(block + 1) * 256].long()
        expected += integer.float() * scales[:, :, block, None] * weight_scales[:, block, None]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.count_nonzero(weights.codes[:, 300:]) == 0
    assert stats["padded_channels"] == 512


def test_cross_plane_rounding_uses_the_same_native_block_scale_contract():
    torch.manual_seed(86)
    weights = prepare(torch.randn(16, 37, 3))
    geometry = prepare_geometry(weights.reconstructed(), rank=1)
    result, stats = multiply(torch.randn(16, 2, 37), weights, bits=4, geometry=geometry)
    assert result.shape == (16, 2, 3)
    assert stats["activation_groups"] == 32
    assert stats["rounding"]["source_zeros_changed"] == 0
    zeros, _ = multiply(torch.zeros(16, 2, 37), weights, bits=8)
    assert torch.count_nonzero(zeros) == 0
