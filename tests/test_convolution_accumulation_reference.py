"""Verify lowering order, metadata reuse and convolution numerical semantics."""

import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from convolution_accumulation_reference import convolve, encode_source, encode_weight, lower
from primitive_accumulation_reference import Counters


def test_lowering_moves_source_metadata_without_recomputing_it():
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(2, 64, 7, 8, generator=generator)
    layer = torch.nn.Conv2d(64, 129, 3, stride=(2, 1), padding=2, dilation=2)
    source = encode_source(x)
    codes, exponent, (maximum, l1) = lower(source, layer)
    assert torch.equal(maximum, codes.abs().amax(-1))
    assert torch.equal(l1, codes.abs().sum(-1))
    reconstructed = torch.ldexp(source.codes.float(), source.exponents[..., None])
    reconstructed = reconstructed.reshape(2, 7, 8, 64).permute(0, 3, 1, 2)
    expected = F.unfold(reconstructed, 3, dilation=2, padding=2, stride=(2, 1))
    expected = expected.reshape(2, 2, 32, 9, -1).permute(0, 4, 3, 1, 2).reshape_as(codes)
    assert torch.equal(torch.ldexp(codes.float(), exponent[..., None]), expected)


@pytest.mark.parametrize("reduction_width", [128, 256])
def test_integer_convolution_matches_direct_torch_with_column_tail(reduction_width):
    generator = torch.Generator().manual_seed(29)
    layer = torch.nn.Conv2d(32, 131, 3, stride=(2, 1), padding=1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.randint(-2, 3, layer.weight.shape, generator=generator).float())
    x = torch.randint(-3, 4, (1, 32, 5, 6), generator=generator).float()
    counters = Counters()
    actual = convolve(x, layer, encode_weight(layer), reduction_width=reduction_width,
                      accumulator_width=32, counters=counters)
    expected = F.conv2d(x, layer.weight, stride=layer.stride, padding=layer.padding)
    assert torch.equal(actual, expected)
    assert counters.source_groups_quantized == 30
    assert counters.valid_lowered_group_uses == 112
    assert counters.lowered_group_slots == 162


def test_unaligned_and_depthwise_contracts_are_not_silently_padded():
    with pytest.raises(ValueError, match="divisible"):
        encode_source(torch.zeros(1, 3, 5, 5))
    with pytest.raises(ValueError, match="dense"):
        encode_weight(torch.nn.Conv2d(32, 32, 3, groups=32))
