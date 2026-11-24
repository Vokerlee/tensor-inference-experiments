"""Validate transforms, convolution indexing and tail tiles independently."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as functional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from winograd_reference import convolution, transforms


@pytest.mark.parametrize("tile", [2, 4])
@pytest.mark.parametrize("shape,padding", [((1, 2, 7, 9), 1), ((2, 3, 8, 11), 0), ((1, 1, 3, 3), 0)])
def test_reference_matches_independent_direct_convolution(tile, shape, padding):
    torch.manual_seed(367)
    source = torch.randn(shape, dtype=torch.float64)
    weight = torch.randn(5, shape[1], 3, 3, dtype=torch.float64)
    bias = torch.randn(5, dtype=torch.float64)
    actual = convolution(source, weight, bias, tile=tile, padding=padding)
    expected = functional.conv2d(source, weight, bias, padding=padding)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("tile", [2, 4])
def test_every_input_and_filter_basis_pair_obeys_the_filter_identity(tile):
    extent = tile + 2
    sources = torch.eye(extent ** 2, dtype=torch.float64).reshape(extent ** 2, 1, extent, extent)
    filters = torch.eye(9, dtype=torch.float64).reshape(9, 1, 3, 3)
    actual = convolution(sources, filters, tile=tile, padding=0)
    expected = functional.conv2d(sources, filters)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-14)


def test_unsupported_transform_and_source_contracts_are_rejected():
    with pytest.raises(ValueError):
        transforms(6)
    with pytest.raises(ValueError):
        convolution(torch.ones(1, 1, 2, 2), torch.ones(1, 1, 3, 3), padding=0)
