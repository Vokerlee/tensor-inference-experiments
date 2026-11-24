"""Known one-axis identity and discrete array tails against independent oracles."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from winograd_mapping import BF16, FP8, Convolution, work as full_work
from winograd_one_axis import convolution, work


@pytest.mark.parametrize("shape,padding", [((2, 3, 7, 9), 1), ((1, 2, 6, 8), 0), ((1, 1, 3, 3), 0)])
def test_one_axis_identity_matches_direct_outputs_including_odd_edges(shape, padding):
    torch.manual_seed(21)
    values, weight, bias = torch.randn(shape).double(), torch.randn(5, shape[1], 3, 3).double(), torch.randn(5).double()
    expected = torch.nn.functional.conv2d(values, weight, bias, padding=padding)
    torch.testing.assert_close(convolution(values, weight, bias, padding=padding), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("interface,channels,ratio", [(FP8, 64, .5), (FP8, 128, 1), (BF16, 32, .5), (BF16, 64, 1)])
def test_array_vector_savings_do_not_imply_equal_useful_multiplications(interface, channels, ratio):
    shape = Convolution(1, 8, 8, channels, 128)
    one, full = work(shape, interface), full_work(shape, interface, 2)
    assert one["issued_vectors"] / full["issued_vectors"] == ratio
    assert one["logical_macs"] / full["logical_macs"] == 1.5
    assert one["weight_payload_bytes"] / full["weight_payload_bytes"] == .75
