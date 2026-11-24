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


def test_integer_one_axis_bounded_batches_retain_exact_center_filter_outputs():
    from integer_winograd import prepare
    from integer_winograd_one_axis import convolution as integer_convolution
    from winograd_one_axis import transform_weights
    from winograd_shift_rounding import ShiftGeometry

    torch.manual_seed(73)
    source = torch.randint(-2, 3, (1, 32, 5, 7)).float()
    weight = torch.zeros(3, 32, 3, 3)
    weight[:, :, 1, 1] = 1
    prepared = prepare(transform_weights(weight))
    geometry = ShiftGeometry(torch.zeros(256, 4, dtype=torch.int16), rank=0)
    reference = torch.nn.functional.conv2d(source, weight, padding=1)
    outputs = []
    for batch in (3, 13):
        actual, stats = integer_convolution(source, prepared, geometry, tile_batch=batch)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        assert stats["rounding"]["source_zeros_changed"] == 0
        assert stats["lowered_reduction"] == 96
        outputs.append(actual)
    assert torch.equal(*outputs)
