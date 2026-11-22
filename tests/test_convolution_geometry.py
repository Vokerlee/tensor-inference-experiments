"""Check physical padding, grouping and alternative channel-reduction schedules."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import torch
from torch.nn import functional as F

from convolution_geometry import Convolution, attention_counter, inventory, valid_axis


def convolution(channels=64, outputs=128, groups=1):
    return Convolution("test", 1, channels, outputs, groups, 8, 8, 8, 8,
                       3, 3, 1, 1, 1, 1, 1, 1)


def test_spatial_kernel_is_not_the_complete_dense_reduction():
    layer = convolution()
    assert layer.reduction == 576
    assert layer.schedule(128)["k_segments_per_group"] == 5
    assert layer.schedule(128, True)["k_segments_per_group"] == 64
    assert layer.schedule(256)["k_segments_per_group"] == 3


def test_depthwise_underfills_both_dimensions_without_free_partitioning():
    layer = convolution(channels=64, outputs=64, groups=64)
    assert layer.reduction == 9
    assert layer.schedule(128)["dense_product_utilization"] == 9 / (128 * 128)
    assert layer.schedule(256)["dense_product_utilization"] == 9 / (256 * 128)


def test_boundary_support_differs_from_array_padding():
    layer = convolution(channels=1, outputs=1)
    assert valid_axis(8, 8, 3, 1, 1, 1) == 22
    assert layer.valid_macs() == 22 * 22
    assert layer.macs == 64 * 9
    assert valid_axis(5, 3, 3, 2, 2, 2) == 7


def test_attention_counts_both_products_and_restores_function():
    original = F.scaled_dot_product_attention
    with attention_counter() as counts, torch.device("meta"):
        F.scaled_dot_product_attention(torch.empty(2, 4, 3, 8),
                                      torch.empty(2, 4, 5, 8),
                                      torch.empty(2, 4, 5, 8))
    assert counts == {"macs": 2 * 4 * 3 * 5 * (8 + 8), "calls": 1}
    assert F.scaled_dot_product_attention is original


def test_explicit_attention_and_sdpa_are_counted_without_double_counting():
    with torch.device("meta"):
        model = torch.nn.Linear(8, 8)
        x = torch.empty(2, 3, 8)

        def execute():
            y = model(x)
            manual = y @ y.transpose(-1, -2)
            F.scaled_dot_product_attention(y, y, y)
            return manual

        report = inventory(model, execute)
    assert report["linear_macs"] == 2 * 3 * 8 * 8
    assert report["other_explicit_matrix_macs"] == 2 * 3 * 3 * 8
    assert report["attention_macs"] == 2 * 3 * 3 * (8 + 8)
