"""Guard physical granularity and joint argument-memory accounting."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from winograd_mapping import BF16, FP8, Convolution, retention, work


def test_fp8_and_winograd_savings_do_not_multiply_without_k_padding():
    shape = Convolution(1, 32, 32, 128, 128)
    direct = work(shape, FP8)
    winograd = work(shape, FP8, 2)
    assert direct["k_segments"] == 5
    assert winograd["issued_vectors"] / direct["issued_vectors"] == 0.8
    assert winograd["logical_macs"] / direct["logical_macs"] == 4 / 9


def test_bf16_nonmultiple_channels_and_spatial_tails_are_counted():
    shape = Convolution(1, 5, 7, 192, 129)
    direct = work(shape, BF16)
    transformed = work(shape, BF16, 2)
    assert direct["k_segments"] == 14
    assert transformed["k_segments"] == 2
    assert transformed["n_tiles"] == 2
    assert transformed["rows_per_plane"] == 12
    assert transformed["issued_vectors"] == 12 * 16 * 2 * 2


def test_two_argument_banks_are_not_assumed_and_m_is_streamed():
    one = retention(BF16, 2, 128, 16)
    assert one["shared_argument_bytes"] == 1 << 20
    assert one["retained_planes_bytes"] == 1 << 20
    assert retention(BF16, 2, 256, 16)["lhs_bytes"] == 2 * one["lhs_bytes"]
    too_big = retention(BF16, 4, 64, 36, double_buffer_inputs=True)
    assert not too_big["arguments_fit_2mib"]
    assert retention(BF16, 4, 64, 4, double_buffer_inputs=True)["arguments_fit_2mib"]


def test_weight_overflow_is_a_tiling_warning_not_an_infeasibility_claim():
    record = work(Convolution(1, 32, 32, 1024, 1024), BF16, 4)
    assert record["weight_payload_bytes"] == 72 * (1 << 20)
    assert record["weight_payload_exceeds_64mib"]
    assert retention(BF16, 4, 32, 1)["arguments_fit_2mib"]


@pytest.mark.parametrize("tile,rows,planes", [(3, 1, 1), (2, 0, 1), (2, 1, 17)])
def test_invalid_retention_is_rejected(tile, rows, planes):
    with pytest.raises(ValueError):
        retention(BF16, tile, rows, planes)
