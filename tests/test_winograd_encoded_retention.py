"""Reusable encoded cache must fit alongside RHS and control metadata."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from winograd_encoded_retention import retention


def test_larger_transform_needs_smaller_streamed_batch_to_keep_full_source_reuse():
    too_large = retention(1024, 1024, tile=4, rows=64, plane_batch=1)
    feasible = retention(1024, 1024, tile=4, rows=32, plane_batch=4)
    assert too_large["encoded_source_cache_bytes"] == int(2.25 * (1 << 20))
    assert not too_large["arguments_with_colocated_controller_fit_2mib"]
    assert feasible["arguments_with_colocated_controller_fit_2mib"]
    assert feasible["products_plus_spatial_fit_2mib"]
    assert feasible["receiver_column_tiles_reusing_source"] == 8
    assert feasible["full_prepared_weight_payload_int8_padded_bytes"] == 36 * (1 << 20)


def test_f2_fits_but_zero_lane_payload_is_not_implicitly_removed():
    result = retention(1024, 1024)
    assert result["encoded_source_cache_bytes"] == 1 << 20
    assert result["arguments_with_colocated_controller_fit_2mib"]
    small = retention(32, 3)
    assert small["encoded_source_cache_bytes"] == 16 * 64 * 256
    assert small["receiver_column_tiles_reusing_source"] == 1
    assert not retention(1024, 1024, tile=4, rows=32, plane_batch=36)["arguments_with_colocated_controller_fit_2mib"]
    with pytest.raises(ValueError):
        retention(0, 128)
