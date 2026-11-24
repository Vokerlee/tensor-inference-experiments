"""Physical source width and state ports must survive amortization accounting."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from plane_controller_cost import ControllerTile, ledger


def test_retention_and_feed_width_are_not_confused_with_encoded_bus_width():
    result = ledger(ControllerTile())
    assert result["state_payload_bytes"] == 64 * 1024
    assert result["immutable_metadata_bytes"] == 8 * 1024
    assert result["state_port_capacity_bytes_per_vector_if_full_reduction_parallel"] == 2048
    assert result["bf16_source_bytes_per_vector"] == 512
    assert result["int4_codes_in_int8_words_bytes_per_vector"] == 256
    assert result["metadata_preload_bytes_per_encoded_element_if_reused_across_rows"] == 2 / 64


def test_output_reuse_amortizes_controller_without_erasing_its_state_traffic():
    narrow, wide = ledger(ControllerTile(outputs=128)), ledger(ControllerTile(outputs=1024))
    assert wide["state_payload_bytes"] == narrow["state_payload_bytes"]
    assert wide["extra_fp32_add_sub_per_useful_mac"] == narrow["extra_fp32_add_sub_per_useful_mac"] / 8
    with pytest.raises(ValueError):
        ControllerTile(outputs=0)
