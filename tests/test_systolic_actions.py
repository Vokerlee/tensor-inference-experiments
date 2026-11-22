"""Small independently enumerable schedules and fixed-tile compaction limits."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from systolic_actions import Schedule, compare


def test_one_processing_element_and_intermediate_results():
    actions = Schedule(1, 1, 3, 1, 1, 1).actions()
    assert actions["internal_operand_word_hops"] == {"a": 0, "b": 0}
    assert actions["issued_padded_products"] == 3
    assert actions["pe_resident_partial_sum_sram_bytes"] == 4
    assert actions["sram_mediated_partial_sum_bytes"] == 20


def test_two_by_two_array_with_padded_rows():
    actions = Schedule(3, 2, 3, 2, 2, 2).actions()
    assert actions["tile_calls"] == 4
    assert actions["useful_dense_products"] == 18
    assert actions["issued_padded_products"] == 32
    assert actions["internal_operand_word_hops"] == {"a": 16, "b": 16}
    assert actions["sram_mediated_partial_sum_bytes"] == 72


def test_gating_does_not_remove_schedule_and_small_k_cannot_remove_a_call():
    result = compare(Schedule(128, 128, 96), 48, .5)
    for name in ["tile_calls", "operand_edge_slots", "internal_operand_word_hops",
                 "sram_mediated_partial_sum_bytes"]:
        assert result["dense"][name] == result["gate_only"][name]
        assert result["dense"][name] == result["common_k_compaction"][name]


def test_full_reduction_compaction_and_validation():
    result = compare(Schedule(128, 768, 3072), 1536, .5)
    assert result["dense"]["tile_calls"] == 144
    assert result["common_k_compaction"]["tile_calls"] == 72
    assert (result["dense"]["sram_mediated_partial_sum_bytes"]
            - result["common_k_compaction"]["sram_mediated_partial_sum_bytes"]) == 9 * 1024**2
    with pytest.raises(ValueError):
        Schedule(0, 1, 1).actions()
    with pytest.raises(ValueError):
        compare(Schedule(1, 1, 1), 2, .5)
