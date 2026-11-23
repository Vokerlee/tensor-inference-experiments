"""Check energy bookkeeping against explicit small action ledgers and boundaries."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from accumulation_cost_envelope import Counts, Costs, box_envelope, evaluate, required_merge_ratio


def test_large_merge_reduction_can_still_increase_total_cost():
    counts = Counts(products=128, reference_merges=3, selected_merges=1, decisions=1)
    result = evaluate(counts, Costs(4, 2, 2, 0.05))
    # Independent ledger: reference128+3*4+256=396;
    # candidate128+1*4+256+2+6.4=396.4 despite removing2/3 of merges.
    assert result["reference_cost_mac_units"] == 396
    assert result["candidate_cost_mac_units"] == pytest.approx(396.4)
    assert result["net_fraction_saved"] < 0


def test_target_threshold_is_a_true_cost_crossing():
    counts = Counts(16384, 384, 48, 1)
    threshold = required_merge_ratio(counts, target=0.1, common=1, guard=32, extra=0.01)
    below = evaluate(counts, Costs(threshold * 0.999, 1, 32, 0.01))
    above = evaluate(counts, Costs(threshold * 1.001, 1, 32, 0.01))
    assert below["net_fraction_saved"] < 0.1 < above["net_fraction_saved"]


def test_no_finite_merge_price_can_exceed_eliminated_merge_fraction():
    counts = Counts(128, 4, 3, 1)
    assert required_merge_ratio(counts, target=0.25, common=0) is None
    assert required_merge_ratio(counts, target=0.3, common=4) is None
    assert required_merge_ratio(counts, target=0.1, common=0) is not None


def test_negative_activity_results_are_retained_and_box_bounds_cover_interior():
    counts = Counts(128, 2, 3, 1)
    bounds = {"merge_to_mac": [1, 32], "common_nonmac_to_mac_total": [0, 4],
              "guard_to_mac": [0, 128], "extra_per_product_to_mac": [0, 0.05]}
    result = box_envelope(counts, bounds)
    middle = evaluate(counts, Costs(8, 1, 32, 0.01))["net_fraction_saved"]
    assert result["status"] == "nonpositive_throughout_box"
    assert result["minimum"]["net_fraction_saved"] <= middle <= result["maximum"]["net_fraction_saved"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1])
def test_uncharacterized_cost_is_never_silently_replaced_by_valid_number(bad):
    with pytest.raises(ValueError):
        Costs(bad, 0, 0, 0)
