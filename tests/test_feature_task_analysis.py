"""Paired inference must preserve discordance and avoid degenerate certainty."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from feature_task_analysis import exact_mcnemar, paired


def test_equal_marginal_accuracy_can_have_different_paired_uncertainty():
    reference = [True] * 40 + [False] * 40
    same = paired(reference, reference, [0] * 80)
    switched = paired(list(reversed(reference)), reference, [0] * 80)
    assert same["accuracy_difference"] == switched["accuracy_difference"] == 0
    assert same["bootstrap_degenerate"]
    assert switched["stratified_bootstrap_95"][0] < 0 < switched["stratified_bootstrap_95"][1]
    assert same["primary_family_one_sided_lower"] < -0.01


def test_stratified_bootstrap_holds_class_allocations_fixed():
    result = paired([True] * 10 + [False] * 10, [False] * 20, [0] * 10 + [1] * 10)
    assert result["stratified_bootstrap_95"] == [0.5, 0.5]
    assert result["benefit"] == 10 and result["harm"] == 0
    assert result["exact_mcnemar_two_sided"] == 2 / 1024


def test_exact_discordance_tail_and_invalid_inputs():
    assert exact_mcnemar(0, 0) == exact_mcnemar(3, 3) == 1
    assert exact_mcnemar(1, 5) == 14 / 64
    with pytest.raises(ValueError):
        paired([True], [True, False], [0])
    with pytest.raises(ValueError):
        paired([2], [0], [0])
    assert np.isfinite(exact_mcnemar(1000, 1100))


def test_one_query_per_class_cannot_produce_a_within_class_interval():
    result = paired([True, False, True], [False, True, False], [0, 1, 2])
    assert not result["stratified_bootstrap_estimable"]
    assert result["stratified_bootstrap_95"] is None
    assert result["primary_family_one_sided_lower"] is None
