"""Distinguish changed numerical predictions from changed source/data in a replay."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from reproduce_primitive_pilots import validate


def records():
    report = {
        "checkpoint": {"sha256": "checkpoint"}, "manifest": [{"target": 1}],
        "experiments": {"commit": "source", "dirty": False},
        "engine": {"commit": "engine", "dirty": False}, "source_sha256": {"script": "hash"},
        "cases": [{"name": "baseline", "predictions": [1, 2, 3]}],
    }
    expected = copy.deepcopy(report)
    expected["predictions"] = {"baseline": [1, 9, 3]}
    return report, expected


def test_prediction_changes_are_reported_after_source_and_data_validation():
    report, expected = records()
    result = validate(report, expected, report["manifest"])
    assert result["sources_match"]
    assert result["prediction_differences"] == {"baseline": 1}


@pytest.mark.parametrize("field", ["manifest", "checkpoint", "source_sha256", "engine"])
def test_changed_data_or_source_is_rejected(field):
    report, expected = records()
    report[field] = [] if field == "manifest" else {}
    with pytest.raises(ValueError):
        validate(report, expected, expected["manifest"])
