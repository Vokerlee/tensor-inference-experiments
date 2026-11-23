"""Prevent incomplete or corrupted decoder evidence from entering summaries."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_decoder_feedback import load_report, rows


def fixture():
    call = {"invalid_events": 0, "source_groups": 2, "source_scalars": 64,
            "exponent_histogram": {"-1": 1, "0": 1}, "clipped_values": 0,
            "introduced_nonzeros": 0}
    return {
        "status": "complete",
        "environment": {"experiments": {"dirty": False}, "engine": {"dirty": False}},
        "settings": {"weight_bits": None, "scope": "spatial"},
        "eligible_macs": 80, "convolution_macs": 100,
        "cases": [{"case": name, "raw_vs_original": {"identical": name == "original"},
                   "image_vs_original": {"rmse": error, "psnr_unit_range": None},
                   "per_input": [{"index": 0, "image_vs_original": {"rmse": error}}],
                   "calls": [] if name == "original" else [copy.deepcopy(call)]}
                  for name, error in (("original", 0), ("none4", 0.3), ("coarse4", 0.2))],
    }


def test_paired_mse_ratios_and_full_histograms_are_preserved():
    cases, images, calls = rows(fixture(), "test.json")
    assert cases[-1]["mse_ratio_to_no_feedback"] == pytest.approx(4 / 9)
    assert images[-1]["mse_ratio_to_no_feedback"] == pytest.approx(4 / 9)
    assert cases[-1]["eligible_convolution_mac_fraction"] == 0.8
    assert cases[0]["mse_ratio_to_no_feedback"] is None
    assert json.loads(calls[-1]["exponent_histogram"]) == {"-1": 1, "0": 1}


@pytest.mark.parametrize("corruption", ["incomplete", "dirty", "invalid", "histogram"])
def test_inadmissible_reports_are_rejected(tmp_path, corruption):
    report = fixture()
    if corruption == "incomplete":
        report["status"] = "running"
    elif corruption == "dirty":
        report["environment"]["experiments"]["dirty"] = True
    elif corruption == "invalid":
        report["cases"][-1]["calls"][0]["invalid_events"] = 1
    else:
        report["cases"][-1]["calls"][0]["exponent_histogram"] = {}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        load_report(path)
