"""Reject incomplete/misattributed data before numerical interpretation."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import summarize_paired_precision as summary


def test_summaries_weight_work_counts_and_preserve_histograms(monkeypatch):
    monkeypatch.setattr(summary, "verify_sources", lambda report: None)
    layers = {name: {**dict.fromkeys(summary.COUNTERS, 0),
                     "useful_products": work, "fp8_products": low,
                     "reference_vectors": 10, "selected_vectors": selected,
                     "activation_scale_exponents": {"-8": 17},
                     "normalized_bound_histogram": {"3": 7}}
              for name, work, low, selected in (("large", 900, 90, 9), ("small", 100, 100, 5))}
    case = {"case": "adaptive-paired:0.04", "layers": layers,
            "activation_summaries": {name: [] for name in layers},
            "skipped_dtype_calls": {name: 0 for name in layers}}
    report = {"model": "example", "target_layers": list(layers), "cases": [case],
              "case_order": [case["case"]], "prepared_weight_statistics": {},
              "dataset": {"manifest": [{"path": "image"}]}}
    rows, details = summary.summarize(report)
    assert rows[0]["fp8_product_fraction"] == 0.19
    assert rows[0]["output_vector_reduction_fraction"] == 0.3
    assert len(details) == 2
    assert details[0]["activation_scale_exponents"] == '{"-8": 17}'
    report["settings"] = {"minimum_fp8_rows": 2, "cohort": 32}
    ablation_rows, ablation_layers = summary.summarize(report)
    assert ablation_rows[0]["minimum_fp8_rows"] == 2
    assert ablation_rows[0]["record_sha256"] != rows[0]["record_sha256"]
    assert ablation_layers[0]["record_sha256"] == ablation_rows[0]["record_sha256"]
    report["case_order"].append("unfinished")
    with pytest.raises(ValueError, match="incomplete"):
        summary.summarize(report)
