#!/usr/bin/env python3
"""Export measured resize-folding fidelity and declared dense arithmetic."""

import argparse
import json
from pathlib import Path

from activation_pilot import digest, write_json
from summarize_decoder_feedback import write_csv


def rows(report: dict, source: str) -> list[dict]:
    if report.get("status") != "complete":
        raise ValueError("incomplete resize report")
    if any(report["environment"][repo]["dirty"] for repo in ("experiments", "engine")):
        raise ValueError("dirty measurement source")
    if not report["cases"][0]["raw_vs_original"]["identical"]:
        raise ValueError("original replay mismatch")
    cases = {case["case"]: case for case in report["cases"]}
    result = []
    for name, case in cases.items():
        quantizers = [call["quantizer"] for call in case["calls"] if "quantizer" in call]
        if any(q["invalid_events"] for q in quantizers):
            raise ValueError("invalid quantizer event")
        if any(sum(q["exponent_histogram"].values()) != q["source_groups"] for q in quantizers):
            raise ValueError("incomplete source-group histogram")
        original_macs = sum(call["expanded_macs"] for call in case["calls"])
        reference_macs = sum(call["executed_reference_macs"] for call in case["calls"])
        base = cases.get(name.removesuffix("-feedback")) if name.endswith("-feedback") else None
        denominator = base["image_vs_original"]["rmse"] ** 2 if base else None
        result.append({
            "source": source, "case": name, "inputs": len(case["per_input"]),
            **case["image_vs_original"],
            "feedback_mse_ratio": case["image_vs_original"]["rmse"] ** 2 / denominator
            if denominator else None,
            "original_branch_macs": original_macs, "reference_branch_macs": reference_macs,
            "reference_branch_mac_ratio": reference_macs / original_macs,
            "source_scalars_quantized": sum(q["source_scalars"] for q in quantizers),
            "source_groups_quantized": sum(q["source_groups"] for q in quantizers),
            "clipped_values": sum(q["clipped_values"] for q in quantizers),
            "introduced_nonzeros": sum(q["introduced_nonzeros"] for q in quantizers),
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary, provenance = [], []
    for path in args.input:
        report = json.loads(path.read_text())
        summary.extend(rows(report, str(path)))
        provenance.append({"path": str(path), "sha256": digest(path),
                           "environment": report["environment"],
                           "source_sha256": report["source_sha256"],
                           "input_sha256": report["input_sha256"], "checkpoint": report["checkpoint"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output.with_suffix(".csv"), summary)
    write_json(args.output.with_suffix(".provenance.json"), {"sources": provenance,
                                                           "case_records": len(summary)})


if __name__ == "__main__":
    main()
