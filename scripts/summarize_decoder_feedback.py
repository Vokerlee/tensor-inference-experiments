#!/usr/bin/env python3
"""Export paired decoder fidelity, full-call statistics and auditable provenance."""

import argparse
import csv
import json
from pathlib import Path

from activation_pilot import digest, write_json


def load_report(path: Path) -> dict:
    report = json.loads(path.read_text())
    if report.get("status") != "complete":
        raise ValueError(f"incomplete report: {path}")
    if report["environment"]["experiments"]["dirty"] or report["environment"]["engine"]["dirty"]:
        raise ValueError(f"dirty measurement source: {path}")
    original = next(row for row in report["cases"] if row["case"] == "original")
    if not original["raw_vs_original"]["identical"]:
        raise ValueError(f"non-identical decoder replay: {path}")
    for case in report["cases"]:
        for call in case["calls"]:
            if call["invalid_events"]:
                raise ValueError(f"invalid native events: {path}")
            if sum(call["exponent_histogram"].values()) != call["source_groups"]:
                raise ValueError(f"incomplete group histogram: {path}")
    return report


def rows(report: dict, source: str) -> tuple[list[dict], list[dict], list[dict]]:
    cases = {case["case"]: case for case in report["cases"]}
    case_rows, image_rows, call_rows = [], [], []
    for name, case in cases.items():
        baseline = cases.get(f"none{name[-1]}") if name[-1:] in ("4", "8") else None
        error = case["image_vs_original"]["rmse"] ** 2
        denominator = baseline["image_vs_original"]["rmse"] ** 2 if baseline else None
        info = {
            "source": source, "case": name,
            "weight_bits": report["settings"]["weight_bits"] or "bf16",
            "inputs": len(case["per_input"]), "scope": report["settings"]["scope"],
            "eligible_convolution_macs": report["eligible_macs"],
            "all_convolution_macs": report["convolution_macs"],
            "eligible_convolution_mac_fraction": report["eligible_macs"] / report["convolution_macs"],
            **case["image_vs_original"],
            "mse_ratio_to_no_feedback": error / denominator if denominator else None,
            "clipped_values": sum(call["clipped_values"] for call in case["calls"]),
            "introduced_nonzeros": sum(call["introduced_nonzeros"] for call in case["calls"]),
            "source_scalars": sum(call["source_scalars"] for call in case["calls"]),
            "source_groups": sum(call["source_groups"] for call in case["calls"]),
        }
        case_rows.append(info)
        for image in case["per_input"]:
            base = baseline["per_input"][image["index"]] if baseline else None
            denom = base["image_vs_original"]["rmse"] ** 2 if base else None
            image_rows.append({
                "source": source, "case": name, "weight_bits": info["weight_bits"],
                "input_index": image["index"], **image["image_vs_original"],
                "mse_ratio_to_no_feedback": image["image_vs_original"]["rmse"] ** 2 / denom
                if denom else None,
            })
        for call in case["calls"]:
            call_rows.append({"source": source, "case": name,
                              **{key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list))
                                 else value for key, value in call.items()}})
    return case_rows, image_rows, call_rows


def write_csv(path: Path, records: list[dict]):
    with path.open("w", newline="") as stream:
        fields = list(dict.fromkeys(key for record in records for key in record))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True, help="output basename")
    args = parser.parse_args()
    collections = [[], [], []]
    provenance = []
    for path in args.input:
        report = load_report(path)
        for combined, values in zip(collections, rows(report, str(path))):
            combined.extend(values)
        provenance.append({"path": str(path), "sha256": digest(path),
                           "environment": report["environment"],
                           "checkpoint": report["checkpoint"], "input_sha256": report["input_sha256"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix, values in zip((".csv", ".images.csv", ".calls.csv"), collections):
        write_csv(Path(str(args.output) + suffix), values)
    write_json(Path(str(args.output) + ".provenance.json"), {"sources": provenance,
               "case_records": len(collections[0]), "image_records": len(collections[1]),
               "call_records": len(collections[2])})


if __name__ == "__main__":
    main()
