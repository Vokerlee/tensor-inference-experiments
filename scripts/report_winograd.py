#!/usr/bin/env python3
"""Audit preserved Winograd outputs and regenerate complete comparison tables."""

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np

from activation_pilot import ROOT, digest, write_json
from archive_selective_reference_evidence import attachment_paths, verify_all


PRELIMINARY = {
    "native-integer-winograd-discovery-20261124": "superseded scale-retirement contract; interrupted",
    "winograd-mapping-20261124": "source marked dirty; clean ledger archived separately",
}


def scalar_metrics(values, reference):
    """Independent NumPy calculation; no producer metric helper is imported."""
    error = values.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(np.square(error)))
    energy = float(np.mean(np.square(reference.astype(np.float64))))
    return {
        "rmse": math.sqrt(mse), "mean_error": float(error.mean()),
        "maximum_absolute_error": float(np.abs(error).max()),
        "relative_mse": mse / energy if energy else None,
        "psnr_unit_range": -10 * math.log10(mse) if mse else None,
        "identical": bool(np.array_equal(values, reference)),
    }


def audit_fields(actual, recorded):
    for key, value in actual.items():
        expected = recorded[key]
        if value is None or isinstance(value, bool):
            valid = value == expected
        else:
            valid = expected is not None and math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-12)
        if not valid:
            raise ValueError(f"preserved output disagrees with recorded {key}: {value} != {expected}")


def write_table(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def decoder_rows(report, campaign, attachments):
    records, inputs, calls = [], [], []
    original = next(case for case in report["cases"] if case["case"] == "original")
    with np.load(attachments[original["arrays"].removeprefix("results/")], allow_pickle=False) as data:
        reference = {key: data[key].copy() for key in ("raw", "images")}
    for case in report["cases"]:
        with np.load(attachments[case["arrays"].removeprefix("results/")], allow_pickle=False) as data:
            for array, metric in (("raw", "raw_vs_original"), ("images", "image_vs_original")):
                audit_fields(scalar_metrics(data[array], reference[array]), case[metric])
                for index, recorded in enumerate(case["per_input"]):
                    audit_fields(scalar_metrics(data[array][index], reference[array][index]), recorded[metric])
        common = {"campaign": campaign, "case": case["case"],
                  "preliminary": campaign in PRELIMINARY,
                  "minimum_outputs": report["settings"].get("minimum_outputs", 0),
                  "software_tile_batch": report["settings"]["tile_batch"]}
        records.append({**common, "inputs": report["settings"]["count"],
                        **case["image_vs_original"], "layer_calls": len(case["calls"]),
                        "integer_partial_vectors": sum(call.get("integer_partial_vectors", 0)
                                                       for call in case["calls"])})
        for row in case["per_input"]:
            inputs.append({**common, "input_index": row["index"], **row["image_vs_original"]})
        for call in case["calls"]:
            calls.append({**common, "layer": call["layer"], "input_index": call["input_index"],
                          **{key: call.get(key) for key in (
                              "K", "N", "activation_bits", "weight_bits", "original_channels", "lowering", "lowered_reduction",
                              "padded_channels", "activation_groups", "zero_groups", "integer_partial_vectors",
                              "output_dequant_scalar_multiplications")},
                          "rounding": json.dumps(call.get("rounding"), sort_keys=True, separators=(",", ":")),
                          "exponent_histogram": json.dumps(call.get("exponent_histogram"), sort_keys=True)})
    return records, inputs, calls


def operator_rows(report, campaign, attachments):
    rows = []
    for operator in report["operators"]:
        with np.load(attachments[operator["arrays"].removeprefix("results/")], allow_pickle=False) as data:
            reference = data["reference"].astype(np.float64)
        is_axis = "method" in operator["variants"][0]
        baselines = {row["bits"]: row["squared_error"] for row in operator["variants"]
                     if (row["method"] == "two-axis" if is_axis else not row["rank"])}
        for variant in operator["variants"]:
            with np.load(attachments[variant["arrays"].removeprefix("results/")], allow_pickle=False) as data:
                error = data["output"].astype(np.float64) - reference
                squared = float(np.square(error).sum())
                energy = float(np.square(reference).sum())
                audit_fields({"squared_error": squared, "relative_mse": squared / energy if energy else None,
                              "maximum_absolute_error": float(np.abs(error).max())}, variant)
            rows.append({"campaign": campaign, "layer": operator["layer"],
                         "bits": variant["bits"], "rank": variant.get("rank"),
                         "method": variant.get("method", "plane-proxy"),
                         "squared_error": squared, "relative_mse": variant["relative_mse"],
                         "sse_ratio_to_comparator": squared / baselines[variant["bits"]],
                         "comparator": "two-axis" if is_axis else "nearest",
                         "source_zeros_changed": variant["statistics"].get("source_zeros_changed", 0)})
    return rows


def generate(archives, output):
    if isinstance(archives, Path):
        archives = [archives]
    verified = [verify_all(archive) for archive in archives]
    attachments, entries = {}, []
    for archive in archives:
        current = attachment_paths(archive)
        if attachments.keys() & current.keys():
            raise ValueError("overlapping logical attachment paths")
        attachments.update(current)
        index = json.loads((archive / "index.json").read_text())
        entries.extend((archive, entry) for entry in index["files"])
    output.mkdir(parents=True, exist_ok=True)
    cases, inputs, calls, operators, mapping, retention, provenance = [], [], [], [], [], [], []
    campaigns = set()
    for archive, entry in entries:
        report = json.loads(gzip.decompress((archive / entry["archive"]).read_bytes()))
        campaign = Path(entry["source"]).parts[0]
        if campaign in campaigns:
            raise ValueError("duplicate campaign identity")
        campaigns.add(campaign)
        if report.get("status") != "complete" and campaign not in PRELIMINARY:
            raise ValueError(f"unexpected incomplete campaign {campaign}")
        if report["experiments"]["dirty"] and campaign not in PRELIMINARY:
            raise ValueError(f"unexpected dirty source {campaign}")
        provenance.append({"campaign": campaign, "status": report["status"],
                           "source_commit": report["experiments"]["commit"],
                           "source_dirty": report["experiments"]["dirty"],
                           "interpretation": PRELIMINARY.get(campaign, "canonical"),
                           "native_sha256": report.get("native_sha256"),
                           "json_payload_sha256": entry["payload_sha256"]})
        if "cases" in report:
            current = decoder_rows(report, campaign, attachments)
            cases.extend(current[0])
            inputs.extend(current[1])
            calls.extend(current[2])
        elif "operators" in report:
            operators.extend(operator_rows(report, campaign, attachments))
            for operator in report["operators"]:
                mapping.extend({"campaign": campaign, "layer": operator["layer"], **row}
                               for row in operator.get("work", []))
        elif "totals" in report:
            mapping.extend({"campaign": campaign, **row} for row in report["totals"])
            retention.extend({"campaign": campaign, **row} for row in report["retention_variants"])
        else:
            raise ValueError(f"unrecognized report {campaign}")
    for name, rows in (("decoder-cases", cases), ("decoder-inputs", inputs), ("layer-calls", calls),
                       ("operators", operators), ("mapping", mapping), ("retention", retention)):
        write_table(output / f"{name}.csv", rows)
    operator_summary = []
    for bits, rank in sorted({(row["bits"], row["rank"]) for row in operators if row["rank"] is not None}):
        rows = [row for row in operators if (row["bits"], row["rank"]) == (bits, rank)]
        baseline = [row for row in operators if row["bits"] == bits and row["rank"] == 0]
        operator_summary.append({"bits": bits, "rank": rank, "layers": len(rows),
                                 "aggregate_sse_ratio": sum(row["squared_error"] for row in rows)
                                                        / sum(row["squared_error"] for row in baseline),
                                 "improved_layers": sum(row["sse_ratio_to_comparator"] < 1 for row in rows)})
    summary = {"schema": 1, "source_sha256": digest(Path(__file__)), "provenance": provenance,
               "archive_records": sum(row[0] for row in verified),
               "logical_attachments": sum(row[1] for row in verified),
               "decoder_container_audits": len(cases), "per_input_audits": len(inputs),
               "operator_container_audits": len(operators), "operator_summary": operator_summary,
               "limitations": [
                   "Decoder cases reuse one discovery and four older holdout inputs; not independent new prompts",
                   "Pixel fidelity is not generation task quality or an energy estimate",
                   "Preliminary records are preserved and audited, but excluded from canonical interpretation",
                   "Floating group32 and proposed integer K256 numerical contracts must not be pooled",
                   "Shape ledgers describe interface capacity, not energy or mandatory SRAM transactions"]}
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archives = args.archive or sorted((ROOT / "reports/statistics").glob("2026-11-24-winograd-*"))
    if not archives:
        parser.error("at least one statistical archive required")
    summary = generate(archives, args.output)
    print(f"Audited {summary['decoder_container_audits']} decoder and "
          f"{summary['operator_container_audits']} operator containers")


if __name__ == "__main__":
    main()
