#!/usr/bin/env python3
"""Regenerate rounding tables and independently audit preserved statistics."""

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np

from activation_pilot import ROOT, digest, write_json
from archive_selective_reference_evidence import verify_all


CAMPAIGNS = (
    "activation-rounding-operators-20261124",
    "activation-rounding-adjacent-operators-20261124",
    "activation-rounding-dinov3-20261124",
    "attention-rounding-dinov3-20261124",
)


def table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def attachment(root, original):
    parts = Path(original).parts
    if not parts or parts[0] != "results" or ".." in parts:
        raise ValueError("unexpected historical attachment path")
    return root / Path(*parts[1:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path,
                        default=ROOT / "reports/statistics/2026-11-24-activation-rounding")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verified = verify_all(args.archive)
    args.output.mkdir(parents=True, exist_ok=True)
    operators, models, means, provenance = [], [], [], []
    array_checks = 0
    for campaign in CAMPAIGNS:
        path = args.archive / campaign / "report.json.gz"
        report = json.loads(gzip.decompress(path.read_bytes()))
        if report["status"] != "complete" or report["invalid_cases"]:
            raise ValueError("incomplete campaign or recorded invalid cases")
        provenance.append({"campaign": campaign, "source_commit": report["experiments"]["commit"],
                           "source_dirty": report["experiments"]["dirty"], "archive_sha256": digest(path)})
        for operator in report.get("operators", []):
            for row in operator["records"]:
                with np.load(attachment(args.archive, row["arrays"]), allow_pickle=False) as arrays:
                    for field, array in (("source_sse", "source_group_sse"),
                                         ("receiving_sse", "output_tile_receiving_sse"),
                                         ("isolated_group_sse", "isolated_group_receiving_sse")):
                        actual = float(arrays[array].astype(np.float64).sum())
                        if not math.isclose(actual, row[field], rel_tol=1e-10, abs_tol=1e-9):
                            raise ValueError(f"raw array disagrees with {campaign}: {field}")
                    array_checks += 1
                operators.append({"campaign": campaign, "model": operator["model"],
                                  "layer": operator["layer"], "bits": row["bits"],
                                  "rank": row.get("rank", 0), "method": row["method"],
                                  **{key: row[key] for key in (
                                      "source_sse", "receiving_sse", "isolated_group_sse", "cross_group_sse_term")},
                                  "receiving_sse_ratio": row.get("receiving_sse_ratio", 1),
                                  "worse_isolated_groups": row.get("worse_isolated_groups", 0),
                                  "exact_source_zeros_changed": row["exact_source_zeros_changed"]})
        for row in report.get("runs", []):
            reference = np.asarray(row.get("split_reference", row.get("reference")), dtype=np.float64)
            output = np.asarray(row["output"], dtype=np.float64)
            error = row.get("matched_error", row)
            squared = float(np.square(output - reference).sum())
            if not math.isclose(squared, error["squared_error"], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("complete output vector disagrees with error statistic")
            models.append({"campaign": campaign, "image_id": row["image_id"],
                           "policy": row["policy"], "finite": row["finite"],
                           "squared_error": squared, "relative_mse": error["relative_mse"],
                           "cosine_distance": error["cosine_distance"], "maximum_error": error["maximum_error"]})
    for campaign in CAMPAIGNS[2:]:
        selected = [row for row in models if row["campaign"] == campaign]
        for policy in sorted({row["policy"] for row in selected}):
            rows = [row for row in selected if row["policy"] == policy]
            means.append({"campaign": campaign, "policy": policy, "images": len(rows),
                          "mean_relative_mse": float(np.mean([row["relative_mse"] for row in rows])),
                          "mean_cosine_distance": float(np.mean([row["cosine_distance"] for row in rows]))})
    table(args.output / "operators-preliminary.csv", [row for row in operators if row["campaign"] == CAMPAIGNS[0]])
    table(args.output / "operators-adjacent.csv", [row for row in operators if row["campaign"] == CAMPAIGNS[1]])
    table(args.output / "model-cases.csv", models)
    table(args.output / "model-means.csv", means)
    write_json(args.output / "summary.json", {
        "schema": 1, "source_sha256": digest(Path(__file__)), "provenance": provenance,
        "archive_records": verified[0], "archive_attachments": verified[1],
        "operator_array_checks": array_checks, "complete_descriptor_checks": len(models),
        "means": means, "limitations": [
            "Preliminary exact-grid alternative differs from corrected adjacent rounding; do not pool",
            "Both model campaigns reuse the same ten Imagenette images",
            "Local receiving improvement is not full-model task quality or target-chip energy",
            "Frozen-feature CIFAR100 task results are archived and analyzed separately"]})
    print(f"Audited {array_checks} operator containers and {len(models)} complete descriptors")


if __name__ == "__main__":
    main()
