#!/usr/bin/env python3
"""Rebuild compact CSV tables from the complete archived controller statistics."""

import argparse
import csv
import gzip
import json
from pathlib import Path

from archive_statistics import verify


PREFIX = "2026-11-24-controller-replay"


def read_record(directory, name):
    return json.loads(gzip.decompress((directory / f"{name}.json.gz").read_bytes()))


def write_table(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def contrasts(analysis):
    for row in analysis["analyses"]:
        for comparison in row["pairwise"]:
            yield "pixel_mse", row["bits"], comparison
    for row in analysis["descriptors"]["analyses"]:
        for comparison in row["pairwise"]:
            yield f"descriptor_{row['metric']}", row["bits"], comparison


def generate(statistics, output):
    verify(statistics)
    output.mkdir(parents=True, exist_ok=True)
    runs, calls = [], []
    for bits in (4, 8):
        report = read_record(statistics, f"controller-replay-a{bits}-20261124/report")
        for row in report["runs"]:
            identity = {"bits": bits, **{key: row[key] for key in (
                "image_id", "bundle", "bundle_input_index", "policy", "backend", "control")}}
            runs.append({**identity, "pixel_mse": row["image_vs_original"]["rmse"] ** 2,
                         "pixel_rmse": row["image_vs_original"]["rmse"],
                         "raw_rmse": row["raw_vs_original"]["rmse"],
                         "trace_sha256": row["trace_sha256"]})
            for call in row["calls"]:
                calls.append({**identity, **{key: call.get(key) for key in (
                    "event", "layer", "dense_macs", "source_scalars", "source_groups",
                    "clipped_values", "introduced_nonzeros", "invalid_events",
                    "scale_groups", "scale_mismatches", "mode_decisions", "mode_mismatches",
                    "mode_switches", "nonzero_feedback_positions",
                    "maximum_mean_state_difference", "maximum_second_state_difference")}})
    guard = read_record(statistics, "scale-stability-20261124")
    patches = [{key: row[key] for key in (
        "bundle", "input_index", "patch_index", "layer", "bits", "groups", "changed_scales",
        "local_bound_certified", "global_bound_certified", "global_bound")} for row in guard["records"]]
    analysis = read_record(statistics, "controller-replay-analysis-20261124")
    paired = []
    for metric, bits, row in contrasts(analysis):
        paired.append({"metric": metric, "bits": bits, **{key: row[key] for key in (
            "first", "second", "context", "n", "mean", "exact_sign_flip_p", "negative", "zero", "positive")},
            "bootstrap_95_low": row["bootstrap_percentile_95"][0],
            "bootstrap_95_high": row["bootstrap_percentile_95"][1],
            "original_vs_folded_sign_changes": row.get("original_vs_folded_sign_changes"),
            "folded_live_vs_frozen_sign_changes": row.get("folded_live_vs_frozen_sign_changes")})
    tables = {"runs": runs, "calls": calls, "patches": patches, "contrasts": paired}
    for name, rows in tables.items():
        write_table(output / f"{PREFIX}.{name}.csv", rows)
    return {name: len(rows) for name, rows in tables.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, default=Path(f"reports/statistics/{PREFIX}"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    print(generate(args.statistics, args.output))


if __name__ == "__main__":
    main()
