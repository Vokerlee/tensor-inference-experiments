#!/usr/bin/env python3
"""Regenerate paired task tables from complete archived raw reports."""

import argparse
import csv
import gzip
import io
import json
from pathlib import Path

from activation_pilot import ROOT, digest
from archive_feature_task import MANIFEST, load_report, verify
from feature_task_analysis import summarize


def table(path: Path, rows: list[dict]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    payload = stream.getvalue().encode()
    path.write_bytes(gzip.compress(payload, mtime=0) if path.suffix == ".gz" else payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs=2, action="append", metavar=("NAME", "ARCHIVE"), required=True)
    parser.add_argument("--replication", default="replication")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = [name for name, _ in args.input]
    if len(names) != len(set(names)):
        parser.error("duplicate run names")
    policies, comparisons, queries = [], [], []
    summary = {"schema": 1, "runs": {},
               "source_sha256": {name: digest(ROOT / "scripts" / name) for name in (
                   "report_feature_task.py", "feature_task_analysis.py", "archive_feature_task.py")}}
    allocations = set()
    for name, directory in args.input:
        directory = Path(directory)
        verify(directory)
        report = load_report(directory)
        current = set(report["test_dataset"]["indices"])
        if current & allocations:
            raise ValueError("discovery and replication query allocations overlap")
        allocations |= current
        result = summarize(report, primary_run=name == args.replication)
        result["manifest_sha256"] = digest(directory / MANIFEST)
        result["experiment_source"] = report["experiments"]
        result["checkpoint"] = report["checkpoint"]
        result["settings"] = report["settings"]
        summary["runs"][name] = result
        for row in result["policies"]:
            policies.append({"run": name, **{key: row[key] for key in (
                "policy", "queries", "correct", "accuracy", "mean_relative_mse", "median_relative_mse")}})
        for row in result["primary_comparisons"]:
            comparisons.append({"run": name, "candidate": row["candidate"], "reference": row["reference"],
                                "queries": row["queries"], "benefit": row["benefit"], "harm": row["harm"],
                                "accuracy_difference": row["accuracy_difference"],
                                "bootstrap_95_lower": row["stratified_bootstrap_95"][0] if row["stratified_bootstrap_95"] else None,
                                "bootstrap_95_upper": row["stratified_bootstrap_95"][1] if row["stratified_bootstrap_95"] else None,
                                "family_lower": row["primary_family_one_sided_lower"],
                                "margin": row["margin"],
                                "mcnemar_two_sided": row["exact_mcnemar_two_sided"],
                                "meets_predeclared_bootstrap_screen": row["meets_predeclared_bootstrap_screen"]})
        for row in report["queries"]:
            for policy in report["settings"]["policies"]:
                queries.append({"run": name, "query": row["image_id"], "target": row["target"],
                                "policy": policy, "prediction": row["predictions"][policy],
                                "correct": row["predictions"][policy] == row["target"],
                                "relative_mse": row["relative_mse"][policy]})
    table(args.output / "policies.csv", policies)
    table(args.output / "paired.csv", comparisons)
    table(args.output / "queries.csv.gz", queries)
    payload = (json.dumps(summary, separators=(",", ":"), allow_nan=False) + "\n").encode()
    (args.output / "summary.json.gz").write_bytes(gzip.compress(payload, mtime=0))
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
