#!/usr/bin/env python3
"""Independently reconstruct task decisions from archived full descriptors."""

import argparse
import gzip
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from activation_pilot import digest, write_json
from archive_feature_task import MANIFEST, load_report, verify


def audit(directory: Path, batch_size: int = 256) -> dict:
    count = verify(directory)
    report = load_report(directory)
    manifest = json.loads((directory / MANIFEST).read_text())
    with np.load(directory / "gallery.npz", allow_pickle=False) as arrays:
        gallery, labels = arrays["features"].astype(np.float64), arrays["labels"].copy()
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
    expected = {row["file"]: row for row in report["queries"]}
    pending, seen, outputs = [], set(), 0
    encoding_counts = defaultdict(int)
    correct = Counter()
    maximum_similarity_difference = 0.0
    reference_differences = 0
    selected_macs = set()

    def flush():
        nonlocal outputs, maximum_similarity_difference
        if not pending:
            return
        features = np.concatenate([np.asarray(row["output"], dtype=np.float64) for row in pending])
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        similarities = features @ gallery.T
        for record, similarity in zip(pending, similarities, strict=True):
            neighbors = np.argsort(-similarity, kind="stable")[:report["settings"]["neighbors"]]
            stored = np.asarray(record["neighbors"])[0]
            if not np.array_equal(neighbors, stored):
                raise ValueError("independent complete-gallery neighbor order differs")
            counts = Counter(int(label) for label in labels[neighbors])
            highest = max(counts.values())
            prediction = next(int(label) for label in labels[neighbors] if counts[int(label)] == highest)
            if prediction != record["prediction"]:
                raise ValueError("independent class vote differs")
            difference = float(np.abs(similarity[neighbors] - np.asarray(record["similarity"])[0]).max())
            maximum_similarity_difference = max(maximum_similarity_difference, difference)
            if difference > 1e-12:
                raise ValueError("independent cosine similarity differs beyond tolerance")
            outputs += 1
        pending.clear()

    for shard_record in manifest["shards"]:
        with zipfile.ZipFile(directory / shard_record["file"]) as shard:
            for member in shard_record["members"]:
                name = member["file"]
                query = json.loads(gzip.decompress(shard.read(name)))
                index = expected[name]
                query_id = query["image_id"]
                if (query_id in seen or query["target"] != index["target"]
                        or query["dataset_index"] != report["test_dataset"]["indices"][query_id]):
                    raise ValueError("query identity or target differs from producer selection")
                seen.add(query_id)
                reference = np.asarray(query["split_reference"], dtype=np.float64)
                original = np.asarray(query["original_reference"], dtype=np.float64)
                reference_differences += int(not np.array_equal(reference, original))
                if {row["policy"] for row in query["runs"]} != set(report["settings"]["policies"]):
                    raise ValueError("missing or unexpected policy output")
                for row in query["runs"]:
                    policy = row["policy"]
                    if not row["finite"] or row["prediction"] != index["predictions"][policy]:
                        raise ValueError("nonfinite output or prediction mismatch")
                    actual = np.asarray(row["output"], dtype=np.float64)
                    squared = float(np.square(actual - reference).sum())
                    relative = squared / float(np.square(reference).sum())
                    if (not math.isclose(squared, row["matched_error"]["squared_error"], rel_tol=1e-10, abs_tol=1e-12)
                            or not math.isclose(relative, index["relative_mse"][policy], rel_tol=1e-10, abs_tol=1e-12)):
                        raise ValueError("independent descriptor error differs")
                    if policy == "bf16" and not np.array_equal(actual, reference):
                        raise ValueError("matched BF16 control differs")
                    correct[policy] += row["prediction"] == query["target"]
                    selected_macs.add(sum(call["macs"] for call in row["other_calls"] + row["attention_calls"]))
                    encodings = set()
                    for call in row["attention_calls"]:
                        if call["bits"] != 16:
                            encodings.add((call["layer"], call["bits"], call["rank"], call["metric"]))
                    encoding_counts[policy] += len(encodings)
                    pending.append(row)
                    if len(pending) >= batch_size:
                        flush()
    flush()
    if len(seen) != count or len(selected_macs) != 1:
        raise ValueError("incomplete output audit or unmatched selected matrix MACs")
    return {"schema": 1, "queries": count, "complete_output_checks": outputs,
            "correct": dict(sorted(correct.items())),
            "matched_reference_different_from_fused": reference_differences,
            "selected_linear_macs_per_forward": selected_macs.pop(),
            "non_bf16_distinct_attention_encodings": dict(sorted(encoding_counts.items())),
            "maximum_independent_cosine_difference": maximum_similarity_difference,
            "manifest_sha256": digest(directory / MANIFEST), "auditor_sha256": digest(Path(__file__)),
            "limitations": ["Audits preserved outputs/decisions, not a new model inference or population accuracy",
                            "Attention encoding counts deduplicate role sharing; they are not hardware energy"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
