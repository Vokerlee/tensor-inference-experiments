#!/usr/bin/env python3
"""Paired, input-level analysis of complete controller-replay experiments."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from activation_pilot import digest, write_json


CONTEXTS = (("original", "live"), ("folded", "live"), ("folded", "frozen"))


def paired_summary(values, *, samples=20000, seed=20261124):
    """Exploratory paired bootstrap and exact two-sided sign-flip test of a mean."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not 1 <= values.size <= 16 or not np.isfinite(values).all():
        raise ValueError("expected 1..16 finite paired observations")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(values.size, size=(samples, values.size))].mean(1)
    observed = abs(float(values.mean()))
    permutation = [abs(float(np.dot(signs, values) / values.size))
                   for signs in itertools.product((-1, 1), repeat=values.size)]
    return {"n": values.size, "mean": float(values.mean()),
            "bootstrap_percentile_95": np.quantile(means, [0.025, 0.975]).tolist(),
            "exact_sign_flip_p": sum(value >= observed for value in permutation) / len(permutation),
            "negative": int((values < 0).sum()), "zero": int((values == 0).sum()),
            "positive": int((values > 0).sum()), "per_input": values.tolist()}


def summarize(report):
    if report.get("status") != "complete":
        raise ValueError("only complete reports accepted")
    count = report["processed_inputs"]
    if count < 1 or len({row["image_id"] for row in report["runs"]}) != count:
        raise ValueError("invalid image coverage")
    grouped = {}
    for row in report["runs"]:
        key = (row["policy"], row["backend"], row["control"])
        cases = grouped.setdefault(key, {})
        if row["image_id"] in cases:
            raise ValueError("duplicate run identity")
        cases[row["image_id"]] = row
    if any(set(rows) != set(range(count)) for rows in grouped.values()):
        raise ValueError("incomplete paired coverage")
    errors = {key: np.array([rows[i]["image_vs_original"]["rmse"] ** 2
                            for i in range(count)]) for key, rows in grouped.items()}
    policies = report["policies"]
    for policy in policies:
        if not np.array_equal(errors[(policy, "original", "live")],
                              errors[(policy, "original", "replay")]):
            raise ValueError("same-backend replay metrics differ")
    result = {"bits": report["bits"], "inputs": count, "recorded_decodes": len(report["runs"]),
              "mean_pixel_mse": {"/".join(key): float(value.mean()) for key, value in errors.items()},
              "pairwise": [], "control": [], "decompositions": [], "layer_control": []}
    for first, second in itertools.combinations(policies, 2):
        differences = {}
        for backend, control in CONTEXTS:
            values = errors[(first, backend, control)] - errors[(second, backend, control)]
            differences[(backend, control)] = values
            result["pairwise"].append({"first": first, "second": second,
                                      "context": f"{backend}/{control}", **paired_summary(values)})
        original = differences[("original", "live")]
        live = differences[("folded", "live")]
        frozen = differences[("folded", "frozen")]
        result["pairwise"].append({"first": first, "second": second,
            "context": "folded live-minus-frozen policy difference", **paired_summary(live - frozen),
            "original_vs_folded_sign_changes": int(((original * live) < 0).sum()),
            "folded_live_vs_frozen_sign_changes": int(((frozen * live) < 0).sum())})
    for policy in policies:
        rows = grouped[(policy, "folded", "live")]
        by_layer = {}
        for image_id, row in rows.items():
            for call in row["calls"]:
                by_layer.setdefault(call["layer"], []).append(call)
        sums = {field: sum(call[field] for row in rows.values() for call in row["calls"])
                for field in ("scale_groups", "scale_mismatches", "mode_decisions", "mode_mismatches")}
        result["control"].append({"policy": policy, **sums,
            "scale_flip_fraction": sums["scale_mismatches"] / sums["scale_groups"],
            "mode_flip_fraction": sums["mode_mismatches"] / sums["mode_decisions"]})
        for layer, calls in by_layer.items():
            result["layer_control"].append({"policy": policy, "layer": layer,
                **{field: sum(call[field] for call in calls) for field in sums},
                "maximum_mean_state_difference": max(call["maximum_mean_state_difference"] for call in calls),
                "maximum_second_state_difference": max(call["maximum_second_state_difference"] for call in calls)})
        decompositions = [row for row in report["decompositions"] if row["policy"] == policy]
        if {row["image_id"] for row in decompositions} != set(range(count)):
            raise ValueError("incomplete decomposition coverage")
        result["decompositions"].append({"policy": policy,
            **{space: {field: sum(row[space][field] for row in decompositions)
                       for field in ("data_squared", "control_squared", "twice_inner_product", "total_squared")}
               for space in ("raw", "pixels")},
            "maximum_identity_residual": max(row[space]["maximum_identity_residual"]
                for row in decompositions for space in ("raw", "pixels"))})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"schema": 1, "status": "complete", "analyses": [], "limitations": [
        "Unit of resampling is an input, never a repeated replay or individual decision",
        "Eight-input comparisons are exploratory; no population or corrected confirmatory significance claim",
        "Percentile intervals assume exchangeable sampled inputs; prompts here are curated, not random population draws",
        "Sign-flip p values assume symmetry/exchangeability under the null; all comparisons are retained without selection",
        "Pixel fidelity to one declared backend is not perceptual quality or target-device correctness",
        "Counterfactual components interact; their squared norms are not unique error fractions",
        "Decision counts and backend elapsed time are not target energy or latency",
    ]}
    for path in args.input:
        report["analyses"].append({"source": str(path), "source_sha256": digest(path),
                                   **summarize(json.loads(path.read_text()))})
    write_json(args.output, report)


if __name__ == "__main__":
    main()
