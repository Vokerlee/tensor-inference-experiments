"""Paired fixed-class task analysis; descriptive bootstrap, not universal fidelity."""

import math

import numpy as np


PRIMARY = (
    ("qkv4-balanced-r4", "qkv4-nearest", 0.0),
    ("q4-k8-v8", "uniform8", -0.01),
    ("q8-k4-v8", "uniform8", -0.01),
)


def exact_mcnemar(benefit: int, harm: int) -> float:
    if min(benefit, harm) < 0:
        raise ValueError("discordant counts must be nonnegative")
    count = benefit + harm
    if count == 0:
        return 1.0
    tail = sum(math.comb(count, value) for value in range(min(benefit, harm) + 1))
    return min(1.0, 2 * (tail / (1 << count)))


def paired(candidate, reference, classes, *, samples: int = 20_000, seed: int = 20261124) -> dict:
    candidate, reference, classes = map(np.asarray, (candidate, reference, classes))
    if (candidate.ndim != 1 or candidate.size == 0 or candidate.shape != reference.shape
            or candidate.shape != classes.shape or samples < 100
            or not np.isin(candidate, [False, True]).all()
            or not np.isin(reference, [False, True]).all()):
        raise ValueError("matched nonempty binary outcomes, classes and enough bootstrap samples required")
    difference = candidate.astype(np.int8) - reference.astype(np.int8)
    rng = np.random.default_rng(seed)
    draws = np.zeros(samples, dtype=np.int64)
    for label in np.unique(classes):
        values = difference[classes == label]
        counts = np.bincount(values + 1, minlength=3)
        resampled = rng.multinomial(len(values), counts / len(values), size=samples)
        draws += resampled[:, 2] - resampled[:, 0]
    means = draws / len(difference)
    _, class_counts = np.unique(classes, return_counts=True)
    estimable = bool(np.all(class_counts >= 2))
    benefit, harm = int((difference == 1).sum()), int((difference == -1).sum())
    alpha = 0.05 / len(PRIMARY)
    result = {
        "queries": len(difference), "candidate_correct": int(candidate.sum()),
        "reference_correct": int(reference.sum()), "benefit": benefit, "harm": harm,
        "accuracy_difference": float(difference.mean()),
        "stratified_bootstrap_estimable": estimable,
        "stratified_bootstrap_95": np.quantile(means, [0.025, 0.975]).tolist() if estimable else None,
        "primary_family_one_sided_lower": float(np.quantile(means, alpha)) if estimable else None,
        "bootstrap_samples": samples, "seed": seed, "primary_lower_alpha": alpha,
        "exact_mcnemar_two_sided": exact_mcnemar(benefit, harm),
        "bootstrap_degenerate": bool(benefit + harm == 0),
    }
    if result["bootstrap_degenerate"]:
        # With zero observed harmful discordances, independence within fixed
        # strata and AM-GM give P(no harms) <= (1 - average_harm_probability)**n.
        # Use this finite-sample bound instead of a falsely exact zero interval.
        result["zero_discordance_harm_upper"] = -math.expm1(math.log(alpha) / len(difference))
        if estimable:
            result["primary_family_one_sided_lower"] = -result["zero_discordance_harm_upper"]
    return result


def summarize(report: dict, *, primary_run: bool = False) -> dict:
    if report["status"] != "complete" or not report["queries"]:
        raise ValueError("complete nonempty task report required")
    records = report["queries"]
    labels = np.asarray([row["target"] for row in records])
    policies = report["settings"]["policies"]
    outcomes, summaries = {}, []
    for policy in policies:
        predictions = np.asarray([row["predictions"][policy] for row in records])
        outcomes[policy] = predictions == labels
        confusion = np.zeros((100, 100), dtype=np.int64)
        np.add.at(confusion, (labels, predictions), 1)
        error = np.asarray([row["relative_mse"][policy] for row in records])
        summaries.append({
            "policy": policy, "queries": len(labels), "correct": int(outcomes[policy].sum()),
            "accuracy": float(outcomes[policy].mean()),
            "mean_relative_mse": float(error.mean()), "median_relative_mse": float(np.median(error)),
            "relative_mse_quantiles": np.quantile(error, [0, .25, .5, .75, .95, 1]).tolist(),
            "per_class_correct": {str(label): int(outcomes[policy][labels == label].sum())
                                  for label in np.unique(labels)},
            "confusion_matrix": confusion.tolist(),
        })
    comparisons = []
    for index, (candidate, reference, margin) in enumerate(PRIMARY):
        if candidate not in outcomes or reference not in outcomes:
            continue
        row = paired(outcomes[candidate], outcomes[reference], labels, seed=20261124 + index)
        row.update({"candidate": candidate, "reference": reference, "margin": margin,
                    "predeclared_primary_run": primary_run})
        lower = row["primary_family_one_sided_lower"]
        row["meets_predeclared_bootstrap_screen"] = bool(primary_run and lower is not None and lower > margin)
        comparisons.append(row)
    return {"policies": summaries, "primary_comparisons": comparisons,
            "limitations": [
                "Fixed CIFAR100 class composition, fixed gallery and checkpoint; not a universal quality guarantee",
                "One observation per class cannot estimate a within-class bootstrap interval",
                "Percentile bootstrap is approximate; Bonferroni screening does not make it an exact test",
                "Discovery outcomes are descriptive and cannot pass a replication acceptance screen",
                "Exact McNemar tests equality on discordant pairs, not the 1pp noninferiority margin",
                "Classifier labels enter evaluation only, not activation encoding or scale selection",
            ]}
