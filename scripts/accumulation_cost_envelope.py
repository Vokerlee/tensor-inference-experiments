#!/usr/bin/env python3
"""Conditional energy break-even envelopes from recorded arithmetic counters.

All costs are dimensionless hypothetical ratios, not device characterization.
The baseline already computes exact integer group dots. Only group-boundary
work is replaced; products and mandatory primitive FP32 outputs are unchanged.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Counts:
    products: int
    reference_merges: int
    selected_merges: int
    decisions: int

    def __post_init__(self):
        if self.products <= 0 or min(self.reference_merges, self.selected_merges, self.decisions) < 0:
            raise ValueError("positive products and nonnegative recorded actions required")

    @property
    def reference_per_product(self) -> float:
        return self.reference_merges / self.products

    @property
    def saved_per_product(self) -> float:
        return (self.reference_merges - self.selected_merges) / self.products


@dataclass(frozen=True)
class Costs:
    merge_to_mac: float
    common_nonmac_to_mac_total: float
    guard_to_mac: float
    extra_per_product_to_mac: float

    def __post_init__(self):
        if any(not math.isfinite(value) or value < 0 for value in asdict(self).values()):
            raise ValueError("finite nonnegative dimensionless costs required")


def evaluate(counts: Counts, costs: Costs) -> dict:
    reference = (counts.products * (1 + costs.common_nonmac_to_mac_total)
                 + counts.reference_merges * costs.merge_to_mac)
    extra = counts.decisions * costs.guard_to_mac + counts.products * costs.extra_per_product_to_mac
    selected = (counts.products * (1 + costs.common_nonmac_to_mac_total)
                + counts.selected_merges * costs.merge_to_mac + extra)
    saving = reference - selected
    return {"reference_cost_mac_units": reference, "candidate_cost_mac_units": selected,
            "added_cost_mac_units": extra, "saved_cost_mac_units": saving,
            "net_fraction_saved": saving / reference,
            "gross_merge_fraction_of_baseline": counts.reference_merges * costs.merge_to_mac / reference}


def required_merge_ratio(counts: Counts, *, target: float, common: float,
                         guard: float = 0, extra: float = 0) -> float | None:
    """Minimum merge/MAC cost for a finite target saving under declared costs.

    Null means no finite nonnegative ratio suffices. This includes an ideal
    target exceeding the fraction of merges replaced, regardless of hardware.
    """
    Costs(0, common, guard, extra)
    if not math.isfinite(target) or not 0 <= target < 1:
        raise ValueError("target must be finite within [0,1)")
    numerator = target * (1 + common) + counts.decisions / counts.products * guard + extra
    denominator = counts.saved_per_product - target * counts.reference_per_product
    if denominator <= 0:
        return 0.0 if numerator == 0 and denominator == 0 else None
    return numerator / denominator


def box_envelope(counts: Counts, bounds: dict) -> dict:
    """Extrema of this linear-fractional model over a declared rectangular box.

    This is a conditional mathematical envelope, not a confidence interval.
    Positive denominators imply extrema occur among the box's corners.
    """
    names = tuple(Costs.__dataclass_fields__)
    if set(bounds) != set(names):
        raise ValueError("bounds must specify every cost parameter")
    for name in names:
        lower, upper = bounds[name]
        if not 0 <= lower <= upper or not math.isfinite(upper):
            raise ValueError("finite ordered nonnegative cost bounds required")
    corners = []
    for values in itertools.product(*(bounds[name] for name in names)):
        costs = Costs(*values)
        corners.append({"costs": asdict(costs), **evaluate(counts, costs)})
    low = min(corners, key=lambda row: row["net_fraction_saved"])
    high = max(corners, key=lambda row: row["net_fraction_saved"])
    status = ("positive_throughout_box" if low["net_fraction_saved"] > 0 else
              "nonpositive_throughout_box" if high["net_fraction_saved"] <= 0 else
              "sign_depends_on_costs")
    return {"status": status, "minimum": low, "maximum": high, "corners": corners}


def records(path: Path):
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if not row["case"].startswith("guarded-"):
                continue
            for accounting in ("raw", "postzero"):
                products, decisions = int(row["useful_products"]), int(row["vector_groups"])
                if accounting == "raw":
                    reference = int(row["reference_group_merges"])
                    selected = reference - int(row["replaced_group_merges"])
                elif row.get("zero_gated_reference_merges"):
                    reference = int(row["zero_gated_reference_merges"])
                    selected = int(row["zero_gated_selected_merges"])
                else:
                    continue
                identity = {key: row[key] for key in ("run", "model", "case", "k_slice", "strategy", "bound_kind")}
                yield {**identity, "accounting": accounting}, Counts(products, reference, selected, decisions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Deliberately broad illustrative ratios; NONE is a calibrated chip value.
    grid = {"merge_to_mac": [1, 2, 4, 8, 16, 32],
            "common_nonmac_to_mac_total": [0, 1, 4],
            "guard_to_mac": [0, 8, 32, 128],
            "extra_per_product_to_mac": [0, 0.01, 0.05]}
    bounds = {name: [min(values), max(values)] for name, values in grid.items()}
    report = {
        "schema": 1, "status": "complete", "kind": "conditional_cost_sensitivity",
        "input": str(args.input), "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "costs_are_characterized_hardware": False, "grid": grid, "box": bounds, "cases": [],
        "limitations": [
            "No measured, synthesized or technology-calibrated hardware coefficients",
            "Raw and postzero records are alternative accounting conventions, never additive",
            "Postzero counters do not establish a free causal zero-detection implementation",
            "Same encoded operands and integer group dots; products/output boundaries unchanged",
            "Guard charged per recorded vector decision, not per cycle or GPU launch",
            "Extra per-product cost must cover alignment, changed accumulator state, metadata and timing effects",
            "Shared non-MAC cost is a scenario assumption, not SRAM traffic inferred from tensor sizes",
            "Reduced activity can affect common clock/timing costs; such effects need explicit characterization",
            "The cost box is hypothetical and rectangular; its bounds are not empirical uncertainty estimates",
            "Network quality is inherited from the original cases; no new neural network was executed",
        ],
    }
    for identity, counts in records(args.input):
        scenarios = []
        for values in itertools.product(*grid.values()):
            costs = Costs(*values)
            scenarios.append({"costs": asdict(costs), **evaluate(counts, costs)})
        thresholds = []
        for common, guard, extra, target in itertools.product((0, 1, 4), (0, 32), (0, 0.01), (0, 0.05, 0.10, 0.20)):
            thresholds.append({"common": common, "guard": guard, "extra": extra, "target": target,
                               "required_merge_to_mac": required_merge_ratio(
                                   counts, target=target, common=common, guard=guard, extra=extra)})
        report["cases"].append({**identity, "counts": asdict(counts),
                                "reference_merges_per_product": counts.reference_per_product,
                                "saved_merges_per_product": counts.saved_per_product,
                                "scenarios": scenarios, "target_thresholds": thresholds,
                                "envelope": box_envelope(counts, bounds)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{len(report['cases'])} alternative-accounting cases; "
          f"{sum(len(row['scenarios']) for row in report['cases'])} hypothetical cost scenarios")


if __name__ == "__main__":
    main()
