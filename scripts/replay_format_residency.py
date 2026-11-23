#!/usr/bin/env python3
"""Replay a constrained setup policy in abstract primitive-opportunity units.

No model is re-executed and no energy/cycle value is inferred. Both strategies
receive the same frozen eligibility stream and declared residency boundaries.
"""

import argparse
import gzip
import json
from pathlib import Path

from summarize_primitive_pilots import digest, verify_sources, write_csv


SETUP_UNITS = (0.0, 8.0, 32.0, 128.0, 512.0)
SWITCH_UNITS = (0.0, 0.25, 1.0, 4.0)


def replay(rows, eligible, width, setup, switch, *, split=False):
    """Known current window; once prepared, the format stays until phase end.

    A conservative window contract returns to BF16 at every boundary and charges
    two transitions whenever any low-format row is used. This differs from the
    greedy partition schedule's actual within-phase transition count.
    """
    high_calls = (width + 127) // 128
    low_calls = high_calls if split else 1
    reference = sum(rows) * high_calls
    credit = saved = opportunity_total = 0.0
    active = False
    used_low_rows = 0
    for size, count in zip(rows, eligible, strict=True):
        if not 0 <= count <= size:
            raise ValueError("invalid eligibility count")
        opportunity = max(0.0, count * (high_calls - low_calls) - (2 * switch if count else 0))
        opportunity_total += opportunity
        if opportunity and not active and credit + opportunity >= setup:
            active = True
        if active and opportunity:
            saved += opportunity
            used_low_rows += count
        elif not active:
            credit += opportunity
    online = reference - saved + (setup if active else 0)
    offline = reference - max(0.0, opportunity_total - setup)
    if not offline - 1e-8 <= online <= 2 * offline + 1e-8:
        raise ValueError("conditional fixed-trace cost invariant failed")
    return {"reference": reference, "opportunity": opportunity_total,
            "online": online, "offline": offline,
            "always_prepared": reference - opportunity_total + setup,
            "setup_performed": int(active), "used_low_rows": used_low_rows,
            "regret": online - offline}


def analyze(report):
    verify_sources(report)
    if [case["case"] for case in report["cases"]] != report["case_order"]:
        raise ValueError("incomplete case sequence")
    if not report["settings"].get("trace_schedule"):
        raise ValueError("complete schedule traces required")
    rows, details = [], {}
    for case in report["cases"]:
        if case["case"] in ("original", "bf16"):
            continue
        phases = [(name, index, phase) for name, layer in case["layers"].items()
                  for index, phase in enumerate(layer["schedule"]["phase_trace"])]
        if not phases:
            raise ValueError("missing phase traces")
        for setup in SETUP_UNITS:
            for switch in SWITCH_UNITS:
                total = {}
                maximum_ratio, worst = 1.0, None
                for layer, index, phase in phases:
                    result = replay(phase["rows_per_window"], phase["fp8_rows_per_window"],
                                    phase["width"], setup, switch, split=case["case"].endswith("split"))
                    for key, value in result.items():
                        total[key] = total.get(key, 0) + value
                    ratio = result["online"] / result["offline"] if result["offline"] else 1.0
                    if ratio > maximum_ratio:
                        maximum_ratio, worst = ratio, {"layer": layer, "phase_index": index,
                                                       "trace": phase, "result": result}
                row = {"model": report["model"], "case": case["case"],
                       "executed_commit": report["experiments"]["commit"],
                       "cohort": report["settings"]["cohort"], "setup_units": setup,
                       "switch_units": switch, "phases": len(phases), **total,
                       "maximum_phase_ratio": maximum_ratio}
                rows.append(row)
                details[f"{case['case']}:{setup}:{switch}"] = {"aggregate": row, "worst_phase": worst}
    return rows, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    evidence = {"schema": 1, "units": "abstract avoided-primitive opportunities; not energy or cycles",
                "conditions": ["Frozen eligibility; no changed-network execution",
                               "Known setup cost, no holding/eviction within recorded phases",
                               "Identical statistics cost and phase boundaries for comparators",
                               "Two mode transitions per profitable window; not the greedy packed ledger",
                               "MAC precision energy, encoding, queue, SRAM and entry/exit costs unpriced"],
                "replayer_sha256": digest(Path(__file__).read_bytes()), "runs": {}}
    for path in args.input:
        raw = path.read_bytes()
        if path.suffix == ".gz":
            raw = gzip.decompress(raw)
        report = json.loads(raw)
        current, details = analyze(report)
        fingerprint = digest(raw)
        rows.extend({"record_sha256": fingerprint, **row} for row in current)
        evidence["runs"][fingerprint] = details
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_prefix.with_suffix(".csv"), rows)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    print(f"Replayed {len(rows)} aggregate cost-grid cases; full input traces remain separate")


if __name__ == "__main__":
    main()
