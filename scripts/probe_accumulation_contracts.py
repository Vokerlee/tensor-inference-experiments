#!/usr/bin/env python3
"""Construct finite counterexamples separating accumulator research contracts.

These are deliberately adversarial arithmetic witnesses, not sampled networks,
accuracy measurements or new mathematical results. The reference computes
products solely to validate contracts, never to select deployed precision.
"""

import argparse
from pathlib import Path

import torch

from activation_pilot import ENGINE, ROOT, digest, revision, write_json
from primitive_accumulation_reference import Counters, multiply
from spatial_feedback_reference import quantize


def operands(groups: int, scale: int):
    a = torch.zeros(1, groups, 32, dtype=torch.int8)
    w = torch.zeros(128, groups, 32, dtype=torch.int8)
    ae = torch.full((1, groups), scale, dtype=torch.int32)
    we = torch.zeros(128, groups, dtype=torch.int32)
    negative = 2 if groups == 4 else 4
    last = 3 if groups == 4 else 5
    for group, sign in ((0, 1), (negative, -1)):
        a[:, group] = 64
        w[:, group] = sign * 32
        ae[:, group] += 8
    for group in (1, last):
        a[:, group, 0] = 1
        w[:, group, 0] = 1
    return a, ae, w, we


def measure(groups: int, scale: int) -> dict:
    args = operands(groups, scale)
    records = []
    for k, width in ((128, None), (128, 24), (128, 32), (256, 32)):
        counters = Counters()
        output = multiply(*args, reduction_width=k, accumulator_width=width, counters=counters)
        if not torch.isfinite(output).all() or not (output == output[0, 0]).all():
            raise ValueError("unexpected nonfinite or column-varying witness output")
        records.append({"k_slice": k, "width": width, "scalar_output": float(output[0, 0]),
                        "vector_groups": counters.vector_groups,
                        "accepted_vectors": counters.accepted_vectors,
                        "changed_primitive_scalars": counters.changed_output_scalars})
    by_key = {(row["k_slice"], row["width"]): row for row in records}
    if groups == 4:
        observed = (by_key[128, None]["scalar_output"] == 2.0 ** scale
                    and by_key[128, 24]["scalar_output"] == 2.0 ** scale
                    and by_key[128, 32]["scalar_output"] == 2.0 ** (scale + 1))
        kind = "range_safety_does_not_imply_fp32_reference_equivalence"
    else:
        observed = (by_key[128, 32]["scalar_output"] == 2.0 ** scale
                    and by_key[256, 32]["scalar_output"] == 2.0 ** (scale + 1))
        kind = "removing_primitive_output_rounding_changes_the_contract"
    if not observed:
        raise ValueError("constructed contract witness was not reproduced")
    return {"kind": kind, "groups": groups, "power_of_two_scale": scale,
            "a_codes": args[0].tolist(), "a_exponents": args[1].tolist(),
            "one_weight_column_codes": args[2][0].tolist(),
            "weight_columns": 128, "weight_exponents": args[3][0].tolist(),
            "exact_real_sum": 2.0 ** (scale + 1), "records": records,
            "contract_difference_observed": observed}


def controller_witness() -> dict:
    """Separate data-path change from recomputing a dynamic quantizer decision."""
    source = torch.full((1, 32, 1, 1), 0.75, dtype=torch.bfloat16)
    source[:, 0] = 7
    changed = source.clone()
    changed[:, 0] = 7.03125  # Exactly one BF16 step; the receiver ignores this coordinate.
    coefficient = torch.zeros(32)
    original = quantize(source, coefficient, bits=4)
    live = quantize(changed, coefficient, bits=4)
    frozen_scale = torch.ldexp(torch.ones(()), original.exponents.flatten()[0])
    frozen = (changed.float() / frozen_scale).round().clamp(-7, 7) * frozen_scale
    weight = torch.zeros(32)
    weight[1] = 1

    def receiver(values):
        return float(values.flatten() @ weight)

    baseline, frozen_output, live_output = map(receiver, (original.reconstructed, frozen, live.reconstructed))
    if not (receiver(source.float()) == receiver(changed.float()) == 0.75
            and baseline == frozen_output == 1 and live_output == 0):
        raise ValueError("controller-path witness was not reproduced")
    return {
        "kind": "small_source_change_can_change_control_without_direct_receiver_error",
        "source": source.flatten().float().tolist(), "changed_source": changed.flatten().float().tolist(),
        "receiver_weight": weight.tolist(), "source_max_absolute_difference": 0.03125,
        "original_exponent": int(original.exponents.flatten()[0]),
        "changed_exponent": int(live.exponents.flatten()[0]),
        "original_unquantized_output": receiver(source.float()),
        "changed_unquantized_output": receiver(changed.float()),
        "baseline_quantized_output": baseline, "frozen_control_output": frozen_output,
        "live_control_output": live_output,
        "data_path_difference_at_fixed_control": frozen_output - baseline,
        "control_path_difference_at_changed_input": live_output - frozen_output,
        "total_difference": live_output - baseline,
        "limitations": ["Constructed discontinuity, not a natural-input occurrence rate",
                        "Frozen control is a diagnostic intervention, not an implementable replacement policy"],
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = {
        "schema": 1, "status": "complete", "kind": "constructed_arithmetic_contract_witnesses",
        "experiments": revision(ROOT), "engine": revision(ENGINE), "torch": torch.__version__,
        "source_sha256": {name: digest(ROOT / "scripts" / name) for name in (
            "probe_accumulation_contracts.py", "primitive_accumulation_reference.py",
            "accumulation_scale_diagnostics.py", "accumulation_segments.py",
            "spatial_feedback_reference.py")},
        "cases": [measure(groups, scale) for groups in (4, 8) for scale in range(-8, 9)],
        "controller_path_witness": controller_witness(),
        "limitations": [
            "Constructed CPU arithmetic references; not hardware or natural-input failure rates",
            "Exact real sum, FP32 reference and alternate arithmetic are distinct comparators",
            "A changed result can be intentional approximation rather than an implementation bug",
            "No task accuracy, energy, timing or novelty result follows from these witnesses",
        ],
    }
    write_json(args.output, report)
    print(f"Verified {len(report['cases'])} constructed contract differences")


if __name__ == "__main__":
    main()
