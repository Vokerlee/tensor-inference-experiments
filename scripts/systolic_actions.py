#!/usr/bin/env python3
"""Count actions for an explicitly specified output-stationary tile schedule.

This is a schedule ledger, not a cycle simulator or a characterized energy model.
"""

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

from activation_pilot import write_json


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class Schedule:
    rows: int
    columns: int
    reduction: int
    tile_rows: int = 128
    tile_columns: int = 128
    tile_reduction: int = 128
    accumulator_bytes: int = 4

    def actions(self) -> dict:
        if min(asdict(self).values()) <= 0:
            raise ValueError("dimensions and accumulator width must be positive")
        mt = ceil_div(self.rows, self.tile_rows)
        nt = ceil_div(self.columns, self.tile_columns)
        kt = ceil_div(self.reduction, self.tile_reduction)
        calls = mt * nt * kt
        a_slots = calls * self.tile_rows * self.tile_reduction
        b_slots = calls * self.tile_reduction * self.tile_columns
        output_bytes = self.rows * self.columns * self.accumulator_bytes
        issued = calls * self.tile_rows * self.tile_columns * self.tile_reduction
        return {
            "schedule": asdict(self), "tile_calls": calls,
            "useful_dense_products": self.rows * self.columns * self.reduction,
            "issued_padded_products": issued,
            "operand_edge_slots": {"a": a_slots, "b": b_slots},
            "internal_operand_word_hops": {
                "a": a_slots * (self.tile_columns - 1),
                "b": b_slots * (self.tile_rows - 1)},
            "operand_register_latches_including_edge": {"a": issued, "b": issued},
            "final_output_write_bytes": output_bytes,
            "pe_resident_partial_sum_sram_bytes": output_bytes,
            "sram_mediated_partial_sum_bytes": (2 * kt - 1) * output_bytes,
            "isolated_tile_feed_fill_drain_cycles": calls * (
                self.tile_reduction + self.tile_rows + self.tile_columns - 2),
            "continuous_k_feed_fill_drain_cycles": mt * nt * (
                kt * self.tile_reduction + self.tile_rows + self.tile_columns - 2),
        }


def compare(schedule: Schedule, retained_k: int, active_fraction: float) -> dict:
    if not 0 < retained_k <= schedule.reduction or not 0 <= active_fraction <= 1:
        raise ValueError("retained K and active fraction must be within the dense range")
    dense = schedule.actions()
    gated = {**dense, "useful_active_products": dense["useful_dense_products"] * active_fraction}
    compact = Schedule(**{**asdict(schedule), "reduction": retained_k}).actions()
    return {
        "schema": 1, "dense": dense, "gate_only": gated, "common_k_compaction": compact,
        "assumptions": [
            "Output-stationary schedule; full fixed-size tiles including padded edge slots",
            "No cross-output-tile operand reuse; counts are array boundary slots, not SRAM reads",
            "Gate-only keeps the dense schedule and B delivery; active products are an upper opportunity",
            "Common K compaction uses the same mask across tile rows, selected before B delivery",
            "Raw A selection reads, packing, metadata, scale conversion and control are additional work",
            "SRAM-mediated case reads/writes each valid output between consecutive K tiles",
            "Cycle alternatives exclude stalls, output unloading and overlap between output tiles",
            "Counts exclude leakage, idle clock power and data-dependent bit transitions",
            "No coefficients or joules; activation width does not determine FP32 accumulator cost",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs=3, metavar=("M", "N", "K"), required=True)
    parser.add_argument("--tile", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--retained-k", type=int, required=True)
    parser.add_argument("--active-fraction", type=float, default=.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, compare(Schedule(*args.shape, *args.tile),
                                    args.retained_k, args.active_fraction))


if __name__ == "__main__":
    main()
