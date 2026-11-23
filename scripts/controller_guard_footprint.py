#!/usr/bin/env python3
"""Map sampled source-scale uncertainty to receiving BF16 convolution cohorts.

This is a derived diagnostic footprint, not scheduled reference work or energy.
M is a software cohort length; the physical reduction/output dimensions are128.
"""

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np

from activation_pilot import ROOT, digest, revision, write_json


COHORTS = (1, 8, 32, 128)
LAYOUTS = ("kernel-channel", "channel-kernel")


def footprint(flags, *, kernel=3, cohort=1, layout="kernel-channel"):
    """Map one declared dense lowering, with channel group32 uncertainty.

    kernel-channel puts128 channels of one spatial kernel position in a K128
    slice. channel-kernel orders all spatial kernel positions within a channel.
    Both are chosen mappings, not assumptions about the target im2col unit.
    """
    flags = np.asarray(flags, dtype=bool)
    if (flags.ndim != 3 or flags.shape[-1] != 4 or kernel < 1 or cohort < 1
            or layout not in LAYOUTS):
        raise ValueError("expected H,W,4 source mask and supported kernel/cohort/layout")
    height, width = flags.shape[:2]
    output_height, output_width = height - kernel + 1, width - kernel + 1
    if min(output_height, output_width) < 1:
        raise ValueError("source patch smaller than convolution kernel")
    rows, columns = np.indices((output_height, output_width)).reshape(2, -1)
    planes = []
    for begin in range(0, 128 * kernel * kernel, 128):
        indices = np.arange(begin, begin + 128)
        if layout == "kernel-channel":
            channels, positions = indices % 128, indices // 128
        else:
            channels, positions = indices // (kernel * kernel), indices % (kernel * kernel)
        coordinates = np.unique(np.stack((positions // kernel, positions % kernel, channels // 32), 1), axis=0)
        vectors = flags[rows[:, None] + coordinates[:, 0],
                        columns[:, None] + coordinates[:, 1], coordinates[:, 2]].any(1)
        counts = [int(vectors[start:start + cohort].sum())
                  for start in range(0, len(vectors), cohort)]
        planes.append({"k_begin": begin, "k_end": begin + 128,
                       "source_group_coordinates": coordinates.tolist(),
                       "row_vectors": len(vectors), "flagged_row_vectors": int(vectors.sum()),
                       "cohorts": math.ceil(len(vectors) / cohort),
                       "flagged_cohorts": sum(value > 0 for value in counts),
                       "flagged_rows_per_cohort": counts})
    return {"kernel": kernel, "cohort_rows": cohort, "lowering": layout,
            "source_groups": flags.size, "flagged_source_groups": int(flags.sum()),
            "row_vectors": sum(row["row_vectors"] for row in planes),
            "flagged_row_vectors": sum(row["flagged_row_vectors"] for row in planes),
            "cohorts": sum(row["cohorts"] for row in planes),
            "flagged_cohorts": sum(row["flagged_cohorts"] for row in planes), "planes": planes}


def analyze(source):
    if source.get("status") != "complete":
        raise ValueError("complete scale-stability record required")
    records = []
    for row in source["records"]:
        shape = tuple(row["group_shape"])
        if shape != (1, 8, 66, 4):
            raise ValueError("this study requires the declared C128,8x66 sampled patches")
        masks = {
            "changed-scale": np.asarray(row["original_exponents"]) != np.asarray(row["candidate_exponents"]),
            "local-bound-unknown": ~np.asarray(row["local_certified"], dtype=bool),
            "global-bound-unknown": ~np.asarray(row["global_certified"], dtype=bool),
        }
        identity = {key: row[key] for key in ("bundle", "input_index", "patch_index", "layer", "bits")}
        for criterion, values in masks.items():
            flags = values.reshape(shape)[0]
            for layout in LAYOUTS:
                for cohort in COHORTS:
                    records.append({**identity, "criterion": criterion,
                                    **footprint(flags, cohort=cohort, layout=layout)})
    aggregates = []
    for bits in (4, 8):
        for criterion in ("changed-scale", "local-bound-unknown", "global-bound-unknown"):
            for layout in LAYOUTS:
                for cohort in COHORTS:
                    rows = [row for row in records if (row["bits"], row["criterion"], row["lowering"], row["cohort_rows"])
                            == (bits, criterion, layout, cohort)]
                    totals = {key: sum(row[key] for row in rows) for key in (
                        "source_groups", "flagged_source_groups", "row_vectors", "flagged_row_vectors",
                        "cohorts", "flagged_cohorts")}
                    aggregates.append({"bits": bits, "criterion": criterion, "lowering": layout, "cohort_rows": cohort,
                        **totals, "flagged_group_fraction": totals["flagged_source_groups"] / totals["source_groups"],
                        "flagged_vector_fraction": totals["flagged_row_vectors"] / totals["row_vectors"],
                        "flagged_cohort_fraction": totals["flagged_cohorts"] / totals["cohorts"]})
    return records, aggregates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    source = json.loads(gzip.decompress(raw) if args.input.suffix == ".gz" else raw)
    records, aggregates = analyze(source)
    write_json(args.output, {"schema": 1, "status": "complete", "kind": "derived-diagnostic-footprint",
        "experiments": revision(ROOT), "script_sha256": digest(Path(__file__)),
        "source": str(args.input), "source_sha256": digest(args.input),
        "source_experiments": source["experiments"], "records": records, "aggregates": aggregates,
        "contract": {"input_format": "reconstructed-bf16", "K": 128, "N": 128,
                     "output_scalars": 128, "output_format": "fp32", "kernel": [3, 3],
                     "padding": "valid patch interior", "cohort_rows": list(COHORTS), "lowerings": list(LAYOUTS)},
        "limitations": [
            "M is a software batching choice, not a fixed physical128-row dimension",
            "Two chosen dense reduction orders; actual target im2col layout/padding/stationarity remain unspecified",
            "Changing reduction order can change numerical outputs; these are alternative footprints of fixed recorded masks",
            "INT4/INT8 source codes are reconstructed BF16; no native FP8 K256 capability inferred",
            "Changed-scale criterion is known only after paired execution; it is not a deployable detector",
            "Unknown-scale footprint is not a guarantee that outputs change or need recomputation",
            "Receiving-convolution footprint does not model the upstream producer or full DAG repair schedule",
            "No compacting/reordering kernel, setup/launch cost, memory traffic, latency or energy is measured",
            "Repeated cohort lengths and correlated patches are alternative accounting, never additive workloads",
        ]})
    if args.table is not None:
        with args.table.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(aggregates[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(aggregates)
    print(len(records), "patch/criterion/cohort records", len(aggregates), "aggregate alternatives")


if __name__ == "__main__":
    main()
