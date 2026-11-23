# Source-scale filtering at convolution-matrix granularity — 2026-11-24

About99% of sampled source groups are conditionally certified scale-stable,
but this does not imply99% fewer reference matrix operations. Under one
explicit dense convolution lowering, the remaining uncertainty reaches81–85%
of consecutive128-row cohorts. Another lowering flags more individual vectors
but fewer large cohorts. Granularity and clustering matter alongside selectivity.

This is a derived diagnostic from the [complete scale-filter study](2026-11-24-controller-replay.md),
not a new neural run or hardware timing measurement. It retains the same24
predeclared local patches of one receiver, with their correlated group masks.
All [48 aggregate alternatives](2026-11-24-controller-guard-footprint.csv) and
[1152 patch/criterion/cohort records](statistics/2026-11-24-guard-footprint/README.md)
are preserved, including every K-slice/source-group incidence and cohort count.

## Declared mappings and denominators

A patch has128 channels and8x66 source positions. Valid3x3 convolution produces
6x64=384 output positions. Its dense receiving matrix has reduction dimension
1152=9x128; each K128 slice feeds128 output channels. There are nine K slices
and384 streamed row vectors per patch, or82,944 vectors across24 patches.
The128 FP32 output scalars per vector follow the declared interface contract.

Two alternative dense reduction orders are evaluated on the fixed masks:

- `kernel-channel`: put all128 channels of one spatial kernel position in one
  K128 slice. Four group32 scale decisions affect each vector.
- `channel-kernel`: order the nine spatial positions within each channel,
  then partition that1152-element reduction into K128 slices. A slice can
  depend on more spatially distributed group decisions.

These are chosen lowerings. Actual target im2col ordering, padding and
stationarity remain unspecified. Alternative reduction orders can change
rounded numerical results; this analysis neither executes those alternatives
nor claims bit equivalence. It maps existing uncertainty masks only.

For each slice, a row vector is flagged if any contributing source group is
flagged. A consecutive row cohort is flagged if any of its row vectors is
flagged. Cohort length `M` is a software batching choice, not a fixed physical
128-row dimension. INT4/INT8 source codes here are reconstructed to BF16; native
FP8 K256 capacity is not assigned to these integer-code experiments.

## Local-bound uncertainty footprint

The source filter leaves500/50,688 INT4 groups and572/50,688 INT8 groups unknown.
The following percentages use receiving-vector/cohort denominators rather than
source-group denominators:

| Precision | Lowering | M1 | M8 | M32 | M128 |
|---|---|---:|---:|---:|---:|
| INT4 | kernel-channel | 3.68% | 23.23% | 56.79% | 81.48% |
| INT4 | channel-kernel | 9.40% | 24.14% | 48.96% | 68.83% |
| INT8 | kernel-channel | 4.42% | 26.35% | 60.19% | 84.72% |
| INT8 | channel-kernel | 11.09% | 26.79% | 52.55% | 70.06% |

Larger M cohorts contain more opportunities for at least one uncertain source.
The channel-first alternative raises individual-vector coverage but concentrates
it differently across K slices, reducing large-cohort coverage in these samples.
A scalar selectivity statistic alone therefore cannot determine a matrix-level
validation budget.

Even using the after-the-fact *actual changed-scale* mask, the kernel-channel
M128 footprint covers61.57% of INT4 and57.56% of INT8 cohorts. This mask is an
oracle diagnostic, not a deployable detector. Global-bound uncertainty, all
other cohort lengths, both layouts and every unfavorable case are in the CSV.
The alternatives are never summed as independent workloads.

## Limits and reproduction

A flagged scale need not change any reconstructed value or receiving output.
A flagged cohort need not require an entire reference GEMM. Conversely, this
receiving footprint does not implement upstream producer checks, causal-state
repair or downstream DAG replay. No row-compaction kernel, reference-work
saving, memory transfer, hardware cycles or energy is measured.

The source bounds still require both observed trajectories. The evidence rejects
an inference from group certification percentage to matrix-work savings under
these declared mappings; it does not disprove selective numerical emulation.
A useful implementation must account for decision sensitivity, valid error
bounds, reference execution granularity and scheduling overhead together.

Canonical analysis source7500a85d4f5cb5954c9f1d20025ae85000867549 is clean;
the parent scale record is clean source96eca307368d611c8e5cb92e909799c6fb6eb534.
The earlier one-layout analysis is retained as recorded. All196 experiment tests
pass, including a constructed case where changing reduction order changes a
single group's receiving footprint from9 to27 vectors at equal dense work.

```sh
.venv/bin/python scripts/controller_guard_footprint.py \
    --input reports/statistics/2026-11-24-controller-replay/scale-stability-20261124.json.gz \
    --output results/controller-guard-footprint-20261124.json \
    --table results/controller-guard-footprint-20261124.csv
```

This analysis needs no model checkpoint, GPU or ignored input tensors.
