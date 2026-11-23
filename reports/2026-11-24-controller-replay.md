# Full-decoder decision/state replay and scale filtering — 2026-11-24

Eight exact SANA-Sprint decoder inputs were evaluated under three activation
quantizers, two precisions and declared original/folded BF16 implementations.
Replaying recorded decisions reproduces every unchanged-backend output bit for
bit. Changed implementations alter scales and causal modes, but the close fixed
versus adaptive feedback comparison has no established quality winner in this
small input set. An interval filter identifies most unchanged local scales when
given bounds measured from both trajectories; it has not saved reference work.

## Scope and concrete implementations

- [Native scale replay and experiment-side decision/state tapes](../docs/controller-replay.md).
- [Paired complete-model analysis](../scripts/controller_replay_analysis.py),
  including all exploratory policy contrasts and input-level uncertainty.
- [Exhaustive finite-BF16 scale contract and conditional interval filter](../docs/decision-stability.md).
- [Pinned DINOv3 descriptor observer](../scripts/controller_descriptor_fidelity.py).
- [Compact run table](2026-11-24-controller-replay.runs.csv),
  [all4256 controlled calls](2026-11-24-controller-replay.calls.csv),
  [all60 paired contrasts](2026-11-24-controller-replay.contrasts.csv) and
  [all48 local patch records](2026-11-24-controller-replay.patches.csv).
- [All complete JSON statistics and restorable tapes](statistics/2026-11-24-controller-replay/README.md).

The checkpoint, two-step1024px pipeline, prompts, seeds, input-array hashes,
source/native hashes, software versions and per-call statistics are retained in
the producer reports. These are the same eight inputs used in earlier studies,
not eight newly sampled independent holdouts. INT4 and INT8 reuse these inputs.

The19 controlled dense3x3 receivers account for2,783,440,797,696 dense MACs per
original decode, about66.52% of the decoder profile. The five nearest-resize
receivers are excluded structurally so original and folded implementations have
identical controlled event identities. Their known BF16 coefficient-folding
change perturbs downstream trajectories. This is a declared numerical/backend
change, not a measured mismatch against the target chip.

Weights remain BF16. All INT4/INT8 source values are reconstructed to BF16 for
convolution; no packed-integer throughput or native FP8 geometry is inferred.
`none` dynamically chooses group32 POT scales without feedback. `coarse` adds
fixed weight-derived half-step feedback. `live` chooses half-step feedback from
preceding32-row terminal-error mean/second-moment state, with smoothing1/4.
There is no activation calibration or label-based policy tuning.

## Controlled execution

Each precision has112 recorded decodes:14 variants per input. Every policy has
original/live, original/exact-replay, folded/live and folded/frozen execution.
The live policy additionally freezes scales alone or coefficients alone.
Unquantized original/folded reference decodes are separate. All48 same-backend
quantized replays across both precisions are bit-exact. Frozen runs still consume
changed activations and update numerical carry; only selected configuration is
replayed. They are counterfactual diagnostics, not deployable policies.

| Precision | Policy | Changed scale groups / total | Changed coefficient choices / total |
|---|---|---:|---:|
| INT4 | none | 51,546,156 / 352,329,728 (14.63%) | 0 / 43,264 |
| INT4 | coarse | 62,825,485 / 352,329,728 (17.83%) | 0 / 43,264 |
| INT4 | live | 65,105,045 / 352,329,728 (18.48%) | 14,362 / 590,080 (2.43%) |
| INT8 | none | 7,202,029 / 352,329,728 (2.04%) | 0 / 43,264 |
| INT8 | coarse | 7,846,300 / 352,329,728 (2.23%) | 0 / 43,264 |
| INT8 | live | 7,883,808 / 352,329,728 (2.24%) | 13,089 / 590,080 (2.22%) |

Counts compare folded/live with matching original/live events. Groups and
coefficient choices are correlated internal decisions, not independent model
samples. Full state differences and per-layer histograms remain available.
The local unquantized-trajectory patch study has a different intervention and
denominator; its smaller flip count does not contradict this full-policy result.

## Output fidelity and ranking uncertainty

Pixel MSE below is averaged over the eight input images against the same
unquantized original output. Lower values mean better fidelity to that declared
reference, not necessarily better generation quality.

| Precision | Policy | Original/live MSE | Folded/live MSE | Folded/frozen MSE |
|---|---|---:|---:|---:|
| INT4 | none | 0.001133762 | 0.001133282 | 0.001115100 |
| INT4 | coarse | 0.001092766 | 0.001089775 | 0.001073319 |
| INT4 | live | 0.001091691 | 0.001093714 | 0.001077035 |
| INT8 | none | 0.00001411545 | 0.00001436487 | 0.00001427319 |
| INT8 | coarse | 0.00001405641 | 0.00001419751 | 0.00001414663 |
| INT8 | live | 0.00001414810 | 0.00001421076 | 0.00001416173 |

At INT4, coarse-minus-live mean MSE changes from+1.0751e-6 on original/live to
-3.9389e-6 on folded/live. The observed average ordering therefore reverses, and
four of eight input-level orderings change. However, paired95% bootstrap
intervals include zero in both contexts: [-7.5191e-6,7.4090e-6] and
[-1.0796e-5,1.8961e-6]. The live-minus-frozen change in this policy contrast also
includes zero. These data do not establish a robust ranking reversal or an
adaptive-feedback advantage. Curated prompts are not random population samples;
intervals and exact sign-flip tests are exploratory diagnostics, with all
contrasts retained and no confirmatory multiple-testing claim.

The DINOv3-S observer uses the pinned384-dimensional FP32 descriptor with
matched official preprocessing of rounded8-bit RGB. All224 descriptors and
eight original references are retained;176 output-array artifacts have unique hashes.
Descriptor fidelity supplies another observer, not GenEval, FID or human quality.
Its close coarse/live ordering also varies by implementation. For mean
coarse-minus-live cosine distance:

| Precision | Context | Mean difference | Exploratory95% paired interval | Exact sign-flip p |
|---|---|---:|---|---:|
| INT4 | original/live | -1.2314e-5 | [-1.6231e-4,1.2860e-4] | 0.9297 |
| INT4 | folded/live | +4.7865e-6 | [-1.4481e-4,1.2695e-4] | 0.9531 |
| INT4 | folded/frozen | -7.0897e-6 | [-1.3451e-4,1.2453e-4] | 0.9141 |
| INT8 | original/live | -5.1281e-6 | [-1.0238e-5,-1.5169e-6] | 0.0234 |
| INT8 | folded/live | +3.0919e-6 | [-2.0300e-6,9.7084e-6] | 0.4453 |
| INT8 | folded/frozen | -3.5984e-6 | [-7.0790e-6,-8.7062e-7] | 0.0234 |

The INT8 original/frozen comparisons show a nominal signal favoring fixed
feedback for this observer, whereas folded/live is unresolved. This is not
confirmatory significance: the observer was added during exploratory analysis,
inputs are curated, multiple metrics/contrasts are retained, and the altered
implementation is not the target-device truth. No task-quality winner follows.

For original quantized output `a`, changed-backend frozen output `b`, and
changed-backend live output `c`, the report records data difference `d=b-a`,
control difference `h=c-b`, total `t=c-a`, and
`||t||² = ||d||² + ||h||² + 2<d,h>`.
At INT4/live, summed pixel terms are16,951.03,17,871.66,-15,780.27 and19,042.43,
respectively. The negative interaction is substantial. Neither component's
squared norm is a unique fraction of total error. Freezing scales and modes
separately is also order-dependent and need not produce additive effects.

## Exhaustive scale contract and local filtering

For each precision, all65,280 finite BF16 bit patterns were evaluated in the
native scale selector. Sign symmetry and positive-domain exponent monotonicity
hold, and no exponent differs from the CPU Torch formula in this enumeration.
Finite input alone does not guarantee finite reconstructed output:62 extreme
BF16 words at INT4 and two at INT8 produce nonfinite reconstruction/state.
These deliberately tested extremes are retained; normal accepted inference
rejects invalid native events. They do not occur in the accepted decoder runs.

The interval filter uses a complete measured decision table and an enclosing
source-error bound, with outward rounding. On24 predeclared local patches:

| Precision | Groups | Actual scale changes | Certified unchanged, per-group bound | Certified unchanged, patch-global bound |
|---|---:|---:|---:|---:|
| INT4 | 50,688 | 145 | 50,188 (99.01%) | 47,506 (93.72%) |
| INT8 | 50,688 | 120 | 50,116 (98.87%) | 47,094 (92.91%) |

Neither filter certifies any observed changed scale as unchanged. Bounds come
from both already computed trajectories, so these fractions are diagnostic
selectivity, not measured reference-work savings. An independently justified
forward bound, controller-state treatment, downstream recomputation and a
reference-budget comparison are still required for a useful selective emulator.
Unchanged scales alone do not certify unchanged codes or model outputs.

The [matrix-granularity follow-up](2026-11-24-controller-guard-footprint.md)
shows why these source-group percentages are not reference-work savings:
under one declared lowering, remaining local uncertainty touches81–85% of
128-row cohorts. A different lowering changes both vector and cohort coverage.
Actual target im2col ordering remains unspecified.

## What the evidence supports

Decision/state instrumentation exposes differences hidden by tensor-only or
fixed-configuration comparisons. It also prevents a stronger unsupported claim:
these eight inputs do not establish that decision-aware validation changes a
robust optimization winner. Live feedback remains a weak variant relative to
the fixed control. The interval filter is a concrete conditional prototype;
its generic principle is established prior art, including
[FLUCTUAT](https://arxiv.org/abs/1309.3910v1) and
[PRECiSA stable-test transformations](https://arxiv.org/abs/1808.04289v2).

The existing [cost-envelope study](2026-11-24-contract-validation.md) remains
necessary: decision counts, dense MACs and CUDA time do not establish target
energy or latency. The precision of each run is fixed, and the evaluated feedback policies do not
change its dense convolution MAC count. Adaptive precision, buffer/format
transitions and a changed memory schedule have not been evaluated by this
study. No energy-ranking or reference-work savings are claimed.

## Reproduction

Full decoder source is8950a98e16ad448af058f6e89ab9d6f49a17b7ea; the later
runner versions retain recorded source hashes. The native operator must first
be built with the recorded engine revision. Exact original input capture is
documented in the earlier decoder study; replay reports embed both manifests.

```sh
.venv/bin/python scripts/sana_controller_replay.py \
    --inputs results/decoder-inputs-20261123 results/decoder-inputs-holdout-20261123 \
    --bits 4 --output results/controller-replay-a4-20261124
# Repeat with --bits 8 and the corresponding output directory.
.venv/bin/python scripts/controller_descriptor_fidelity.py \
    --input results/controller-replay-a4-20261124/report.json \
            results/controller-replay-a8-20261124/report.json \
    --output results/controller-descriptor-fidelity-20261124.json
.venv/bin/python scripts/controller_replay_analysis.py \
    --input results/controller-replay-a4-20261124/report.json \
            results/controller-replay-a8-20261124/report.json \
    --descriptor results/controller-descriptor-fidelity-20261124.json \
    --output results/controller-replay-analysis-20261124.json
.venv/bin/python scripts/report_controller_replay.py
```

The last command needs only the checked-in full JSON archive; it regenerates all
four CSV tables without checkpoints, CUDA or ignored workspace files. Scale
filter reproduction and trace restoration are linked above. All190 experiment
tests passed, including replay causality, identity, interval boundaries,
full-exponent-range error bounds and independently restorable archive shards.
