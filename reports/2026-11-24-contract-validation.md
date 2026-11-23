# Accumulation cost screening and controller-path diagnostics — 2026-11-24

A large merge-count reduction need not produce a large total-energy gain.
Separately, a small source perturbation can change the adaptive quantizer's
scale as well as its numerical input. The experiments below expose these two
limits with conditional cost scenarios, constructed arithmetic witnesses and
sampled real decoder trajectories. They do not establish a new hardware or
validation-method contribution.

## Evidence and reproduction

- [Cost model and runner](../docs/accumulation-cost-envelope.md),
  [all target thresholds](2026-11-24-accumulation-cost-thresholds.csv).
- [Arithmetic/control protocol and witnesses](../docs/arithmetic-contract-validation.md),
  [all sampled decoder records](2026-11-24-controller-paths.csv).
- [Complete JSON archive](statistics/2026-11-24-contracts/README.md), preserving
  every hypothetical cost-grid result, constructed operand, sampled metric and
  the preliminary run. The separate [negative-results register](limitations-and-negative-results.md)
  keeps unfavorable variants and revisit conditions searchable.

Cost runner source e9b415277f47df3366497aac48fa9b6d76d95b8e uses the immutable
primitive-accumulation CSV and records its SHA256 plus the script SHA256. It
executes no new network. Arithmetic witnesses use clean source
29d6e388f78765437be6615348a1246c3efca987. Canonical real-trajectory diagnostics
use clean source039da48465beb8e6cfab477b906de37f79b20252, engine642089cbeeb77f6a6719c201330b6eb70e6534ff.
All164 experiment tests passed.

## Conditional energy significance

The frozen CSV produces23 alternative raw/postzero accounting records and4968
hypothetical cost scenarios. These are not independent network experiments or
characterized devices. Raw greedy K128 group boundaries number about0.023 per
scalar product; removing most boundaries still leaves every MAC and mandatory
primitive output intact.

For illustration ONLY, suppose unchanged non-MAC energy equals total MAC energy,
a vector decision costs32 scalar-MAC energy units, and all remaining added cost
is1% of baseline MAC energy. The boundary/MAC energy ratio needed for target
TOTAL savings of the evaluated workload is then:

| Recorded control model | 5% target | 10% target | 20% target |
|---|---:|---:|---:|
| ResNet18 | 5.52 | 11.05 | 24.38 |
| Swin-T | 6.15 | 12.42 | 27.87 |
| ViT-B/16 | 6.16 | 12.48 | 28.13 |

At boundary/MAC ratio1, these same assumptions give only0.36–0.44% savings;
at ratio8, about6.5–7.3%. Neither scenario predicts the target accelerator.
The denominator concerns evaluated layers; whole-model assessment must include
unmodified layers in shared cost. The boundary ratio must characterize the
actual eliminated conversion/scaling/addition bundle, or explicitly only its
floating-add portion. Shared multiplier costs, changed accumulator activity,
metadata, buffering and clock effects cannot be silently omitted.

The current evidence therefore does not justify calling accumulation alone a
major energy optimization. It also does not reject the mechanism for every
architecture. Characterizing the boundary/MAC ratio, common workload energy and
added per-product state/alignment cost would resolve much more than additional
accuracy-only sweeps. Rectangular cost-box extrema are conditional mathematical
bounds, not measured uncertainty or confidence intervals.

## Constructed contract witnesses

Thirty-four finite CPU cases separate range safety, FP32 reference equivalence
and mandatory matrix-primitive output rounding. An exact range-safe I32 sum can
differ from sequential FP32; changing a K128 output boundary to K256 can also
change the result. The exact-real comparator is recorded separately. These
are intentional arithmetic-contract differences, not automatic implementation
bugs or natural-input failure frequencies.

One additional BF16 witness changes a source coordinate7->7.03125 that the
unquantized receiver ignores. Its direct receiving output stays0.75. The INT4
POT scale changes1->2, however, and the represented output changes1->0; freezing
the old control leaves1. This demonstrates why numerical input error and
recomputed-control effects can require different diagnostics.

## Sampled decoder trajectories

Use the same eight exact SANA-Sprint decoder inputs as the earlier studies.
Compare original versus known folded-BF16 decoder trajectories at the unchanged
receiver `up_blocks.0.1.conv1`. Select three predeclared8x66 source patches per
image, at quarter/half/three-quarter height and central width. These24 patches
are not random spatial sampling or a whole-layer estimate. Quantize channel
groups32 independently with POT INT4/INT8, without residual feedback.

| Recorded quantity, summed over24 patches | INT4 | INT8 |
|---|---:|---:|
| Source groups inspected | 50,688 | 50,688 |
| Groups with changed scale | 145 | 120 |
| Patches containing a scale change | 20/24 | 21/24 |
| Fixed-control data-path squared output difference | 151,408,227.41 | 16,842,066.76 |
| Recomputed-control squared output difference | 28,526,624.06 | 170,104.41 |
| Twice the inner product between paths | 36,564.01 | -80,288.69 |
| Total squared output difference | 179,971,415.48 | 16,931,882.48 |

The source relative RMS difference is about0.3156%. Changed scales affect about
0.286%/0.237% of these sampled groups. Small scale-flip counts can coexist with
a noticeable INT4 control-path difference. Squared component norms are NOT
unique fractions of explained error: the cross term matters and can be negative.
The decomposition is tied to the chosen intervention order.

Original decoding was bit-exact to captured outputs. The preliminary diagnostic
recorded an untracked statistics directory and is retained with dirty status.
An unchanged-code clean rerun reproduced all48 sampled records exactly; the
canonical interpretation uses that clean record. No status was rewritten.
The fixed-old-scale INT8 diagnostic clipped61 scalars; that counterfactual is
not a safe deployable policy. Extra receiving products are validation work,
not controller operations.

This is local evidence from two actual trajectories, not an end-to-end
quantized-policy quality comparison, a hardware mismatch rate or proof that
folding is unsuitable. It motivates checking controller decisions alongside
numerical outputs when evaluating adaptive arithmetic. Portable emulation,
differential testing and quality/energy exploration already have substantial
prior work; a new methodology would need a demonstrated additional capability
and a mechanism-level comparison.
