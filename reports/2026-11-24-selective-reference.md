# Selective exact BF16 observations: correctness survives, acceleration does not

The implemented selective-reference variant is not competitive with complete
execution of the same declared reference. Both candidates lose on all12 sampled
operator cases and all18 complete-model input/policy cases. Full-model speedup
ratios are0.285–0.634, equivalent to1.58–3.51 times longer software execution.
The result rejects this implementation as an acceleration mechanism in the
recorded regime. It does not reject every possible selective-emulation method.

Correctness and performance have different outcomes. No BF16 mismatch, false
acceptance or bound violation appeared in the canonical operator audits. All
2214 compared intermediate-output events, complete model outputs and recorded
controller arrays matched the full reference. This finite campaign is evidence
for the specified implementation, not a proof for every input or vendor kernel.

[Complete summary](2026-11-24-selective-reference.json),
[operator cases](2026-11-24-selective-reference.operators.csv),
[all model timings](2026-11-24-selective-reference.models.csv),
[all fidelity events](2026-11-24-selective-reference.fidelity.csv) and
[component measurements](2026-11-24-selective-reference.stages.csv) are generated
from the [complete evidence archive](statistics/2026-11-24-selective-reference/README.md).
No unfavorable case or preliminary unsafe-filter result was discarded.

## What the algorithm actually does

The reference accepts BF16 matrices `A[M,K]`, `B[K,N]`. For each output, it uses
sequential round-to-nearest FP32 fused multiply-add inside K128 primitives and
ordered FP32 addition between primitives. It then adds optional bias in FP32
and explicitly observes BF16. Matrix-product outputs remain FP32 before that
observation. This is a chosen finite arithmetic contract; the actual accelerator
arithmetic has not been specified or measured in this campaign.

A candidate uses either four independent FP32 accumulator chains with a balanced
merge, or native K128 Tensor Core products with ordered FP32 merging. Runtime
source norms and immutable weight norms produce an upper bound on the difference
between candidate and reference. The filter accepts an output only when its
whole error interval lies strictly inside one BF16 rounding cell. Unresolved
scalar output indices are compacted; the sequential reference computes those
outputs exactly. The actual reconstructed tensor feeds subsequent operators.
There is no reference-output oracle in the runtime selector.

The [algorithm description](../docs/selective-bf16-reference.md) defines reduction
depths, gamma bounds, underflow/domain handling and bias treatment. Norm and
interval operations use directed FP32 rounding in CUDA. The four-chain graph
is source-defined. The Tensor Core argument is conditional on its stated FP32
rounding model; empirical agreement does not establish an unconditional vendor
arithmetic certificate. Arbitrary FP32 consumers are outside this BF16 shortcut.

## Baselines, models and inputs

Both complete reference kernels are native CUDA: direct scalar-output threads
and a shared-memory tiled implementation. Every reported speedup divides the
faster complete-reference median for that case by the candidate median. This
allows the reference to choose its better implementation and avoids a CPU or
unoptimized Python reference as a timing baseline.

Pinned pretrained ConvNeXt-V2 Tiny and DINOv3 ViT-S use their existing checkpoint
and preprocessing revisions. Operator inputs come from the first declared
Imagenette loader batch, using at most256 deterministically spaced source rows
and the first unique receiving-weight shape: eight ConvNeXt-V2 cases and four
DINOv3 cases. These are correlated layer cases, not12 independent networks.

Complete-model audits use the same first three images for both networks,
allocation offset250, batch1. The selected34 ConvNeXt-V2 and48 DINOv3 receiving
operators use the declared reference; other operators retain Torch behavior.
This is a complete forward graph with selected arithmetic substitutions, not
complete device-arithmetic emulation. DINOv3 returns384-dimensional descriptors.
The small input set assesses fidelity and timing, not population task accuracy.

Three source policies run through the actual graph:

- `bf16`: unchanged BF16 sources.
- `dynamic8`: current group32 power-of-two scales and reconstructed integer
  activations, with zero feedback.
- `live4`: four-bit reconstruction, coefficients selected from preceding
  32-row terminal moments, smoothed state and reset per consumer/image.

The last policy has subsequent row blocks in early CNN layers. DINOv3 sources
are represented with height1, so there is no later row block on which its state
can adapt. Its exact state comparison is useful, but is not evidence for useful
sequential adaptation in that model. No native packed INT4/INT8 speed is claimed.

## Repeated complete-model performance

All methods have11 rotated timing trials per input/policy case. Each entry below
is the mean of the three per-input medians. `Full` averages the better direct or
tiled reference median per input. Speedup is `Full / selective`; values below1
mean slower execution. These times are software-emulator wall latency on one
RTX4070, Torch2.11.0+cu128, eight host threads. All samples and CUDA elapsed times
are retained separately in the complete records.

| Model | Source policy | Full, ms | Four-chain selective, ms | Tensor Core selective, ms | Four-chain speedup | Tensor Core speedup |
|---|---|---:|---:|---:|---:|---:|
| ConvNeXt-V2 | bf16 | 9.155 | 24.785 | 27.456 | 0.369 | 0.333 |
| ConvNeXt-V2 | dynamic8 | 11.317 | 29.822 | 29.231 | 0.379 | 0.387 |
| ConvNeXt-V2 | live4 | 20.520 | 43.911 | 41.856 | 0.467 | 0.490 |
| DINOv3 | bf16 | 10.786 | 30.702 | 29.326 | 0.351 | 0.368 |
| DINOv3 | dynamic8 | 16.564 | 41.412 | 32.452 | 0.400 | 0.510 |
| DINOv3 | live4 | 31.316 | 53.364 | 52.105 | 0.587 | 0.601 |

Timing includes source control, runtime norms, cell filtering, index compaction,
exact repair, conversions and the rest of the model. Immutable weight/norm
preparation is excluded and recorded separately. Intermediate CPU snapshots and
complete controller collection occur in separate audit passes. Timed outputs
and deferred invalid counters are checked after the timer. The earlier single
trial per audit is preserved but does not determine this performance conclusion.

## Why the reference work does not disappear

Across1,720,552 operator outputs, unrepaired candidates differed from the
observed reference in116 four-chain outputs and163 Tensor Core outputs. Yet
conservative bounds request repair for6.89–48.34% of four-chain outputs and
13.60–72.39% of Tensor Core outputs, depending on the case. A small observed
error frequency does not provide an independently justified runtime bound.
Dropping the guard would change the correctness contract.

Component experiments preserve all24 operator/backend alternatives. With
prerequisites precomputed, source norms average approximately70–86 microseconds,
cell filtering57–83 microseconds, index compaction171–197 microseconds and
indexed repair142–183 microseconds across the four model/backend groups. These
are warm, isolated wall-time averages of medians; they must not be summed into
a causal decomposition of complete latency. Candidate-only evaluation is also
measured and still requires its later observation/repair work. The complete
selective execution, rather than a saved-output count, determines usefulness.

The implemented improvements already include native directed arithmetic,
compact scalar-output repair and a comparison against both full-reference
kernels. Repeating a group-certification percentage or comparing with a slow
CPU reference would not overturn the negative result. A future reconsideration
requires a materially different bound, scheduling/compaction mechanism or
reference contract, followed by the same complete-cost comparison.

## A precision-cast failure retained as evidence

The compiler can remove intermediate downcast/upcast operations. A rounding-cell
guard depends on the BF16 round trip, so this changes its meaning. In the
constructed three-value witness, eager execution rejects all intervals. Compiled
execution with `emulate_precision_casts=False` accepts all three; `True` reproduces
all eager rejections. Exact FP32 input words, FP64 bounds and results are archived.

Earlier unsafe compiled prototypes produced actual repaired-output mismatches;
those preliminary logs remain distinct from the canonical native-filter runs.
[The pinned upstream configuration](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/torch/_inductor/config.py)
documents the cast-elimination behavior. This is a semantic hazard for this
predicate, not evidence of a new general compiler bug. Faster execution after
removing a required rounding cannot count as a valid speedup.

## Relationship to existing methods and device conclusions

[Rounding-cell acceptance](https://perso.citi-lab.fr/fdedinec/recherche/publis/2013-TOMS-OnZivRoundingTest.pdf)
and classical forward-error bounds are established methods.
[Arbitor](https://www.usenix.org/conference/atc23/presentation/jiang-chenhao) already
performs numerically accurate accelerator emulation on GPUs.
[Recent fixed-point GEMM emulation guarantees](https://arxiv.org/abs/2611.19008v1)
already use operand-dependent accuracy selection. Matching an observation of a
finite operation graph differs from rounding an exact real dot product, but
that specification difference and this implementation do not by themselves
establish a new method. The present implementation has no demonstrated
performance advantage against its directly measured same-contract baseline.

None of these timings estimates systolic-array latency or energy. The K128
reference is an arithmetic partition, not an assertion of a fixed physical
M128 dimension. Native FP8 K256 capacity is not assigned to reconstructed
integer activations. Hardware optimizations must still account for MACs,
operand/result movement, conversion, buffers and controller costs separately.

## Reproduction and verification

Use the pinned experiment/engine commits in each raw record. Build the engine
Torch extension and install the existing pinned experiment requirements. The
producers are `selective_reference_operator_pilot.py`,
`selective_reference_model_pilot.py`, `selective_reference_model_timing.py`,
`selective_reference_stage_pilot.py` and `precision_cast_audit.py`; each supports
`--help`. Run canonical GPU producers sequentially without source changes.
The model timing producer consumes its complete audit report; the stage producer
consumes its complete operator report and hash-verified captured operands.

Archive validation and table regeneration need only the committed evidence and
are documented in the [archive README](statistics/2026-11-24-selective-reference/README.md).
Local validation:212 experiment Python tests,211 engine Python tests and four
native CTest cases passed. Existing warnings remain. A bare unscoped pytest
invocation collected ignored historical checkouts and failed collection; the
reported suite results use the explicit `tests` directories.
