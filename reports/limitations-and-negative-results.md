# Unfavorable results and limitations register

Updated2026-11-24. Retain weak variants and their complete evidence. Each finding
is conditional on its recorded models, inputs, precision, baseline and mapping;
it does not disprove the entire family. Revisit conditions below identify what
would justify another experiment rather than repeating the same measurement.

| Variant / observation | Evidence | What was learned | Condition for revisiting |
|---|---|---|---|
| Forced2:4 activation sparsity | [Representation report](2026-11-22-activation-representations.md) | Accuracy fell on the evaluated ViT/Swin controls. A sparse mask alone does not establish useful array skipping. | A different live mask/codec with matched-quality evidence and a realizable shared-mask schedule. |
| Full receiving-geometry versus diagonal selectors | [Representation report](2026-11-22-activation-representations.md) | Some local/diffusion fidelity gains did not consistently improve classification. | Demonstrable end-task improvement sufficient to repay added statistics/state. |
| Conservative live BF16/FP8 guard | [Modern precision report](2026-11-23-paired-precision.md) | Tested strict thresholds retained almost all work in BF16. | A better justified, cheaper acceptance criterion with actual complete-model validation. |
| More FP8 rows through bounded partitioning | [Full adaptive archive](statistics/2026-11-23-adaptive/README.md) | More low-precision work and fewer mode changes require extra weight-format capacity; the1000-image ConvNeXt-V2 comparison showed no quality advantage over uniform FP8. | A workload/quality regime where adaptation beats a strong uniform-low-precision control, including format/setup costs. |
| Centered-moment precision criterion | [Complete modern campaigns](statistics/2026-11-23-adaptive/README.md) | The more complex criterion did not clearly outperform the cheaper diagonal surrogate. | A measured predictive/task advantage and explicitly accounted collection cost. |
| Spatial feedback evaluated only locally | [Spatial feedback report](2026-11-23-spatial-feedback.md) | Reduced receiving error did not consistently improve final classifier outputs. | Distinguish local versus propagated error and test an appropriate full workload; modern decoder evidence is separately retained. |
| Causal terminal-moment feedback controller | [Decoder report](2026-11-23-decoder-feedback.md) | Relative to fixed feedback, initial/additional pixel-MSE ratios were1.0014760/0.9977987, with mixed per-image signs. Extra adaptation has no consistent demonstrated benefit. | A better-supported causal sensor/objective, specified before new quality comparisons. |
| Uniform coarse feedback at INT8 | [Decoder report](2026-11-23-decoder-feedback.md) | One examined W8/A8 output worsened slightly despite favorable aggregate fidelity. | Evaluate per-input effects; aggregate gains are not a guarantee. |
| Strong merge-count reduction in guarded accumulation | [Primitive report](2026-11-23-primitive-accumulation.md), [cost assumptions](../docs/accumulation-cost-envelope.md) | Fewer group boundaries preserve all MACs and primitive outputs. Total energy benefit remains uncharacterized. | Characterized component/mapping costs or a bounded scenario that establishes usefulness. |
| Wider or range-safe accumulation | [Primitive report](2026-11-23-primitive-accumulation.md) | Range safety and unchanged predictions do not imply FP32 bit equivalence. Some wider variants changed logits. | Declare the intended arithmetic contract and assess actual task/compatibility tolerances. |
| Exact-output claim from a real-arithmetic rewrite | [Resize report](2026-11-23-resize-folding.md) | Known folding reduces dense MACs but changes rounded coefficients/operation order; image fidelity is finite. | Appropriate numerical validation plus phase/weight/buffer costs; no novelty claim from the known rewrite itself. |

| Decision divergence alone as evidence of a better validation method | [Full replay study](2026-11-24-controller-replay.md) | Scale/mode differences and close mean-order reversals occur, but the eight-input study does not establish a robust quality winner, energy ranking or novel validation capability. | A declared useful comparison/reference budget, stronger independent task evidence and precise comparison against existing testing/stability methods. |
| Adaptive feedback versus fixed feedback under backend changes | [Full replay study](2026-11-24-controller-replay.md) | Pixel/descriptor observers and declared implementations can disagree on the close ordering. All contrasts and unfavorable inputs remain visible. | A causal objective with a predeclared task metric and sufficient independent inputs; include implementation uncertainty rather than selecting a favorable observer. |
| Interval scale filter with observed pairwise bounds | [Conditional guard](../docs/decision-stability.md), [all patch results](2026-11-24-controller-replay.patches.csv) | About99% local and93–94% global-bound certification is diagnostic selectivity because both trajectories were already computed. It certifies scales only. | Independently justified forward bounds, state/downstream treatment and a measured reference-work budget; compare established FLUCTUAT/PRECiSA principles. |

| Source-group certification as a matrix-work saving estimate | [Convolution footprint](2026-11-24-controller-guard-footprint.md) | Roughly99% group certification still leaves81–85% of M128 cohorts flagged under one declared lowering; another lowering changes clustering. | A complete selective-reference implementation with explicit reduction order, stream batching, state/repair behavior and measured overhead; group rates alone are insufficient. |

CUDA implementation cost is not target hardware time or energy. The hardware-optimization variants above are not rejected solely because their
experimental CUDA implementation is slow. The selective-reference variant below
explicitly targets SOFTWARE emulator acceleration, for which software time is
the intended metric.
An unavailable implementation/full text is not evidence that prior research
lacks hardware. Unmeasured ideas remain separate from measured negative results.

Original source/input identities and complete statistical archives remain linked
from the reports. Do not silently relabel old exploratory inputs as independent
holdouts, discard unfavorable cases, or collapse incompatible counting ledgers.

## Software-reference implementation results

| Additional variant | Evidence | What was learned | Condition for revisiting |
|---|---|---|---|
| Operand-bounded exact BF16 observation with selective reference repair | [Complete campaign](2026-11-24-selective-reference.md) | No tested fidelity errors, but all12 operator and18 full-model cases are slower; full-model speedup ratios0.285–0.634. The v0 acceleration claim fails. | A materially different useful bound or execution mechanism, measured against the better complete same-contract CUDA reference with all overhead charged. |
| Compiled guard with default precision-cast elimination | [Constructed witness and retained prototypes](2026-11-24-selective-reference.md#a-precision-cast-failure-retained-as-evidence) | Removing the BF16 round trip yields false cell acceptance; apparent faster execution can be semantically invalid. | Explicitly preserve required rounding and validate boundary witnesses after compilation. This does not imply a new general compiler bug. |

## Shared receiving-geometry activation rounding

[Complete comparison](2026-11-24-activation-rounding.md): first-QKV rank1 INT4
receiving SSE falls to0.1052 of nearest rounding, but the ten-image full-model
descriptor mean worsens from0.080346 to0.080774 relative MSE. Rank4 improves this
comparator but remains far from uniform INT8. Local metric improvement is
insufficient as an acceptance criterion. The preliminary exact-grid alternative
and all corrected variants are preserved separately; no target-energy saving
is inferred from reconstructed operand bits or software execution.

## Attention rounding fails the declared recognition screen

[Complete3000-query replication](2026-11-24-attention-task.md) rejects the tested
Q-only/K-only INT4 configurations under the1pp-loss acceptance rule. Their point
losses against uniform INT8 are2.0/2.1pp. Balanced rank4 gains1.167pp over nearest
QKV4, but its declared family lower bound remains below zero. All primary
screens fail. Retain the complete result; do not select a favorable100-image
point estimate or descriptor-MSE ranking as a replacement acceptance criterion.

## Transform precision and source support

[Complete Winograd study](2026-11-24-winograd.md) retains all comparisons.
Shape-fit one-axis selection loses follow-up pixel fidelity despite favorable
discovery; all-layer one-axis improves fidelity while increasing vector work.
Support-conditioned causal geometry gives small mixed per-input changes and no
consistent free quality improvement. INT4 feedback does not recover INT8 pixel
fidelity. Larger-tile local group32/BF16-weight improvements do not establish
full W8/K256 quality, whose numerical contract differs.

The shape-only mixed BF16-F4/eight-bit-F2 control is a stronger baseline, not a
new adaptive algorithm. Its improved four-input pixel-MSE/packet point does not
establish energy savings: multiplier precision, gating, weight movement,
state/ports and array scheduling remain necessary costs. Preserve both positive
and negative points rather than treating a local metric or fewer packets as a
whole-system result.
