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
