# Runtime activation representation pilots — 2026-11-22

These exploratory experiments test numerical representations at pretrained
MLP input edges. They identify quality and implementation tradeoffs; they do
not measure low-bit execution, ASIC energy, or establish a novel algorithm.

Reproduce the main comparisons with the
[fixed command pipeline](../docs/reproduce-activation-pilots.md). Individual
runs retain exact checkpoint/source hashes, settings, sample manifests and
original execution revisions/dirty flags. The
[recorded provenance](2026-11-22-activation-provenance.json) retains those fields.
Detailed generated reports, images
and trajectories stay in ignored `results/`; they can be regenerated. The
snapshot tag `snapshot/activation-pilots-2026-11-22` preserves this complete
experiment checkout and its nested engine revision after PR squash merges.

## Scope and controls

Classifier pilots use official pretrained ViT-B/16 and Swin-T, Imagenette 2-160
validation images with official preprocessing and the original 1000-class
predictions. The main holdout has 1000 images, disjoint from diagnostic images
and 20/class screening examples. Batch size 16 is fixed for these comparisons.
They are not full ImageNet measurements. All eligible MLP linear input edges
are transformed; weights remain BF16, and products remain ordinary BF16.

Main DiT-XL/2 comparisons use 40 images, eight conditioning classes repeated
five times, seed 20261125, batch 2,50 DDIM steps, guidance 4 and the pinned
checkpoint revision `eab87f77abd5aef071a632f08807fbaab0b704d0`. Only 56 feed-forward
linear input edges are transformed. Initial noise/batching is paired.
Runtime choices use current values and fixed receiving-weight metadata, with
no fitted activation calibration or task labels. Scale groups cover 32 values;
format decisions cover 128 × 128 input blocks. E2M1 scales remain FP32: this is
not the complete NVFP4 or MXFP4 recipe.

## Main classifier results

Accuracy uses all 1000 output classes. E2M1 MAC share weights each mode by the
receiving product size, rather than averaging layer tile counts. It is a
representation share, not saved arithmetic/energy. All adaptive rows below use
the same global local-score tolerance 0.01; that score is not a task bound.

| Model | Representation / score | Top-1 | E2M1 dense-equivalent MAC share |
| --- | --- | ---: | ---: |
| ViT-B/16 | `baseline` | 86.2% | — |
| ViT-B/16 | `affine:4` | 86.2% | — |
| ViT-B/16 | `fp4:4` | 85.5% | — |
| ViT-B/16 | `precision-diagonal:0.01` | 85.6% | 53.73% |
| ViT-B/16 | `precision-gram:0.01` | 86.1% | 65.60% |
| ViT-B/16 | `adaptive-diagonal:0.01` | 85.2% | 53.29% |
| ViT-B/16 | `adaptive-gram:0.01` | 85.4% | 65.19% |
| ViT-B/16 | `precision-gram-numerator:0.01` | 85.7% | 51.37% |
| ViT-B/16 | `precision-gram-denominator:0.01` | 86.3% | 66.38% |
| Swin-T | `baseline` | 80.9% | — |
| Swin-T | `affine:4` | 80.4% | — |
| Swin-T | `fp4:4` | 80.3% | — |
| Swin-T | `precision-diagonal:0.01` | 80.7% | 33.23% |
| Swin-T | `precision-gram:0.01` | 80.3% | 36.10% |
| Swin-T | `adaptive-diagonal:0.01` | 80.5% | 33.24% |
| Swin-T | `adaptive-gram:0.01` | 80.6% | 36.05% |
| Swin-T | `precision-gram-numerator:0.01` | 80.4% | 33.28% |
| Swin-T | `precision-gram-denominator:0.01` | 80.6% | 36.06% |

Affine INT4 is a strong baseline, particularly on ViT. Full Gram geometry does
not consistently improve classifier quality over a cheaper diagonal score.
The hybrid policies are different decision rules: `gram-numerator` keeps full
geometry only for four-bit error; `gram-denominator` keeps it only for signal
energy. They are not algebraic equivalents. Multiple exploratory comparisons
on the same holdout do not support a confirmatory significance claim.

Forced 2:4 at every MLP edge loses substantial classifier quality. On the
same 1000-image holdout, ViT baseline 86.2% became 82.0% with Gram-selected BF16
masks; Swin baseline 80.9% became 68.6%. Strict adaptive fallback retains quality
with only a modest sparse-work fraction; sparsity is not a free companion to
low precision.

## Main paired diffusion results

PSNR is the mean of per-image same-seed PSNR values, not PSNR of pooled MSE.
Class consistency uses two unmodified ImageNet classifiers and the requested
conditioning class. Neither metric measures FID, aesthetics or diversity;
40 samples across eight classes do not establish general generation quality.

| Representation | Mean PSNR vs BF16 | E2M1 MAC share | ViT class matches | Swin class matches |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | reference | — | 40/40 | 40/40 |
| `dense:8` | 25.61 dB | — | 40/40 | 40/40 |
| `affine:4` | 13.33 dB | — | 25/40 | 21/40 |
| `fp4:4` | 13.33 dB | — | 22/40 | 19/40 |
| `precision-diagonal:0.01` | 15.33 dB | 43.72% | 39/40 | 39/40 |
| `precision-gram:0.01` | 17.38 dB | 45.50% | 40/40 | 40/40 |
| `adaptive-gram:0.01` | 15.34 dB | 45.04% | 39/40 | 40/40 |
| `precision-gram-numerator:0.01` | 18.43 dB | 42.41% | 40/40 | 40/40 |
| `sparse-residual:4` | 25.50 dB | — | 40/40 | 40/40 |
| `double-sparse:4` | 16.57 dB | — | 39/40 | 40/40 |
| `double-sparse-p2:4` | 16.14 dB | — | 39/40 | 40/40 |

The full local Gram score improves paired DiT fidelity over diagonal scoring
in this pilot, while selecting a slightly larger E2M1 share. Keeping geometry
only for low-format error improves fidelity further, selects somewhat more
INT8 and evaluates 18 rather than 30 quadratic terms/quartet. This is not a
matched-format-budget comparison. The 40% reduction concerns score terms,
not total controller cycles or energy.
Known-family residual representations are competitive and must be retained
as strong baselines rather than crediting their decomposition as a new idea.

The sparse+dense reference follows the backbone/residual mechanism of
[SharQ](https://arxiv.org/abs/2606.26587v1). The two-sparse reference uses known
greedy error-gain assignment, with
[VR-SparseQuant](https://arxiv.org/abs/2611.25717v1) a close comparator.
The tested variants use current-value FP32 scales, no smoothing/calibration,
and unchanged BF16 weights; they are not full reproductions of those methods.
The P2 variant restricts residual/coarse scale ratios to 2^(-s), s in 0..8.
The hook adds reconstructions before BF16 casting; separate product-rounding
semantics are not tested. Known GPU implementations already share weights
and accumulation; two completely independent GEMMs are not a sufficient
optimized baseline.

## Cheap-decision and prediction diagnostics

The static coherence gate at delta 0.1 marks nearly every quartet for full
geometry and is ineffective as a cheap-path selector on these inputs. A
runtime best-versus-runner diagonal-mask certificate certifies about 53% of
sampled groups, with no observed local-objective violations. This is neither
a floating-point proof nor a whole-network guarantee.

At the useful precision tolerance 0.01, sufficient interval bounds certify only
about 11%/6% of ViT/Swin tiles when averaging layers equally; at 0.001 and 0.03
they certify many more. The middle regime does not justify assuming that
most full-score work disappears. Exact-bin scale prediction also misses often:
previous-block hit means are about 52%/37%; previous-row/same-channel means
about 61%/51%. A buffered 32-value current-group encoder already reads raw input
once, so prediction does not remove another global read from that baseline.

## Native controller checks and GPU timing

The native CUDA controller buffers an input 128 × 128 block (approximately 41KB
shared storage, with padding/reductions), evaluates candidate scalar scores,
then encodes the chosen mode. It reads global input once and performs no
receiving matrix product. Metadata generation and count collection are
separate work. All four geometry variants have matched numerical-reference
checks. On the tested 1000-image classifier runs, native/full-reference
predictions and mode/MAC counts match exactly. For the paired eight-image
DiT format run, images and every observed latent state match exactly.

Final CUDA graph timing was run without concurrent research GPU jobs. Values
are medians of five graph trials on RTX 4070; they are not ASIC timing/energy.

| M,N,K | BF16 GEMM | Fused fixed INT4 + Gram mask | Native diagonal precision | Native full precision | Native low-error geometry |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128,3072,768 | 17.23 us | 2.99 us | 23.48 us | 24.43 us | 23.56 us |
| 128,768,3072 | 20.47 us | 8.37 us | 23.89 us | 24.55 us | 23.67 us |
| 128,128,1024 | 7.85 us | 3.58 us | 23.60 us | 24.37 us | 23.65 us |

For these shapes the full native precision-controller kernel alone is slower
than BF16 GEMM. This implementation has not met a low-overhead GPU deployment
criterion; it does not demonstrate an inference speedup. CUDA latency also
does not determine a dedicated ASIC controller's cost.

A native scalar controller still costs work. Its measured latency must not be
replaced by the much slower unfused reference, nor assumed free because it
avoids a GEMM. Metadata is 10K bytes (5/N of BF16 weight bytes); construction
uses approximately 2.5KN products and can be amortized across its lifetime.

## Systolic interpretation

The targeted MLP edges account for approximately 63.51% (ViT),61.79% (Swin),
and 64.13% (DiT) of inventoried matrix/convolution FLOPs. This is not an energy
fraction, complete model FLOPs, or physically padded work. These pilots target
a substantial part of matrix work but do not yet demonstrate total-energy gains.

The [action ledger](../docs/systolic-actions.md) separates multipliers,
wide accumulation, operand forwarding and result SRAM actions. Gate-only 2:4
keeps the dense schedule/B delivery; supported sparse skipping is a different
contract. A common mask across 128 rows can compact both operands and remove
K128 calls, but needs quality checks plus charged selection/gathering.
For M=128, N=768, K=3072 and common half-K, calls change 144 to 72. If every K tile
updates result SRAM, valid-output traffic changes 17.625 to 8.625 MiB; if partial
sums remain in PEs, final-output traffic is unchanged. No DRAM spill is assumed.

Sparse+dense residual has 1.5 dense FP4 nonzero-product count; two 2:4 streams
have 1.0, before metadata/scales/combination. Ordinary dense hardware may need
two passes. Independently scaled 32-value groups require block-scaled arithmetic
or explicit partial-product rescaling within K128. Wide FP32 accumulation,
clocks and operand registers do not become proportionally cheaper when only
activation codes shrink. Fixed BF16 weights in these numerical pilots do not
establish a cheap native integer datapath. Integer weights, actual packed
products, format-dependent array shapes/coalescing and accumulation semantics
require their own implementation and numerical validation.

Measured common-mask call reductions at tolerance 0.01 are 0.0067% for ViT
(top-1 unchanged 86.2%) and 2.20% for Swin (81.0% versus 80.9% baseline). At 0.03
they are 2.90%/5.12%, with top-1 84.6%/80.7%. These are explicit padded-call
counts, not the nominal 50% reduction of every accepted window. Small global
savings leave little room for selection/packing overhead on these workloads.

Even without a receiving GEMM, the current controller needs throughput.
Its 18-term straight quadratic expansion has a pair product and coefficient
product/FMA per term:147,456 general multiply contributions per 128 × 128 block,
before quantization, reductions and encoding, ignoring constant factors.
At 32 multiply lanes this expansion alone needs at least 4608 lane-issue cycles;
ideal dense 128 × 128 feed across N=768 takes 6*128=768 cycles before bubbles.
This illustrates a possible controller bottleneck; it is not an ASIC circuit
lower bound or a claim that this is the best algebraic implementation.

Current hooks preserve original tensors for other consumers. Encoding before
operand memory still reads raw BF16 from general memory; earlier producer-side
compression would need graph/vector-consumer format semantics and separate
quality tests. Narrow values must remain narrow through the chosen path to
obtain its physical savings. Compute-bound latency does not establish an
arithmetic-dominated energy split, or eliminate possible weight streaming.

No universal energy-component percentages or target joules are inferred.
Use characterized action costs, including new detectors/codecs/control,
operand/result memory and leakage, before claiming a hardware energy win.

## Local validation

The final engine checkout passes 156 Python tests, including CUDA graph replay
of cached-table references; both compiled native CTests passed. The experiments
checkout passes 22 tests, including full reference-policy graph replay, native
mode equivalence, common-mask compaction and fixed-tile action boundaries.
Both CMake and C++/CUDA formatting checks passed. Optional dependency
installation satisfies `pip check`.
