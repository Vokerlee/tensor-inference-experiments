# DINOv3 ViT-S

2025 vision feature backbone; axial position handling and transformer projection/MLP products. The pinned checkpoint returns384-dimensional global descriptors, with no classifier head.

- Source: [paper](https://arxiv.org/abs/2508.10104v1).
- Checkpoint/configuration: `timm/vit_small_patch16_dinov3.lvd1689m` at `3bf4720a82ec2066db88137180ff1f83a675cef0`.
- Profile input: 256×256 image, batch1.
- Parameters in this scope: 21,586,944; hypothetical BF16 parameter storage: 43,173,888 bytes. This excludes buffers, activations, workspace and additional encoded copies.
- Evidence: meta-device shape execution, without numerical pretrained weights.

| Measured shape statistic | Value |
|---|---:|
| Matrix and convolution MACs | 6,245,286,912 |
| Dense linear +1×1-convolution MAC share | 88.7387% |
| Spatial-convolution MAC share | 1.2089% |
| Depthwise MAC share | 0.0000% |

The denominator includes matrix and convolution products, not all operations.
These are neither energy shares nor a compute-bound classification. Operator
normalization, activation, memory and control costs need separate evidence.

[Complete immutable profile](geometry-2026-11-23.json) retains every recorded convolution, dense
matrix shape, interface-capacity comparison, source revision/hash and scope
limitation. FP8 K256 capacity is hypothetical: its numerical suitability is
not established by this shape profile. Output-vector capacity is not SRAM
traffic or output cadence.

[Paired precision numerical report](../../reports/2026-11-23-paired-precision.md)
records the fixed-K128 baseline, uniform FP8 and live-bound policies. Complete
statistics, exact input/checkpoint conditions and limitations accompany the report.

[Preserved scientific statistics](../../reports/statistics/2026-11-23/models.md#dinov3-small)
include the complete dated records available at this snapshot, including earlier
exploratory runs. Consult their original conditions and source status.

[Extended moment/cohort/phase statistics](../../reports/statistics/2026-11-23-adaptive/README.md)
preserve all subsequent comparisons. ConvNeXt-V2 includes a separate1000-image
Imagenette holdout at allocation offset100; DINO records descriptor diagnostics.
Preliminary and unfavorable results remain distinguishable from clean evidence.

[Controller-replay descriptor observation](../../reports/2026-11-24-controller-replay.md)
uses this pinned model in FP32 with official preprocessing to describe224 SANA
control variants and eight original images. Full384-dimensional vectors and
paired contrasts are preserved. Descriptor consistency is a fidelity diagnostic,
not a generation-quality benchmark or an adaptive-policy quality guarantee.

[Selective finite-reference observations](../../reports/2026-11-24-selective-reference.md)
validate intermediate/final BF16 fidelity and dynamic-controller arrays on the
same three declared Imagenette inputs. Both selective implementations are slower
than the complete native reference. All per-case timings and full controller
arrays are preserved; this is software-emulation evidence, not device energy
or population task accuracy. DINOv3 height1 control has no subsequent row block.

[Shared activation rounding](../../reports/2026-11-24-activation-rounding.md)
compares low-rank receiving metrics, matched Q/K/V roles and complete descriptors.
A large local QKV improvement does not transfer to rank1 full-model fidelity.
Both descriptor campaigns reuse the same ten images; task recognition is a
separate experiment. All favorable, unfavorable and preliminary statistics
remain recoverable.

[Frozen-feature CIFAR100 task](../../reports/2026-11-24-attention-task.md)
adds100 discovery and3000 disjoint replication images with a fixed2000-training-
image gallery. Uniform INT8 gives64.367%; Q-only/K-only INT4 give62.367%/62.267%
and fail the declared1pp-loss screen. Balanced rank4 superiority is not
established. This is a constructed descriptor-recognition task, not the model's
published classification benchmark. All27900 policy outputs are independently
audited and preserved.
