# ConvNeXt-V2 Tiny

2023 architectural control: depthwise spatial filtering, dense channel mixing and global response normalization. The pinned checkpoint has a1000-class ImageNet head.

- Source: [paper](https://arxiv.org/abs/2301.00808v1).
- Checkpoint/configuration: `timm/convnextv2_tiny.fcmae_ft_in22k_in1k` at `b1dd46230e80bf4cc3fa0c3c905db2c3ec53a817`.
- Profile input: 224×224 image, batch1.
- Parameters in this scope: 28,635,496; hypothetical BF16 parameter storage: 57,270,992 bytes. This excludes buffers, activations, workspace and additional encoded copies.
- Evidence: meta-device shape execution, without numerical pretrained weights.

| Measured shape statistic | Value |
|---|---:|
| Matrix and convolution MACs | 4,455,531,264 |
| Dense linear +1×1-convolution MAC share | 93.4247% |
| Spatial-convolution MAC share | 6.5753% |
| Depthwise MAC share | 2.3590% |

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

[Preserved scientific statistics](../../reports/statistics/2026-11-23/models.md#convnextv2-tiny)
include the complete dated records available at this snapshot, including earlier
exploratory runs. Consult their original conditions and source status.

[Extended moment/cohort/phase statistics](../../reports/statistics/2026-11-23-adaptive/README.md)
preserve all subsequent comparisons. ConvNeXt-V2 includes a separate1000-image
Imagenette holdout at allocation offset100; DINO records descriptor diagnostics.
Preliminary and unfavorable results remain distinguishable from clean evidence.

[Selective finite-reference observations](../../reports/2026-11-24-selective-reference.md)
validate intermediate/final BF16 fidelity and dynamic-controller arrays on the
same three declared Imagenette inputs. Both selective implementations are slower
than the complete native reference. All per-case timings and full controller
arrays are preserved; this is software-emulation evidence, not device energy
or population task accuracy. Early CNN layers contain successive32-row control blocks.
