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

Numerical experiment reports will be linked here as they are recorded.

[Preserved scientific statistics](../../reports/statistics/2026-11-23/models.md#dinov3-small)
include the complete dated records available at this snapshot, including earlier
exploratory runs. Consult their original conditions and source status.
