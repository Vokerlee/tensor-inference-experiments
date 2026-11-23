# SANA-Sprint0.6B

2025 text-conditioned few-step diffusion model. This profile covers its denoiser only: linear/cross attention, pointwise channel mixing and depthwise spatial processing. Text encoding and VAE decoding are excluded.

- Source: [paper](https://arxiv.org/abs/2503.09641v1).
- Checkpoint/configuration: `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers` at `aa76e7f4f4928f378716b6716a2130fba3caf5b1`.
- Profile input: 1024×1024 image, batch1;32×32 latent;300 text tokens.
- Parameters in this scope: 593,501,216; hypothetical BF16 parameter storage: 1,187,002,432 bytes. This excludes buffers, activations, workspace and additional encoded copies.
- Evidence: meta-device shape execution, without numerical pretrained weights.

| Measured shape statistic | Value |
|---|---:|
| Matrix and convolution MACs | 560,745,676,800 |
| Dense linear +1×1-convolution MAC share | 95.8119% |
| Spatial-convolution MAC share | 0.2651% |
| Depthwise MAC share | 0.2651% |

The denominator includes matrix and convolution products, not all operations.
These are neither energy shares nor a compute-bound classification. Operator
normalization, activation, memory and control costs need separate evidence.

[Complete immutable profile](geometry-2026-11-23.json) retains every recorded convolution, dense
matrix shape, interface-capacity comparison, source revision/hash and scope
limitation. FP8 K256 capacity is hypothetical: its numerical suitability is
not established by this shape profile. Output-vector capacity is not SRAM
traffic or output cadence.

Numerical experiment reports will be linked here as they are recorded.
