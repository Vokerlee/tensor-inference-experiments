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

[Paired precision numerical report](../../reports/2026-11-23-paired-precision.md)
records the fixed-K128 baseline, uniform FP8 and live-bound policies. Complete
statistics, exact input/checkpoint conditions and limitations accompany the report.

[Preserved scientific statistics](../../reports/statistics/2026-11-23/models.md#sana-sprint-06b)
include the complete dated records available at this snapshot, including earlier
exploratory runs. Consult their original conditions and source status.

## Complete cold-prompt request

The [complete component profile](pipeline-geometry-2026-11-23.json) counts one
300-token prompt encoding, two denoiser calls and one1024×1024 image decode.
Every component agrees with an independent Torch dispatch FLOP count.

| Component | Request MACs | Share of matrix/convolution MACs |
|---|---:|---:|
| Gemma2 prompt encoder | 616,867,468,800 | 10.4154% |
| SANA denoiser, two calls | 1,121,491,353,600 | 18.9356% |
| DC-AE image decoder | 4,184,305,827,840 | 70.6490% |

Spatial convolutions contribute62.2204% of this complete-request denominator.
The earlier0.2651% spatial share concerns the denoiser only. These are not
energy shares. Encoder lookup/vector work, tokenization, scheduling, memory
and masked-kernel work elision need separate accounting; cached prompts or
other step counts change the request balance. The text shape path uses FP32
CPU-fake eager attention; decoder/denoiser use BF16 meta tensors. Numerical
weights/quality are not evaluated by this profile.

[Extended numerical and schedule evidence](../../reports/statistics/2026-11-23-adaptive/README.md)
retains all moment/cohort/phase-trace comparisons, including unfavorable results.
See the separate [DC-AE decoder card](../dc-ae-sana-decoder/README.md).

[Exact-input decoder feedback results](../../reports/2026-11-23-decoder-feedback.md)
cover eight1024px outputs, BF16/INT8 weights, INT4/INT8 activations and a causal
row-statistic controller ablation. Full JSON statistics are archived; pixel
fidelity and MAC coverage do not establish generation quality or energy.

[Resize folding and source-placement results](../../reports/2026-11-23-resize-folding.md)
validate the known nearest-neighbor transformation, coefficient precision and
conversion before duplication, with complete per-branch statistics. Dense MAC
reductions and pixel fidelity remain distinct from hardware energy/task quality.

[Joint precision/transform and complete Winograd study](../../reports/2026-11-24-winograd.md)
compares native rounding, one-axis lowering, FP8 controls, producer support and
mixed BF16/F2/F4 execution on five reused decoder conditions. The shape-only
mixed controls improve all four follow-up pixel-MSE comparisons and issue17.66%
fewer input vectors than their uniform eight-bit F2 controls. This is not a
hardware-energy or population-level generation-quality result. All full output
arrays, coefficient/format metadata and unfavorable cases are archived.
