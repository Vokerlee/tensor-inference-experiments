# Resize folding and source quantization placement — 2026-11-23

Known nearest-neighbor kernel folding works numerically in the pinned modern
SANA-Sprint image decoder. A BF16 folded-weight variant has55.55/53.20dB pixel
PSNR relative to original decoding on the initial/additional four images.
Declared dense MACs decrease by11.6879% of complete decoder matrix/convolution
work. This is a validated numerical/arithmetic baseline, not a new decomposition
or a measured hardware-energy reduction.

## Evidence and conditions

[Runner and transformation scope](../docs/sana-resize-folding.md),
[case table](2026-11-23-resize-folding.csv),
[full provenance](2026-11-23-resize-folding.provenance.json),
[complete JSON statistics](statistics/2026-11-23-resize/README.md).
Two clean-source campaigns use56d06dcac246dccdf68c1feccc0ef911806cb375,
engine642089cbeeb77f6a6719c201330b6eb70e6534ff. Immutable tag:
`snapshot/sana-resize-folding-2026-11-23`. All150 experiment tests passed,
including double-precision phase/border tests and exact branch/placement checks.

Inputs/checkpoint match the [decoder feedback study](2026-11-23-decoder-feedback.md):
eight exact final decoder inputs from two-step1024px SANA-Sprint, same pinned
weights/revision, prompts and seeds. Original branch replay remains bit-exact.
There are24 cases,96 per-input case records and480 complete branch records;
repeated controls are not independent input sets. No activation calibration,
retraining or quality-fitted controller parameters were used.

## Numerical comparisons

`expanded-a*` quantizes after nearest duplication; `source-a*` quantizes before
it. Both retain original BF16 weights. The folded cases use four2x2 phase banks:
FP32 weights/arithmetic, rounded BF16 weights, or prepared INT8 weights with
BF16 reconstruction. `fold-w8-a*` additionally converts current source groups;
its feedback variant uses coarse effective-kernel coefficients. Other decoder
operators and the upsampling residual shortcuts are preserved.

| Inputs | Variant | Pixel PSNR (dB) | Feedback MSE ratio |
|---|---|---:|---:|
| Initial four | fold-fp32 | 55.01936 | — |
| Initial four | fold-bf16 | 55.54745 | — |
| Initial four | expanded-a8 | 51.61345 | — |
| Initial four | source-a8 | 51.61345 | — |
| Initial four | fold-w8 | 52.80272 | — |
| Initial four | fold-w8-a8 | 50.52273 | — |
| Initial four | fold-w8-a8-feedback | 50.54535 | 0.9948038 |
| Initial four | expanded-a4 | 31.50875 | — |
| Initial four | source-a4 | 31.50875 | — |
| Initial four | fold-w8-a4 | 31.54421 | — |
| Initial four | fold-w8-a4-feedback | 31.64700 | 0.9766111 |
| Additional four | fold-fp32 | 52.74748 | — |
| Additional four | fold-bf16 | 53.20127 | — |
| Additional four | expanded-a8 | 48.75393 | — |
| Additional four | source-a8 | 48.75393 | — |
| Additional four | fold-w8 | 50.09702 | — |
| Additional four | fold-w8-a8 | 47.62031 | — |
| Additional four | fold-w8-a8-feedback | 47.63051 | 0.9976534 |
| Additional four | expanded-a4 | 28.87062 | — |
| Additional four | source-a4 | 28.87062 | — |
| Additional four | fold-w8-a4 | 28.87664 | — |
| Additional four | fold-w8-a4-feedback | 28.91666 | 0.9908278 |

Full raw AND postprocessed arrays for `source-a4`/`expanded-a4` and
`source-a8`/`expanded-a8` are bit-identical on every examined input. For each
four-image campaign, quantizer groups fall from57,147,392 to14,286,848.
This4x reduction concerns source conversions only; it is not4x fewer matrix
operations or4x lower energy. Spatial feedback does not generally commute with
replication and needs coefficients for the effective receiving transformation.

Folded INT8 weights/source INT8 plus feedback give50.54535/47.63051dB pixel
PSNR on initial/additional inputs. INT4 feedback reduces MSE versus its same-
weight folded no-feedback control by2.34%/0.92%; each of the eight individual
outputs improves. This gain is modest. The much stronger overall error change
between INT4 and INT8 remains; no matched-quality4-bit claim is established.
FP32 folding is not closer to the original than BF16 folding in these runs;
the original finite-precision execution is the comparison, not an exact-real
reference. The existing rounding order changes in both transformed variants.

## Dense arithmetic and hardware boundary

Five nearest2x-plus3x3 branches originally account for889,058,230,272 MACs per
decode. Four folded2x2 phases reduce the ideal branch MAC count to4/9, an
established result from [Colbert et al., Section4.2](https://arxiv.org/html/2107.07647v1).
Our implementation additionally computes one border row/column per phase plane.
Its actual declared dense branch count is400,000,811,008 MACs (ratio0.4499152),
saving489,057,419,264 MACs per decode.

Against the independently profiled4,184,305,827,840 decoder MACs, this is11.6879%.
Against the cold two-step full-request5,922,664,650,240 MACs, this is8.2574%.
Text/denoiser/other decoder work is unchanged. These fractions describe dense
arithmetic only; hardware runtime and energy depend on tiling, bandwidth and
execution. The prepared bank contains16/9 as many weight scalars; buffers,
weight delivery, phase-output addressing, metadata and data-layout copies are
not free. Four phase banks do not imply a physically four-times-wider array.
A matrix engine still processes fixed output windows/reduction slices.

All compute is Torch on reconstructed BF16/FP32 values, not packed integer
hardware. No target energy, controller cost, SRAM traffic or CUDA-to-chip time
conversion was measured. Pixel fidelity is not FID, GenEval or semantic quality.
Folding and moving memoryless quantization before deterministic duplication
are baselines; their use here does not establish PhD novelty. Runtime adaptation
and a costed precision/phase/memory co-design remain separate research questions.
