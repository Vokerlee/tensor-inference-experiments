# Source feedback in a modern image decoder — 2026-11-23

The pinned SANA-Sprint DC-AE decoder provides a modern high-compute spatial
convolution workload. Fixed kernel-derived feedback reduced INT4 pixel MSE
on every one of eight examined outputs, with both BF16 and INT8 weight cases.
This is a small, reproducible fidelity result. The causal terminal-moment
controller did not demonstrate a consistent improvement over fixed feedback.
Neither result establishes a new algorithm, general generation quality or
energy savings.

## Reproduction and evidence

[Runner and exact-input capture](../docs/sana-decoder-feedback.md);
[case CSV](2026-11-23-decoder-feedback.csv),
[per-image CSV](2026-11-23-decoder-feedback.images.csv),
[every quantized call](2026-11-23-decoder-feedback.calls.csv),
[complete source/checkpoint/input fingerprints](2026-11-23-decoder-feedback.provenance.json).
The [immutable complete JSON archive](statistics/2026-11-23-decoder/README.md)
contains both input manifests and all six campaign reports, including unfavorable
results, full exponent histograms and all recorded controller statistics.
Large image/latent arrays remain under ignored local paths with hashes/replay.

Four fixed-policy campaigns use source de8ea0588ba044571d784ed1621d5a3b4789d99b
(tag `snapshot/sana-decoder-fixed-feedback-2026-11-23`). Two live campaigns use
0312f49639fe4699d013bf54e5ce15d8edc3484f. All record clean experiment/engine
sources; engine642089cbeeb77f6a6719c201330b6eb70e6534ff. CUDA-source validation
passed178 engine Python tests and4 CTests; the later experiment suite passes136.
There are40 case records,160 per-input case records and2880 quantized calls.
These include repeated controls, not40 independent model/input datasets.

## Conditions and coverage

Checkpoint `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers`, revision
`aa76e7f4f4928f378716b6716a2130fba3caf5b1`; BF16,1024px,2 denoising steps,
guidance4.5. Four initial prompts/seeds20261123..20261126 and four additional
prompts/seeds20261223..20261226 are retained in the manifests. Additional inputs
were selected after the first results, without fitting coefficients/thresholds;
this is an exploratory replication, not a preregistered statistical holdout.

Capture the actual `vae.decode` argument after final denoising/scaling.
Scheduler callback latents are different. Original raw decoding was reproduced
bit for bit for every input before accepting modified comparisons. BF16 image
denormalization is reproduced before NumPy conversion. Unmodified, weights-only,
no-feedback and fixed/dynamic feedback controls use identical final latents.
The decoder has no recurrent effect on the denoising trajectory.

The shape rule selects24 dense spatial convolutions, input channels divisible
by32 and at least128 outputs, with zero padding. Per decode, eligible work is
3,672,499,027,968 MACs out of4,033,952,612,352 convolution MACs (91.0397%).
The separate complete decoder profile counts4,184,305,827,840 matrix/convolution
MACs; eligible work is87.77% of that denominator and approximately62.01% of the
cold two-step full request. These are distinct denominators and are NOT energy
proportions. Grouped, pointwise and small-output consumers remain unchanged.
Copies preserve other DAG consumers and the convolution input memory format.

## Results

PSNR uses aggregate unit-range pixel MSE over four equal-size images. MSE ratios
compare with no feedback at the SAME precision, weights and input set.
An infinite PSNR for identical controls is stored as null plus `identical=true`.

| Inputs | Weights | Mode | Pixel PSNR (dB) | MSE / no feedback |
|---|---|---|---:|---:|
| Initial four | bf16 | none4 | 28.96368 | 1.0000000 |
| Initial four | bf16 | coarse4 | 29.29401 | 0.9267606 |
| Initial four | bf16 | kernel4 | 29.34970 | 0.9149526 |
| Initial four | 8 | none4 | 28.96745 | 1.0000000 |
| Initial four | 8 | coarse4 | 29.28177 | 0.9301819 |
| Initial four | 8 | kernel4 | 29.34692 | 0.9163321 |
| Additional four | bf16 | none4 | 26.49999 | 1.0000000 |
| Additional four | bf16 | coarse4 | 26.69536 | 0.9560109 |
| Additional four | bf16 | kernel4 | 26.70748 | 0.9533454 |
| Additional four | 8 | none4 | 26.52114 | 1.0000000 |
| Additional four | 8 | coarse4 | 26.70352 | 0.9588751 |
| Additional four | 8 | kernel4 | 26.69269 | 0.9612693 |
| Initial four | 8 | none4 (live campaign) | 28.96745 | 1.0000000 |
| Initial four | 8 | coarse4 (live campaign) | 29.28177 | 0.9301819 |
| Initial four | 8 | live4 (live campaign) | 29.27536 | 0.9315549 |
| Additional four | 8 | none4 (live campaign) | 26.52114 | 1.0000000 |
| Additional four | 8 | coarse4 (live campaign) | 26.70352 | 0.9588751 |
| Additional four | 8 | live4 (live campaign) | 26.71309 | 0.9567644 |

Coarse feedback reduces INT4 MSE by7.32%/6.98% on the initial BF16/INT8-weight
runs and4.40%/4.11% on the additional runs. Continuous kernel feedback also
improves every examined INT4 image, but does not uniformly outperform coarse
feedback on the additional INT8-weight set. INT4 fidelity remains much lower
than INT8; this does not establish matched-quality4-bit execution.

INT8 no-feedback PSNR is48.79217/45.95208dB for initial/additional BF16 weights
and46.85097/44.17390dB with INT8 weights. Coarse feedback slightly worsens one
initial INT8-weight/INT8-activation image (MSE ratio1.00064) despite improving
its four-image aggregate. A favorable aggregate is not a per-input guarantee.

The causal variant chooses the next32-row block's half-step coefficient from
preceding terminal residual moments, smoothing1/4, with resets per image and
consumer. Compared to fixed coarse feedback, its MSE ratios are1.0014760 on
the initial set and0.9977987 on the additional set. Per-image signs are mixed.
It makes9674/9101 coefficient switches across1596 blocks per four-image run;
these are algorithm counters, not cycles or measured hardware cost. This
version provides no convincing fidelity advantage to justify its extra state
and decision arithmetic. Terminal row-end sampling, finite-sample bias and
omitted residual correlations remain possible mechanisms, not measured causes.

## Hardware interpretation and remaining limits

Quantization uses current original source groups32 and POT INT4/INT8 scales.
Fixed coefficients come from weights without activation calibration or retraining.
Coarse coefficients need sign/half/zero modes; live state and scale arithmetic
still need a real datapath design. Maximum eligible input channels1024 imply
4096 bytes of FP32 carry per independently active row stream, before statistic,
metadata, pipeline or buffer overhead. The software launches many concurrent
streams; this does not prescribe a hardware schedule.

The largest eligible source is[1,256,1024,1024]: BF16 payload512MiB, INT8
256MiB, INT4 128MiB before scale metadata. Even[1,128,1024,1024] at INT4 is
64MiB BEFORE metadata, residuals and weights; it is not evidence of fitting a
64MB memory. Spatial tiling and live consumer/weight state remain necessary.

All matrix/convolution arithmetic uses reconstructed BF16 operands in Torch;
there is no packed integer array deployment, energy result, SRAM transaction
measurement or CUDA-to-ASIC cost conversion. Native FP8 geometry does not
imply an INT4 datapath. Pixel fidelity is not FID, GenEval, semantic accuracy or
perceptual equivalence. DitherNN already studies spatial activation feedback
and hardware, while PassionSR quantizes VAE components with calibration.
The present measurements establish a building block and a negative controller
ablation, not a complete PhD novelty claim.
