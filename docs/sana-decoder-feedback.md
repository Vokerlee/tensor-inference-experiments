# Exact-input decoder feedback experiment

This pilot intervenes only in the pinned SANA-Sprint DC-AE decoder. Capture the
actual argument of `vae.decode`, after final denoising and latent scaling;
scheduler callback latents are not this argument. Save the unmodified raw output
and final image alongside the inputs. The replay must reproduce every original
raw output bit for bit before accepting quantized comparisons.

```sh
.venv/bin/python scripts/sana_decoder_feedback.py prepare \
    --output results/decoder-inputs-20261123
.venv/bin/python scripts/sana_decoder_feedback.py run \
    --inputs results/decoder-inputs-20261123 \
    --output results/decoder-feedback-20261123
```

The default four prompts, seeds20261123..20261126, two denoising steps, guidance4.5
and1024px resolution match the earlier pilot conditions. Use `--prompt` and
`--seed` to create independent additional inputs. Preparation executes the full
pipeline with unchanged weights. Frozen final inputs are a valid intervention
because decoding does not feed back into denoising.

Default eligibility is dense spatial convolution, input channels divisible
by32, at least128 output channels, zero padding. `--scope all` includes1x1
convolutions under the same channel rule. The rule is selected from shapes,
without activation calibration, labels or manually selected module names.
The report retains all actual convolution shapes/MACs and eligible coverage.

Cases compare unchanged BF16 with source INT8/INT4 reconstruction, no feedback,
continuous kernel-derived feedback and coarse feedback in{-0.5,0,0.5}. The
current original source group of32 channels determines each power-of-two scale;
carry runs horizontally and resets at every spatial row. Kernel coefficients
come from existing weights, without input calibration or retraining. Consumer
inputs are copied rather than mutated, preserving residual/other DAG consumers.
Convolution input memory format is preserved to avoid a layout-only comparison.

Optional `--weight-bits 8` or4 first adds a weights-only baseline and then holds
that weight representation fixed across activation cases. Weight groups are32
input channels at each kernel position, matching source-before-im2col grouping.
Weight preparation changes only eligible consumers. Static weight preparation
is distinct from the causal activation quantizer. A4/W8 is an experimental
format combination, not an assertion of native FP8/FP4 hardware support.

Each case preserves per-input and aggregate raw/pixel errors, image arrays and
PNG previews, every quantized call's shape, clipping/zero counters, complete
scale histogram, terminal carry statistics, all coefficient values and exact
source/checkpoint/native-library fingerprints. Invalid native events reject the
result before its convolution. Reports are written after each completed case;
an interrupted run remains explicitly incomplete. Large arrays remain ignored
locally, with hashes in the report.

Image postprocessing reproduces Diffusers' BF16 denormalization before NumPy
conversion. Pixel PSNR/RMSE describe fidelity to the original generated image;
they do not establish FID, GenEval, semantic or perceptual quality. The execution
uses Torch convolutions on reconstructed BF16 operands, not packed integer
matrix products. CUDA time and these numerical results do not determine target
hardware latency, SRAM transactions, energy or controller overhead. The kernel
feedback criterion assumes independent white residuals; actual network errors
may violate that approximation. No novelty claim follows from this pilot.
