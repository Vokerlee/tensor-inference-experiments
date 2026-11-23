# Source activation feedback — 2026-11-23

Receiving-kernel-derived spatial feedback reduced local convolution output
error on the tested inputs, while increasing activation reconstruction error.
Complete-model results were mixed. The method is retained as an exploratory
component, not a validated low-bit inference solution or a novelty claim.

## Method and scope

The [reference and pilot](../docs/spatial-feedback.md) use online symmetric
INT4/INT8 reconstruction with power-of-two scales per source spatial
coordinate and32 input channels. Carry propagates horizontally within each
row/channel and resets at every row boundary. A current source value plus
scaled previous residual is quantized; the new residual is their difference.
The current original group maximum chooses the scale. No future activation,
reference output, label or activation calibration chooses the representation.

The receiving kernel supplies a deterministic coefficient: horizontal lag-one
autocorrelation divided by kernel energy, per input channel, clipped
to[-0.5,0.5]. The `kernel-coarse` variant rounds to{-0.5,0,0.5}; the `-zero`
variant additionally forces exact source zeros to remain zero and clears their
carry. The criterion assumes white independent equal-variance residuals as an
approximation; actual residuals need not satisfy it. No model is retrained.
Static weight statistics are not fitted to the validation images.

Controls are ordinary no-feedback quantization, fixed coefficient1 (`unit`),
fixed coefficient0.5 (`half`), continuous kernel coefficients, coarse kernel
coefficients, and coarse coefficients with zero preservation. The `unit`
control is not a reproduction of a complete trained DitherNN system.

Eligible dense, zero-padded convolutions with channels divisible by32 are
transformed:19 in ResNet18 and52 in ResNet50. Each model has16 eligible spatial
3×3 convolutions; other eligible convolutions are1×1. Kernel-derived feedback
is zero for1×1 kernels, so those still receive ordinary quantization. The
unaligned first convolution is unchanged. Weights remain BF16 and all
receiving products are Torch convolutions on reconstructed BF16 activations.
This does not evaluate integer-weight arithmetic or a packed low-bit array.

Local diagnostics quantize identical original-trajectory inputs and compare
each receiving convolution output with its unmodified output. Separately,
each policy executes through the complete modified classifier graph. A
consumer-specific source representation is used; sharing one representation
across different DAG consumers has not been implemented.

## Local error

Ratios below divide the sum of squared errors for a policy by the sum for
no-feedback quantization over the same16 spatial convolutions and same inputs.
They are not means of per-layer ratios, full-network losses or energy ratios.

| Model / activation bits / images | Policy | Activation error ratio | Receiving-output error ratio |
|---|---|---:|---:|
| ResNet18 /4 /100 | unit | 1.994 | 1.318 |
| ResNet18 /4 /100 | half | 1.255 | 0.907 |
| ResNet18 /4 /100 | kernel | 1.093 | 0.860 |
| ResNet18 /4 /100 | kernel-coarse | 1.144 | 0.885 |
| ResNet18 /4 /100 | kernel-coarse-zero | 1.142 | 0.884 |
| ResNet50 /4 /10 | kernel | 1.072 | 0.884 |
| ResNet50 /4 /10 | kernel-coarse | 1.092 | 0.889 |
| ResNet50 /8 /10 | kernel | 1.033 | 0.887 |
| ResNet50 /8 /10 | kernel-coarse | 1.046 | 0.914 |

Both kernel-derived versions reduced receiving-output squared error in all16
spatial layers in each of these three runs. This is consistent with changing
the spatial correlation of quantization errors to better match a receiving
operator. It does not establish white residuals or validate that approximation
as a general model. Different output errors can propagate differently through
the remainder of a network.

The spatial-only denominator matters. Including unchanged-feedback1×1
convolutions, the continuous kernel version's aggregate output-error ratio is
0.867 for ResNet18/4,0.995 for ResNet50/4 and0.993 for ResNet50/8. Its advantage
on the latter models is concentrated in a subset of their convolutions.
These sums do not estimate each subset's share of inference energy.

Unconstrained feedback can create nonzero values where the source was zero.
In the ResNet18 spatial diagnostics, coarse feedback introduced103991 such
values; the zero-preserving version introduced none and had a similar local
output-error ratio. Clipping and zero-change counters for every policy/layer
are retained in the exported data. Changing feedback around zeros also changes
its noise-shaping behavior; zero preservation is not assumed free numerically.

## Complete-model results

Images are paired Imagenette validation subsets, with the original1000-class
classifier outputs and official checkpoint preprocessing. ResNet18 uses100
images, ten/class; ResNet50 uses10, one/class, with the same allocation offset20.
The subsets overlap previous exploratory pilots and are not an independent
confirmatory holdout or full ImageNet accuracy measurement.

| Policy | ResNet18 INT4, correct/100 | ResNet18 logit RMSE vs original | ResNet50 INT4, correct/10 | ResNet50 INT8, correct/10 |
|---|---:|---:|---:|---:|
| Original BF16 | 57 | 0 | 7 | 7 |
| No feedback | 58 | 0.9901 | 0 | 7 |
| Unit | 51 | 1.4782 | 0 | 6 |
| Half | 54 | 1.1822 | 0 | 6 |
| Kernel | 60 | 1.0054 | 2 | 7 |
| Kernel-coarse | 54 | 0.9859 | 1 | 7 |
| Kernel-coarse-zero | 58 | 0.9873 | 2 | 7 |

A few extra correct classifications do not establish a reliable improvement.
The continuous kernel version slightly worsened ResNet18 logit RMSE despite
reducing every spatial layer's local output error. ResNet50's INT4 reference
failed badly, and feedback did not restore usable quality. INT8 retained the
number of correct predictions for the kernel-derived variants on ten images,
but predictions/logits need not match the original. There is no demonstrated
bit-width reduction at matched task quality here.

No diffusion network or generation-quality metric was evaluated. Adding
adaptive high-precision fallback would be a separate algorithm requiring new
measurements, not an inferred benefit of these results.

## Hardware interpretation and prior work

No CUDA timing is used to assess hardware feasibility. A source feedback
implementation needs a declared carry schedule, coefficient delivery, scale
availability, clipping/zero handling and buffering. The coarse coefficient can
use sign/power-of-two operations; updating feedback still needs additions and
residual computation. Carry creates dependencies between consecutive pixels
of a channel. Interleaving independent streams can alter throughput and state
requirements and must be specified, not assumed.

For one active row stream, FP32 carry across C input channels uses4C bytes,
where C denotes the input-channel count. Multiple concurrent streams multiply
that state. The reference stores a complete residual tensor for diagnostics;
this is not the required live-state size. Quantizing before im2col permits
reuse of the resulting codes inside one convolution, but does not remove
statistics/control/metadata traffic or define an SRAM/banking implementation.

[DitherNN](https://doi.org/10.1587/transinf.2019PAP0009) already includes spatial
activation error feedback and hardware co-design.
[Minami et al.](https://doi.org/10.9746/sicetr.56.425) study noise-shaping filter
design for weight quantization; that comparison remains abstract-only here.
[PREFILT](https://openaccess.thecvf.com/content/ICCV2025W/BiVision/papers/Makhov_PREFILT_Prefiltering_for_Fully_Quantized_Image_Restoration_Neural_Networks_ICCVW_2025_paper.pdf)
studies preprocessing, reconstruction and quantization-aware training for image
restoration. This pilot differs in its coefficient information and execution
contract; those distinctions do not prove novelty of the specific variant.

## Reproduction and data

The [case CSV](2026-11-23-spatial-feedback.csv) contains21 model/policy cases;
the [layer CSV](2026-11-23-spatial-feedback.layers.csv) contains738 paired
layer/policy records. [Provenance](2026-11-23-spatial-feedback.provenance.json)
retains exact clean executed revisions/hashes, engine gitlinks, checkpoint
fingerprints, image manifests, paired predictions and raw report hashes.

All three runs executed experiment commit
`04e64341361974bf8550022b7a07d5a423c3e8b2`, preserved by the pushed tag
`snapshot/spatial-feedback-2026-11-23`. The later reference rejects scale
underflow and corrects normalization for extremely small weight energies;
the measurements retain their original execution version.

To replay the original source without altering the current development branch:

```sh
git worktree add --detach results/spatial-feedback-source snapshot/spatial-feedback-2026-11-23
git -C results/spatial-feedback-source submodule update --init --recursive
.venv/bin/python results/spatial-feedback-source/scripts/spatial_feedback_pilot.py \
    --model resnet18 --bits 4 --samples-per-class 10 --output results/replay-r18.json
.venv/bin/python results/spatial-feedback-source/scripts/spatial_feedback_pilot.py \
    --model resnet50 --bits 4 --samples-per-class 1 --output results/replay-r50-int4.json
.venv/bin/python results/spatial-feedback-source/scripts/spatial_feedback_pilot.py \
    --model resnet50 --bits 8 --samples-per-class 1 --output results/replay-r50-int8.json
```

Use the pinned Python/Torch environment. These numerical references do not
need a compiled native engine library. Raw outputs stay in ignored `results/`.

A fresh detached-source replay of the ResNet50 INT8 run reproduced all seven
prediction lists, logit comparison metrics and paired local-error records.
This checks source/data restoration and execution, not independent statistical
replication. The current experiment suite passed92 tests.
