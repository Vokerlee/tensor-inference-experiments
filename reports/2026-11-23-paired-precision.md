# Paired precision on modern models — 2026-11-23

The max/L1 runtime selector is too conservative in these pilots: at tolerance
0.04 it selects FP8 for at most0.0013% of eligible products. Even that small
fraction can change a diffusion trajectory. Preserving one-row operations in
BF16 substantially reduced that numerical drift in a paired SANA ablation.
Neither result establishes an energy-saving algorithm or a novelty claim.

## Arithmetic and conditions

The [reference contract](../docs/paired-precision.md) compares BF16 K128
primitives with FP8-represented K256 primitives, with N128 and FP32 outputs.
One current power-of-two scale per activation row/K256 slice and per prepared
weight column/K256 slice encodes E4M3FN. Reconstructed BF16 operands execute
through the custom engine's CUDA GEMM with FP32 output. This is not packed FP8
execution or a model of the target's internal FP8 accumulation semantics.

The live criterion combines source maximum/error-maximum with prepared
weight-L1/error-L1 summaries over128 output columns. Its exact-arithmetic bound
is normalized by an operand upper scale, not the true receiving-output norm.
The FP32 implementation is not a bit-level certificate or a task-quality
bound. M cohorts of32 rows share a mode; this is a scheduling choice, not a
physical M dimension. No additional receiving GEMM, activation calibration,
retraining or label-fitted threshold selects the mode. Tolerances0.02/0.04/0.06
were declared before the full runs.

Eligible operators are linears or dense1×1 convolutions with K>128. Adaptive
policies keep K<=128 tails in BF16. Uniform FP8 controls can encode those tails.
DINOv3 has48 eligible modules, ConvNeXt-V2 has34, and SANA has288. All eligible
calls in the recorded runs used the modified path; no dtype calls were skipped.
Spatial/depthwise convolutions, nonlinear/vector operations and other operators
remain unchanged.

DINOv3-S and ConvNeXt-V2 Tiny use100 paired Imagenette validation images,
ten/class, allocation offset20, batch10 and official checkpoint preprocessing.
The original1000-class ConvNeXt head is retained. DINO returns384-dimensional
global descriptors with no classifier. Neighbor agreement concerns nearest
neighbors within this100-image set, not an external retrieval benchmark.
These exploratory inputs overlap earlier pilots and are not an independent
confirmatory holdout or full ImageNet evaluation.

SANA-Sprint0.6B uses four recorded prompts/seeds,1024×1024 images, two denoising
steps, guidance4.5 and unchanged prompt embeddings. Only its denoiser's eligible
operations change; text encoding and VAE remain unchanged. Numerical errors are
measured through the changed trajectory and decoded image. PSNR and paired
similarity are not FID, GenEval or evidence of perceptual-quality loss/gain.
Pinned checkpoints and architecture profiles are in the
[model cards](../models/README.md).

## Precision selection and outputs

FP8 product share divides selected useful FP8 scalar products by all eligible
useful products. Vector reduction compares selected dense logical FP32 output
vectors with the two-K128 reference schedule, including tails. These are not
physical SRAM writes, energy savings, latency or PE-activity measurements.
The numerical oracle may compute both precision candidates before masking a
mixed selection. Its actual GEMM-call count is recorded separately from the
selected logical schedule.

| Model | Policy | FP8 products,% | Output-vector reduction,% | Output diagnostic vs original |
|---|---|---:|---:|---|
| DINOv3-S | bf16 | 0.000000 | 0.000000 | descriptor relative MSE 2.66395e-05; neighbor agreement 0.99 |
| DINOv3-S | fp8-paired | 100.000000 | 38.888889 | descriptor relative MSE 0.00644937; neighbor agreement 0.91 |
| DINOv3-S | adaptive-paired:0.04 | 0.001135 | 0.000568 | descriptor relative MSE 2.70116e-05; neighbor agreement 0.99 |
| DINOv3-S | adaptive-paired:0.06 | 0.716771 | 0.358385 | descriptor relative MSE 0.000143084; neighbor agreement 1.00 |
| ConvNeXt-V2 Tiny | bf16 | 0.000000 | 0.000000 | 80/100 correct; 0 changed predictions |
| ConvNeXt-V2 Tiny | fp8-paired | 100.000000 | 43.982619 | 81/100 correct; 1 changed predictions |
| ConvNeXt-V2 Tiny | adaptive-paired:0.04 | 0.000000 | 0.000000 | 80/100 correct; 0 changed predictions |
| ConvNeXt-V2 Tiny | adaptive-paired:0.06 | 0.082630 | 0.040045 | 80/100 correct; 0 changed predictions |

Original ConvNeXt-V2 has80/100 correct; uniform FP8 gives81/100 with one changed
prediction. This single change does not demonstrate an accuracy improvement.
The DINO BF16 split baseline already changes descriptors relative to Torch
(relative MSE2.66e-5, neighbor agreement0.99), so quantization effects must also
be compared with that baseline. The0.06 selector's neighbor agreement1.00 does
not establish better general descriptors.

| SANA policy | FP8 products,% | Output-vector reduction,% | Trajectory RMSE vs BF16 split |
|---|---:|---:|---:|
| bf16 | 0.000000 | 0.000000 | 0.00000000 |
| fp8-paired | 100.000000 | 45.062222 | 0.14567594 |
| fp8-split | 100.000000 | 0.000000 | 0.14575844 |
| adaptive-paired:0.02 | 0.000000 | 0.000000 | 0.00000000 |
| adaptive-paired:0.04 | 0.001281 | 0.000638 | 0.05177570 |
| adaptive-paired:0.06 | 0.762640 | 0.380402 | 0.14678938 |

At0.02 the SANA selector accepts no FP8 work and reproduces the BF16 split
output/trajectory. At0.04, only three receiving modules accept any FP8 work,
including `time_embed.timestep_embedder.linear_2`. A local operand bound does
not measure how an error in conditioning later affects the graph. A tiny MAC
share therefore cannot be assumed to have a tiny effect on the final output.

At identical current inputs, `fp8-split` uses the same K256-encoded operands as
`fp8-paired` but emits two K128 results. It is a reduction-boundary ablation,
not a strong optimized baseline. Complete-model inputs can subsequently diverge
as earlier rounding differences propagate. The two policies produce35,642 different DINO descriptor
scalars (RMSE0.008288) and84,606 different ConvNeXt logits (RMSE0.026153).
Their SANA decoded-image MSE against each other is0.005362. Thus regrouping is
not numerically neutral; the experiment also includes its downstream effects. The adaptive
0.04 paired/split SANA runs happened to match; this is not a general identity.

## Single-row ablation

An additional clean-source run uses `--minimum-fp8-rows 2`: one-row operations
keep the custom BF16 K128 contract. The condition uses current matrix shape,
without module-name rules or sensitivity calibration. Other settings, four
prompts, seeds and checkpoint identities match. Original and BF16 control
images and every retained denoising state reproduce exactly across the runs.

| SANA adaptive0.04 | Default minimum1 | Minimum2 |
|---|---:|---:|
| FP8 product share,% | 0.001281 | 0.001464 |
| Output-vector reduction,% | 0.000638 | 0.000729 |
| Trajectory RMSE vs BF16 split | 0.05177570 | 0.00181198 |
| Mean decoded-image MSE vs BF16 split | 0.00296590 | 0.00001076 |

The shape-only intervention reduces trajectory RMSE by28.57× and mean image
MSE by275.69× relative to the split control on these four requests. It changes
later activations and their mode decisions as well: the minimum2 run selects
FP8 in two attention-output projections instead of the default's three
receiving modules. It is not a fixed-input single-layer injection experiment,
nor proof that one named layer accounts for every downstream difference.
Preserving conditioning arithmetic is not asserted to be a new principle.

The result identifies an important limitation of this selector: local range/
error summaries alone do not express downstream influence. It still selects
a negligible fraction of useful work, so this is a diagnostic improvement,
not a demonstrated practical mixed-precision saving. Dense uniform FP8 is the
stronger work-count baseline, and requires separate numerical/task validation.

## Complete evidence and replay

[Case CSV](2026-11-23-paired-precision.csv),
[layer/case CSV](2026-11-23-paired-precision.layers.csv), and
[paired-control validation](2026-11-23-paired-precision.validation.json)
retain27 cases and3,824 layer/case records. CSV rows include full-record hashes,
executed revisions and row/cohort conditions. Complete per-call ranges,
moments, sampled quantiles, zero counts, mode/scale/bound histograms and prepared
weight statistics remain in the full JSON records linked through the
[117-record archive](statistics/2026-11-23/README.md).

Original full runs execute experiments
`496b5dce0965ba25b82d2853fc888ca3ee2f4b7f`, immutable tag
`snapshot/paired-precision-2026-11-23`. The single-row ablation executes
`aebc4fe99123c4301bf14a6c30149433a047573c`, tag
`snapshot/paired-single-row-2026-11-23`. Both record clean source trees, engine
`84f9158c262af68eff09060edbe4b8f74f771952`, source/library hashes, input manifests
and exact checkpoint revisions. Source hashes were checked against Git blobs
and the recorded engine gitlinks. Earlier smoke runs are also archived with
their original dirty flags; they are not the evidence for these tables.

Install the pinned optional modern environment and build the native engine as
in the [baseline instructions](../docs/benchmark-gemm.md). From a separate
checkout at the original full-run tag, run:

```sh
.venv/bin/python scripts/paired_precision_pilot.py --model dinov3-small \
    --samples-per-class 10 --output results/paired-dinov3
.venv/bin/python scripts/paired_precision_pilot.py --model convnextv2-tiny \
    --samples-per-class 10 --output results/paired-convnextv2
.venv/bin/python scripts/paired_precision_pilot.py --model sana-sprint \
    --output results/paired-sana
```

At the separate single-row tag, run:

```sh
.venv/bin/python scripts/paired_precision_pilot.py --model sana-sprint \
    --minimum-fp8-rows 2 --case bf16 --case adaptive-paired:0.04 \
    --output results/paired-sana-single-row
```

Hardware evaluation remains separate: charge candidate encoding/error
statistics, source buffering, prepared metadata, weight-format variants,
mode changes, power-of-two output rescaling, array fill/drain and the actual
accumulator/output schedule. Logical FP32 vectors are not automatically SRAM
transactions. No reported CUDA runtime estimates these costs or device energy.
