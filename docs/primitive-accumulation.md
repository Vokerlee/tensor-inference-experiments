# Primitive accumulation numerical reference

`scripts/primitive_accumulation_reference.py` evaluates two arithmetic contracts
using identical group32 integer codes and power-of-two scales. It is a research
oracle, with Python control and Torch products; its latency does not estimate
a deployed hardware controller.

The group-FP32 baseline computes each group dot product exactly, scales that
sum and merges groups in FP32. Code magnitudes at most 127 keep every prefix
of a 32-element integer dot product below 2^24, so FP32 represents it exactly.
The guarded path uses activation maximum/L1 values and prepared weight extrema
to bound every integer prefix across the current K slice. If one signed width
suffices for all 128 output columns, it aligns group sums in a common integer
domain and converts at the primitive output. Otherwise it uses the baseline.
No receiving GEMM is used to make the guard decision.

The exact-integer oracle uses FP64 to represent bounded integers of at most
48 bits. This is a testing device, not a proposal to deploy FP64. Both paths
emit FP32 after **every** K slice and merge emitted slices in FP32. A test
distinguishes this behavior from preserving an exact integer sum over all K.

K=128 and K=256 are explicitly selectable design contracts with the
[fixed vector interface](matrix-unit-interface.md). INT8/E2M1 code support in
those geometries is a proposed datapath option; it does not follow from the
existing BF16/FP8 input modes. Required encoding, alignment, buffering,
integer state, control and fallback hardware remain costs to characterize.

Counters separate accepted vector groups, useful products and baseline
floating group merges replaced. Products and fixed-size calls are retained;
fewer floating merges is not an energy percentage. Numerical output changes
are measured before any model-output BF16 cast, using the same represented
inputs for the local comparison. Whole-model quality must be measured
separately, following the same separation of local and propagated errors.

Integer alignment/deferred conversion already occur in
[Bucket Getter](https://doi.org/10.1145/3613424.3614249) and intermediate precision
codesign in [MIAO](https://arxiv.org/abs/2611.19722v1). Passing this oracle is
not evidence of novelty or a hardware energy advantage over those designs.

## Paired classifier execution

```sh
.venv/bin/python scripts/primitive_accumulation_pilot.py --model vit_b_16 \
    --k-slice 128 --width 24 --width 32 --samples-per-class 10 \
    --output results/vit-primitive-accumulation.json
```

`--model swin_t` selects the other cached classifier. Use the existing pinned
base environment. Both candidates use fixed INT8 weight codes and current-input
INT8 activation codes; `--activation-format e2m1` changes only the latter.
Source scales are explicit powers of two, not a complete OCP MX/NVFP4 recipe.

The pilot first executes the unmodified BF16 model, then the group-FP32
quantized reference, then each guarded variant on identical images. It replaces
receiving MLP linears only, retains their bias and casts their final output back
to the model's BF16 type. Other operators and graph consumers are preserved.
The report records checkpoint/source hashes, exact images, local primitive
differences and propagated final-logit/prediction differences separately.

The fixed holdout offset separates these images from earlier screening inputs.
Neither labels nor agreement with the reference choose scales or accumulator
widths. Ten images/class is a small diagnostic subset; it cannot establish a
general accuracy guarantee or a diffusion generation-quality result.

## Reused source statistics

The guard accepts cached maximum/L1 statistics for the exact encoded source
groups. The caller must preserve their association with codes through lowering.
The default `min` bound uses the smaller of Amax times weight L1 and activation
L1 times weight maximum. The `max` alternative uses only the first expression,
so it needs no activation L1 reduction or L1 metadata. It may reject more vectors.
Both are sufficient bounds and neither uses a receiving product as a predictor.

Metadata movement, register reads and decision timing remain costs even when
statistics are reused. A source maximum can be exposed by the quantizer; this
Python oracle's explicit reductions do not establish free hardware collection.

## Exporting paired measurements

`scripts/summarize_primitive_pilots.py` accepts complete `--input` JSON reports,
optional `--geometry` reports and an `--output-prefix`. It writes case/layer
CSVs, a geometry CSV and provenance JSON with source revisions, image manifests,
checkpoint fingerprints, settings and paired predictions. For U-Net geometry,
pass its recorded repository/revision/file JSON as `--unet-source`.

The exporter verifies source hashes against the executed Git objects and checks
the engine gitlink. It rejects dirty arithmetic executions and incomplete runs.
Raw action counts and zero-output-gated comparisons remain separate, and older
runs without the latter have empty fields rather than inferred measurements.

`scripts/reproduce_primitive_pilots.py --provenance RECORD --output DIRECTORY`
replays arithmetic runs in detached worktrees at their recorded revisions,
initializes the exact engine submodules and uses the current pinned environment.
`--dry-run` lists commands; `--run NAME` selects a recorded case group. It checks
source/data/checkpoint identity and reports prediction differences separately.
Worktrees and replay logs remain in the requested output directory; the active
development checkout is preserved. Geometry is regenerated with the recorded
script revision and, for U-Net, the recorded external configuration.
