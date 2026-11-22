# Grouped activation pilot

Use the pinned environment and engine submodule. No additional Python package
is required. The first download retrieves official Torchvision checkpoints and
Imagenette2-160 into the user cache, outside Git.

```sh
.venv/bin/python scripts/activation_pilot.py --model vit_b_16 --phase operator \
    --output results/vit-operator.json
.venv/bin/python scripts/activation_pilot.py --model swin_t --phase operator \
    --output results/swin-operator.json
```

`--phase geometry` only reads weights. Operator mode captures eight token rows
per image at every MLP linear input on four validation images per class. A
fixed shuffle reserves the first eight images/class for diagnostics; subsequent
task evaluations use a disjoint allocation. Reports contain checkpoint/source
SHA256 hashes, numerical settings, exact image paths and geometry statistics.

All comparisons keep represented BF16 weights and max-derived quantization
scales fixed. The engine reconstructs symmetric INT4/INT8 values with group
size 32, then the model-input BF16 cast is applied. `--bits 16` means unchanged
BF16 values before masking. Every sparse policy retains two slots per quartet;
retained zero values need not become nonzero. Compare magnitude, receiving-row
norm, quantization-aware diagonal gain, full local Gram score and gated score.
This is not a reproduction of any complete published method with correction,
rotation, calibration or a different numerical format.

Reported full-product errors include interactions between quartets. Sum of
group losses is also recorded, explicitly as a different quantity. Per-row
errors enable paired inspection. The scripts issue diagnostic FP32 products;
those products are never proposed as a deployed selector.

The input images originate at 160-pixel resolution and are transformed by each
checkpoint's official preprocessing. This pilot is neither full ImageNet nor
evidence for diffusion quality. It measures numerical effects, not native
INT4/INT8 or 2:4 execution, target-chip throughput, or target-chip energy.

## Whole-model checks

```sh
.venv/bin/python scripts/activation_pilot.py --model vit_b_16 --phase quality \
    --samples-per-class 100 --case baseline --case dense:4 \
    --case diagonal:16 --case gram:16 --case diagonal:4 --case gram:4 \
    --output results/vit-quality.json
```

`--scope down` limits modifications to each MLP's second linear operation;
the default modifies both MLP linear inputs. The classifier remains the
checkpoint's full 1000-class classifier. Primary accuracy uses that prediction
and the original ImageNet class indices. Restricted ten-class accuracy is a
secondary diagnostic and must not be presented as ImageNet accuracy.

Cases use the same image manifest. Reports retain predictions and correctness
per image for paired comparisons. A small screening subset does not establish
statistical significance or a general quality advantage. No calibration,
fine-tuning or threshold selection from evaluation labels occurs.

## Runtime mode policies

`--selector-backend cuda` uses the engine's native mask evaluator; other
reference policies remain explicit Torch operations. `--holdout-offset 20`
skips the screening allocation before selecting the next100 images/class.

Cases `precision-diagonal:0.01`, `adaptive-diagonal:0.01` and
`sparsity-diagonal:0.01` select precision, both precision/masks, or BF16 masks.
Replace `diagonal` with `gram` for local signed interactions. The number is a
global relative group-error tolerance, never a task-quality guarantee. Format
decisions cover128 rows by128 input channels by default. `--mode-rows 32`
measures a different format-granularity requirement; it does not create free
subarrays on an existing128-row array.

The finite mode ordering uses nominal representation/work ranks, not measured
hardware joules. Weights remain BF16, so cheaper integer MAC support is a
separate architectural assumption.32-element scales require block-scaled
arithmetic or explicit scaled partial products within a128-element reduction.
The reference does not price either. Mode counts and dense-equivalent MACs
are recorded separately; weighting every layer's input tile equally is wrong.

`common-diagonal:16` and `common-gram:16` use one BF16 mask for each128 token
rows across the input channels. Their `common-adaptive-*:0.01` variants retain
the original input window when the global proxy fails, and disable compaction
when fixed128-wide reductions would save no call. The selected weight rows can
be gathered into a dense product; tests verify that product's numerical
equivalence. This is distinct from splitting smaller row cohorts. Model runs
inject zeros; counters describe prospective full128x128x128 matrix calls after
compaction, not executed GPU sparse instructions. Gathering, buffering and
physical scheduling remain additional costs.

Hooks transform only the receiving MLP edge and preserve the original tensor
for other consumers. A deployed dataflow needs equivalent tensor/version
semantics and must charge any additional storage.

## Alternative four-bit baselines

Fixed cases `affine:4` and `fp4:4` compare integer-zero-point affine INT4 and
E2M1 FP4. The latter uses FP32 scales and is not a complete NVFP4 recipe.
`--low-format affine4` / `--low-format e2m1` changes the low-format candidates
of runtime policies; the higher low-precision candidate stays symmetric INT8.
Use the recorded mode names when comparing counts across different menus.
Native fixed symmetric CUDA cases use the fused engine preparation kernel;
affine/FP4 and multi-mode decisions use numerical references. Their GPU
latencies are not comparable without accounting for these implementations.

## Preparation and stream diagnostics

```sh
.venv/bin/python scripts/benchmark_selection.py --output results/preparation-timing.json
.venv/bin/python scripts/activation_stream_diagnostics.py --model vit_b_16 \
    --output results/vit-streams.json
```

Run timing without concurrent GPU jobs. CUDA graph trials compare selectors,
unfused/fused current-group preparation and ordinary BF16 GEMM. Reconstructed
outputs, metadata availability and supported inputs are explicit. Multi-mode
evaluation, sparse packing and target-chip control are outside that benchmark.

Stream diagnostics capture complete token rows from ten reserved images. A
real-arithmetic sufficient condition compares the best two diagonal-mask losses
using a normalized Gram off-diagonal bound. The script checks certified choices
against the full local objective with stated FP32 tolerances; this is not a
floating-point proof or a whole-model certificate.

Scale-bin prediction is evaluated in two declared orders: previous32-value
block in row-major order and the previous token row's corresponding channels.
It reports ideal saved bytes against a hypothetical separate raw-data pass,
and explicitly zero such saving against a producer-buffered current-bin path.
Hit rate alone does not establish an efficient predictor or replay mechanism.

## Native precision-policy backend

`--selector-backend cuda-tile` runs precision-only INT4/E2M1 versus INT8 with
the engine's buffered128x128 CUDA controller. It requires group32,128 mode
rows and the selected engine revision. Other policy families still use their
reference orchestration and native mask evaluators. The adapter records mode
and MAC counters separately from the kernel. Its model output remains a BF16
reconstruction. The benchmark distinguishes a bare native controller from the
same policy including counter collection; setup metadata regeneration has its
own timing and is not assumed to occur on every inference.

## Separating error and signal geometry

Precision cases also accept `precision-gram-numerator:0.01` and
`precision-gram-denominator:0.01`. The former uses full quartet geometry only
for the four-bit error, with diagonal INT8 error and diagonal reference signal;
the latter uses full geometry only for reference signal energy. These are
separate policies, not algebraic rearrangements of the full score. Each uses
18 quadratic terms/quartet rather than30; preparation, encoding, reductions and
metadata still cost work. CUDA tile and numerical reference paths are checked
against one another for all four geometry choices.

Stream reports additionally test sufficient interval decisions for precision
at three fixed tolerances. Certified-choice agreement is checked against the
explicit full local score; low certification rates do not imply a cheap path.
No intervals here certify final network quality.

## Targeted matrix-work inventory

```sh
.venv/bin/python scripts/profile_matrix_work.py --model vit_b_16 --output results/vit-work.json
```

Use `swin_t` or `dit` for the other supported models; DiT needs the optional
pinned diffusion environment. This checks how much matrix/convolution work
belongs to the selected MLP edges. It forces math attention and disables MHA
fusion for the inventory only. Counted operators are mm/addmm/bmm/conv2d;
other profiler FLOP categories are listed separately. One synthetic input
provides shapes, not quality, bit activity, padded physical operations, a
complete FLOP count, or an energy fraction. See the separate
[systolic action ledger](systolic-actions.md) for execution-contract limits.
