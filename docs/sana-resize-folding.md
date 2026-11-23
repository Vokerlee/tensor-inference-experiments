# Resize-convolution arithmetic and source placement

This experiment implements the established nearest-neighbor coefficient
folding of [Colbert et al., Section4.2](https://arxiv.org/html/2107.07647v1).
For2x nearest-neighbor expansion followed by dense3x3 convolution, aggregate
weights acting on identical source samples. Four output phases each use a2x2
low-resolution kernel. The ideal MAC ratio is4/9; prepared weights grow16/9.
Our reference computes an extra border row/column and counts that additional
work rather than claiming the ideal ratio as its executed arithmetic.

```sh
.venv/bin/python scripts/sana_resize_pilot.py \
    --inputs results/decoder-inputs-20261123 \
    --output results/sana-resize-20261123
```

Use the exact-input bundles from the [decoder pilot](sana-decoder-feedback.md).
The original branch replay must be bit-exact. All selected upsampling blocks
have2x nearest interpolation and dense3x3 stride1/padding1 convolution, with
channels aligned32. Preserve their residual shortcuts and all other operators.
The code constructs low-resolution phase planes and writes the corresponding
high-resolution output positions; it does not allocate an interpolated source
for folded cases. This software materialization is not a target hardware design.

Cases isolate FP32 folded weights/arithmetic, BF16 rounded folded weights,
INT8 folded weights, and current-source group32 INT8/INT4 activation conversion.
Compare source conversion before duplication with conversion after duplication
at unchanged original weights. Without feedback these conversions commute
because each source-pixel channel group has the same values/scale in every
replica. Tests verify identical output and fourfold fewer quantizer groups.
This identity does not extend automatically to spatial feedback.

Folded-feedback cases use coarse coefficients from the effective phase bank,
without activation calibration or retraining. Other decoder convolutions are
unchanged. Original, folded-BF16 and folded-INT8 controls separate errors from
weight aggregation, coefficient rounding and activation conversion. No claim
of general bit equivalence follows from real-arithmetic folding.

Reports preserve full branch shape/MAC/weight/source counts, quantizer
histograms/counters, coefficients, per-input raw/image errors, arrays and
source/checkpoint/library fingerprints. MACs are dense arithmetic, not array
call counts, speed or energy. The output phases still require fixed-width
matrix tiles and memory routing. Kernel growth, borders, precision, weight
residency and output ordering must be priced on the target hardware.
