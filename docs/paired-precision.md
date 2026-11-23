# Paired precision and reduction-boundary reference

`paired_precision_reference.py` compares two BF16 K128 primitives with a
single FP8-represented K256 primitive, keeping128 FP32 output columns. M rows
are streamed. This is a numerical experiment contract, not a claim that the
target already implements all proposed scale/control behavior.

One current power-of-two scale encodes each activation row/K256 slice in
E4M3FN. Prepared immutable weights use the same format per output column/K256
slice. The reference reconstructs these values into BF16 and uses the engine's
CUDA GEMM with FP32 outputs. It does not execute a packed FP8 hardware kernel
or establish the target's accumulation semantics. Fallback merges two K128
outputs sequentially in FP32; the paired path emits one K256 result. Every
primitive keeps the FP32 boundary.

The live selector uses activation maximum/reconstruction-error maximum and
prepared original/error weight-L1 maxima over128 output columns. Its
exact-arithmetic formula bounds the output error from replacing both operands;
the implementation evaluates statistics in FP32 and is not a bit-level error
certificate. The normalization uses an operand-product upper scale, not the
true output norm, so it is not a relative-output or task-quality guarantee.
Only scalar reductions/products/comparisons are used, without an additional
receiving matrix product. No activation calibration or fitted labels are used.

A configurable cohort of M rows shares one mode per128-column window. The
cohort is a control/buffering choice, not a physical M dimension. Adaptive
policies leave K<=128 tails in BF16 because there is no pair to combine.
Uniform FP8 controls can still encode those tails. `*-split` controls retain
the exact same K256-encoded operands but emit two K128 results, isolating the
reduction-boundary change. Uniform FP8 paired execution is the stronger
standalone baseline; a split ablation is not claimed to be an optimal baseline.

Statistics retain selected/useful work, output-vector counts, within-pair
cohort mode transitions, source-scale exponents and selector-bound histograms.
The reference may compute both candidates for mixed masks; its actual GEMM
call count is separate from selected logical interface actions. No GPU timing
is interpreted as controller, array or ASIC cost.

Hardware evaluation must charge activation encoding/error statistics, prepared
metadata, source buffering, weight-format copies, mode changes, output
power-of-two rescaling, register state and array fill/drain. FP32 vectors are
not automatically result-SRAM transactions. Counts assume a dense issue
schedule; neither zero gating nor a free mixed-format weight schedule is
provided. Internal accumulator precision/stationarity remain explicit target
questions. Final-model quality must be evaluated separately from local bounds.

## Modern model pilots and complete statistics

```sh
.venv/bin/python scripts/paired_precision_pilot.py --model dinov3-small \
    --samples-per-class 10 --output results/paired-dinov3
.venv/bin/python scripts/paired_precision_pilot.py --model convnextv2-tiny \
    --samples-per-class 10 --output results/paired-convnextv2
.venv/bin/python scripts/paired_precision_pilot.py --model sana-sprint \
    --output results/paired-sana
```

Install the optional modern-model environment. DINOv3 returns descriptors;
the pilot measures descriptor change and nearest-neighbor agreement, not
classification accuracy. ConvNeXt-V2 retains its original1000-class head.
SANA-Sprint uses paired prompt embeddings, initial seeds, two denoising steps
and1024×1024 decoded images. Denoiser linears/dense1×1 convolutions with K>128
are eligible; text encoder, VAE and other operators remain unchanged.

Each case retains complete per-layer mode/vector counters, activation range,
mean/RMS/zero statistics, deterministic-sample quantiles, scale/bound histograms,
weight statistics, exact checkpoint/source identities and outputs or trajectory
fingerprints. Raw numerical arrays/images stay under the output directory.
Original-model and fixed-K128 custom-engine comparisons are separate from
quantized policies. Paired image/feature similarity is a numerical diagnostic,
not FID, GenEval or a downstream-task guarantee. None of these software
executions is a target-hardware timing measurement.

`--minimum-fp8-rows 2` is an explicit shape-only ablation: a matrix with one
streamed M row retains the custom-engine BF16 K128 contract, even when the
selected policy would allow FP8. It does not use module names, calibration,
labels or an offline sensitivity table. Compare against the default minimum1
with identical inputs and all other settings. Small conditioning operations
can influence many later activations; low MAC count does not establish low
numerical importance. This ablation tests that possibility, not a universal
rule that one-row operations must use BF16. Its lost FP8 work and output-vector
counts remain in the same per-layer statistics.
