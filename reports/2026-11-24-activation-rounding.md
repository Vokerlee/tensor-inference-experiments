# Shared activation rounding: local error and complete-model behavior

The corrected rank1 rule reduces the receiving error of the first DINOv3 QKV
projection by about9.5× at INT4, yet does not improve the final descriptor on
the ten tested images. Rank4 improves descriptor error relative to nearest
INT4 QKV rounding, but remains far from uniform INT8 fidelity. These measurements
justify separate task evaluation; they do not establish an acceptable low-bit
model or target-device energy saving.

## Mechanism and matched comparisons

Current activations use group32 symmetric power-of-two grids. Immutable receiving
weights prepare a rank1/2/4 factor and diagonal surrogate. Adjacent floor/ceil
choices update a few error states; the completed group can retain nearest
rounding. Runtime selection never evaluates a receiving matrix multiplication.
The spectral-tail alternative is an FP64 diagnostic with a real-algebra screen,
not a finite-hardware certificate. Rank32 is a costly diagnostic comparator.

Each source group is shared across all receiving output columns. Isolated-group
surrogate costs omit cross-group terms, which remain present in complete measured
receiving errors. The attention study also tests separate Q/K/V precision and
role-normalized weight metrics. Every policy, including BF16, uses the same
three-way output-column split; the matrix MAC count is unchanged by that split.

## Evidence and independence

The [archive](statistics/2026-11-24-activation-rounding/README.md) preserves five
JSON records and631 attachments. Two operator campaigns each contain12 captured
operators,264 variants and312 array containers. Both reuse earlier operands.
The first campaign allowed an upward choice at an exactly representable level;
it is retained as a preliminary wider-feasible-set variant. Corrected floor/ceil
rounding forces exact grid points, including exact zeros, to remain unchanged.

Two model campaigns contain70 and110 actual forwards, respectively, on the
**same ten Imagenette images**, not twenty independent examples. These descriptor
measurements are separate from the frozen-feature CIFAR100 task. Full output
vectors, code/range statistics and geometry are preserved. The weight-only sketch
is preliminary and lacks complete historical environment metadata.

## Selected corrected results

For the captured first QKV projection, rank1 proxy receiving-SSE ratios to nearest
rounding are0.1052 at INT4 and0.1857 at INT8. Other operator shapes generally show
smaller or mixed changes. The spectral screen reports no worsening isolated group
in this campaign; it does not protect cross-group terms or complete model quality.

Mean final descriptor relative MSE on the ten matched images:

| Policy | Relative MSE |
|---|---:|
| Uniform INT8 linear activations |0.00034024|
| INT4 QKV, nearest; remaining linears INT8 |0.080346|
| INT4 QKV, raw rank1 |0.080774|
| INT4 QKV, raw rank4 |0.059930|
| INT4 QKV, role-balanced rank4 |0.058688|
| INT4 Q/K and INT8 V |0.061182|
| INT4 Q only, INT8 K/V |0.019487|
| INT4 K only, INT8 Q/V |0.014248|

The raw rank1 local improvement fails to transfer to this full-model descriptor
metric. Role balancing and higher rank help relative to the INT4 comparator,
without approaching uniform INT8 descriptor fidelity. A descriptor-distance gap
alone does not determine recognition accuracy: use the separate task comparison
before rejecting or accepting a policy for that task.

## Hardware and scope limits

INT4/8 values reconstruct BF16 operands here. They do not demonstrate native
FP8 K256 geometry, an INT4 K512 interface, target cycles or joules. The controller
needs a32-step decision dependency, coefficient residency, state bandwidth and
scale observation. Rank4 factor plus diagonal metadata occupy20 bytes per input
channel in FP32 before alignment or diagnostic metadata. Reading that anew for
every activation would be substantial; reuse is part of the design requirement.

A Q-only or K-only precision change covers one third of QKV work. It therefore
has a much smaller full-model opportunity than quantizing all large matrix
products. Do not equate the changed projection's error or bit count with an
end-to-end energy result. Scalar controller work must be charged alongside any
matrix and memory savings.

## Reproduce and audit

```sh
.venv/bin/python scripts/report_activation_rounding.py \
    --output reports/2026-11-24-activation-rounding
```

The generator verifies every indexed byte and independently recomputes SSE from
528 operator containers and180 full descriptors. It emits all operator variants,
all per-image model cases and policy means in [four CSV tables](2026-11-24-activation-rounding/),
with a source-identified JSON summary. It does not require the ignored original
results directory. Model/checkpoint and historical source identities remain in
the raw records.
