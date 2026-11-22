# Common-exponent accumulation headroom — 2026-11-23

A conservative integer-range guard accepts a substantial part of the sampled
MLP work. This is a numerical-range screen; it does not yet execute changed
accumulation, measure model quality or establish energy savings or novelty.

Interface clarification: these historical full-K square-window results are
not an evaluation of the target FP8 geometry, which has K=256 and 128 FP32
outputs. See the [matrix interface](../docs/matrix-unit-interface.md). Retaining
integer state across primitive outputs would be an explicitly costed extension;
the original measured data and execution provenance below remain unchanged.

## Reproduction and scope

Use the [diagnostic instructions](../docs/accumulation-scale-diagnostics.md).
Both models ran unchanged on one complete ten-image diagnostic batch, one
reserved image/class, using official checkpoint preprocessing. All token rows
from every MLP edge were captured. Static weights are hypothetically encoded
as grouped INT8; activation candidates are INT8 and E2M1 with explicitly
specified power-of-two scales. Neither quantization nor a changed accumulator
was injected into the forward pass. The conditions below therefore apply to
these original-model inputs, not a modified inference trajectory.

[Execution provenance](2026-11-23-accumulation-provenance.json) retains the clean
executed revision, script/engine revision, image manifest and raw report hashes.
[Per-layer data](2026-11-23-accumulation-headroom.csv) preserves all cases.

## Guard coverage

Each output window must have sufficient width for every valid row. Percentage
weights accepted windows by useful M*N*K products across the 24 MLP edges of
each model. 128/256 are alternative square-window contracts, not assumed free
subarrays or measured native format support. No fixed-size matrix calls are
removed by passing the guard.

| Model | Activation codes | Accumulator bits |128-window coverage |256-window coverage |
|---|---|---:|---:|---:|
| ViT-B/16 | int8 | 24 | 0.00% | 0.00% |
| ViT-B/16 | int8 | 32 | 99.01% | 97.28% |
| ViT-B/16 | int8 | 48 | 100.00% | 100.00% |
| ViT-B/16 | e2m1 | 24 | 45.74% | 41.84% |
| ViT-B/16 | e2m1 | 32 | 100.00% | 100.00% |
| ViT-B/16 | e2m1 | 48 | 100.00% | 100.00% |
| Swin-T | int8 | 24 | 6.80% | 3.19% |
| Swin-T | int8 | 32 | 91.67% | 91.67% |
| Swin-T | int8 | 48 | 99.67% | 98.61% |
| Swin-T | e2m1 | 24 | 75.14% | 68.17% |
| Swin-T | e2m1 | 32 | 92.99% | 91.67% |
| Swin-T | e2m1 | 48 | 100.00% | 100.00% |

The guard uses group exponent ranges and triangle bounds from activation
maximum/L1 statistics and prepared weight statistics. It does not evaluate
a receiving matrix product. Every integer prefix is bounded; final-output
cancellation is not used as an overflow argument. Zero groups are ignored
when selecting the common exponent.

## Execution question exposed by this result

A successful complete-K mode could retain one integer domain across fixed
K tiles and perform floating conversion only at the end. An optimized
block-scaled baseline can already use integer intra-group accumulation;
compare against that baseline, including group-boundary alignment/merging.
The screened headroom is not a saving in multiplier products or matrix calls.

If an existing 32-bit partial-sum register can be reused, the integer mode need
not add another 32-bit state. If a design retains both the floating partial sum
and an integer chunk, it adds state instead. An output SRAM path also needs
an explicit word interpretation, anchor metadata and final conversion for
vector consumers.24-bit packing only saves actual SRAM actions if the memory
organization supports it. Wider fallback hardware remains present.

Full-K metadata must be available before issue; its preparation, buffering and
latency can matter even when the scalar guard is small. Alignment shifts,
integer adds, control, registers and fallback all need pricing against the
floating alignment/normalization they replace.

## Numerical and prior-work boundaries

Exact summation of represented operands can differ from a sequential FP32
reference: adding 2^24, then 1, then -2^24 in FP32 yields 0, whereas the exact sum
is 1. A new contract therefore needs fixed-input arithmetic checks followed by
cumulative trained-model/task evaluation. Range coverage alone cannot make
that choice for the application.

[MIAO](https://arxiv.org/abs/2611.19722v1) and
[Bucket Getter](https://doi.org/10.1145/3613424.3614249) are close hardware
comparators. Integer exponent grouping and narrow accumulation are established
primitives. The exact relationship to Bucket Getter remains pending full-text
review; these data do not establish a new hardware mechanism.
