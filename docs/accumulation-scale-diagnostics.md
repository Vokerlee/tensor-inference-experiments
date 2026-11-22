# Common-exponent integer accumulation headroom

The historical square windows below are policy cohorts, not the native FP8
array shape. The [matrix interface](matrix-unit-interface.md) has a K=256 FP8
input and 128 FP32 outputs. Full-K coverage does not show that integer state
can persist across emitted FP32 vectors; that requires a separate ISA contract.

This diagnostic measures a sufficient bound on real pretrained MLP inputs. It
is not a new-arithmetic claim, optimized kernel, hardware simulation or energy
measurement. Run it with the existing pinned base environment:

```sh
.venv/bin/python scripts/accumulation_scale_diagnostics.py --model vit_b_16 \
    --output results/vit-accumulation-scales.json
```

`--model swin_t` selects the other classifier. One reserved image/class forms
a single ten-image batch; all token rows at each MLP edge are retained. Thus
rows come from an actual full invocation, rather than concatenated sampled
rows. The model itself runs unchanged. Labels do not choose a format.

The diagnostic encodes static weight rows as signed INT8 in groups of32 along
K. Activation alternatives are signed INT8 and E2M1, with power-of-two scales
chosen upward from the current maximum to avoid clipping. E2M1 is represented
by twice its element values (integers up to12) and an exponent reduced by one.
This explicit recipe is not a complete OCP MX or NVFP4 implementation.

For row i, output column j and group g, the represented group contribution is
an integer dot product times2^(eA[i,g]+eW[j,g]). Choose a common anchor no larger
than any participating exponent. All contributions can then be left-shifted
into one integer domain. A sufficient absolute group bound is the smaller of
Amax[i,g]*sum(abs(W[j,g])) and sum(abs(A[i,g]))*Wmax[j,g]. These are standard
triangle bounds, not new theorems. The implementation uses conservative weight
norm/exponent extrema across the declared output-column window, avoiding a
bound computation for every individual output element.

A candidate width passes only when the sum of all shifted absolute group
bounds fits its signed positive limit. This also bounds all partial sums,
regardless of cancellation. The guard uses integer comparisons/shifts with
explicit overflow avoidance. Zero groups do not enlarge the exponent range.
The final mode requires every valid row of a square output window to pass;
otherwise that whole window rejects the width. Both128 and256 windows are
reported as alternative contracts. They do not create free subarrays or
establish that either format has native support in a particular chip.

The reported fraction weights accepted windows by actual M*N*K work; padded
scheduled calls do not disappear. One-row guard fractions are a separate,
optimistic diagnostic, not the supported whole-window fraction.

Crucial costs are intentionally unresolved here. Full-K lookahead requires
scale/statistic metadata before issue. Integer-domain merging needs shifts
and integer accumulation. Retaining both a narrow chunk and an outer floating
partial sum can add state; replacing the complete reduction's register needs
a compatible datapath and arithmetic contract. A floating fallback retains its
hardware area even if gated. Existing optimized block-scaled reduction,
[Bucket Getter](https://doi.org/10.1145/3613424.3614249) and
[MIAO](https://arxiv.org/abs/2611.19722v1) are relevant comparators.

This test does not evaluate the altered reduction or whole-model quantized
quality. Exact integer accumulation of represented operands can still produce
a different rounded result from a sequential floating-point reference. Before
an execution claim, test local fixed-input arithmetic and cumulative model/task
behavior separately, with the same quantized operands on both paths.
