# Conditional scale stability diagnostics

The prototype has two parts. First, `scale_stability_probe.py` enumerates all
65,280 finite BF16 bit patterns, repeats each across one source group, and records
the actual native POT scale decisions for INT4 and INT8. It checks sign symmetry
and positive-domain monotonicity, compares CPU Torch exponent selection, and
retains nonfinite reconstruction cases at extreme magnitudes. These extremes
are diagnostic contract cases, not accepted neural-inference outputs.

Second, `decision_stability.ScaleTable` filters a reference group using an
absolute source-perturbation bound. For finite source vectors `x` and `x'`, let
`m = max(abs(x))` and suppose `max(abs(x' - x)) <= epsilon`. Then the candidate
maximum lies in `[max(0, m - epsilon), m + epsilon]`. If every BF16 maximum in
this interval belongs to the same measured scale-decision region, the scale
cannot change under this bound. Outward FP64 rounding encloses the interval;
zero uncertainty is exact. Decision regions include the special zero group.

The exhaustive table follows the actual binary's FP32 division/logarithm
behavior rather than silently treating it as an exact real logarithm. The table
is specific to that binary/device and is complete only for the stated finite
BF16 source domain. It certifies only amax-selected scale, not reconstructed
codes, feedback mode/state, a receiving product or model output. An unknown
result means the condition could not certify stability, not that a flip occurred.

The real-trajectory probe uses three predeclared local source patches per SANA
input. Both per-group and patch-global bounds are measured from the actual pair
of original/folded trajectories. All group maxima, observed bounds, exponents
and guard outcomes are retained. Since both trajectories were already evaluated,
this measures diagnostic selectivity, not saved reference work. A future
selective emulator needs an independently justified forward error bound and
end-to-end validation, including state and downstream changed activations.

```sh
python scripts/scale_stability_probe.py \
    --inputs results/decoder-inputs-20261123 results/decoder-inputs-holdout-20261123 \
    --output results/scale-stability-20261124.json
```

Interval guarding is established numerical analysis. Relevant primary baselines
are [FLUCTUAT](https://arxiv.org/abs/1309.3910v1),
[stable-test transformation](https://arxiv.org/abs/1808.04289v2) and
[PRECiSA code generation](https://arxiv.org/abs/2001.02981v1).
This specialization does not claim a new theorem or a hardware implementation.
