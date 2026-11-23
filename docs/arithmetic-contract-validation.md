# Arithmetic and causal-controller validation

The [finite witness runner](../scripts/probe_accumulation_contracts.py) separates
range safety, compatibility with a floating reference, primitive-boundary
rounding and state-dependent controller decisions. It executes numerical
references; it is not a hardware implementation or sampled reliability study.

```sh
.venv/bin/python scripts/probe_accumulation_contracts.py \
    --output results/arithmetic-contract-witnesses.json
```

For a local diagnostic let x be a source tensor, u a complete controller
configuration/decision, and H(x,u) the receiving computation. Compare original
x,u with changed source x', recomputed decision u'. The identity

    H(x',u') - H(x,u)
      = [H(x',u)-H(x,u)] + [H(x',u')-H(x',u)]

separates a fixed-control data-path effect from recomputing control at the
changed input. It is ordinary algebra, not a new theorem, and depends on this
intervention order. It does not establish causal uniqueness or a norm bound.
Freeze identical preceding state for a local comparison. Across a whole
network, past changes affect subsequent activations, statistics and caches;
a frozen trace is not interchangeable with reexecuting a live policy.

The constructed BF16 witness changes one source coordinate from7 to7.03125.
A linear receiver ignores that coordinate, so its unquantized output remains
0.75. Current-group INT4 POT scaling nevertheless changes from1 to2, changing
the represented receiver output from1 to0. Keeping the original scale yields1.
The fixed-control effect is0; the recomputed-control effect is−1. This
illustrates a potential confound; it does not measure its frequency in models.

The other witnesses scale the finite pattern[2^24,1,-2^24,1] across powers of
two. Exact range-safe I32 accumulation and sequential FP32 give different
outputs. Separating positive/negative groups across mandatory primitive FP32
outputs demonstrates why removing that boundary changes the arithmetic contract.
Exact real sum, FP32 comparator and intended hardware output are distinct
references; a difference is not automatically an implementation defect.

Useful checks for adaptive experiments include:

1. Specify the arithmetic, output boundaries, source layout, stream resets and
   information available when each decision is made.
2. Match operands and prior state for local arithmetic comparisons; separately
   execute the changed complete DAG and measure its final outputs/task.
3. Preserve controller decisions and resource actions as well as tensor errors.
   Perturb future blocks to check that preceding decisions cannot depend on them.
4. Compare frozen-control diagnostics with live policy execution. Include strong
   uniform precision controls and the same policy with/without scheduling changes.
5. Keep modeled activity separate from characterized component costs. Expose
   cost-sign changes and unresolved parameters rather than reporting GPU timings
   as hardware energy.

Portable numerical emulation already exists in
[QPyTorch](https://arxiv.org/html/1910.04540v1) and
[Arbitor](https://www.usenix.org/conference/atc23/presentation/jiang-chenhao).
Generated differential tests and quantization robustness search also exist in
[NNSmith](https://arxiv.org/abs/2207.13066v2) and
[DiverGet](https://arxiv.org/abs/2207.06282v1).
These checks are an experimental protocol and set of constructed diagnostics,
not a claim of a new emulation or testing methodology.
