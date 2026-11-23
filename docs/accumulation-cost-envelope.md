# Conditional accumulation cost envelope

This tool asks which component costs would make the recorded accumulation
optimization worthwhile. It does not predict a chip's energy or use CUDA time.

```sh
.venv/bin/python scripts/accumulation_cost_envelope.py \
    --input reports/2026-11-23-primitive-accumulation.csv \
    --output results/accumulation-cost-envelope.json
```

Let P be recorded scalar products, F reference scalar group merges, R selected
merges and V pre-issue vector decisions. Products and mandatory FP32 primitive
outputs are unchanged. Normalize energy to one baseline scalar MAC:

- lambda: effective energy of a counted group boundary / scalar MAC energy;
- mu: unchanged non-MAC energy / total baseline MAC energy;
- gamma: one vector decision's energy / scalar MAC energy;
- omega: all remaining added energy / total baseline MAC energy.

For raw counters, eliminated merges and eliminated group conversions have the
same difference when the first conversion per output is common. Lambda may
represent that characterized bundle, or only the floating add if conversion
savings are deliberately excluded. Declare which convention is characterized;
never multiply these counts by an unspecified operation's energy.

The conditional costs in MAC-energy units are

    B = P*(1 + mu) + F*lambda
    C = P*(1 + mu + omega) + R*lambda + V*gamma
    gain = (B-C)/B.

With f=F/P, d=(F-R)/P and h=V*gamma/P+omega, the required ratio for target
fraction g is

    lambda >= [g*(1+mu) + h] / [d-g*f],

provided d-g*f>0. Otherwise that positive target has no finite solution.
These are elementary accounting identities, not a new energy model or theorem.
The zero-overhead case is an optimistic limit for the represented benefit
family, not a bound on every possible additional architectural optimization.

Both raw and postzero accounting are preserved as ALTERNATIVES. Postzero
counters observe actual zero group/chunk outputs; they do not establish a free
causal detector or remove its energy. Never add the two ledgers. These counts
also describe the evaluated layers, not all operations of an entire network.
The common energy must include unmodified layers when estimating whole-model
benefit; component and system fractions cannot be interchanged.

The grid/rectangular box is explicitly hypothetical. It spans ratios rather
than importing energy per operation from another technology or accelerator.
Its extrema are conditional mathematical bounds, not confidence intervals.
Correlated costs may make some box corners unrealizable. Negative gains remain
in the output. A measured operation-count improvement is therefore allowed to
be energy-negative under a declared cost scenario.

To obtain a hardware estimate, characterize the actual group-boundary bundle,
MAC/accumulator path, vector guard and metadata handling. Omega must cover
alignment, changed state width, additional buffers/register activity and any
clock/timing effects not included elsewhere. Mu must use actual mapping and
communication, including the unmodified workload. State/traffic counts and
coefficients need validation before attaching joules or energy-saving claims.
Action-based component estimation is established by
[Accelergy](https://accelergy.mit.edu/paper.pdf); this script is a preliminary
sensitivity screen, not a replacement for characterized components.
