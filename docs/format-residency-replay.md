# Conditional format-setup replay

`replay_format_residency.py` consumes complete preserved resident-phase traces.
Its units are abstract avoided-primitive opportunities, not joules, cycles or
CUDA time. It does not re-execute a network or certify generation quality.

A restricted phase has known constant setup cost C, no holding/eviction cost,
and the same frozen eligibility stream and boundaries for all comparators.
The current window is known before issue. A paired eligible row saves one
primitive for K>128; short tails and split controls save none in this proxy.
Each profitable window conservatively starts/ends in BF16 and pays two mode
transitions. This contract differs from the greedy partition ledger.

The causal strategy accumulates missed positive net opportunities until the
current opportunity repays setup, then retains the format until the phase ends.
Compare never-prepare, always-prepare and a restricted clairvoyant strategy.
The rent-or-buy idea is established theory, not a new result. Cost-grid values
are explicit hypothetical parameters; they are not fitted to labels or claimed
to describe an accelerator. Frozen-trace checks include conditional cost bounds
under these assumptions. The comparator cannot move phase boundaries or evict
representations differently.

Statistics/encoding costs are shared by these replay strategies and remain
unpriced against a no-controller baseline. MAC precision energy, queues,
physical memory actions and phase entry/exit are also unpriced. Future network
activations depend on earlier precision decisions; replay cannot establish the
quality or total energy of actually applying extra BF16 fallbacks.

The output preserves aggregate grid statistics and worst-phase witnesses.
Complete input window counts remain in the archived source records; every
per-phase calculation can be regenerated from them and this script.
