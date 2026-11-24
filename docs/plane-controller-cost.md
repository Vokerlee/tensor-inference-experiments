# Causal plane-controller cost boundary

`scripts/plane_controller_cost.py` records logical incremental work above nearest
rounding. The signed-power-of-two greedy criterion uses two residual subtractions
and three criterion/state additions per encoded scalar. It adds five logical
power-of-two adjustments in this operation graph. Code floor/ceil, clipping,
comparisons, metadata decode, source staging and finite-precision conversion are
additional costs; the ledger does not silently treat them as zero.

The generic CUDA implementation is an executable numerical reference. A compiler
may implement an exponent adjustment using multiplication; its instruction count
or execution time is not a target-hardware cost measurement.

For M64/K256/P16 with one FP32 state per source row/channel:

| Quantity | Payload or capacity |
|---|---:|
| Live state |64KiB|
| Immutable packed coefficients |8KiB|
| State read/write capacity at256 parallel updates |2048bytes per vector|
| BF16 source width for256 values |512bytes per vector|
| Encoded INT8 operand width |256bytes per vector|
| INT4 codes still stored in INT8 words |256bytes per vector|

The2048-byte figure describes logical state-port access capacity. A dedicated
register file or pipeline may implement it differently from SRAM, but its ports,
wires, area and switching must be charged. Merely fitting64KiB is insufficient.
Coefficient preloading can amortize across64 rows, provided a resident broadcast
path exists. A256-byte encoded operand port does not establish a512-byte source
frontend. Software batches larger than64 do not establish a physical schedule.

If one encoding serves O actual receiver columns, incremental additions per
useful MAC are5/O:0.0390625 at O128,0.009765625 at O512 and0.0048828125 at O1024.
An RGB receiver with O3 needs1.667 additions per useful MAC before other costs;
shape-based bypass is therefore a meaningful comparator. Padding to128 columns
does not turn these into128 useful receiver operations.

For a candidate INT4-versus-INT8 deployment, the break-even condition per encoded
scalar is:

    O * (energy_MAC8 - energy_MAC4) + saved_memory_energy
        > incremental_controller_energy + other_added_energy

The right side includes state accesses and metadata delivery as well as arithmetic.
The current numerical oracle stores INT4 codes in INT8 words and proposes no
K512 mode; it demonstrates neither packed-storage savings nor increased throughput.
A real low-bit arithmetic/gating path, or a different validated precision switch,
is required to turn code range into an energy advantage. No numerical value for
this inequality is asserted without a hardware implementation or cost model.

Compare against ordinary live nearest rounding, prepared tap-wise transforms,
existing streaming inverse transforms and shape-specific lowerings. Extra
receiving GEMMs are not part of this controller. None of these counts is a claim
that an individual rounding mechanism is novel or that whole-model energy falls.
