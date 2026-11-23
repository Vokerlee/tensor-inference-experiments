# Bounded row partitioning by precision

A cohort-wide decision can reject otherwise acceptable rows. This experiment
separates per-row precision selection (`--cohort 1`) from a bounded schedule
that partitions the next32 available M rows by their chosen mode. Within each
mode row order remains stable; row IDs restore the original output positions.
The reduction coordinate K and each row's accumulation order are unchanged.
A row is not a physical PE-array M dimension.

`--row-buffer 32 --resident-rows 1024` records an explicit residency option:
for each K slice and N128 output window, retain the required original/FP8 weight
tiles while streaming up to1024 rows through32-row lookahead windows. The next
window is partitioned using its own decisions and the previous mode, without
future-window information. Initial ordering follows the current window's first
row. Statistics retain natural/partitioned mode changes within each resident
phase; entry/exit costs between phases are not included. No globally optimal
schedule is claimed.

For a full K256/N128 receiver tile, original BF16 weights occupy64KiB and the
candidate FP8 weights32KiB. A32-row source buffer needs16KiB original payload
and8KiB candidate payload. The combined payload is120KiB before row IDs, scales,
statistics, alignment, ports, queues or double buffering. A1024-row/N128 FP32
partial-result payload occupies512KiB. These are declared storage requirements,
not a model of an existing cache or an assumption of free accumulation state.

The ledger separately records weight-interface capacities with residency and
with reloads per lookahead window. Mixed modes can require BOTH weight formats,
increasing weight load/storage relative to uniform BF16 or uniform FP8. Source
candidate storage is charged even when every candidate is rejected, if encoding
was attempted. BF16-only/short-tail paths avoid that candidate allocation.
Output capacity is not automatically SRAM traffic; masked edges, forwarding and
actual buffer accesses require a concrete datapath/schedule implementation.

`--execution oracle` executes the numerical dense-candidate reference and maps
its decisions to this ledger. It can compute both candidates before masking.
`--execution packed` actually gathers selected row sets, executes only their
chosen products through the engine and restores row IDs. It is a software
numerical check, using BF16-reconstructed FP8 operands. Full-model oracle runs
must not be described as packed hardware execution. Tests compare gathered
native products with identical decisions in the reference, including tails.

Compare cohort32 and cohort1 at the same score, threshold and model conditions.
Also compare identical row decisions before/after partitioning: grouping can
change mode runs, not be credited with another quantization-quality benefit.
Quantization/statistics, queue operations, weight-format preparation, format
switches, array fill/drain and result handling all need separate cost evidence.
CUDA times do not estimate any of those target costs. No energy improvement is
established by these counters alone.

`--trace-schedule` preserves every resident phase's source matrix/K position,
window row counts, accepted FP8-row counts and partitioned mode changes.
This supports causal replay of setup/residency policies on a frozen eligibility
stream. Such replay does not execute a changed network policy: later activations
could differ if a real controller chose additional BF16 fallbacks. Frozen-trace
cost analysis and modified-model quality must be reported separately.
