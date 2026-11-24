# Retaining one encoded source across output-column tiles

`scripts/winograd_encoded_retention.py` checks a proposed K256/N128 integer
storage layout. It keeps the complete encoded source for a streamed spatial
batch, so every output-column tile can reuse the same codes. It explicitly
includes padded zero lanes, INT16 scale exponents, buffered RHS blocks, causal
controller state and packed coefficients. This is a capacity check; it does not
prove a port schedule, DMA overlap, source delivery rate or energy savings.

The argument memory budget is2MiB shared by both arguments. Result memory is a
separate2MiB. Controller state can use dedicated storage, or be colocated with
arguments; both totals are reported. No extra2MiB bank for each argument is
assumed. The general64MiB memory is on chip.

For C1024/O1024, an F(2,3) source with M64 occupies1MiB in encoded INT8 form.
The same M64 under F(4,3) occupies2.25MiB before adding RHS or metadata and
therefore fails the shared argument budget. Reducing M to32 gives1.125MiB;
processing four RHS planes at a time permits double buffering while retaining
codes, exponents and the current-K controller state below2MiB. The retained
product planes plus a separate spatial result for one N128 tile also fit2MiB.
This arrangement lets eight output-column tiles share each encoded source.

The largest layer's full padded F(4,3) W8 representation is36MiB. We report
whether weights plus a materialized source tile fit64MiB, but this is not a
whole-layer feasibility test: complete high-resolution activations do not
necessarily fit, so spatial blocking and halo traffic still matter. Gathered
source-tile bytes are a possible materialized footprint, not compulsory traffic;
a streaming transform can avoid materializing every overlapping patch.

An alternative inverse-before-K-sum schedule is already known and can reduce
intermediate retention further. Its changed operation graph, extra transforms,
output cadence and inter-K accumulation placement remain separate questions.
Software evaluation batches may be larger than these proposed hardware batches;
no CUDA timing or allocator footprint is used to establish this storage layout.
