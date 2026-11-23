# Contiguous accumulator chunks from group metadata

`scripts/accumulation_segments.py` plans boundaries inside one K slice from
nonnegative group bounds and exponent intervals. It does not inspect a receiving
matrix product. The input is the same conservative maximum-only or maximum/L1
metadata used by the whole-slice guard.

Starting with a group, extend the current integer chunk while its common-anchor
absolute bound fits the configured signed width. If it would not fit, end the
previous chunk and start another. A single active group that cannot fit uses
the baseline floating path; zero groups can be bypassed on either path. The
reference accepts at most eight groups, matching K=128/256 and group32.

Adding a nonzero group cannot decrease this bound; trimming an interval cannot
make a feasible interval fail. Longest feasible prefixes therefore minimize
the number of contiguous chunks under this bound and singleton fallback rule.
Tests compare the plan against exhaustive partitions. This elementary greedy
property is not a novelty claim or a proof of minimum hardware energy.

Every integer chunk needs a conversion when merged into its floating carry.
The matrix primitive still emits FP32 at the original K boundary; the plan
does not keep integer state across calls. Both integer chunk state and floating
carry may be live, so register/clock costs must be counted. A four/eight-stage
scalar planner could be pipelined, with operand buffering and boundary tags;
Python loop latency does not establish its cycle cost or throughput.

Chunking changes floating-point parentheses. Exact integer sums inside each
chunk do not guarantee equality to sequential FP32 group accumulation. Local
fixed-input tests and complete-model evaluation are required. Existing
[Bucket Getter](https://doi.org/10.1145/3613424.3614249),
[MIAO](https://arxiv.org/abs/2611.19722v1), and
[precision-boundary work](https://doi.org/10.1016/j.sysarc.2026.103704)
are comparators for deferred conversion and chunked accumulation. This planner
alone does not establish a new architecture.

The primitive oracle accepts `strategy="greedy"`. It sums each selected chunk
using the bounded exact reference, converts that chunk to FP32, then merges it
into a floating carry. A cancellation test deliberately produces a different
answer from sequential FP32 grouping, so numerical equality is not assumed.

Counters retain whole-slice guard coverage as a separate diagnostic. Chunk
benefit is measured by selected chunk count and remaining group merges, not
by that coverage. Both raw merge counts and a comparator that bypasses zero
FP32 group outputs are recorded. All-zero inputs must show zero saving against
that zero-gated comparator. These are action counts, not joules or skipped MACs.

```sh
.venv/bin/python scripts/primitive_accumulation_pilot.py --model resnet18 \
    --strategy greedy --bound-kind max --k-slice 128 --width 24 \
    --samples-per-class 10 --output results/resnet18-greedy-chunks.json
```

Use the same images and code format for the `--strategy whole` comparator.
`--model vit_b_16` and `--model swin_t` exercise MLP consumers instead. Each run
includes the unchanged model and the same group-FP32 quantized reference. Fixed
INT8 weights and online INT8 activation scales remain the default. The model
is modified during these executions; this is more than a range-only screen.

`zero_gated_whole_slice_merges` records the whole-slice fallback comparator on
the **same represented inputs** used by the selected execution. This permits
an action comparison without silently changing the input trajectory. The counts
concern useful output scalars; padded interface capacity and common clocks,
registers, SRAM activity and control remain separate hardware costs.

## Native planner cross-check

`scripts/integer_segment_backend.py` prepares output-window statistics and
calls the engine's fused CUDA planner. Python metadata preparation and model
selection stay in this repository; the engine contains only the generic
scalar and CUDA interfaces. The engine revision is pinned by the submodule.

```sh
.venv/bin/python -m pytest -q tests/test_integer_segments_cuda.py
```

Build the optional engine Torch adapter first, as in the GEMM instructions.
These checks compare byte masks against independently constructed tensor
bounds, including tail slices, both norm modes, zero groups and fallback.
They also exercise streams, graph capture and the Meta interface. They validate
planning only: the numerical oracle still executes the selected arithmetic
separately, and source-statistics preparation is not free.
