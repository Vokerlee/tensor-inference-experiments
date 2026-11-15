# BF16 CUDA GEMM baseline — 2026-11-15

Custom CUTLASS-based kernels are compared with `torch.mm` on one RTX 4070
(SM89), under WSL2. This is an operator baseline; no trained-model task or
accelerator-energy result is reported.

## Reproduce

Use the [build and benchmark instructions](../docs/benchmark-gemm.md).
The measured source revisions are experiments
`ce1b8bcf9d4c4f1f960861dabb717ab6b5e67c24` and engine
`16b3de6c7ef4953f7cd2e8d778432969d8ee43bf`, both clean at run time.
The JSON files preserve full versions, build/package fingerprints, settings,
all five timing samples per backend and numerical differences.

```sh
.venv/bin/python scripts/benchmark_gemm.py --output results/gemm-graph-baseline.json
.venv/bin/python scripts/benchmark_gemm.py --mode eager --shape 128x128x128 \
  --shape 1024x1024x1024 --shape 2048x2048x2048 \
  --output results/gemm-eager-baseline.json
```

The environment uses Torch 2.11.0+cu128, NVCC 12.8.93, GCC 13.3.0 and driver
610.88. Shared dependency pins remain owned by the engine. No new Python
package was needed for these measurements.

## CUDA graph results

[Raw graph data](2026-11-15-gemm-graph.json). Medians are microseconds per
allocating call, with ten calls captured per graph replay and fifty replays
per trial. Captured outputs stay alive until replay completes. A ratio above
1 means the custom path was faster. Dimensions below are `(M, N, K)`.

| Shape | RHS layout | Custom µs | Torch µs | Torch/custom |
|---|---|---:|---:|---:|
| 128 × 128 × 128 | row | 2.63 | 2.18 | 0.831 |
| 128 × 128 × 128 | column | 2.61 | 2.21 | 0.847 |
| 197 × 2304 × 768 | row | 22.38 | 22.79 | 1.018 |
| 197 × 2304 × 768 | column | 22.65 | 21.37 | 0.944 |
| 197 × 768 × 3072 | row | 38.38 | 30.15 | 0.786 |
| 197 × 768 × 3072 | column | 38.77 | 30.13 | 0.777 |
| 784 × 256 × 128 | row | 4.75 | 3.99 | 0.839 |
| 784 × 256 × 128 | column | 3.48 | 3.49 | 1.005 |
| 1024 × 1024 × 1024 | row | 51.22 | 43.15 | 0.842 |
| 1024 × 1024 × 1024 | column | 50.75 | 42.79 | 0.843 |
| 2048 × 2048 × 2048 | row | 291.48 | 291.47 | 1.000 |
| 2048 × 2048 × 2048 | column | 292.77 | 293.50 | 1.002 |

Large 2048-square products were approximately tied in this run. The 1024-square
case and the long-reduction 197 × 768 × 3072 case remain slower than Torch.
These results establish a usable baseline with an explicit performance gap;
they do not establish a general speed advantage. The shape heuristic selects
64-square output tiles when either output dimension is at most 256 and
128-square tiles otherwise. It was chosen after exploratory measurements on
these shapes, so this suite is development evidence rather than held-out
performance validation.

## Eager results

[Raw eager data](2026-11-15-gemm-eager.json). Each trial measures fifty eager
calls, including dispatch and output allocation on the event timeline.

| Shape | RHS layout | Custom µs | Torch µs | Torch/custom |
|---|---|---:|---:|---:|
| 128 × 128 × 128 | row | 6.39 | 5.84 | 0.913 |
| 128 × 128 × 128 | column | 13.70 | 15.05 | 1.099 |
| 1024 × 1024 × 1024 | row | 56.64 | 48.29 | 0.853 |
| 1024 × 1024 × 1024 | column | 56.56 | 48.11 | 0.850 |
| 2048 × 2048 × 2048 | row | 318.40 | 295.28 | 0.927 |
| 2048 × 2048 × 2048 | column | 289.44 | 291.04 | 1.006 |

Tiny eager products show substantial launch/host variability; their ratios
should not be interpreted as isolated kernel throughput. Graph and eager
results are different execution conditions, not interchangeable measurements.

## Numerical validation and limits

All 18 benchmark cases passed elementwise comparison with Torch at
`rtol=0.008, atol=0.02`, with BF16 inputs/output, FP32 accumulation, TF32 disabled
and BF16 reduced-precision reduction disabled. The largest absolute BF16
output difference from Torch was 1.0 in the graph cases; bitwise equality is
not claimed. Raw records also include differences from a separate FP32 product.
These tolerances check this operator baseline, not final task accuracy.

The final engine also passed 176 native reference cases and 78 Torch tests,
including both output types, four layout pairs, tails/pitched inputs, current
streams, graph replay, registration checks and a linear/bias/nonlinear chain.
A separate Torch-disabled native build passed both CTest checks.

The GPU was shared with the desktop; clocks and external load were not locked.
Backends were timed consecutively, not randomly interleaved. Five samples do
not establish confidence intervals; inspect the raw spread and repeat on
controlled hardware before drawing performance conclusions. No board power,
ASIC cycles, convolution, batching or complete-network accuracy was measured.
