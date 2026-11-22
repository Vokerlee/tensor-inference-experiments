# Matrix-unit vector interface

The target input vector has **256 bytes per clock**: 128 BF16 or 256 FP8
elements. Its corresponding output vector has **128 FP32 scalars** in both
formats. In GEMM notation the format-dependent input length is a K slice,
not M. M counts streamed input rows; N has 128 output positions. The FP8
unit is therefore 256 by 128, with no implied 256-wide output.

INT8/FP4 geometry, RHS bandwidth, stationarity, internal accumulator format,
retained state between K slices and output cadence remain unspecified. A
fixed FP32 output does not establish persistent integer state across calls.

Earlier [activation pilot](activation-pilot.md) and
[accumulation headroom](accumulation-scale-diagnostics.md) use software policy
windows. Their 128/256 square windows are historical grouping alternatives,
not the native FP8 shape. A 256-column policy window would cover two native
output vectors. The native CUDA precision controller's 128 by 128 input block
is also a software decision unit; a K=256 issue needs compatible decisions
across its two K=128 pieces or explicitly defined handling. Its M cohort does
not follow from the array's physical reduction width.

For a first changed-accumulation execution test, preserve the FP32 output of
each matrix primitive. Holding a common integer domain across several emitted
FP32 vectors needs an explicit ISA/state extension, with its costs counted.
