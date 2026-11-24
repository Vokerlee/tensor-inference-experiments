# Block-scaled FP8 control

`fp8-k256-f2` and `fp8-k256-f4` evaluate E4M3FN activation and weight operands
under the K256/N128 geometry. Current activation scales are observed per
plane/row/K256 block; weight scales are prepared per plane/column/K256 block.
Both are powers of two, with symmetric clipping to448 before FP8 conversion.
No activation calibration, labels or fitted transform parameters are used.

The numerical oracle decodes every FP8 value into BF16, where it is exactly
representable, and uses BF16 products with FP32 partial output. Scales are
retired after each K block before FP32 accumulation and inverse transformation.
These floating partial sums do not inherit the exactness argument for bounded
INT8 integer products. No bit equality with a particular FP8 array operation
graph or native FP8 GPU execution time is claimed.

This is a format control for existing FP8 geometry, distinct from the proposed
INT8 arithmetic mode used by the feedback candidates. Weights also change
representation, so a system-level quality comparison cannot attribute the
whole difference to activation rounding. All final outputs and actual layer
statistics are preserved. A physical cost comparison must include block scales,
conversion, transforms, partial-sum retirement and the actual multiplier design.
