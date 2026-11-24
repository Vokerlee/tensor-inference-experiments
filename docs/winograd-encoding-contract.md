# Encoded planes carry their scales

Every plane encoder returns a `PlaneEncoding`: reconstructed values, the actual
per-plane/row/K-block exponents used by that backend, and statistics. Integer
retirement uses these returned exponents. It checks finite integral recovered
codes and the signed symmetric code range before any int8 cast or product.
These checks are numerical-oracle validation, not proposed hardware work.

Recomputing the exponent in Torch is unsafe for a native encoding: a maximum
just above 127 can produce exponent 0 in one FP32 operation graph and 1 in the
other. Decoding with the wrong scale can turn a valid positive code into a
wrapped negative int8 code. A directed CUDA test covers this boundary.

The `int8-k256-a{4,8}-native` policies provide native nearest-rounding controls.
They use zero metric metadata in the causal encoder, so ties select nearest
and the scale operation graph is shared with native proxy/shift policies.
A comparison against the older Torch nearest policy is not a matched-grid
comparison. Zero-metric use of the generic software kernel does not imply
feedback-state hardware is needed for ordinary nearest rounding, and its GPU
runtime does not estimate that hardware cost.
