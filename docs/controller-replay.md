# Capturing and replaying source-quantizer control

`scripts/controller_replay.py` records the actual POT exponents, coefficient
choices and preceding mean/second-moment sensor state for each controlled
source. It supports memoryless conversion, fixed kernel feedback and causal
32-row feedback updates. State resets per consumer call and image.

A tape binds each event to its layer identity, NCHW shape, precision and row
block size. It stores INT16 exponents and FP32 coefficients/sensor values in
compressed NumPy arrays without pickles. The engine consumes externally supplied
INT32 exponents after exact conversion. Invalid native events reject results.

On unchanged tensors and prior execution, replaying both scales and coefficients
must reproduce output and terminal state bit for bit. Tests also change future
row blocks to verify that earlier outputs/decisions cannot depend on them.
On changed tensors, replayed control is an explicit counterfactual intervention:
input values, clipping and feedback arithmetic remain live. Frozen scales can
clip, and frozen modes need not minimize the current surrogate.

Scale-only and mode-only freezing are separate interventions; their effects
need not add because decisions and future activations interact. Preceding
sensor moments are stored even when frozen coefficients bypass mode selection.
Mode occupancy and nonzero-feedback positions describe chosen configurations,
not measured gated operations, hardware latency or energy.

The full-decoder runner `scripts/sana_controller_replay.py` uses known BF16
resize folding as a backend change and excludes its convolution receivers
structurally from control instrumentation. This preserves matching event
identities across both backends. Other dense aligned spatial receivers remain
controlled. Same-backend trace replay must be bit-exact before accepting the
original/folded live/frozen comparisons. Pixel fidelity remains distinct from
semantic generation quality and target-device arithmetic.
