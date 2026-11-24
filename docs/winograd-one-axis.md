# One-axis convolution as a precision/utilization control

The known F(2,3) identity can transform only one spatial axis. A 2x1 output
strip then uses four GEMMs with reduction3C. A two-axis2x2 tile uses sixteen
GEMMs with reductionC. For the same four spatial outputs, one-axis lowering
performs24C products rather than16C, before multiplying by output channels.
It stores12C transformed weights instead of16C.

Discrete K padding changes the comparison. With K256, C128 gives equal issued
vector counts for both lowerings; C64 halves the one-axis vector count, whereas
C256 makes it1.5 times larger. Equal issued vectors do not mean equal energy:
useful products, zero-lane gating, transforms, weight reuse and result retention
must all be charged. Inverse-before-K-sum is an existing strong storage baseline.

`winograd_axis_pilot.py` compares direct, one-axis and two-axis encodings on
identical real decoder tiles. Each uses its own prepared W8 representation,
current native K256 activation scales and the exact integer-partial oracle.
Full outputs and work ledgers are retained; local improvement does not establish
adapted full-decoder quality.

`winograd_decoder_pilot.py` supports `int8-k256-a{4,8}-axis-all` and `axis-fit`.
The latter uses one-axis lowering only when ceil(3C/256)<=2ceil(C/256); this
shape-only rule cannot exceed the even-width two-axis vector count. Other
layers retain two-axis lowering. No activation calibration or label-based
policy preparation is used. The original-network output remains the reference.
The software tile batch is an oracle throughput parameter, not a claim that
all its temporary tensors fit a specified hardware memory bank simultaneously.
