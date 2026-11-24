# Producer support and bounded receiving geometry

A nearest2x producer followed by aligned, padding1, stride1 3x3 convolution
restricts the transformed input's reachable support. In F(2,3), the middle two
source entries of each four-value window are equal; their difference vanishes.
Seven of sixteen two-dimensional planes are therefore zero. The corresponding
algebraic support for the F(4,3) transform has25 of36 planes.

`winograd_support.py` derives these masks from the transform and the duplication
map, and identifies compatible producer/consumer modules by their declared
attributes. No layer-name list, activation calibration or trained mask is used.
The executable oracle checks that every declared inactive source plane is
actually zero. Real-algebra identities alone do not establish cancellation in
every finite-precision transform backend; a violated precondition is rejected.
A deployed hint needs a producer contract, or a separately costed detector and
fallback. The software check itself is not assumed free hardware.

The support-conditioned policy projects the receiving metric onto this declared
support before extracting its bounded-rank factor. Otherwise, scarce rank can
be spent on directions in which the producer never creates a quantization error.
Eigenvector numerical residue on inactive coordinates is explicitly zeroed.
Current activation scales, code decisions and causal feedback remain online;
only immutable receiving coefficients and producer traits prepare the metric.

The numerical replay retains the original weight representation and product
layout. It does not implement a new sparse array schedule or claim that its
CUDA kernels skip the zero products. Potential whole-vector elisions, source
code retention, ports and actual gating must be counted separately. Known
Winograd/transposed-convolution hardware and quantized upsampling methods are
strong comparators; this support restriction is not claimed as a new identity.
