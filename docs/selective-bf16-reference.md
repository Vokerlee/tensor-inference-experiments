# Selective execution at an explicit BF16 observation

The reference matrix product accepts BF16 `A[M,K]`, `B[K,N]`, returns FP32
primitive outputs, and merges K primitives in order. Each primitive starts at
positive zero and uses sequential RN FP32 FMA. The observation adds optional
FP32/BF16 bias in FP32, then converts to BF16. This is a declared software
reference, not an assertion about an unspecified accelerator.

`selective_bf16_reference.hybrid` computes a candidate, builds an error bound
from operands alone, accepts outputs whose entire interval has one BF16
rounding, and executes the ordered reference only for unresolved outputs.
Modified outputs feed actual consumers. No full reference output is used by
the selector. Weight metadata require immutable operands; prepare again after
changing weights. This protects the BF16 observation, not arbitrary FP32 reads.

For primitive length `T` (default128), define `P=ceil(K/T)`, `w=min(K,T)` and
`gamma(n,u)=n*u/(1-n*u)`, with `n*u<1`. Sequential reference depth is bounded by
`h_ref=w+P`. Four independent RN FMA chains plus a balanced final reduction have
`h_four=ceil(w/4)+2+P`. Both use `u=2^-24`. Thus their difference is bounded by
`(gamma(h_ref,u)+gamma(h_four,u))*S + delta`, where
`S[i,j]=sum_k(abs(A[i,k]*B[k,j]))` and `delta` conservatively covers underflow.
These are standard forward-error/norm bounds, not a new theorem.

The native filter upper-bounds S by the minimum of row/column L1/Linf products
and an L2 product. Norm sums, square roots, bound arithmetic and cell endpoints
use explicit directed FP32 rounding. Bad/subnormal source domains, overflow
risk, ties and unsupported output cells require fallback. Optional bias includes
its own rounding error. Exactly zero products have a separate exact observation.

The `chains4` candidate's operation graph is defined in source. `tensorcore`
uses K128 native products and ordered FP32 merging, with a more conservative
minimum-FP32 rounding model (`u=2^-23` and a larger underflow allowance).
The Tensor Core guarantee is conditional on that model: numerical audit does
not prove an unspecified vendor operation graph. Both paths retain FP32 matrix
outputs before the explicit observation.

The eager/compiled FP64 filter is a comparison implementation. Compiled use
explicitly sets `emulate_precision_casts=True`; default fusion may elide an
intermediate BF16 round trip and invalidate a predicate relying on it. The
unsafe factory is test-only. Recheck predicates after compilation, including
boundary witnesses. [Pinned upstream semantics](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/torch/_inductor/config.py)
explain this behavior.

The [operator pilot](../scripts/selective_reference_operator_pilot.py) captures
first unique receiving-weight shapes from pinned ConvNeXt-V2/DINOv3, deterministic
source rows and complete provenance. It compares direct/tiled full reference
with both selective candidates. Timing includes dynamic norms, filtering,
compaction, repair and conversions. Immutable weight preparation is separately
reported. These are software emulator times, not target-device energy/latency.

Primary mathematical context: [Ziv rounding tests](https://perso.citi-lab.fr/fdedinec/recherche/publis/2013-TOMS-OnZivRoundingTest.pdf),
classical forward-error analysis, and [recent emulation guarantees](https://arxiv.org/abs/2611.19008v1).
Correctly rounding a real dot product and reproducing one finite-reference
observation are different specifications. This implementation alone is not
claimed as a new PhD contribution.
