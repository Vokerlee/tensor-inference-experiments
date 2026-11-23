# Runtime receiver-moment surrogate

`precision_moments.py` compares a centered, finite-population second-moment
surrogate with a simpler diagonal/noise control. Both use live scalar moments
of reconstructed activations Z and residuals E=A-Z, plus six prepared statistics
for each K slice and <=128-column receiver window. Weights W have reconstruction
Wq and residual D=W-Wq. The represented product difference is E@W+Z@D before
accumulation rounding. The selector never computes these receiving products.

The centered formula accounts for mean errors, E/Z covariance and W/D
correlation. It is exact after averaging over uniform common permutations of
paired (E,Z) coordinates relative to fixed (W,D). Actual trained channels are
not generally exchangeable. The diagonal control drops coherent means and
cross terms. It is an assumption-based comparator, not a reproduction of the
complete FGMP system or a novel second-moment principle.

The score is predicted RMS receiving error divided by predicted RMS receiving
output. It averages over each output window and can hide a sensitive column.
A cohort passes only if every row's score meets the same global tolerance.
Neither predictor is a worst-case bound, an actual output-error measurement or
a downstream quality certificate. Cancellation/roundoff is clamped in FP32;
exact-arithmetic identities do not certify this implementation.

Six receiver coefficients are prepared without activation calibration. Live
means, centered squares and cross-products require real datapath/state cost;
FP32 scalar arithmetic cannot be assumed cheap merely from operation counts.
No GPU timing is converted to target controller cost. Quantization, metadata,
format switches, buffering, array issue and FP32 output handling remain separate.

`moment-paired`, `moment-split`, `diagonal-paired` and `diagonal-split` reuse the
existing numerical K128/K256 contract. BF16 reconstructed FP8 products are not
native packed FP8 execution. The max/L1 selector and uniform FP8 remain
comparators. A different score cannot be credited with the uniform FP8 benefit.

Tests use exhaustive common permutations, coherent bias, zero-sum receiving
weights and native precision-policy extremes. These check algebra and declared
execution contracts; final-model tests are still required.
