"""Receiver-window error surrogates from live scalar activation moments.

The centered formula is exact only after averaging over uniform common
permutations of paired activation/residual coordinates relative to fixed
weights. Trained channels are not generally exchangeable. It is not a bound.
"""

import torch
from torch.nn import functional as F


def weight_statistics(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """Six prepared means per <=128-column receiving window for one K slice.

    Inputs are [N,K]. Columns in the mathematical A @ W convention correspond
    to rows here. Statistics retain squared column sums, their cross product,
    and centered squared norms/cross products for W and D=W-Wq.
    """
    weight = original.float() if original.dtype != torch.float64 else original
    residual = weight - reconstructed.to(weight.dtype)
    weight_sum = weight.sum(-1)
    residual_sum = residual.sum(-1)
    centered_weight = weight - weight.mean(-1, keepdim=True)
    centered_residual = residual - residual.mean(-1, keepdim=True)
    terms = torch.stack((weight_sum.square(), residual_sum.square(), weight_sum * residual_sum,
                         centered_weight.square().sum(-1), centered_residual.square().sum(-1),
                         (centered_weight * centered_residual).sum(-1)), -1)
    return torch.stack([terms[begin:begin + 128].mean(0) for begin in range(0, len(weight), 128)])


def predicted_energy(original, reconstructed, statistics, *, centered=True):
    """Return [M,Nwindows] error and reference-output energy surrogates.

    Z=Aq, E=A-Z and D=W-Wq give delta=E@W+Z@D before accumulation rounding.
    Centered moments include live E/Z covariance and prepared W/D correlation.
    The diagonal control assumes zero-mean independent coordinate contributions.
    Neither formula measures actual receiving outputs or graph sensitivity.
    """
    a = original.to(statistics.dtype)
    z = reconstructed.to(statistics.dtype)
    e = a - z
    k = a.shape[-1]
    if k <= 1:
        raise ValueError("moment surrogate requires at least two reduction coordinates")
    sw2, sd2, swsd, wc2, dc2, wcdc = statistics.T
    if not centered:
        w2 = wc2 + sw2 / k
        d2 = dc2 + sd2 / k
        error = e.square().mean(-1, keepdim=True) * w2 + z.square().mean(-1, keepdim=True) * d2
        reference = a.square().mean(-1, keepdim=True) * w2
        return error, reference
    me = e.mean(-1, keepdim=True)
    mz = z.mean(-1, keepdim=True)
    ec, zc = e - me, z - mz
    ve, vz = ec.square().mean(-1, keepdim=True), zc.square().mean(-1, keepdim=True)
    covariance = (ec * zc).mean(-1, keepdim=True)
    correction = k / (k - 1)
    error = (me.square() * sw2 + mz.square() * sd2 + 2 * me * mz * swsd
             + correction * (ve * wc2 + vz * dc2 + 2 * covariance * wcdc))
    reference = ((me + mz).square() * sw2
                 + correction * (ve + vz + 2 * covariance) * wc2)
    return error.clamp_min(0), reference.clamp_min(0)


def select_cohorts(original, reconstructed, statistics, *, tolerance, cohort, centered=True):
    error, reference = predicted_energy(original, reconstructed, statistics, centered=centered)
    fits = error <= tolerance**2 * reference
    padded = F.pad(fits, (0, 0, 0, (-len(original)) % cohort), value=True)
    selected = padded.reshape(-1, cohort, statistics.shape[0]).all(1)
    relative = torch.where(reference > 0, error / reference,
                           torch.where(error == 0, 0, torch.inf)).sqrt()
    return selected, relative
