"""Numerical reference for source-group quantization with row-local feedback.

This experiment-side oracle defines arithmetic and state; its Torch execution
does not model the schedule, latency or energy of a hardware quantizer.
"""

from dataclasses import dataclass

import torch


@dataclass
class FeedbackResult:
    reconstructed: torch.Tensor
    residual: torch.Tensor
    exponents: torch.Tensor
    clipped_values: int
    introduced_nonzeros: int


def kernel_coefficient(weight: torch.Tensor, *, coarse: bool = False) -> torch.Tensor:
    """Horizontal lag-one autocorrelation, per input channel of a dense kernel.

    The white, independent, equal-variance residual approximation predicts
    alpha = correlation / energy. Those assumptions are not imposed on real
    tensors. Clipping limits feedback strength; optional half-step rounding
    leaves only -1/2, 0, +1/2, each implementable without a general multiplier.
    """
    if weight.ndim != 4:
        raise ValueError("expected dense [output,input,height,width] weights")
    values = weight.detach().float()
    energy = values.square().sum((0, 2, 3))
    correlation = (values[..., 1:] * values[..., :-1]).sum((0, 2, 3))
    coefficient = torch.where(energy > 0, correlation / energy.clamp_min(1e-30), 0)
    coefficient = coefficient.clamp(-0.5, 0.5)
    return (coefficient * 2).round() / 2 if coarse else coefficient


def quantize(values: torch.Tensor, coefficient: torch.Tensor, *, bits: int = 4,
             preserve_zeros: bool = False) -> FeedbackResult:
    """Quantize NCHW with POT scales per source pixel/channel-group32.

    For each row and channel, u[j] = x[j] + alpha * r[j-1], q[j] = Q(u[j]),
    r[j] = u[j] - q[j]. Carry starts at zero at EVERY spatial row. Scales use
    the current original source group, not future pixels or a reference model.
    If preserve_zeros, an exactly zero source emits zero and resets its carry.
    The returned residual tensor is a diagnostic; live state needs only one
    carry per active row/channel stream, rather than this full tensor.
    """
    if (values.ndim != 4 or values.shape[1] % 32 or bits not in (4, 8)
            or coefficient.shape != (values.shape[1],)):
        raise ValueError("expected NCHW with aligned channels, one coefficient/channel, bits4/8")
    if (coefficient.device != values.device or not torch.isfinite(values).all()
            or not torch.isfinite(coefficient).all() or (coefficient.abs() > 1).any()):
        raise ValueError("finite inputs and same-device feedback coefficients within [-1,1] required")
    source = values.float().permute(0, 2, 3, 1)
    batch, height, width, channels = source.shape
    grouped = source.reshape(batch, height, width, channels // 32, 32)
    maximum = grouped.abs().amax(-1)
    limit = (1 << (bits - 1)) - 1
    exponent = torch.where(maximum > 0, (maximum / limit).log2().ceil(), 0).int()
    scales = torch.ldexp(torch.ones_like(maximum), exponent)
    reconstructed = torch.empty_like(source)
    residual = torch.empty_like(source)
    carry = torch.zeros(batch, height, channels, device=values.device, dtype=torch.float32)
    clipped = torch.zeros((), dtype=torch.int64, device=values.device)
    introduced = torch.zeros_like(clipped)
    for column in range(width):
        original = source[:, :, column]
        scale = scales[:, :, column, :, None].expand(-1, -1, -1, 32).reshape_as(original)
        incoming = original + coefficient * carry
        unbounded = torch.round(incoming / scale)
        current = unbounded.clamp(-limit, limit) * scale
        carry = incoming - current
        if preserve_zeros:
            current = torch.where(original == 0, 0, current)
            carry = torch.where(original == 0, 0, carry)
        clipped += (unbounded.abs() > limit).sum()
        introduced += ((original == 0) & (current != 0)).sum()
        reconstructed[:, :, column] = current
        residual[:, :, column] = carry
    return FeedbackResult(reconstructed.permute(0, 3, 1, 2).contiguous(),
                          residual.permute(0, 3, 1, 2).contiguous(), exponent,
                          int(clipped), int(introduced))
