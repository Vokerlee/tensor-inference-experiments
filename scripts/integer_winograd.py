"""Numerical contract for proposed block-scaled INT8 Winograd primitives.

A K256/N128 integer interface is a proposed mode, not the existing FP8 format.
One activation scale per row/K block and one weight scale per column/K block
are applied after each integer partial product. No hidden group32 correction
products are assumed inside that primitive.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from winograd_rounding import round_planes
from winograd_streaming import product


@dataclass(frozen=True)
class IntegerWeights:
    codes: torch.Tensor
    exponents: torch.Tensor
    channels: int
    block: int

    def reconstructed(self):
        planes, padded, outputs = self.codes.shape
        values = self.codes.float().reshape(planes, padded // self.block, self.block, outputs)
        scale = torch.ldexp(torch.ones_like(self.exponents, dtype=torch.float32), self.exponents)
        return (values * scale[:, :, None, :]).reshape(planes, padded, outputs)


def prepare(coefficients: torch.Tensor, block: int = 256) -> IntegerWeights:
    if coefficients.ndim != 3 or block not in (32, 128, 256):
        raise ValueError("[planes,C,O] and supported integer K block required")
    if not torch.isfinite(coefficients).all() or min(coefficients.shape) < 1:
        raise ValueError("finite nonempty weights required")
    planes, channels, outputs = coefficients.shape
    padded = functional.pad(coefficients.float(), (0, 0, 0, (-channels) % block))
    grouped = padded.reshape(planes, -1, block, outputs)
    maximum = grouped.abs().amax(2)
    exponent = torch.where(maximum > 0, (maximum / 127).log2().ceil(), 0).int()
    scale = torch.ldexp(torch.ones_like(maximum), exponent)
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("weight scale outside finite FP32 domain")
    codes = (grouped / scale[:, :, None, :]).round().clamp(-127, 127).to(torch.int8)
    return IntegerWeights(codes.reshape(planes, -1, outputs), exponent, channels, block)


def multiply(source: torch.Tensor, weights: IntegerWeights, *, bits: int = 8,
             geometry=None) -> tuple[torch.Tensor, dict]:
    if (source.ndim != 3 or source.shape[0] != weights.codes.shape[0]
            or source.shape[2] != weights.channels or bits not in (4, 8)
            or not torch.isfinite(source).all()):
        raise ValueError("compatible finite [planes,M,C] source and INT4/8 required")
    padded = functional.pad(source.float(), (0, weights.codes.shape[1] - source.shape[2]))
    planes, rows, channels = padded.shape
    grouped = padded.reshape(planes, rows, -1, weights.block)
    maximum = grouped.abs().amax(-1)
    limit = (1 << (bits - 1)) - 1
    exponent = torch.where(maximum > 0, (maximum / limit).log2().ceil(), 0).int()
    scale = torch.ldexp(torch.ones_like(maximum), exponent)
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("activation scale outside finite FP32 domain")
    stats = None
    if geometry is not None:
        reconstructed, stats = round_planes(padded, geometry, bits, group=weights.block)
        codes = (reconstructed.reshape(grouped.shape) / scale[..., None]).round().to(torch.int8)
    else:
        codes = (grouped / scale[..., None]).round().clamp(-limit, limit).to(torch.int8)
    weight_scales = torch.ldexp(torch.ones_like(weights.exponents, dtype=torch.float32), weights.exponents)
    accumulated = None
    for block in range(channels // weights.block):
        start = block * weights.block
        lhs = codes[:, :, block].to(torch.bfloat16)
        rhs = weights.codes[:, start:start + weights.block].to(torch.bfloat16)
        # Each int8*int8 K<=256 partial has absolute sum <=127**2*256<2**24.
        # BF16 represents each integer operand exactly; FP32 accumulation is exact
        # for these integer partials. This is a numerical oracle, not INT8 timing.
        partial = product(lhs, rhs)
        partial = partial * scale[:, :, block, None] * weight_scales[:, block, None, :]
        accumulated = partial if accumulated is None else accumulated + partial
    if not torch.isfinite(accumulated).all():
        raise ValueError("nonfinite dequantized partial accumulation")
    levels, counts = torch.unique(exponent, return_counts=True)
    return accumulated, {
        "activation_bits": bits, "weight_bits": 8, "K": weights.block,
        "original_channels": weights.channels, "padded_channels": channels,
        "activation_groups": maximum.numel(), "zero_groups": int((maximum == 0).sum()),
        "exponent_histogram": {str(int(level)): int(count) for level, count in zip(levels, counts)},
        "rounding": stats,
        "integer_partial_vectors": planes * rows * (channels // weights.block)
                                   * ((weights.codes.shape[-1] + 127) // 128),
        "output_dequant_scalar_multiplications": 2 * planes * rows * (channels // weights.block)
                                                * weights.codes.shape[-1],
    }
