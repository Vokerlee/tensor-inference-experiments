"""Numerical contract for proposed block-scaled INT8 Winograd primitives.

A K256/N128 integer interface is a proposed mode, not the existing FP8 format.
One activation scale per row/K block and one weight scale per column/K block
are applied after each integer partial product. No hidden group32 correction
products are assumed inside that primitive.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from winograd_rounding import encode_planes
from winograd_shift_rounding import ShiftGeometry, encode_shift
from winograd_reference import inverse_outputs, transforms
from winograd_streaming import map_tiles, product


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
             geometry=None, partial_transform=None) -> tuple[torch.Tensor, dict]:
    if (source.ndim != 3 or source.shape[0] != weights.codes.shape[0]
            or source.shape[2] != weights.channels or bits not in (4, 8)
            or not torch.isfinite(source).all()):
        raise ValueError("compatible finite [planes,M,C] source and INT4/8 required")
    padded = functional.pad(source.float(), (0, weights.codes.shape[1] - source.shape[2]))
    planes, rows, channels = padded.shape
    grouped = padded.reshape(planes, rows, -1, weights.block)
    maximum = grouped.abs().amax(-1)
    limit = (1 << (bits - 1)) - 1
    stats = None
    if geometry is not None:
        encoder = encode_shift if isinstance(geometry, ShiftGeometry) else encode_planes
        encoded = encoder(padded, geometry, bits, group=weights.block)
        exponent, stats = encoded.exponents, encoded.statistics
        if exponent.shape != maximum.shape:
            raise ValueError("encoder returned incompatible group exponents")
        scale = torch.ldexp(torch.ones_like(maximum), exponent)
        recovered = encoded.reconstructed.reshape(grouped.shape) / scale[..., None]
        rounded = recovered.round()
        valid = torch.isfinite(recovered) & (recovered == rounded) & (rounded.abs() <= limit)
        if not bool(valid.all()):
            raise ValueError("encoded code range/grid does not match its authoritative scale")
        codes = rounded.to(torch.int8)
    else:
        exponent = torch.where(maximum > 0, (maximum / limit).log2().ceil(), 0).int()
        scale = torch.ldexp(torch.ones_like(maximum), exponent)
        if not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("activation scale outside finite FP32 domain")
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
        if partial_transform is not None:
            partial = partial_transform(partial)
        accumulated = partial if accumulated is None else accumulated + partial
    if not torch.isfinite(accumulated).all():
        raise ValueError("nonfinite dequantized partial accumulation")
    levels, counts = torch.unique(exponent, return_counts=True)
    return accumulated, {
        "activation_bits": bits, "weight_bits": 8, "K": weights.block,
        "original_channels": weights.channels, "padded_channels": channels,
        "activation_groups": maximum.numel(), "zero_groups": int((maximum == 0).sum()),
        "exponent_histogram": {str(int(level)): int(count) for level, count in zip(levels, counts)},
        "rounding": stats, "code_grid_validated": geometry is not None,
        "integer_partial_vectors": planes * rows * (channels // weights.block)
                                   * ((weights.codes.shape[-1] + 127) // 128),
        "output_dequant_scalar_multiplications": 2 * planes * rows * (channels // weights.block)
                                                * weights.codes.shape[-1],
    }


def convolution(values: torch.Tensor, weights: IntegerWeights, bias=None, *, tile: int = 2,
                padding: int = 1, tile_batch: int = 64, bits: int = 8, geometry=None,
                streaming_inverse: bool = False):
    scheme = transforms(tile, dtype=torch.float32, device=values.device)
    stats = {"activation_bits": bits, "weight_bits": 8, "K": weights.block, "N": 128,
             "original_channels": weights.channels, "padded_channels": weights.codes.shape[1],
             "activation_groups": 0, "zero_groups": 0, "integer_partial_vectors": 0,
             "output_dequant_scalar_multiplications": 0, "exponent_histogram": {},
             "rounding": None, "streaming_inverse": streaming_inverse}

    def inverse(products):
        return inverse_outputs(products, scheme)

    def consume(source):
        result, current = multiply(source, weights, bits=bits, geometry=geometry,
                                   partial_transform=inverse if streaming_inverse else None)
        for key in ("activation_groups", "zero_groups", "integer_partial_vectors",
                    "output_dequant_scalar_multiplications"):
            stats[key] += current[key]
        for exponent, count in current["exponent_histogram"].items():
            histogram = stats["exponent_histogram"]
            histogram[exponent] = histogram.get(exponent, 0) + count
        if current["rounding"] is not None:
            if stats["rounding"] is None:
                stats["rounding"] = {"bits": bits, "rank": geometry.rank,
                                     "mode": current["rounding"].get("mode", "completed-proxy"),
                                     "backend": current["rounding"].get("backend", "torch")}
            for key in ("candidate_selected", "candidate_decisions", "source_sse", "source_zeros_changed",
                        "selected_proxy_sum", "nearest_proxy_sum", "groups", "zero_groups", "changed_channels"):
                if key in current["rounding"]:
                    stats["rounding"][key] = stats["rounding"].get(key, 0) + current["rounding"][key]
        return result if streaming_inverse else inverse(result)

    result, layout = map_tiles(values, weights.codes.shape[-1], consume, bias,
                               tile=tile, padding=padding, tile_batch=tile_batch)
    return result, {**layout, **stats}
