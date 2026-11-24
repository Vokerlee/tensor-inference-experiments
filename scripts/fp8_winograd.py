"""Block-scaled FP8 Winograd numerical control for a K256/N128 interface.

Decoded FP8 values are exactly representable in BF16. Torch BF16 products with
FP32 output provide a numerical oracle, not native FP8 timing or a specified
accelerator's accumulation order. Unlike INT8 partials, these sums are not
claimed to be exactly representable in FP32.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from winograd_reference import inverse_outputs, transforms
from winograd_streaming import map_tiles, product


LIMIT = 448


def quantize(grouped, dimension):
    maximum = grouped.abs().amax(dimension)
    exponent = torch.where(maximum > 0, (maximum / LIMIT).log2().ceil(), 0).int()
    scale = torch.ldexp(torch.ones_like(maximum), exponent)
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("FP8 scale outside finite FP32 domain")
    normalized = grouped / scale.unsqueeze(dimension)
    decoded = normalized.clamp(-LIMIT, LIMIT).to(torch.float8_e4m3fn).to(torch.bfloat16)
    if not torch.isfinite(decoded).all():
        raise ValueError("nonfinite FP8 reconstruction")
    return decoded, exponent


@dataclass(frozen=True)
class Weights:
    codes: torch.Tensor  # Decoded FP8 numeric values, held exactly in BF16.
    exponents: torch.Tensor
    channels: int
    block: int = 256

    def reconstructed(self):
        planes, padded, outputs = self.codes.shape
        grouped = self.codes.float().reshape(planes, -1, self.block, outputs)
        scales = torch.ldexp(torch.ones_like(self.exponents, dtype=torch.float32), self.exponents)
        return (grouped * scales[:, :, None, :]).reshape(planes, padded, outputs)


def prepare(coefficients):
    if coefficients.ndim != 3 or min(coefficients.shape) < 1 or not torch.isfinite(coefficients).all():
        raise ValueError("finite nonempty [P,C,O] weights required")
    planes, channels, outputs = coefficients.shape
    padded = functional.pad(coefficients.float(), (0, 0, 0, (-channels) % 256))
    decoded, exponent = quantize(padded.reshape(planes, -1, 256, outputs), 2)
    return Weights(decoded.reshape(planes, -1, outputs), exponent, channels)


def multiply(source, weights):
    if (source.ndim != 3 or source.shape[0] != weights.codes.shape[0]
            or source.shape[-1] != weights.channels or not torch.isfinite(source).all()):
        raise ValueError("compatible finite source required")
    padded = functional.pad(source.float(), (0, weights.codes.shape[1] - source.shape[-1]))
    grouped = padded.reshape(*padded.shape[:2], -1, 256)
    codes, exponent = quantize(grouped, -1)
    scale = torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent)
    weight_scale = torch.ldexp(torch.ones_like(weights.exponents, dtype=torch.float32), weights.exponents)
    accumulated = None
    for group in range(grouped.shape[-2]):
        start = group * 256
        partial = product(codes[:, :, group].contiguous(), weights.codes[:, start:start + 256].contiguous())
        partial = partial * scale[:, :, group, None] * weight_scale[:, group, None, :]
        accumulated = partial if accumulated is None else accumulated + partial
    if not torch.isfinite(accumulated).all():
        raise ValueError("nonfinite dequantized FP8 sum")
    levels, counts = torch.unique(exponent, return_counts=True)
    return accumulated, {
        "activation_groups": exponent.numel(),
        "zero_groups": int((grouped.abs().amax(-1) == 0).sum()),
        "exponent_histogram": {str(int(level)): int(count) for level, count in zip(levels, counts)},
        "fp8_partial_vectors": source.shape[0] * source.shape[1] * grouped.shape[-2]
                               * ((weights.codes.shape[-1] + 127) // 128),
        "output_dequant_scalar_multiplications": 2 * source.shape[0] * source.shape[1]
                                                * grouped.shape[-2] * weights.codes.shape[-1],
    }


def convolution(values, weights, bias=None, *, tile=2, padding=1, tile_batch=64):
    scheme = transforms(tile, dtype=torch.float32, device=values.device)
    statistics = {"operand_format": "fp8_e4m3fn", "lowering": "two-axis", "activation_bits": 8,
                  "weight_bits": 8, "K": 256, "N": 128, "original_channels": weights.channels,
                  "padded_channels": weights.codes.shape[1], "activation_groups": 0,
                  "zero_groups": 0, "fp8_partial_vectors": 0,
                  "output_dequant_scalar_multiplications": 0, "exponent_histogram": {}}

    def consume(source):
        result, current = multiply(source, weights)
        for key in ("activation_groups", "zero_groups", "fp8_partial_vectors",
                    "output_dequant_scalar_multiplications"):
            statistics[key] += current[key]
        for exponent, count in current["exponent_histogram"].items():
            histogram = statistics["exponent_histogram"]
            histogram[exponent] = histogram.get(exponent, 0) + count
        return inverse_outputs(result, scheme)

    output, layout = map_tiles(values, weights.codes.shape[-1], consume, bias, tile=tile,
                               padding=padding, tile_batch=tile_batch)
    return output, {**layout, **statistics}
