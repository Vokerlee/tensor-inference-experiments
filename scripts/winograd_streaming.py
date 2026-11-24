"""Bounded tile-batch Winograd numerical oracle and live activation quantizer.

Torch matmuls implement a numerical experiment, not an array timing model.
The alternative inverse-before-K-sum schedule exposes a possible retention
tradeoff without assuming that its extra transforms or changed rounding are free.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from winograd_reference import inverse_outputs, transform_inputs, transform_weights, transforms


@dataclass(frozen=True)
class PreparedWeights:
    coefficients: torch.Tensor
    tile: int
    operand_dtype: torch.dtype


def prepare(weight: torch.Tensor, tile: int, operand_dtype=torch.bfloat16) -> PreparedWeights:
    scheme = transforms(tile, dtype=torch.float32, device=weight.device)
    return PreparedWeights(transform_weights(weight, scheme).to(operand_dtype), tile, operand_dtype)


def encode(values: torch.Tensor, bits: int | None, group: int = 32):
    if bits is None:
        return values, {"groups": 0, "zero_groups": 0, "exponent_histogram": {}}
    if bits not in (4, 8) or group < 1 or values.shape[-1] % group:
        raise ValueError("INT4/8 encoding requires an aligned positive channel group")
    grouped = values.float().reshape(*values.shape[:-1], -1, group)
    maximum = grouped.abs().amax(-1)
    limit = (1 << (bits - 1)) - 1
    exponents = torch.where(maximum > 0, (maximum / limit).log2().ceil(), 0).int()
    scales = torch.ldexp(torch.ones_like(maximum), exponents)
    codes = (grouped / scales[..., None]).round().clamp(-limit, limit)
    result = (codes * scales[..., None]).reshape(values.shape)
    levels, counts = torch.unique(exponents, return_counts=True)
    return result, {"groups": maximum.numel(), "zero_groups": int((maximum == 0).sum()),
                    "exponent_histogram": {str(int(level)): int(count) for level, count in zip(levels, counts)}}


def product(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    if lhs.dtype == torch.bfloat16 and lhs.device.type == "cuda":
        return torch.bmm(lhs, rhs, out_dtype=torch.float32)
    return torch.bmm(lhs.float(), rhs.float())


def extract(padded: torch.Tensor, first: int, last: int, rows: int, columns: int,
            tile: int) -> torch.Tensor:
    indices = torch.arange(first, last, device=padded.device)
    batch = indices // (rows * columns)
    top = (indices // columns) % rows * tile
    left = indices % columns * tile
    channels = torch.arange(padded.shape[1], device=padded.device)
    offset = torch.arange(tile + 2, device=padded.device)
    return padded[batch[:, None, None, None], channels[None, :, None, None],
                  top[:, None, None, None] + offset[None, None, :, None],
                  left[:, None, None, None] + offset[None, None, None, :]]


def map_tiles(values: torch.Tensor, outputs: int, consumer, bias: torch.Tensor | None = None,
              *, tile: int = 2, padding: int = 1, tile_batch: int = 64, source_transform=None):
    """Transform bounded input batches and store a consumer's actual spatial tiles."""
    if (values.ndim != 4 or min(values.shape) < 1 or outputs < 1 or tile not in (2, 4)
            or padding not in (0, 1) or tile_batch < 1):
        raise ValueError("nonempty NCHW source, valid transform/padding and positive tile batch required")
    height, width = values.shape[-2] + 2 * padding - 2, values.shape[-1] + 2 * padding - 2
    if min(height, width) < 1:
        raise ValueError("nonempty convolution output required")
    rows, columns = (height + tile - 1) // tile, (width + tile - 1) // tile
    padded = functional.pad(values, (padding, padding + (-width) % tile,
                                     padding, padding + (-height) % tile))
    scheme = transforms(tile, dtype=torch.float32, device=values.device)
    output = torch.empty(values.shape[0], outputs, rows * tile, columns * tile,
                         dtype=values.dtype, device=values.device)
    stats = {"tile": tile, "tile_batch": tile_batch, "tile_batches": 0,
             "maximum_abs_transformed_input": 0.0}
    count = values.shape[0] * rows * columns
    for first in range(0, count, tile_batch):
        last = min(first + tile_batch, count)
        spatial_tiles = extract(padded, first, last, rows, columns, tile)
        transformed = (transform_inputs(spatial_tiles, scheme) if source_transform is None
                       else source_transform(spatial_tiles))
        stats["maximum_abs_transformed_input"] = max(stats["maximum_abs_transformed_input"],
                                                       float(transformed.abs().max()))
        result = consumer(transformed)
        if result.shape != (last - first, outputs, tile, tile):
            raise ValueError("consumer returned incompatible spatial tiles")
        if bias is not None:
            result = result + bias.float()[None, :, None, None]
        if not torch.isfinite(result).all():
            raise ValueError("nonfinite Winograd output")
        indices = torch.arange(first, last, device=values.device)
        batch = indices // (rows * columns)
        top, left = (indices // columns) % rows * tile, indices % columns * tile
        channels = torch.arange(outputs, device=values.device)
        offset = torch.arange(tile, device=values.device)
        output[batch[:, None, None, None], channels[None, :, None, None],
               top[:, None, None, None] + offset[None, None, :, None],
               left[:, None, None, None] + offset[None, None, None, :]] = result.to(values.dtype)
        stats["tile_batches"] += 1
    return output[:, :, :height, :width].contiguous(), stats


def convolution(values: torch.Tensor, prepared: PreparedWeights,
                bias: torch.Tensor | None = None, *, padding: int = 1, tile_batch: int = 64,
                bits: int | None = None, reduction_block: int | None = None,
                streaming_inverse: bool = False) -> tuple[torch.Tensor, dict]:
    if reduction_block is not None and reduction_block < 1:
        raise ValueError("positive reduction block required")
    if streaming_inverse and reduction_block is None:
        raise ValueError("streaming inverse requires an explicit K block")
    scheme = transforms(prepared.tile, dtype=torch.float32, device=values.device)
    statistics = {"bits": bits, "reduction_block": reduction_block,
                  "streaming_inverse": streaming_inverse, "operand_dtype": str(prepared.operand_dtype),
                  "groups": 0, "zero_groups": 0, "exponent_histogram": {}}

    def consume(transformed):
        quantized, current = encode(transformed, bits)
        lhs = quantized.to(prepared.operand_dtype)
        channels, accumulated = lhs.shape[-1], None
        for start in range(0, channels, reduction_block or channels):
            end = start + (reduction_block or channels)
            partial = product(lhs[..., start:end].contiguous(), prepared.coefficients[:, start:end].contiguous())
            if streaming_inverse:
                partial = inverse_outputs(partial, scheme)
            accumulated = partial if accumulated is None else accumulated + partial
        for key in ("groups", "zero_groups"):
            statistics[key] += current[key]
        for exponent, frequency in current["exponent_histogram"].items():
            histogram = statistics["exponent_histogram"]
            histogram[exponent] = histogram.get(exponent, 0) + frequency
        return accumulated if streaming_inverse else inverse_outputs(accumulated, scheme)

    result, layout = map_tiles(values, prepared.coefficients.shape[-1], consume, bias,
                               tile=prepared.tile, padding=padding, tile_batch=tile_batch)
    return result, {**layout, **statistics}
