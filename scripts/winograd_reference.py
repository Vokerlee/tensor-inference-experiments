"""Small-transform Winograd references for numerical and mapping experiments.

These references materialize transformed tiles and use Torch products. They do
not assert a physical array schedule, device energy or bit equality with a
specific accelerator accumulation graph.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class Transforms:
    input: torch.Tensor
    weight: torch.Tensor
    output: torch.Tensor

    @property
    def tile(self) -> int:
        return self.output.shape[0]

    @property
    def extent(self) -> int:
        return self.input.shape[0]


def transforms(tile: int = 2, *, dtype=torch.float64, device="cpu") -> Transforms:
    if tile == 2:
        input_rows = [[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]]
        weight_rows = [[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]]
        output_rows = [[1, 1, 1, 0], [0, 1, -1, -1]]
    elif tile == 4:
        input_rows = [[4, 0, -5, 0, 1, 0], [0, -4, -4, 1, 1, 0],
                      [0, 4, -4, -1, 1, 0], [0, -2, -1, 2, 1, 0],
                      [0, 2, -1, -2, 1, 0], [0, 4, 0, -5, 0, 1]]
        weight_rows = [[0.25, 0, 0], [-1/6, -1/6, -1/6], [-1/6, 1/6, -1/6],
                       [1/24, 1/12, 1/6], [1/24, -1/12, 1/6], [0, 0, 1]]
        output_rows = [[1, 1, 1, 1, 1, 0], [0, 1, -1, 2, -2, 0],
                       [0, 1, 1, 4, 4, 0], [0, 1, -1, 8, -8, 1]]
    else:
        raise ValueError("reference supports F(2,3) and F(4,3)")
    return Transforms(*(torch.tensor(rows, dtype=dtype, device=device)
                        for rows in (input_rows, weight_rows, output_rows)))


def transform_weights(weight: torch.Tensor, scheme: Transforms) -> torch.Tensor:
    if weight.ndim != 4 or weight.shape[-2:] != (3, 3):
        raise ValueError("dense [O,C,3,3] weights required")
    values = scheme.weight @ weight.to(scheme.weight.dtype) @ scheme.weight.t()
    return values.permute(2, 3, 1, 0).reshape(scheme.extent ** 2, weight.shape[1], weight.shape[0]).contiguous()


def transform_inputs(tiles: torch.Tensor, scheme: Transforms) -> torch.Tensor:
    if tiles.ndim != 4 or tiles.shape[-2:] != (scheme.extent, scheme.extent):
        raise ValueError("[M,C,extent,extent] input tiles required")
    values = scheme.input @ tiles.to(scheme.input.dtype) @ scheme.input.t()
    return values.permute(2, 3, 0, 1).reshape(scheme.extent ** 2, tiles.shape[0], tiles.shape[1]).contiguous()


def inverse_outputs(products: torch.Tensor, scheme: Transforms) -> torch.Tensor:
    if products.ndim != 3 or products.shape[0] != scheme.extent ** 2:
        raise ValueError("[extent**2,M,O] products required")
    values = products.reshape(scheme.extent, scheme.extent, *products.shape[1:]).permute(2, 3, 0, 1)
    return scheme.output @ values @ scheme.output.t()


def input_tiles(values: torch.Tensor, tile: int, padding: int = 1):
    if values.ndim != 4 or padding not in (0, 1) or tile not in (2, 4):
        raise ValueError("NCHW input, zero/one padding and tile2/4 required")
    height, width = values.shape[-2] + 2 * padding - 2, values.shape[-1] + 2 * padding - 2
    if min(height, width) < 1:
        raise ValueError("empty convolution output")
    extra_height, extra_width = (-height) % tile, (-width) % tile
    padded = functional.pad(values, (padding, padding + extra_width, padding, padding + extra_height))
    tiled = padded.unfold(2, tile + 2, tile).unfold(3, tile + 2, tile)
    shape = (values.shape[0], height, width, (height + tile - 1) // tile, (width + tile - 1) // tile)
    # This materialized reference is intentionally separate from a bounded-memory mapping.
    return tiled.permute(0, 2, 3, 1, 4, 5).reshape(-1, values.shape[1], tile + 2, tile + 2), shape


def restore_tiles(values: torch.Tensor, shape: tuple, tile: int):
    batch, height, width, rows, columns = shape
    restored = values.reshape(batch, rows, columns, values.shape[1], tile, tile)
    return restored.permute(0, 3, 1, 4, 2, 5).reshape(batch, values.shape[1], rows * tile, columns * tile)[:, :, :height, :width]


def convolution(values: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
                *, tile: int = 2, padding: int = 1, dtype=torch.float64) -> torch.Tensor:
    scheme = transforms(tile, dtype=dtype, device=values.device)
    tiles, shape = input_tiles(values, tile, padding)
    transformed = transform_inputs(tiles, scheme)
    coefficients = transform_weights(weight, scheme)
    output = restore_tiles(inverse_outputs(transformed @ coefficients, scheme), shape, tile)
    if bias is not None:
        output = output + bias.to(dtype)[None, :, None, None]
    return output
