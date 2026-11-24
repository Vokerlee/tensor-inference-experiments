"""Known one-axis F(2,3) lowering for precision/utilization comparisons.

A 2x1 output tile uses four transformed products with reduction 3C. This
is a numerical reference, not a new Winograd identity or a hardware schedule.
"""

from math import ceil

import torch
import torch.nn.functional as functional

from winograd_mapping import Convolution, Interface
from winograd_reference import transforms


def transform_weights(weight):
    if weight.ndim != 4 or weight.shape[-2:] != (3, 3):
        raise ValueError("[O,C,3,3] weights required")
    scheme = transforms(dtype=weight.dtype, device=weight.device)
    values = scheme.weight @ weight
    return values.permute(2, 1, 3, 0).reshape(4, 3 * weight.shape[1], weight.shape[0]).contiguous()


def transform_inputs(tiles):
    if tiles.ndim != 4 or tiles.shape[-2:] != (4, 3):
        raise ValueError("[M,C,4,3] tiles required")
    scheme = transforms(dtype=tiles.dtype, device=tiles.device)
    values = scheme.input @ tiles
    return values.permute(2, 0, 1, 3).reshape(4, tiles.shape[0], 3 * tiles.shape[1]).contiguous()


def inverse_outputs(products):
    if products.ndim != 3 or products.shape[0] != 4:
        raise ValueError("[4,M,O] products required")
    scheme = transforms(dtype=products.dtype, device=products.device)
    return torch.einsum("ip,pmo->moi", scheme.output, products)[..., None]


def convolution(values, weight, bias=None, *, padding=1):
    """Materialized FP64 reference for independent direct-convolution checks."""
    if values.ndim != 4 or padding not in (0, 1) or values.shape[1] != weight.shape[1]:
        raise ValueError("compatible NCHW input and zero/one padding required")
    height, width = values.shape[-2] + 2 * padding - 2, values.shape[-1] + 2 * padding - 2
    if min(height, width) < 1:
        raise ValueError("empty output")
    padded = functional.pad(values.double(), (padding, padding, padding, padding + height % 2))
    tiled = padded.unfold(2, 4, 2).unfold(3, 3, 1)
    tiles = tiled.permute(0, 2, 3, 1, 4, 5).reshape(-1, values.shape[1], 4, 3)
    products = transform_inputs(tiles) @ transform_weights(weight.double())
    spatial = inverse_outputs(products)
    output = spatial.reshape(values.shape[0], ceil(height / 2), width, weight.shape[0], 2)
    output = output.permute(0, 3, 1, 4, 2).reshape(values.shape[0], weight.shape[0], -1, width)
    output = output[:, :, :height]
    return output if bias is None else output + bias.double()[None, :, None, None]


def work(shape: Convolution, interface: Interface):
    rows = shape.batch * ceil(shape.height / 2) * shape.width
    reduction = 3 * shape.channels
    segments = ceil(reduction / interface.reduction)
    vectors = rows * 4 * segments * ceil(shape.outputs / interface.columns)
    logical = rows * 4 * reduction * shape.outputs
    return {"method": "F(2,3)xDirect(1,3)", "interface": interface.name,
            "rows_per_plane": rows, "transform_planes": 4, "reduction": reduction,
            "k_segments": segments, "logical_macs": logical, "issued_vectors": vectors,
            "padded_product_slots": vectors * interface.reduction * interface.columns,
            "lhs_interface_capacity_bytes": vectors * interface.reduction * interface.input_bytes,
            "fp32_output_capacity_bytes_if_every_segment_emits": vectors * interface.columns * interface.output_bytes,
            "weight_payload_bytes": 4 * reduction * shape.outputs * interface.input_bytes}
