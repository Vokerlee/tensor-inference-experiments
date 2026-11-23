"""Source-group quantization and reused-statistic lowering for dense convolutions."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from accumulation_scale_diagnostics import encode
from convolution_geometry import valid_axis
from primitive_accumulation_reference import Counters, multiply


@dataclass(frozen=True)
class SourceBlocks:
    codes: torch.Tensor       # B, H, W, C/32, 32
    exponents: torch.Tensor   # B, H, W, C/32
    maximum: torch.Tensor     # Exact maximum absolute encoded value per source group.
    l1: torch.Tensor | None


def encode_source(values: torch.Tensor, format_name: str = "int8",
                  *, bound_kind: str = "min") -> SourceBlocks:
    """Encode one channel group at one source coordinate, before any reuse."""
    if values.ndim != 4 or values.shape[1] % 32:
        raise ValueError("source reference requires NCHW and channels divisible by32")
    if bound_kind not in {"min", "max"}:
        raise ValueError("bound kind must be min or max")
    codes, exponents = encode(values.permute(0, 2, 3, 1).contiguous(), format_name)
    return SourceBlocks(codes, exponents, codes.abs().amax(-1),
                        codes.abs().sum(-1) if bound_kind == "min" else None)


def encode_weight(layer: torch.nn.Conv2d):
    """R,S,C order aligns consumer groups with source channel groups."""
    if layer.groups != 1 or layer.in_channels % 32 or layer.padding_mode != "zeros":
        raise ValueError("dense, zero-padded convolutions with aligned channels only")
    weight = layer.weight.detach().permute(0, 2, 3, 1).contiguous()
    return encode(weight.reshape(layer.out_channels, -1), "int8")


def lower(source: SourceBlocks, layer: torch.nn.Conv2d):
    """Move codes and existing metadata into matching R,S,C consumer groups.

    This Torch oracle materializes lowering for verification. A streaming feeder
    would gather these same source groups and metadata, rather than allocate the
    expanded tensors. This function does not claim that such storage is free.
    """
    batch, height, width, channel_groups, _ = source.codes.shape
    channels = channel_groups * 32
    options = dict(kernel_size=layer.kernel_size, dilation=layer.dilation,
                   padding=layer.padding, stride=layer.stride)
    values = source.codes.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
    expanded = F.unfold(values.float(), **options)
    positions = expanded.shape[-1]
    taps = layer.kernel_size[0] * layer.kernel_size[1]
    codes = expanded.reshape(batch, channel_groups, 32, taps, positions)
    codes = codes.permute(0, 4, 3, 1, 2).reshape(batch * positions, taps * channel_groups, 32).long()

    def gather(metadata):
        nchw = metadata.permute(0, 3, 1, 2).float()
        expanded = F.unfold(nchw, **options)
        return expanded.reshape(batch, channel_groups, taps, positions).permute(
            0, 3, 2, 1).reshape(batch * positions, taps * channel_groups)

    exponent = gather(source.exponents).to(torch.int32)
    maximum = gather(source.maximum).long()
    l1 = gather(source.l1).long() if source.l1 is not None else None
    return codes, exponent, (maximum, l1)


def convolve(values: torch.Tensor, layer: torch.nn.Conv2d, prepared, *,
             activation_format: str = "int8", reduction_width: int = 128,
             accumulator_width: int | None = None, counters: Counters | None = None,
             bound_kind: str = "min") -> torch.Tensor:
    source = encode_source(values, activation_format, bound_kind=bound_kind)
    codes, exponent, statistics = lower(source, layer)
    weight, weight_exponent = prepared
    output = multiply(codes, exponent, weight, weight_exponent,
                      reduction_width=reduction_width, accumulator_width=accumulator_width,
                      counters=counters, a_statistics=statistics, bound_kind=bound_kind)
    if layer.bias is not None:
        output += layer.bias.float()
    h = (values.shape[2] + 2 * layer.padding[0]
         - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) // layer.stride[0] + 1
    w = (values.shape[3] + 2 * layer.padding[1]
         - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1) // layer.stride[1] + 1
    if counters is not None:
        counters.source_groups_quantized += source.exponents.numel()
        counters.lowered_group_slots += exponent.numel()
        valid_h = valid_axis(values.shape[2], h, layer.kernel_size[0], layer.stride[0],
                             layer.dilation[0], layer.padding[0])
        valid_w = valid_axis(values.shape[3], w, layer.kernel_size[1], layer.stride[1],
                             layer.dilation[1], layer.padding[1])
        counters.valid_lowered_group_uses += values.shape[0] * (values.shape[1] // 32) * valid_h * valid_w
    return output.reshape(values.shape[0], h, w, layer.out_channels).permute(
        0, 3, 1, 2).contiguous().to(values.dtype)
