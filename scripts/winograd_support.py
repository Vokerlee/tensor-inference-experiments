"""Producer-declared support for aligned nearest2x / padded3x3 transforms."""

import torch

from winograd_reference import transforms


def nearest_planes(tile=2):
    scheme = transforms(tile)
    positions = torch.arange(tile + 2).add(1).div(2, rounding_mode="floor")
    expansion = torch.nn.functional.one_hot(positions).double()
    active_axis = (scheme.input @ expansion).ne(0).any(-1)
    return (active_axis[:, None] & active_axis[None, :]).flatten()


def convolution_names(decoder):
    names = set()
    for name, layer in decoder.named_modules():
        conv = getattr(layer, "conv", None)
        if (getattr(layer, "interpolate", False) and getattr(layer, "factor", None) == 2
                and getattr(layer, "interpolation_mode", None) == "nearest"
                and isinstance(conv, torch.nn.Conv2d) and conv.kernel_size == (3, 3)
                and conv.padding == (1, 1) and conv.stride == (1, 1)
                and conv.dilation == (1, 1) and conv.groups == 1):
            names.add(f"{name}.conv" if name else "conv")
    return names


def validate(source, active):
    if active is not None:
        if active.numel() != source.shape[0] or torch.count_nonzero(source[~active]):
            raise ValueError("declared inactive planes contain a nonzero value")
