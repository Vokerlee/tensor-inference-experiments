"""Bounded one-axis numerical oracle using the same K256 integer contract."""

import torch

from integer_winograd import multiply
from winograd_one_axis import inverse_outputs, transform_inputs
from winograd_streaming import map_tiles


def convolution(values, weights, geometry, bias=None, *, padding=1, tile_batch=64, bits=8):
    statistics = {"lowering": "one-axis", "activation_bits": bits, "weight_bits": 8,
                  "K": weights.block, "N": 128, "original_channels": values.shape[1],
                  "lowered_reduction": weights.channels,
                  "padded_channels": weights.codes.shape[1], "activation_groups": 0,
                  "zero_groups": 0, "integer_partial_vectors": 0,
                  "output_dequant_scalar_multiplications": 0, "exponent_histogram": {},
                  "rounding": None}

    def source_transform(tiles):
        strips = torch.cat((tiles[..., :3], tiles[..., 1:]), dim=0)
        return transform_inputs(strips)

    def consume(source):
        products, current = multiply(source, weights, bits=bits, geometry=geometry)
        for key in ("activation_groups", "zero_groups", "integer_partial_vectors",
                    "output_dequant_scalar_multiplications"):
            statistics[key] += current[key]
        for exponent, count in current["exponent_histogram"].items():
            histogram = statistics["exponent_histogram"]
            histogram[exponent] = histogram.get(exponent, 0) + count
        if statistics["rounding"] is None:
            statistics["rounding"] = {"mode": "nearest", "backend": geometry.backend,
                                      "bits": bits, "rank": 0}
        for key in ("source_zeros_changed", "source_sse", "groups", "zero_groups"):
            statistics["rounding"][key] = statistics["rounding"].get(key, 0) + current["rounding"][key]
        result = inverse_outputs(products)[..., 0]
        count = result.shape[0] // 2
        return torch.stack((result[:count], result[count:]), dim=-1)

    output, layout = map_tiles(values, weights.codes.shape[-1], consume, bias, tile=2,
                               padding=padding, tile_batch=tile_batch, source_transform=source_transform)
    return output, {**layout, **statistics}
