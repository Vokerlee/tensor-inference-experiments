"""Thin experiment adapter for the engine's generic floating plane operator."""

import torch

from integer_segment_backend import load_native
from plane_encoding import PlaneEncoding


def encode_native(source, geometry, bits, group):
    load_native()
    output, exponent, selected, invalid = torch.ops.tensor_inference_engine.plane_rounding(
        source.contiguous(), geometry.factor.contiguous(), geometry.diagonal.contiguous(),
        bits, group, False)
    if int(invalid.item()):
        raise ValueError("native plane rounding reported invalid arithmetic")
    levels, counts = torch.unique(exponent, return_counts=True)
    statistics = {
        "bits": bits, "rank": geometry.rank, "mode": "completed-proxy", "backend": "cuda-rn-scalar",
        "groups": exponent.numel(), "zero_groups": int((source.reshape(
            source.shape[0], source.shape[1], -1, group).abs().amax(-1) == 0).sum()),
        "candidate_selected": int(selected.sum()), "candidate_decisions": selected.numel(),
        "source_sse": float((output - source.float()).double().square().sum()),
        "source_zeros_changed": int(((source == 0) & (output != 0)).sum()),
        "exponent_histogram": {str(int(level)): int(count) for level, count in zip(levels, counts)},
    }

    return PlaneEncoding(output, exponent, statistics)
