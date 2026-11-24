"""Causal rank1 plane rounding with immutable signed-power-of-two metadata."""

from dataclasses import dataclass

import torch

from plane_encoding import PlaneEncoding
from winograd_support import validate as validate_support


@dataclass(frozen=True)
class ShiftGeometry:
    metadata: torch.Tensor
    backend: str = "torch"

    rank: int = 1
    active_planes: torch.Tensor | None = None


def prepare(geometry, *, backend="torch") -> ShiftGeometry:
    if geometry.factor.shape[-1] != 1:
        raise ValueError("rank1 receiving geometry required")
    direction = geometry.factor[..., 0].double()
    magnitude = direction.abs()
    exponent = magnitude.clamp_min(2.**-126).log2().floor().clamp(-126, 0).int()
    code = torch.where(magnitude >= 2.**-126, (1 - exponent) * direction.sign().int(), 0)
    diagonal = geometry.metric.diagonal(dim1=1, dim2=2).double().clamp_min(0)
    diagonal_exponent = diagonal.clamp_min(2.**-126).log2().ceil().clamp(-126, 0).int()
    diagonal_code = torch.where(diagonal > 0, 1 - diagonal_exponent, 0)
    packed = ((diagonal_code << 8) | (code & 255)).short().contiguous()
    return ShiftGeometry(packed, backend, active_planes=geometry.active_planes)


def reference(source: torch.Tensor, metadata: torch.Tensor, bits: int = 4, group: int = 256):
    if (source.ndim != 3 or group not in (32, 128, 256) or source.shape[-1] % group
            or metadata.shape != (source.shape[-1], source.shape[0]) or bits not in (4, 8)):
        raise ValueError("compatible source, packed metadata and aligned group required")
    packed = metadata.int()
    raw = packed & 255
    direction = torch.where(raw >= 128, raw - 256, raw)
    diagonal = (packed >> 8) & 255
    if (packed < 0).any() or (direction == -128).any() or (diagonal > 127).any():
        raise ValueError("invalid packed metadata")
    planes, rows, channels = source.shape
    maximum = source.float().reshape(planes, rows, -1, group).abs().amax(-1)
    limit = (1 << (bits - 1)) - 1
    exponent = torch.where(maximum > 0, (maximum / limit).log2().ceil(), 0).int()
    expanded = exponent.repeat_interleave(group, dim=2)
    state = torch.zeros(rows, channels, device=source.device)
    output = torch.empty_like(source, dtype=torch.float32)
    changed = torch.zeros(rows, channels, device=source.device, dtype=torch.bool)
    for plane in range(planes):
        original = source[plane].float()
        scaled = torch.ldexp(original, -expanded[plane])
        low, high = scaled.floor().clamp(-limit, limit), scaled.ceil().clamp(-limit, limit)
        nearest = scaled.round().clamp(-limit, limit)
        low_value, high_value = torch.ldexp(low, expanded[plane]), torch.ldexp(high, expanded[plane])
        low_error, high_error = low_value - original, high_value - original
        current = direction[:, plane]
        projected = torch.where(current != 0, torch.ldexp(state, 1 - current.abs()), 0)
        projected = torch.where(current < 0, -projected, projected)
        h = diagonal[:, plane]
        term = torch.where(h != 0, torch.ldexp(low_error + high_error, 1 - h), 0)
        criterion = torch.ldexp(projected, torch.ones_like(current)) + term
        choose_low = (criterion > 0) | ((criterion == 0) & (nearest == low))
        error = torch.where(choose_low, low_error, high_error)
        update = torch.where(current != 0, torch.ldexp(error, 1 - current.abs()), 0)
        state = state + torch.where(current < 0, -update, update)
        output[plane] = torch.where(choose_low, low_value, high_value)
        changed |= torch.where(choose_low, low, high) != nearest
    if not torch.isfinite(output).all() or not torch.isfinite(state).all():
        raise ValueError("nonfinite shift rounding arithmetic")
    return output, exponent, changed


def encode_shift(source, geometry: ShiftGeometry, bits=4, group=256):
    validate_support(source, geometry.active_planes)
    if geometry.backend == "cuda":
        from integer_segment_backend import load_native
        load_native()
        output, exponent, changed, invalid = torch.ops.tensor_inference_engine.plane_shift_rounding(
            source.contiguous(), geometry.metadata, bits, group, False)
        if int(invalid.item()):
            raise ValueError("native shift rounding reported invalid arithmetic")
    elif geometry.backend == "torch":
        output, exponent, changed = reference(source, geometry.metadata, bits, group)
    else:
        raise ValueError("unknown shift rounding backend")
    levels, counts = torch.unique(exponent, return_counts=True)
    statistics = {"bits": bits, "rank": geometry.rank,
                    "mode": "power2-greedy" if geometry.rank else "nearest", "backend": geometry.backend,
                    "groups": exponent.numel(), "zero_groups": int((source.reshape(
                        source.shape[0], source.shape[1], -1, group).abs().amax(-1) == 0).sum()),
                    "changed_channels": int(changed.sum()), "candidate_decisions": changed.numel() if geometry.rank else 0,
                    "source_sse": float((output - source.float()).double().square().sum()),
                    "source_zeros_changed": int(((source == 0) & (output != 0)).sum()),
                    "exponent_histogram": {str(int(level)): int(count) for level, count in zip(levels, counts)}}

    return PlaneEncoding(output, exponent, statistics)


def round_shift(source, geometry, bits=4, group=256):
    encoded = encode_shift(source, geometry, bits, group)
    return encoded.reconstructed, encoded.statistics
