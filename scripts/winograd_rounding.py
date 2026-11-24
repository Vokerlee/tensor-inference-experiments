"""Bounded-rank live rounding across Winograd planes, shared by all receivers.

Only immutable transformed weights and the inverse transform prepare geometry.
Runtime decisions use current group scales, adjacent levels and r scalar error
states per source channel. They never evaluate a receiving matrix product.
The isolated-channel surrogate omits cross-channel error terms.
"""

from dataclasses import dataclass

import torch

from winograd_reference import transforms
from plane_encoding import PlaneEncoding
from winograd_support import validate as validate_support


@dataclass(frozen=True)
class PlaneGeometry:
    factor: torch.Tensor
    diagonal: torch.Tensor
    normalizer: torch.Tensor
    eigenvalues: torch.Tensor
    metric: torch.Tensor  # Audit-only full metric; not needed by runtime rounding.
    backend: str = "torch"
    active_planes: torch.Tensor | None = None

    @property
    def rank(self):
        return self.factor.shape[-1]

    def to(self, device):
        return PlaneGeometry(*(value.to(device).contiguous() for value in (
            self.factor, self.diagonal, self.normalizer, self.eigenvalues, self.metric)), backend=self.backend,
            active_planes=None if self.active_planes is None else self.active_planes.to(device))


def prepare(coefficients: torch.Tensor, tile: int = 2, rank: int = 4,
            active_planes: torch.Tensor | None = None) -> PlaneGeometry:
    planes, channels, outputs = coefficients.shape
    if planes != (tile + 2) ** 2 or not 1 <= rank <= planes or channels < 1 or outputs < 1:
        raise ValueError("valid [planes,C,O] transformed weights and rank required")
    weight = coefficients.detach().cpu().double()
    if not torch.isfinite(weight).all():
        raise ValueError("finite transformed weights required")
    inverse = transforms(tile).output
    basis = torch.einsum("ip,jq->pqij", inverse, inverse).reshape(planes, tile ** 2)
    gram = torch.einsum("pco,qco->cpq", weight, weight) * (basis @ basis.t())[None]
    if active_planes is not None:
        active_planes = active_planes.detach().cpu().bool()
        if active_planes.shape != (planes,) or not active_planes.any():
            raise ValueError("nonempty compatible active-plane mask required")
        gram *= active_planes[None, :, None] * active_planes[None, None, :]
    normalizer = gram.diagonal(dim1=1, dim2=2).amax(1).clamp_min(torch.finfo(torch.float64).tiny)
    metric = gram / normalizer[:, None, None]
    eigenvalues, directions = torch.linalg.eigh(metric)
    eigenvalues = eigenvalues.clamp_min(0)
    factor = (directions[:, :, -rank:] * eigenvalues[:, None, -rank:].sqrt()).float()
    if active_planes is not None:
        factor[:, ~active_planes] = 0
    diagonal = (metric.diagonal(dim1=1, dim2=2) - factor.double().square().sum(-1)).clamp_min(0).float()
    return PlaneGeometry(factor, diagonal, normalizer, eigenvalues, metric, active_planes=active_planes)


def encode_planes(source: torch.Tensor, geometry: PlaneGeometry, bits: int = 4,
                  group: int = 32) -> PlaneEncoding:
    validate_support(source, geometry.active_planes)
    if geometry.backend == "cuda":
        from plane_rounding_backend import encode_native
        return encode_native(source, geometry, bits, group)
    if geometry.backend != "torch":
        raise ValueError("unknown plane rounding backend")
    if (source.ndim != 3 or source.shape[0] != geometry.factor.shape[1]
            or source.shape[2] != geometry.factor.shape[0] or group < 1 or source.shape[2] % group or bits not in (4, 8) or geometry.factor.device != source.device):
        raise ValueError("compatible [planes,M,C], aligned channel groups and INT4/8 required")
    if not torch.isfinite(source).all():
        raise ValueError("finite source required")
    planes, rows, channels = source.shape
    grouped = source.float().reshape(planes, rows, channels // group, group)
    maximum = grouped.abs().amax(-1)
    limit = (1 << (bits - 1)) - 1
    exponent = torch.where(maximum > 0, (maximum / limit).log2().ceil(), 0).int()
    scale = torch.ldexp(torch.ones_like(maximum), exponent).repeat_interleave(group, dim=2)
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("scale outside finite FP32 domain")
    values = source.float() / scale
    nearest = values.round().clamp(-limit, limit)
    lower, upper = values.floor().clamp(-limit, limit), values.ceil().clamp(-limit, limit)
    state = torch.zeros(rows, channels, geometry.factor.shape[-1], device=source.device)
    nearest_state = torch.zeros_like(state)
    cost, nearest_cost = torch.zeros_like(state[..., 0]), torch.zeros_like(state[..., 0])
    codes = torch.empty_like(values)
    for plane in range(planes):
        direction = geometry.factor[:, plane]
        diagonal = geometry.diagonal[:, plane]
        low = (lower[plane] - values[plane]) * scale[plane]
        high = (upper[plane] - values[plane]) * scale[plane]
        low_state, high_state = state + low[..., None] * direction, state + high[..., None] * direction
        low_cost = low_state.square().sum(-1) + diagonal * low.square()
        high_cost = high_state.square().sum(-1) + diagonal * high.square()
        choose_low = low_cost <= high_cost
        error = torch.where(choose_low, low, high)
        codes[plane] = torch.where(choose_low, lower[plane], upper[plane])
        state = torch.where(choose_low[..., None], low_state, high_state)
        cost += diagonal * error.square()
        nearest_error = (nearest[plane] - values[plane]) * scale[plane]
        nearest_state += nearest_error[..., None] * direction
        nearest_cost += diagonal * nearest_error.square()
    candidate_cost = cost + state.square().sum(-1)
    baseline_cost = nearest_cost + nearest_state.square().sum(-1)
    selected = candidate_cost < baseline_cost
    codes = torch.where(selected[None], codes, nearest)
    reconstructed = codes * scale
    levels, counts = torch.unique(exponent, return_counts=True)
    statistics = {
        "bits": bits, "rank": geometry.factor.shape[-1], "decision_planes": planes,
        "groups": maximum.numel(), "zero_groups": int((maximum == 0).sum()),
        "candidate_selected": int(selected.sum()), "candidate_decisions": selected.numel(),
        "source_sse": float((reconstructed - source.float()).square().sum()),
        "source_zeros_changed": int(((source == 0) & (reconstructed != 0)).sum()),
        "selected_proxy_sum": float(torch.minimum(candidate_cost, baseline_cost).double().sum()),
        "nearest_proxy_sum": float(baseline_cost.double().sum()),
        "exponent_histogram": {str(int(level)): int(count) for level, count in zip(levels, counts)},
    }

    return PlaneEncoding(reconstructed, exponent, statistics)


def round_planes(source, geometry, bits=4, group=32):
    encoded = encode_planes(source, geometry, bits, group)
    return encoded.reconstructed, encoded.statistics
