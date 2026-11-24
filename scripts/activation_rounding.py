"""Reference group rounding with bounded receiving-geometry state.

Only immutable weights prepare the geometry. Runtime decisions use current
activations and a few scalar feedback accumulators, never a receiving GEMM.
This is an experimental reference, not a target-hardware timing model.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GroupGeometry:
    factor: torch.Tensor
    diagonal: torch.Tensor
    residual_norm: torch.Tensor
    normalizer: torch.Tensor
    eigenvalues: torch.Tensor

    @property
    def group_size(self) -> int:
        return self.factor.shape[1]

    @property
    def rank(self) -> int:
        return self.factor.shape[2]

    def to(self, device) -> "GroupGeometry":
        return GroupGeometry(*(value.to(device) for value in (
            self.factor, self.diagonal, self.residual_norm,
            self.normalizer, self.eigenvalues)))


def prepare_geometry(weight: torch.Tensor, rank: int, group_size: int = 32) -> GroupGeometry:
    """Prepare one shared source-group metric across all receiving output columns.

    ``weight[K,N]`` may concatenate convolution positions in N. This represents
    isolated group contributions; spatial/group cross terms are not discarded
    from the actual modified-model evaluation.
    """
    if (group_size < 1 or weight.ndim != 2 or weight.shape[0] % group_size or not weight.numel()
            or not 1 <= rank <= group_size or not torch.isfinite(weight).all()):
        raise ValueError("finite nonempty aligned [K,N] weight and valid rank required")
    blocks = weight.detach().cpu().double().reshape(-1, group_size, weight.shape[1])
    gram = blocks @ blocks.transpose(1, 2)
    normalizer = gram.diagonal(dim1=1, dim2=2).amax(1).clamp_min(torch.finfo(torch.float64).tiny)
    gram = gram / normalizer[:, None, None]
    eigenvalues, directions = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.clamp_min(0)
    factor = (directions[:, :, -rank:] * eigenvalues[:, None, -rank:].sqrt()).float()
    diagonal = (gram.diagonal(dim1=1, dim2=2) - factor.double().square().sum(-1)).clamp_min(0).float()
    approximation = factor.double() @ factor.double().transpose(1, 2) + torch.diag_embed(diagonal.double())
    residual_norm = torch.linalg.matrix_norm(gram - approximation, ord=2)
    return GroupGeometry(factor, diagonal, residual_norm, normalizer, eigenvalues)


@dataclass
class RoundedGroups:
    reconstructed: torch.Tensor
    codes: torch.Tensor
    exponents: torch.Tensor
    candidate_selected: torch.Tensor
    proxy_candidate: torch.Tensor
    proxy_nearest: torch.Tensor
    spectral_margin: torch.Tensor


def group_grid(source: torch.Tensor, group_size: int, bits: int):
    if group_size < 1 or source.ndim != 2 or source.shape[1] % group_size or bits not in (4, 8):
        raise ValueError("aligned [M,K] source and bits4/8 required")
    if not torch.isfinite(source).all():
        raise ValueError("finite source required")
    values = source.float().reshape(source.shape[0], -1, group_size)
    limit = (1 << (bits - 1)) - 1
    maximum = values.abs().amax(-1)
    exponent = torch.where(maximum > 0, (maximum / limit).log2().ceil(), 0).int()
    scale = torch.ldexp(torch.ones_like(maximum), exponent)
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("scale outside finite FP32 domain")
    return values / scale[:, :, None], exponent, scale, limit


def round_groups(source: torch.Tensor, geometry: GroupGeometry, bits: int = 4,
                 selection: str = "proxy") -> RoundedGroups:
    """Round adjacent levels using a fixed-rank running error projection.

    ``proxy`` chooses the candidate only when its completed surrogate cost is
    below nearest rounding. ``spectral`` additionally applies a real-arithmetic
    tail screen evaluated in FP64; this is a diagnostic, not a finite-arithmetic
    hardware certificate. ``candidate`` exposes the raw greedy result.
    """
    if selection not in ("proxy", "spectral", "candidate"):
        raise ValueError("unknown completed-group selection")
    values, exponents, scale, limit = group_grid(source, geometry.group_size, bits)
    if geometry.factor.shape[0] != values.shape[1] or geometry.factor.device != source.device:
        raise ValueError("geometry groups/device must match source")
    nearest = values.round().clamp(-limit, limit)
    lower = values.floor().clamp(-limit, limit)
    upper = values.ceil().clamp(-limit, limit)
    state = torch.zeros(*values.shape[:2], geometry.rank, device=source.device)
    nearest_state = torch.zeros_like(state)
    diagonal_cost = torch.zeros_like(scale)
    nearest_diagonal_cost = torch.zeros_like(scale)
    codes = torch.empty_like(values)
    for channel in range(geometry.group_size):
        direction = geometry.factor[:, channel]
        diagonal = geometry.diagonal[:, channel]
        low_error = lower[:, :, channel] - values[:, :, channel]
        high_error = upper[:, :, channel] - values[:, :, channel]
        inner = (state * direction).sum(-1)
        energy = direction.square().sum(-1) + diagonal
        difference = (2 * inner * (high_error - low_error)
                      + energy * (high_error.square() - low_error.square()))
        code = torch.where(difference < 0, upper[:, :, channel], lower[:, :, channel])
        code = torch.where(difference == 0, nearest[:, :, channel], code)
        error = code - values[:, :, channel]
        nearest_error = nearest[:, :, channel] - values[:, :, channel]
        state += error[:, :, None] * direction
        nearest_state += nearest_error[:, :, None] * direction
        diagonal_cost += diagonal * error.square()
        nearest_diagonal_cost += diagonal * nearest_error.square()
        codes[:, :, channel] = code
    proxy_candidate = state.square().sum(-1) + diagonal_cost
    proxy_nearest = nearest_state.square().sum(-1) + nearest_diagonal_cost
    difference = (codes - nearest).double()
    total_error = (codes + nearest - 2 * values).double()
    tail = geometry.residual_norm * (difference.square().sum(-1) * total_error.square().sum(-1)).sqrt()
    margin = proxy_nearest.double() - proxy_candidate.double() - tail
    if selection == "candidate":
        selected = torch.ones_like(scale, dtype=torch.bool)
    elif selection == "spectral":
        selected = margin > 0
    else:
        selected = proxy_candidate < proxy_nearest
    chosen = torch.where(selected[:, :, None], codes, nearest)
    reconstructed = (chosen * scale[:, :, None]).reshape_as(source).to(source.dtype)
    if not torch.isfinite(reconstructed).all():
        raise ValueError("reconstruction outside source format")
    return RoundedGroups(reconstructed, chosen.to(torch.int8), exponents, selected,
                         proxy_candidate, proxy_nearest, margin)
