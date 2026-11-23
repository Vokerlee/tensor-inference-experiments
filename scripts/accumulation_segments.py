"""Pre-issue contiguous chunk selection from conservative group metadata."""

import torch


def shifted_sum(bounds: torch.Tensor, shifts: torch.Tensor, width: int):
    """Saturate a nonnegative two-term sum without overflowing INT64."""
    limit = (1 << (width - 1)) - 1
    safe = shifts.clamp(0, width - 1).long()
    fits = ((bounds == 0) | (shifts < width - 1)) & (bounds <= (limit >> safe))
    terms = torch.where(fits, bounds, 0) << safe
    total = terms.sum(-1)
    accepted = fits.all(-1) & (total <= limit)
    return accepted, torch.where(accepted, total, limit + 1)


def greedy_ends(bounds: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor,
                width: int) -> torch.Tensor:
    """Return chunk-end flags; extend each prefix while its absolute bound fits.

    The last axis contains at most eight groups of one matrix primitive. A
    single active group that cannot fit stays a floating group, with zero groups
    bypassed on either path. The procedure
    uses only bounds/exponents, never a receiving matrix product. Feasibility
    is hereditary under interval trimming, so longest feasible prefixes minimize
    chunk count for this declared contiguous-partition objective. This is a
    standard greedy property, not a claimed new theorem.
    """
    if (bounds.shape != lower.shape or bounds.shape != upper.shape
            or not 1 <= bounds.shape[-1] <= 8 or not 2 <= width <= 48):
        raise ValueError("matching shapes, one to eight groups, and width2..48 required")
    if torch.any(bounds < 0) or torch.any((bounds > 0) & (upper < lower)):
        raise ValueError("nonnegative bounds and ordered active exponents required")
    ends = torch.zeros_like(bounds, dtype=torch.bool)
    accumulated = torch.zeros_like(bounds[..., 0])
    anchor = torch.zeros_like(lower[..., 0])
    populated = torch.zeros_like(ends[..., 0])
    for group in range(bounds.shape[-1]):
        term = bounds[..., group]
        active = term > 0
        lo, hi = lower[..., group], upper[..., group]
        candidate_anchor = torch.where(active, torch.where(populated, torch.minimum(anchor, lo), lo), anchor)
        accepted, candidate = shifted_sum(
            torch.stack((accumulated, term), -1),
            torch.stack((anchor - candidate_anchor, hi - candidate_anchor), -1), width)
        split = populated & active & ~accepted
        if group:
            ends[..., group - 1] |= split
        _, singleton = shifted_sum(torch.stack((torch.zeros_like(term), term), -1),
                                   torch.stack((torch.zeros_like(lo), hi - lo), -1), width)
        accumulated = torch.where(active, torch.where(split, singleton, candidate), accumulated)
        anchor = torch.where(split, lo, candidate_anchor)
        populated |= active
    ends[..., -1] = True
    return ends
