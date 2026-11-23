"""Compare greedy metadata partitions with exhaustive scalar interval choices."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from accumulation_segments import greedy_ends


def feasible(bounds, lower, upper, begin, end, limit):
    active = [i for i in range(begin, end) if bounds[i]]
    # One active group can use the baseline float path; zero groups are bypasses.
    if len(active) <= 1:
        return True
    anchor = min(lower[i] for i in active)
    return sum(bounds[i] * 2 ** (upper[i] - anchor) for i in active) <= limit


def test_greedy_matches_all_contiguous_partitions():
    generator = torch.Generator().manual_seed(831)
    bounds = torch.randint(0, 10, (80, 8), generator=generator)
    lower = torch.randint(0, 8, (80, 8), generator=generator)
    upper = lower + torch.randint(0, 4, (80, 8), generator=generator)
    width = 7
    ends = greedy_ends(bounds, lower, upper, width)
    for row in range(80):
        b, lo, hi = bounds[row].tolist(), lower[row].tolist(), upper[row].tolist()
        minimum = 8
        for mask in range(1 << 7):
            cuts = [i + 1 for i in range(7) if mask & (1 << i)] + [8]
            begin = 0
            for end in cuts:
                if not feasible(b, lo, hi, begin, end, 63):
                    break
                begin = end
            else:
                minimum = min(minimum, len(cuts))
        assert int(ends[row].sum()) == minimum


def test_large_exponent_gaps_do_not_overflow_the_guard():
    b = torch.tensor([[100, 100, 100, 100]])
    lo = torch.tensor([[-1000, 1000, -1000, 1000]])
    assert greedy_ends(b, lo, lo, 24).all()


def test_zero_padding_does_not_force_an_exponent_anchor():
    b = torch.tensor([[0, 1, 0, 1]])
    lo = torch.tensor([[-1000, 3, 1000, 3]])
    assert greedy_ends(b, lo, lo, 8).tolist() == [[False, False, False, True]]


def test_invalid_primitive_group_count_is_rejected():
    x = torch.zeros(1, 9, dtype=torch.int64)
    with pytest.raises(ValueError, match="eight"):
        greedy_ends(x, x, x, 24)
