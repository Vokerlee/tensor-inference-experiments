"""Independent integer oracles, including guards at signed/shift boundaries."""

import random
import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from accumulation_scale_diagnostics import fits_bound, guarded_rows, summarize_guard


def test_guard_matches_unbounded_integer_oracle():
    generator = random.Random(20261123)
    bounds = [[generator.randrange(0, 100000) for _ in range(7)] for _ in range(1000)]
    shifts = [[generator.randrange(0, 6 if row < 500 else 80) for _ in range(7)]
              for row in range(1000)]
    bounds += [[0] * 7, [32767] + [0] * 6, [32768] + [0] * 6]
    shifts += [[80] * 7, [0] * 7, [0] * 7]
    for width in [16, 24, 32, 48]:
        actual = fits_bound(torch.tensor(bounds), torch.tensor(shifts), width).tolist()
        expected = [sum(value << shift for value, shift in zip(row, offset)) < (1 << (width - 1))
                    for row, offset in zip(bounds, shifts)]
        assert actual == expected


@pytest.mark.parametrize("tile_size", [128, 256])
def test_row_guard_contains_every_column_and_prefix(tile_size):
    torch.manual_seed(51)
    a = torch.randint(-7, 8, (3, 4, 32))
    w = torch.randint(-127, 128, (131, 4, 32))
    ea, ew = torch.randint(-3, 4, (3, 4)), torch.randint(-3, 4, (131, 4))
    for width in [16, 24, 32]:
        fits, anchors = guarded_rows(a, ea, w, ew, width, tile_size)
        for row in range(3):
            for column in range(131):
                tile = column // tile_size
                if not fits[row, tile]:
                    continue
                total = 0
                for group in range(4):
                    shift = int(ea[row, group] + ew[column, group] - anchors[row, tile])
                    for left, right in zip(a[row, group], w[column, group]):
                        total += (int(left) * int(right)) << shift
                        assert abs(total) < (1 << (width - 1))


def test_physical_tile_needs_all_rows_and_padded_rows_are_zero():
    fits = torch.ones(129, 2, dtype=torch.bool)
    fits[0, 0] = False
    result = summarize_guard(fits, 129, 130, 128)
    assert result["supported_output_tiles"] == 3
    assert result["supported_useful_products"] == (128 * 2 + 130) * 128
    zero = torch.zeros(1, 4, 32, dtype=torch.int64)
    accepted, anchors = guarded_rows(zero, torch.zeros(1, 4, dtype=torch.int32),
                                    zero, torch.zeros(1, 4, dtype=torch.int32), 16)
    assert accepted.all() and not anchors.any()
