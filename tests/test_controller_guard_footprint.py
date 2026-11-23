import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from controller_guard_footprint import footprint


def test_single_source_group_reuses_nine_kernel_vectors_at_an_interior_position():
    flags = np.zeros((5, 5, 4), dtype=bool)
    flags[2, 2, 1] = True
    result = footprint(flags, cohort=1)
    assert result["source_groups"] == 100
    assert result["flagged_source_groups"] == 1
    assert result["row_vectors"] == 81
    assert result["flagged_row_vectors"] == result["flagged_cohorts"] == 9
    grouped = footprint(flags, cohort=9)
    assert grouped["flagged_cohorts"] == grouped["cohorts"] == 9


def test_masks_empty_full_and_tail_cohorts_preserve_denominators():
    flags = np.zeros((4, 5, 4), dtype=bool)
    empty = footprint(flags, cohort=4)
    assert empty["row_vectors"] == 54
    assert empty["cohorts"] == 18
    assert empty["flagged_cohorts"] == 0
    full = footprint(~flags, cohort=4)
    assert full["flagged_row_vectors"] == 54
    assert full["flagged_cohorts"] == 18
    assert all(row["flagged_rows_per_cohort"] == [4, 2] for row in full["planes"])


@pytest.mark.parametrize("shape,kernel,cohort", [((8, 8, 8), 3, 1), ((2, 2, 4), 3, 1), ((8, 8, 4), 3, 0)])
def test_unsupported_geometry_rejected(shape, kernel, cohort):
    with pytest.raises(ValueError):
        footprint(np.zeros(shape, dtype=bool), kernel=kernel, cohort=cohort)
