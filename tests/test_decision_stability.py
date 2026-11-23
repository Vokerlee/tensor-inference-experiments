import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from decision_stability import ScaleTable


def test_guard_includes_boundary_and_excludes_changed_decisions_exhaustively():
    maxima = np.arange(17, dtype=np.float64)
    exponents = np.array([0] + [-1] * 3 + [0] * 4 + [1] * 8 + [2])
    table = ScaleTable(maxima, exponents)
    certified = rejected = 0
    for i, first in enumerate(maxima):
        for j, second in enumerate(maxima):
            stable = bool(table.stable(first, abs(second - first)))
            if stable:
                assert exponents[i] == exponents[j]
                certified += 1
            else:
                rejected += 1
    assert certified and rejected
    assert bool(table.stable(7, 0))
    assert not bool(table.stable(7, 1))
    assert not bool(table.stable(0, 1))


def test_vector_bound_and_nonmonotonic_zero_region_are_handled():
    table = ScaleTable([0, 1, 2, 3, 4], [0, -1, 0, 0, 1])
    assert table.stable([0, 2, 3, 4], [0, 0, 0.1, 1]).tolist() == [True, True, True, False]
    assert not table.stable(0, 4)


@pytest.mark.parametrize('maximum,bound', [(1.5, 0), (1, -1), (np.nan, 0), (1, np.inf)])
def test_invalid_contract_inputs_rejected(maximum, bound):
    with pytest.raises(ValueError):
        ScaleTable([0, 1, 2], [0, 0, 1]).stable(maximum, bound)
