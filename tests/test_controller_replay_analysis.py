import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from controller_replay_analysis import paired_summary


def test_paired_statistics_keep_independent_inputs_and_zero_effect():
    result = paired_summary([0, 0, 0, 0])
    assert result['n'] == 4
    assert result['mean'] == 0
    assert result['bootstrap_percentile_95'] == [0, 0]
    assert result['exact_sign_flip_p'] == 1
    assert result['zero'] == 4
    assert paired_summary([1, 1, 1, 1])['exact_sign_flip_p'] == 0.125
    assert paired_summary([1, -2, 3, -4]) == paired_summary([1, -2, 3, -4])


@pytest.mark.parametrize('values', [[], [np.nan], [[1, 2]], list(range(17))])
def test_invalid_independent_input_vector_rejected(values):
    with pytest.raises(ValueError):
        paired_summary(values)
