"""Fixed-trace conditional costs; these checks do not measure network energy."""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from replay_format_residency import replay


def test_exhaustive_short_variable_opportunities_against_restricted_offline():
    for length in range(6):
        for eligible in itertools.product((0, 1, 3), repeat=length):
            for setup in (0, 0.5, 1, 2, 8):
                result = replay([4] * length, eligible, 256, setup, 0)
                expected_offline = 8 * length - max(0, sum(eligible) - setup)
                assert result["offline"] == expected_offline
                assert result["offline"] <= result["online"] <= 2 * result["offline"]


def test_setup_is_avoided_when_costly_and_zero_gain_tails_are_not_used():
    result = replay([32, 32], [2, 3], 256, 128, 0)
    assert result["setup_performed"] == result["used_low_rows"] == 0
    assert result["online"] < result["always_prepared"]
    tail = replay([32], [32], 128, 0, 0)
    assert tail["opportunity"] == 0 and tail["online"] == tail["reference"]
    expensive_switch = replay([32], [8], 256, 0, 4)
    assert expensive_switch["used_low_rows"] == 0
