import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from controller_replay_analysis import descriptor_summary, paired_summary


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


def test_descriptor_aggregation_keeps_paired_identities_and_rejects_duplicates():
    records = []
    for bits in (4, 8):
        for backend, control in (("original", "live"), ("folded", "live"), ("folded", "frozen")):
            for policy, error in (("first", 0.1), ("second", 0.2)):
                for index in range(2):
                    records.append({"bits": bits, "backend": backend, "control": control,
                        "policy": policy, "bundle": "inputs", "bundle_input_index": index,
                        "cosine_distance": error, "relative_mse": 2 * error})
    report = {"status": "complete", "records": records, "references": [
        {"bundle": "inputs", "bundle_input_index": index} for index in range(2)]}
    result = descriptor_summary(report)
    assert len(result) == 4
    assert result[0]["inputs"] == 2
    assert result[0]["pairwise"][0]["per_input"] == [-0.1, -0.1]
    report["records"].append(records[0])
    with pytest.raises(ValueError, match="duplicate"):
        descriptor_summary(report)


def test_sign_flip_ties_survive_catastrophic_float_cancellation():
    result = paired_summary([1e16, 1, -1e16, 2])
    assert result["mean"] == 0.75
    assert result["exact_sign_flip_p"] == 0.75
