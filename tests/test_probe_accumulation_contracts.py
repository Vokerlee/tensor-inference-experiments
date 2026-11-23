"""Check constructed witnesses reproduce the stated finite arithmetic effects."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from probe_accumulation_contracts import controller_witness, measure


@torch.inference_mode()
@pytest.mark.parametrize("groups", [4, 8])
@pytest.mark.parametrize("scale", [-8, 0, 8])
def test_finite_contract_witness(groups, scale):
    record = measure(groups, scale)
    assert record["contract_difference_observed"]
    assert record["exact_real_sum"] == 2.0 ** (scale + 1)
    assert record["weight_columns"] == 128


def test_recomputed_control_is_not_conflated_with_direct_tensor_error():
    record = controller_witness()
    assert record["data_path_difference_at_fixed_control"] == 0
    assert record["control_path_difference_at_changed_input"] == -1
    assert record["total_difference"] == -1
    assert record["original_exponent"] == 0
    assert record["changed_exponent"] == 1
