"""Independent operand bounds, explicit observations and sparse exact repairs."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from selective_bf16_reference import (candidate_product, full_reference, gamma, hybrid,
                                     prepare_native, rounding_cell_accepts)
from integer_segment_backend import load_native

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@torch.inference_mode()
@pytest.mark.parametrize("backend", ["chains4", "tensorcore"])
@pytest.mark.parametrize("reduction", [7, 128, 259, 1024])
def test_independent_bound_and_repair_match_declared_reference(reduction, backend):
    torch.manual_seed(349)
    source = torch.randn(9, reduction, device="cuda").bfloat16()
    weight = torch.randn(reduction, 131, device="cuda").bfloat16()
    bias = torch.randn(131, device="cuda").bfloat16()
    prepared = prepare_native(weight)
    actual = hybrid(source, prepared, bias, backend=backend)
    reference = full_reference(source, weight, bias)
    assert torch.equal(actual.output.view(torch.int16), reference.view(torch.int16))
    assert not actual.invalid_indices.item()
    fast = candidate_product(source, weight, 128, backend) + bias.float()
    target = torch.ops.tensor_inference_engine.ordered_gemm(source, weight)[0] + bias.float()
    error = (fast.double() - target.double()).abs()
    assert (error <= actual.error_bound.double()).all()


@torch.inference_mode()
def test_cancellation_and_bf16_ties_trigger_repair_instead_of_false_acceptance():
    source = torch.tensor([[1e8, 1, -1e8, 1], [1, 2.0 ** -8, 0, 0]], device="cuda").bfloat16()
    weight = torch.ones(4, 1, device="cuda", dtype=torch.bfloat16)
    actual = hybrid(source, prepare_native(weight))
    reference = full_reference(source, weight)
    assert not actual.accepted.any()
    assert torch.equal(actual.output.view(torch.int16), reference.view(torch.int16))
    assert actual.output[0, 0] == 1


@torch.inference_mode()
def test_zero_and_subnormal_sources_have_defined_safe_behavior():
    source = torch.zeros(2, 128, device="cuda", dtype=torch.bfloat16)
    source[1, 0] = 2.0 ** -133
    weight = torch.ones(128, 3, device="cuda", dtype=torch.bfloat16)
    actual = hybrid(source, prepare_native(weight))
    assert actual.accepted[0].all() and not actual.accepted[1].any()
    assert not actual.output[0].view(torch.int16).count_nonzero()
    assert torch.equal(actual.output.view(torch.int16), full_reference(source, weight).view(torch.int16))


@torch.inference_mode()
def test_invalid_contract_parameters_and_bias_are_rejected():
    source = torch.ones(2, 128, device="cuda", dtype=torch.bfloat16)
    prepared = prepare_native(torch.ones(128, 3, device="cuda", dtype=torch.bfloat16))
    with pytest.raises(ValueError):
        hybrid(source, prepared, primitive=0)
    with pytest.raises(ValueError):
        hybrid(source, prepared, torch.ones(4, device="cuda"))


def test_gamma_domain_and_strict_cell_boundaries():
    assert gamma(128) > 128 * 2.0 ** -24
    with pytest.raises(ValueError):
        gamma(2**24)
    values = torch.tensor([1, 1 + 2.0 ** -8, -1, -1 - 2.0 ** -8], device="cuda")
    assert rounding_cell_accepts(values, torch.zeros_like(values, dtype=torch.float64)).tolist() == [True, False, True, False]


def test_compiled_predicate_preserves_the_bf16_boundary_when_configured():
    candidate = torch.tensor([1 + 2.0 ** -8 - 2.0 ** -23, 1 + 2.0 ** -8 + 2.0 ** -23,
                              -1 - 2.0 ** -8 - 2.0 ** -23], device="cuda")
    error = torch.full_like(candidate, 3 * 2.0 ** -23, dtype=torch.float64)
    compiled = torch.compile(rounding_cell_accepts, dynamic=False, fullgraph=True,
                             options={"emulate_precision_casts": True})
    expected = rounding_cell_accepts(candidate, error)
    actual = compiled(candidate, error)
    assert not expected.any() and torch.equal(expected, actual)
