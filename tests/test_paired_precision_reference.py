"""Check mode decisions, primitive boundaries and numerical error accounting."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from paired_precision_reference import Counters, encode_fp8, multiply, prepare_weight, select_cohorts


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA engine required")


def matrices(reduction=513):
    torch.manual_seed(117)
    return (torch.randn(7, reduction, device="cuda").bfloat16(),
            torch.randn(131, reduction, device="cuda").bfloat16())


@torch.inference_mode()
@pytest.mark.parametrize("policy", ["bf16", "fp8-paired", "fp8-split"])
def test_outputs_against_independent_cpu_accumulation(policy):
    a, w = matrices()
    prepared = prepare_weight(w)
    counters = Counters()
    actual = multiply(a, w, prepared, policy=policy, cohort=3, counters=counters)
    expected = torch.zeros(7, 131)
    if policy == "bf16":
        pairs = [(a[:, begin:begin + 256], w[:, begin:begin + 256]) for begin in range(0, 513, 256)]
    else:
        pairs = [(encode_fp8(a[:, begin:begin + 256])[0], prepared.reconstructed[:, begin:begin + 256])
                 for begin in range(0, 513, 256)]
    step = 256 if policy == "fp8-paired" else 128
    for left, right in pairs:
        for offset in range(0, left.shape[1], step):
            # CPU FP64 computes exact represented products before the declared
            # FP32 boundary; compare with the engine within FP32 roundoff.
            partial = left[:, offset:offset + step].cpu().double() @ right[:, offset:offset + step].cpu().double().T
            expected += partial.float()
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=2e-5)
    assert actual.dtype == torch.float32
    assert counters.reference_vectors == 7 * 2 * 5
    assert counters.selected_vectors == 7 * 2 * (3 if policy == "fp8-paired" else 5)


@torch.inference_mode()
def test_zero_tolerance_falls_back_and_large_tolerance_pairs():
    a, w = matrices(512)
    prepared = prepare_weight(w)
    high = multiply(a, w, policy="bf16")
    rejected = Counters()
    actual = multiply(a, w, prepared, tolerance=0, cohort=3, counters=rejected)
    assert torch.equal(actual, high) and rejected.fp8_products == 0
    accepted = Counters()
    actual = multiply(a, w, prepared, tolerance=1000, cohort=3, counters=accepted)
    low = multiply(a, w, prepared, policy="fp8-paired")
    assert torch.equal(actual, low)
    assert accepted.fp8_products == accepted.useful_products
    assert accepted.selected_vectors * 2 == accepted.reference_vectors


@torch.inference_mode()
def test_cohort_rejection_and_bound_vs_true_quantized_output_error():
    a, w = matrices(256)
    prepared = prepare_weight(w)
    q, _ = encode_fp8(a)
    selected, score = select_cohorts(a, q, prepared.l1[0], prepared.error_l1[0], tolerance=0.04, cohort=3)
    expected = torch.nn.functional.pad(score <= 0.04, (0, 0, 0, 2), value=True).reshape(3, 3, 2).all(1)
    assert torch.equal(selected, expected)
    error = ((a.cpu().double() @ w.cpu().double().T)
             - (q.cpu().double() @ prepared.reconstructed.cpu().double().T)).abs()
    scale = a.float().abs().amax(-1)[:, None] * prepared.l1[0]
    bound = (score * scale).repeat_interleave(128, 1)[:, :131].cpu()
    assert (error <= bound.double() * (1 + 1e-5) + 1e-6).all()


@torch.inference_mode()
def test_short_tail_does_not_speculatively_quantize_in_adaptive_mode():
    a, w = matrices(96)
    counters = Counters()
    actual = multiply(a, w, prepare_weight(w), tolerance=1000, counters=counters)
    assert torch.equal(actual, multiply(a, w, policy="bf16"))
    assert counters.candidate_activation_rows == 0 and counters.fp8_products == 0


@torch.inference_mode()
def test_zero_rows_and_exact_fp8_codes_accept_without_nan_scores():
    a = torch.zeros(3, 256, device="cuda", dtype=torch.bfloat16)
    a[1] = 1
    w = torch.ones(128, 256, device="cuda", dtype=torch.bfloat16)
    prepared = prepare_weight(w)
    q, _ = encode_fp8(a)
    selected, score = select_cohorts(a, q, prepared.l1[0], prepared.error_l1[0], tolerance=0, cohort=2)
    assert selected.all() and (score == 0).all()
    actual = multiply(a, w, prepared, tolerance=0, cohort=2)
    assert torch.equal(actual, a.float() @ w.float().T)
