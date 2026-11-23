"""Bounded schedule accounting and gathered execution against dense candidates."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from row_precision_schedule import ScheduleCounters, account, execute
from paired_precision_reference import Counters, encode_fp8, multiply, prepare_weight
from integer_segment_backend import load_native


def test_alternating_rows_expose_residency_and_mode_change_tradeoff():
    selected = (torch.arange(64) % 2).bool()[:, None]
    counters = ScheduleCounters()
    account(selected, 256, counters, buffer_rows=8, resident_rows=64)
    assert counters.windows == counters.mixed_windows == 8
    assert counters.bf16_rows == counters.fp8_rows == 32
    assert counters.natural_mode_changes == 63
    assert counters.packed_mode_changes == 8
    assert counters.reference_weight_capacity_bytes == 65536
    assert counters.selected_weight_capacity_bytes == 98304
    assert counters.selected_weight_capacity_bytes_without_residency == 8 * 98304
    assert counters.peak_argument_payload_bytes == 98304 + 8 * 256 * 3
    assert counters.peak_result_payload_bytes == 64 * 128 * 4


def test_uniform_reference_has_no_candidate_buffer_or_mode_changes():
    counters = ScheduleCounters()
    account(torch.zeros(35, 2).bool(), 256, counters, buffer_rows=8, resident_rows=32,
            candidate_prepared=False)
    assert counters.mixed_windows == counters.packed_mode_changes == 0
    assert counters.selected_weight_capacity_bytes == counters.reference_weight_capacity_bytes
    assert counters.peak_argument_payload_bytes == 65536 + 8 * 256 * 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA engine required")
@torch.inference_mode()
def test_gathered_native_products_restore_rows_and_keep_k_boundaries():
    torch.manual_seed(963)
    a = torch.randn(35, 256, device="cuda").bfloat16()
    w = torch.randn(131, 256, device="cuda").bfloat16()
    z, q = encode_fp8(a)[0], encode_fp8(w)[0]
    prior = torch.randn(35, 131, device="cuda")
    selected = ((torch.arange(35, device="cuda")[:, None]
                 + torch.arange(2, device="cuda")[None, :]) % 2).bool()
    load_native()
    product = lambda left, right: torch.ops.tensor_inference_engine.gemm(left, right.T, True)
    high = prior + product(a[:, :128], w[:, :128])
    high += product(a[:, 128:], w[:, 128:])
    low = prior + product(z, q)
    mask = selected.repeat_interleave(128, 1)[:, :131]
    expected = torch.where(mask, low, high)
    actual = execute(a, z, w, q, prior, selected, product, buffer_rows=8, resident_rows=32)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA engine required")
@torch.inference_mode()
def test_complete_packed_reduction_matches_same_decisions_in_candidate_oracle():
    torch.manual_seed(1963)
    a = torch.randn(35, 513, device="cuda").bfloat16()
    w = torch.randn(131, 513, device="cuda").bfloat16()
    prepared = prepare_weight(w, with_moments=True)
    oracle, packed = Counters(), Counters()
    options = {"policy": "moment-paired", "tolerance": 0.04, "cohort": 1,
               "row_buffer": 8, "resident_rows": 32}
    expected = multiply(a, w, prepared, counters=oracle, **options)
    actual = multiply(a, w, prepared, counters=packed, execution="packed", **options)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5)
    assert packed.fp8_products == oracle.fp8_products
    assert packed.selected_vectors == oracle.selected_vectors
    assert packed.schedule == oracle.schedule
