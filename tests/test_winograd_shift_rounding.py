"""Packed approximation, exact grids and causal emission without terminal replay."""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from winograd_rounding import prepare as prepare_geometry
from winograd_shift_rounding import prepare, reference


def test_prepared_shift_surrogate_has_nonnegative_diagonal_residual():
    torch.manual_seed(43)
    geometry = prepare_geometry(torch.randn(16, 32, 5), rank=1)
    packed = prepare(geometry).metadata.int()
    raw = packed & 255
    direction = torch.where(raw >= 128, raw - 256, raw)
    h = (packed >> 8) & 255
    u = torch.where(direction != 0, torch.ldexp(torch.ones_like(direction).double(), 1 - direction.abs()), 0)
    diagonal = torch.where(h != 0, torch.ldexp(torch.ones_like(h).double(), 1 - h), 0)
    assert torch.all(diagonal >= u.square())
    assert prepare(geometry).metadata.element_size() == 2


def test_future_plane_values_cannot_change_already_emitted_codes():
    torch.manual_seed(8)
    source = torch.randn(16, 2, 32)
    geometry = prepare(prepare_geometry(torch.randn(16, 32, 5), rank=1))
    first = reference(source, geometry.metadata, group=32)[0]
    changed = source.clone()
    changed[-1] *= 300
    second = reference(changed, geometry.metadata, group=32)[0]
    assert torch.equal(first[:-1], second[:-1])


def test_zero_metric_retains_nearest_and_exact_source_zeros():
    source = torch.zeros(4, 2, 32)
    source[0, :, 0] = 0.4
    output, _, changed = reference(source, torch.zeros(32, 4, dtype=torch.int16), group=32)
    assert torch.count_nonzero(output[source == 0]) == 0
    assert not torch.count_nonzero(changed)


def test_shift_decision_matches_the_independent_quadratic_cost_difference():
    torch.manual_seed(334)
    state = torch.randn(100, dtype=torch.float64)
    direction = torch.tensor([.25, -.5, 1., 0.], dtype=torch.float64).repeat(25)
    diagonal_total = torch.tensor([.5, .5, 1., 0.], dtype=torch.float64).repeat(25)
    diagonal_residual = diagonal_total - direction.square()
    low = -torch.rand(100, dtype=torch.float64)
    high = low + torch.rand(100, dtype=torch.float64)
    direct = ((state + direction * high).square() + diagonal_residual * high.square()
              - (state + direction * low).square() - diagonal_residual * low.square())
    criterion = (high - low) * (2 * direction * state + (low + high) * diagonal_total)
    torch.testing.assert_close(direct, criterion, rtol=1e-12, atol=1e-12)


@torch.inference_mode()
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("group", [32, 256])
@pytest.mark.parametrize("bits", [4, 8])
def test_prepared_metadata_native_bridge_matches_live_reference(group, bits):
    from winograd_shift_rounding import ShiftGeometry, round_shift

    torch.manual_seed(31 + group + bits)
    geometry = prepare(prepare_geometry(torch.randn(16, group, 5), rank=1))
    metadata = geometry.metadata.cuda()
    source = torch.randn(16, 3, group, device="cuda")
    source[:, :, ::5] = 0
    expected = reference(source, metadata, bits, group)[0]
    actual, stats = round_shift(source, ShiftGeometry(metadata, "cuda"), bits, group)
    assert torch.equal(actual, expected)
    assert stats["source_zeros_changed"] == 0
