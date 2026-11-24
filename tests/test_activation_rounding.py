"""Independent geometry/rounding checks for the experimental scalar-state rule."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from activation_rounding import group_grid, prepare_geometry, round_groups


def test_diagonal_receivers_keep_nearest_rounding():
    torch.manual_seed(349)
    source = torch.randn(9, 32).bfloat16()
    geometry = prepare_geometry(torch.eye(32), 2)
    result = round_groups(source, geometry)
    values, _, scale, limit = group_grid(source, 32, 4)
    expected = (values.round().clamp(-limit, limit) * scale[:, :, None]).reshape_as(source).bfloat16()
    assert torch.equal(result.reconstructed, expected)


def test_correlated_receiver_can_cancel_error_with_shared_codes():
    source = torch.full((3, 32), 0.49).bfloat16()
    weight = torch.ones(32, 5)
    result = round_groups(source, prepare_geometry(weight, 1))
    values, _, scale, limit = group_grid(source, 32, 4)
    nearest = (values.round().clamp(-limit, limit) * scale[:, :, None]).reshape_as(source)
    original = source.float() @ weight
    assert ((result.reconstructed.float() @ weight - original).square().sum()
            < (nearest @ weight - original).square().sum())
    assert (result.codes >= -7).all() and (result.codes <= 7).all()


def test_completed_proxy_selection_never_selects_a_larger_proxy():
    torch.manual_seed(351)
    geometry = prepare_geometry(torch.randn(64, 16), 4)
    result = round_groups(torch.randn(7, 64).bfloat16(), geometry)
    assert (result.proxy_candidate[result.candidate_selected] < result.proxy_nearest[result.candidate_selected]).all()


def test_spectral_margin_is_a_real_arithmetic_group_screen_not_a_full_product_claim():
    torch.manual_seed(353)
    weight = torch.randn(32, 7)
    geometry = prepare_geometry(weight, 4)
    source = torch.randn(17, 32).bfloat16()
    result = round_groups(source, geometry, selection="spectral")
    values, _, scale, limit = group_grid(source, 32, 4)
    nearest = (values.round().clamp(-limit, limit) * scale[:, :, None]).reshape_as(source)
    actual = (result.reconstructed.double() - source.double()) @ weight.double()
    reference = (nearest.double() - source.double()) @ weight.double()
    assert (actual.square().sum(-1) <= reference.square().sum(-1) + 1e-8).all()


@pytest.mark.parametrize("rank", [1, 2, 4])
def test_zero_sources_are_preserved(rank):
    source = torch.zeros(2, 32, dtype=torch.bfloat16)
    result = round_groups(source, prepare_geometry(torch.randn(32, 9), rank))
    assert not result.reconstructed.count_nonzero()
    assert not result.codes.count_nonzero()


def test_invalid_source_and_geometry_contracts_are_rejected():
    with pytest.raises(ValueError):
        prepare_geometry(torch.ones(31, 4), 2)
    with pytest.raises(ValueError):
        prepare_geometry(torch.ones(32, 4), 33)
    with pytest.raises(ValueError):
        round_groups(torch.full((1, 32), float("nan")), prepare_geometry(torch.ones(32, 4), 1))
