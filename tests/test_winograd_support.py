"""Reachable support, unchanged active objective and producer-contract rejection."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from winograd_reference import input_tiles, transform_inputs, transforms
from winograd_rounding import prepare, round_planes
from winograd_support import nearest_planes, validate


@pytest.mark.parametrize("tile,count", [(2, 9), (4, 25)])
def test_nearest_expansion_has_the_declared_support_including_padded_edges(tile, count):
    torch.manual_seed(9)
    original = torch.randn(1, 32, 5, 7, dtype=torch.float64)
    expanded = torch.nn.functional.interpolate(original, scale_factor=2, mode="nearest")
    tiles, _ = input_tiles(expanded, tile)
    source = transform_inputs(tiles, transforms(tile))
    active = nearest_planes(tile)
    assert active.sum() == count
    validate(source, active)
    geometry = prepare(torch.randn((tile + 2) ** 2, 32, 5), tile=tile, rank=1, active_planes=active)
    assert torch.count_nonzero(geometry.metric[:, ~active]) == 0
    assert torch.count_nonzero(geometry.factor[:, ~active]) == 0
    bad = source.float().clone()
    bad[~active] = .01
    with pytest.raises(ValueError, match="inactive planes"):
        round_planes(bad, geometry)


@pytest.mark.parametrize("case,count", [("int8-k256-a4-support-shift", 9),
                                         ("int8-k256-f4-a8-support-shift", 25)])
@torch.inference_mode()
def test_producer_hint_reaches_the_actual_native_model_encoder(case, count, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from winograd_integer_models import modified
    from winograd_support import convolution_names

    class Block(torch.nn.Module):
        interpolate, factor, interpolation_mode = True, 2, "nearest"

        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(32, 32, 3, padding=1)

        def forward(self, value):
            return self.conv(torch.nn.functional.interpolate(value, scale_factor=2, mode="nearest"))

    model = Block().cuda().eval()
    source = torch.randn(1, 32, 5, 7, device="cuda")
    original = model(source)
    records = []
    with modified({"conv": model.conv}, case, records, 4, tmp_path / "prepared.npz",
                  support_layers=convolution_names(model)) as metadata:
        output = model(source)
    assert torch.isfinite(output).all()
    assert records[0]["rounding"]["source_zeros_changed"] == 0
    assert metadata["weight_preparation"][0]["active_transform_planes"] == count
    assert torch.equal(model(source), original)
