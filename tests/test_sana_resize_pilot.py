"""Check branch preservation, expansion placement and measured work scope."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sana_resize_pilot import changed, prepared, upsampling_blocks


def block():
    from diffusers.models.autoencoders.autoencoder_dc import DCUpBlock2d
    return DCUpBlock2d(32, 128, interpolate=True, shortcut=True).eval()


def test_unmodified_branch_preserves_shortcut_and_restores_forward():
    torch.manual_seed(132)
    layer = block().double()
    values = torch.randn(2, 32, 3, 4, dtype=torch.float64)
    original_forward, expected = layer.forward, layer(values)
    records = []
    with changed({"up": layer}, prepared({"up": layer}, "original"), "original", records):
        assert torch.equal(layer(values), expected)
    assert layer.forward == original_forward
    assert records[0]["executed_reference_macs"] == records[0]["expanded_macs"]
    assert records[0]["interpolated_tensor_materialized"] is True


def test_shape_detection_excludes_pixel_shuffle_and_non_nearest_blocks():
    layer = block()
    assert set(upsampling_blocks(torch.nn.Sequential(layer))) == {"0"}
    layer.interpolation_mode = "bilinear"
    assert not upsampling_blocks(torch.nn.Sequential(layer))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
@pytest.mark.parametrize("bits", [4, 8])
def test_source_quantization_is_identical_before_duplication_and_uses_fourfold_fewer_groups(bits):
    torch.manual_seed(133)
    layer = block().cuda().bfloat16()
    values = torch.randn(1, 32, 3, 4, device="cuda").bfloat16()
    results, records = {}, {}
    for placement in ("source", "expanded"):
        case = f"{placement}-a{bits}"
        records[placement] = []
        with changed({"up": layer}, prepared({"up": layer}, case), case, records[placement]):
            results[placement] = layer(values)
    assert torch.equal(results["source"], results["expanded"])
    assert (records["source"][0]["quantizer"]["source_groups"] * 4
            == records["expanded"][0]["quantizer"]["source_groups"])
