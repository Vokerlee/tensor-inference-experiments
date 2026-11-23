"""Check decoder intervention boundaries without downloading model weights."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sana_decoder_feedback as pilot


def test_shape_rule_excludes_small_output_grouped_and_pointwise_when_spatial():
    decoder = torch.nn.ModuleDict({
        "spatial": torch.nn.Conv2d(32, 128, 3),
        "pointwise": torch.nn.Conv2d(32, 128, 1),
        "rgb": torch.nn.Conv2d(32, 3, 3),
        "depthwise": torch.nn.Conv2d(32, 128, 3, groups=32),
        "unaligned": torch.nn.Conv2d(3, 128, 3),
    })
    assert set(pilot.eligible_layers(decoder, "spatial")) == {"spatial"}
    assert set(pilot.eligible_layers(decoder, "all")) == {"spatial", "pointwise"}


def test_consumer_specific_intervention_restores_forward_after_exception(monkeypatch):
    layer = torch.nn.Conv2d(32, 128, 3, padding=1)
    source = torch.randn(1, 32, 3, 3)
    original_values, original_forward = source.clone(), layer.forward
    monkeypatch.setattr(pilot, "quantize_source", lambda values, coefficient, **kwargs:
                        SimpleNamespace(reconstructed=torch.zeros_like(values),
                                        statistics=lambda: {"invalid_events": 0}))
    records = []
    with pytest.raises(RuntimeError, match="deliberate"):
        with pilot.modified({"test": layer}, {"test": torch.zeros(32)}, "none4", records):
            assert torch.equal(layer(source), layer._conv_forward(torch.zeros_like(source),
                                                                  layer.weight, layer.bias))
            assert torch.equal(source, original_values)
            raise RuntimeError("deliberate")
    assert layer.forward == original_forward
    assert records[0]["input_shape"] == [1, 32, 3, 3]


def test_postprocessing_matches_diffusers_bf16_rounding():
    from diffusers.image_processor import PixArtImageProcessor

    raw = torch.linspace(-2, 2, 3 * 32 * 32).reshape(1, 3, 32, 32).bfloat16()
    expected = PixArtImageProcessor().postprocess(raw, output_type="np")
    assert np.array_equal(pilot.image_values(raw.float().numpy()), expected)


def test_inventory_counts_actual_conv_shape_and_restores_hooks():
    decoder = torch.nn.Sequential(torch.nn.Conv2d(32, 128, 3, padding=1))
    records = []
    with pilot.convolution_inventory(decoder, records):
        decoder(torch.ones(1, 32, 4, 5))
    assert records[0]["macs"] == 1 * 128 * 4 * 5 * 32 * 9
    assert not decoder[0]._forward_hooks
