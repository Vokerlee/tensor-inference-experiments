"""Policies alter actual consumers and restore a branched model after failures."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from activation_rounding_models import modified, policy_parameters, prepare_layers, reconstruct


def test_qkv_roles_select_bits_without_per_layer_calibration():
    assert policy_parameters("blocks.3.attn.qkv", "qkv4-r1") == (4, 1)
    assert policy_parameters("blocks.3.attn.proj", "qkv4-r1") == (8, 0)
    assert policy_parameters("blocks.3.mlp.fc1", "qkv8-r1") == (8, 0)
    with pytest.raises(ValueError):
        policy_parameters("blocks.0.attn.qkv", "unknown")


@torch.inference_mode()
def test_wrapper_feeds_modified_source_and_preserves_original_input():
    torch.manual_seed(357)
    layer = torch.nn.Linear(32, 64).bfloat16()
    layers = {"blocks.0.attn.qkv": layer}
    prepared = prepare_layers(layers)
    source = torch.randn(1, 7, 32).bfloat16()
    original_source = source.clone()
    restored, _ = reconstruct(source, 4, prepared["blocks.0.attn.qkv"][1])
    expected = layer(restored) + source.sum(-1, keepdim=True)
    original_forward = layer.forward
    with modified(layers, prepared, "qkv4-r1") as calls:
        actual = layer(source) + source.sum(-1, keepdim=True)
    assert torch.equal(actual, expected)
    assert torch.equal(source, original_source)
    assert calls[0]["rank"] == 1 and calls[0]["bits"] == 4
    assert layer.forward == original_forward


@torch.inference_mode()
def test_bf16_policy_is_exact_and_restoration_runs_on_exception():
    layer = torch.nn.Linear(32, 32).bfloat16()
    source = torch.randn(2, 32).bfloat16()
    expected = layer(source)
    original = layer.forward
    with pytest.raises(RuntimeError):
        with modified({"layer": layer}, {}, "bf16"):
            assert torch.equal(layer(source), expected)
            raise RuntimeError("caller error")
    assert layer.forward == original
