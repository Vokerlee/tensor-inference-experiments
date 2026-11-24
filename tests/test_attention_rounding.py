"""Role precision, encoding reuse and matched arithmetic topology checks."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from activation_rounding_models import reconstruct
from attention_rounding import component_plan, prepare, split_attention


def test_component_plans_keep_value_precision_explicit():
    assert [bits for bits, _, _ in component_plan("qk4-v8")] == [4, 4, 8]
    assert [bits for bits, _, _ in component_plan("q4-k8-v8")] == [4, 8, 8]
    assert [bits for bits, _, _ in component_plan("q8-k4-v8")] == [8, 4, 8]


@torch.inference_mode()
def test_split_baseline_has_same_explicit_products_as_modified_cases():
    torch.manual_seed(361)
    layer = torch.nn.Linear(32, 96, bias=False).bfloat16()
    layers = {"blocks.0.attn.qkv": layer}
    values = torch.randn(1, 9, 32).bfloat16()
    expected = torch.cat([torch.nn.functional.linear(values, part) for part in layer.weight.chunk(3)], -1)
    with split_attention(layers, {}, "bf16") as calls:
        actual = layer(values)
    assert torch.equal(actual, expected)
    assert sum(row["macs"] for row in calls) == values.numel() * 96
    assert all(row["bits"] == 16 for row in calls)


@torch.inference_mode()
def test_shared_qk_encoding_and_separate_value_encoding_feed_real_products():
    torch.manual_seed(363)
    layer = torch.nn.Linear(32, 96).bfloat16()
    source = torch.randn(1, 5, 32).bfloat16()
    original = source.clone()
    low, _ = reconstruct(source, 4)
    high, _ = reconstruct(source, 8)
    expected = torch.cat([torch.nn.functional.linear(value, weight, bias) for value, weight, bias in
                          zip((low, low, high), layer.weight.chunk(3), layer.bias.chunk(3), strict=True)], -1)
    with split_attention({"blocks.0.attn.qkv": layer}, {}, "qk4-v8") as calls:
        actual = layer(source)
    assert torch.equal(actual, expected) and torch.equal(source, original)
    assert all(row["distinct_encodings_for_projection"] == 2 for row in calls)


def test_balanced_metadata_uses_weights_only_and_is_finite_for_zero_components():
    layer = torch.nn.Linear(32, 96).bfloat16()
    with torch.no_grad():
        layer.weight[:32] = 0
    geometry = prepare({"blocks.0.attn.qkv": layer})
    assert all(torch.isfinite(value.factor).all() for value in geometry["blocks.0.attn.qkv"].values())
