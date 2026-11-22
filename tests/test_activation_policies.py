"""Representation invariants and equivalence to compacted receiving products."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from activation_policies import adaptive_representation, common_representation, selection


@pytest.mark.parametrize("policy", ["diagonal", "gram"])
def test_common_mask_compacts_without_changing_the_reconstructed_product(policy):
    torch.manual_seed(17)
    x, weight = torch.rand(129, 256) + .1, torch.randn(256, 29)
    gram = selection.group_gram(weight)
    output, _, elements = common_representation(x, gram, policy, None)
    assert elements.sum().item() == x.numel()
    for start in [0, 128]:
        block = output[start:start + 128]
        kept = block[0] != 0
        assert torch.all((block != 0) == kept)
        assert kept.sum().item() == 128
        compact_product = x[start:start + 128, kept] @ weight[kept]
        torch.testing.assert_close(block @ weight, compact_product, rtol=1e-5, atol=1e-5)


def test_common_fallback_and_fixed_reduction_granularity():
    torch.manual_seed(19)
    x, weight = torch.randn(129, 256), torch.randn(256, 19)
    result, counts, _ = common_representation(x, selection.group_gram(weight), "gram", 1e-15)
    torch.testing.assert_close(result, x)
    assert counts[5] == 0
    x, weight = torch.randn(129, 96), torch.randn(96, 19)
    result, counts, _ = common_representation(x, selection.group_gram(weight), "gram", 1e9)
    torch.testing.assert_close(result, x)
    assert counts[5] == 0


@pytest.mark.parametrize("tile_rows", [32, 128])
@pytest.mark.parametrize("family", ["precision", "adaptive", "sparsity"])
def test_strict_global_tolerance_retains_original_values(tile_rows, family):
    torch.manual_seed(23)
    x = torch.randn(129, 256).bfloat16()
    gram = selection.group_gram(torch.randn(256, 29))
    args = SimpleNamespace(group_size=32, selector_backend="reference", mode_rows=tile_rows)
    result, _, elements = adaptive_representation(x, gram, None, family, "gram", 1e-20, args)
    torch.testing.assert_close(result, x.float())
    assert elements.sum().item() == x.numel()
