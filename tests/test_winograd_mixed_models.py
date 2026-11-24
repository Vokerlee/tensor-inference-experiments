"""A shape-only strong baseline must alter both branches and restore all hooks."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from winograd_decoder_pilot import modified


@pytest.mark.parametrize("case", ["mixed-bf16-f4-int8-f2", "mixed-bf16-f4-fp8-f2"])
@torch.inference_mode()
def test_mixed_format_baseline_reaches_both_layer_consumers(case):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model = torch.nn.Sequential(torch.nn.Conv2d(32, 256, 3, padding=1), torch.nn.GELU(),
                                torch.nn.Conv2d(256, 128, 3, padding=1)).cuda().eval()
    layers = {str(i): model[i] for i in (0, 2)}
    source = torch.randn(1, 32, 8, 12, device="cuda")
    original = model(source)
    records = []
    with modified(layers, case, records, 4) as metadata:
        result = model(source)
    assert torch.isfinite(result).all() and not torch.equal(result, original)
    assert [row["K"] for row in records] == [128, 256]
    assert records[0]["floating_partial_vectors"] > 0
    assert metadata["mixed_narrow_layers"] == ["0"]
    assert torch.equal(model(source), original)


def test_shape_precondition_failure_removes_hooks_and_restores_original():
    model = torch.nn.Conv2d(32, 32, 3, padding=1).eval()
    source = torch.randn(1, 32, 7, 9)
    original = model(source)
    with pytest.raises(ValueError, match="divisible"):
        with modified({"conv": model}, "mixed-bf16-f4-fp8-f2", [], 4):
            model(source)
    assert torch.equal(model(source), original)
    assert not model._forward_pre_hooks
