"""Real downstream consumption and restoration around proposed integer modes."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from winograd_integer_models import modified


def test_modified_outputs_feed_both_layers_and_context_restores_originals(tmp_path):
    torch.manual_seed(91)
    model = torch.nn.Sequential(torch.nn.Conv2d(32, 32, 3, padding=1),
                                torch.nn.GELU(), torch.nn.Conv2d(32, 32, 3, padding=1)).eval()
    layers = {str(index): model[index] for index in (0, 2)}
    source = torch.randn(1, 32, 5, 7)
    reference = model(source)
    records = []
    with torch.inference_mode(), modified(layers, "int8-k256-a4-r1", records, 3,
                                         tmp_path / "prepared.npz") as metadata:
        result = model(source)
    assert not torch.equal(result, reference)
    assert [row["layer"] for row in records] == ["0", "2"]
    assert all(row["K"] == 256 and row["rounding"] is not None for row in records)
    assert (tmp_path / "prepared.npz").exists() and metadata["geometry_sha256"]
    torch.testing.assert_close(model(source), reference, rtol=0, atol=0)


@torch.inference_mode()
def test_native_proxy_and_causal_shift_reach_the_actual_model_consumers(tmp_path):
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(13)
    model = torch.nn.Conv2d(32, 32, 3, padding=1).cuda().eval()
    source = torch.randn(1, 32, 5, 7, device="cuda")
    original = model(source)
    for case in ("int8-k256-a4-native", "int8-k256-a8-native",
                 "int8-k256-a4-r1-native", "int8-k256-a4-shift", "int8-k256-a8-shift",
                 "int8-k256-a4-axis-fit", "int8-k256-a8-axis-all",
                 "int8-k256-f4-a8-native", "int8-k256-f4-a8-r1-native",
                 "int8-k256-f4-a8-shift"):
        records = []
        with modified({"conv": model}, case, records, 3, tmp_path / f"{case}.npz"):
            result = model(source)
        assert not torch.equal(result, original)
        assert records[0]["rounding"]["backend"].startswith("cuda")
        assert records[0]["rounding"]["source_zeros_changed"] == 0
        assert torch.equal(model(source), original)
