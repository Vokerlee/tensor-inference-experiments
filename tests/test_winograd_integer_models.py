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
