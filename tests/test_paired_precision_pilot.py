"""Check the row-count ablation without model downloads or CUDA execution."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import paired_precision_pilot as pilot


def test_single_row_preservation_uses_bf16_contract_and_restores_forward(monkeypatch):
    layer = torch.nn.Linear(256, 129).bfloat16()
    original = layer.forward
    selected = []
    prepared = object()
    monkeypatch.setattr(pilot, "prepare_weight", lambda weight: prepared)

    def multiply(values, weight, metadata, *, policy, **kwargs):
        selected.append((policy, metadata))
        return torch.zeros(len(values), weight.shape[0], dtype=torch.float32)

    monkeypatch.setattr(pilot, "multiply", multiply)
    cache = {}
    with pilot.applied({"layer": layer}, cache, "adaptive-paired:0.04", 32,
                       minimum_fp8_rows=2) as (_, traces, _):
        assert layer(torch.ones(1, 256).bfloat16()).shape == (1, 129)
        assert cache == {}
        assert layer(torch.ones(2, 256).bfloat16()).shape == (2, 129)
        assert len(traces["layer"]) == 2
    assert selected == [("bf16", None), ("adaptive-paired", prepared)]
    assert layer.forward == original
