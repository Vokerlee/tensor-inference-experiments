"""Validate complete-DAG tape replay and interaction accounting."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from live_source_feedback import weight_metadata
from sana_controller_replay import controlled, decomposition


def test_data_control_squared_errors_include_cancellation():
    original = np.ones((1, 3, 2, 2), dtype=np.float32)
    frozen = original + 1
    live = original.copy()
    result = decomposition(original, frozen, live)["raw"]
    assert result["data_squared"] == result["control_squared"] == 12
    assert result["twice_inner_product"] == -24
    assert result["total_squared"] == 0
    assert result["maximum_identity_residual"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_captured_control_replays_a_branched_decoder_path_exactly():
    torch.manual_seed(630)
    first = torch.nn.Conv2d(32, 64, 3, padding=1).cuda().bfloat16()
    second = torch.nn.Conv2d(64, 128, 3, padding=1).cuda().bfloat16()
    layers = {"first": first, "second": second}
    metadata = {name: weight_metadata(layer.weight) for name, layer in layers.items()}
    source = torch.randn(1, 32, 8, 11, device="cuda").bfloat16()
    before = source.clone()

    def execute():
        hidden = first(source)
        return second(hidden) + source.mean(1, keepdim=True)

    events, calls = [], []
    with controlled(layers, metadata, "live", 4, events, calls):
        expected = execute()
    replay_events, replay_calls = [], []
    with controlled(layers, metadata, "live", 4, replay_events, replay_calls,
                    events, freeze_scales=True, freeze_modes=True):
        actual = execute()
    assert torch.equal(source, before)
    assert torch.equal(expected, actual)
    assert [event.layer for event in replay_events] == ["first", "second"]
    assert all(row["mode_mismatches"] == row["scale_mismatches"] == 0 for row in replay_calls)
    assert all(row["dense_macs"] > 0 for row in calls)
