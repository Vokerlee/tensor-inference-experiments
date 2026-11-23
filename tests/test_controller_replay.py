"""Check captured decisions, causality and bit-exact same-trajectory replay."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from controller_replay import quantize_decisions, read_tape, write_tape
from live_source_feedback import quantize_live_rows, weight_metadata


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
@pytest.mark.parametrize("policy", ["none", "coarse", "live"])
def test_capture_disk_roundtrip_and_exact_replay(tmp_path, policy):
    torch.manual_seed(502)
    values = torch.randn(1, 32, 65, 9, device="cuda").bfloat16()
    metadata = weight_metadata(torch.randn(4, 32, 3, 3, device="cuda"))
    result, event, _ = quantize_decisions(values, metadata, layer="layer", policy=policy, bits=4)
    path = tmp_path / "trace.npz"
    write_tape(path, [event])
    restored = read_tape(path)[0]
    replay, actual, stats = quantize_decisions(values, metadata, layer="layer", policy=policy, bits=4,
                                              reference=restored, freeze_scales=True, freeze_modes=True)
    for name in ("reconstructed", "terminal", "exponents", "counts"):
        assert torch.equal(getattr(result, name), getattr(replay, name))
    assert stats["scale_mismatches"] == stats["mode_mismatches"] == 0
    assert torch.equal(event.means_before, actual.means_before)
    if policy == "live":
        expected, _ = quantize_live_rows(values, metadata)
        assert torch.equal(expected.reconstructed, result.reconstructed)
        assert torch.equal(expected.terminal, result.terminal)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_control_replay_and_live_scale_separate_on_changed_source():
    values = torch.full((1, 32, 1, 1), 0.75, device="cuda", dtype=torch.bfloat16)
    values[:, 0] = 7
    zeros = torch.zeros(32, device="cuda")
    metadata = {"rho": zeros, "dc_gain": zeros}
    _, event, _ = quantize_decisions(values, metadata, layer="test", policy="none", bits=4)
    changed = values.clone()
    changed[:, 0] = 7.03125
    live, _, live_stats = quantize_decisions(changed, metadata, layer="test", policy="none", bits=4,
                                            reference=event)
    frozen, _, frozen_stats = quantize_decisions(changed, metadata, layer="test", policy="none", bits=4,
                                                reference=event, freeze_scales=True, freeze_modes=True)
    assert live_stats["scale_mismatches"] == 1
    assert frozen_stats["scale_mismatches"] == 0
    assert live.reconstructed[0, 1, 0, 0] == 0
    assert frozen.reconstructed[0, 1, 0, 0] == 1
    with pytest.raises(ValueError, match="identity"):
        quantize_decisions(values, metadata, layer="wrong", policy="none", bits=4, reference=event)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_later_rows_do_not_change_earlier_decisions_or_state():
    torch.manual_seed(503)
    values = torch.randn(1, 32, 65, 7, device="cuda").bfloat16()
    metadata = weight_metadata(torch.randn(4, 32, 3, 3, device="cuda"))
    result, event, _ = quantize_decisions(values, metadata, layer="layer", policy="live", bits=8)
    changed = values.clone()
    changed[:, :, 32:] *= 3
    second, next_event, _ = quantize_decisions(changed, metadata, layer="layer", policy="live", bits=8)
    assert torch.equal(result.reconstructed[:, :, :32], second.reconstructed[:, :, :32])
    assert torch.equal(event.coefficients[0], next_event.coefficients[0])
    assert torch.equal(event.means_before[:2], next_event.means_before[:2])
