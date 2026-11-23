"""Experiment-side capture/replay of causal source-quantizer decisions.

Only controller configuration is replayed. Changed tensors and row feedback
remain live; replay on another trajectory is a diagnostic intervention.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from live_source_feedback import select_coefficient
from source_feedback_backend import SourceFeedback, quantize_source


@dataclass
class Decision:
    layer: str
    shape: tuple
    bits: int
    block_rows: int
    exponents: torch.Tensor
    coefficients: torch.Tensor
    means_before: torch.Tensor
    seconds_before: torch.Tensor

    def validate(self, layer: str, values: torch.Tensor, bits: int, block_rows: int):
        if (self.layer != layer or tuple(values.shape) != self.shape or self.bits != bits
                or self.block_rows != block_rows):
            raise ValueError("replay event identity/shape/precision differs")
        _, channels, height, width = self.shape
        blocks = (height + block_rows - 1) // block_rows
        if (self.exponents.shape != (1, height, width, channels // 32)
                or self.coefficients.shape != (blocks, channels)
                or self.means_before.shape != (blocks, channels)
                or self.seconds_before.shape != (blocks, channels)):
            raise ValueError("invalid replay decision payload shape")
        if (self.exponents.dtype != torch.int16 or self.coefficients.dtype != torch.float32
                or self.means_before.dtype != torch.float32 or self.seconds_before.dtype != torch.float32):
            raise ValueError("invalid replay decision payload dtype")
        if not all(torch.isfinite(tensor).all() for tensor in (
                self.coefficients, self.means_before, self.seconds_before)):
            raise ValueError("nonfinite replay sensor/mode payload")


def write_tape(path: Path, decisions: list[Decision]):
    """Lossless integer exponents and FP32 mode/sensor traces, without pickles."""
    metadata, arrays = [], {}
    for index, decision in enumerate(decisions):
        metadata.append({"layer": decision.layer, "shape": list(decision.shape),
                         "bits": decision.bits, "block_rows": decision.block_rows})
        for field in ("exponents", "coefficients", "means_before", "seconds_before"):
            arrays[f"{index}_{field}"] = getattr(decision, field).cpu().numpy()
    arrays["metadata"] = np.frombuffer(json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def read_tape(path: Path) -> list[Decision]:
    with np.load(path, allow_pickle=False) as arrays:
        metadata = json.loads(arrays["metadata"].tobytes())
        return [Decision(row["layer"], tuple(row["shape"]), row["bits"], row["block_rows"],
                         *(torch.from_numpy(arrays[f"{index}_{field}"].copy()) for field in (
                             "exponents", "coefficients", "means_before", "seconds_before")))
                for index, row in enumerate(metadata)]


def quantize_decisions(values: torch.Tensor, metadata: dict, *, layer: str, policy: str,
                       bits: int, reference: Decision | None = None,
                       freeze_scales: bool = False, freeze_modes: bool = False,
                       block_rows: int = 32, smoothing: float = 0.25):
    if (values.ndim != 4 or values.shape[0] != 1 or values.dtype != torch.bfloat16
            or policy not in ("none", "coarse", "live") or bits not in (4, 8)):
        raise ValueError("expected one BF16 NCHW image and none/coarse/live policy")
    _, channels, height, width = values.shape
    if height < 1 or width < 1 or block_rows < 1 or not 0 < smoothing <= 1:
        raise ValueError("nonempty sources, positive blocks and smoothing in (0,1] required")
    block_rows = block_rows if policy == "live" else height
    if freeze_scales or freeze_modes:
        if reference is None:
            raise ValueError("frozen controller requires reference decisions")
    if reference is not None:
        reference.validate(layer, values, bits, block_rows)
    fixed = (reference.exponents.to(device=values.device, dtype=torch.int32)
             if freeze_scales else None)
    old_modes = reference.coefficients.to(values.device) if freeze_modes else None
    mean = torch.zeros(channels, device=values.device)
    second = torch.zeros_like(mean)
    output = torch.empty_like(values)
    exponents = torch.empty(1, height, width, channels // 32, device=values.device, dtype=torch.int32)
    terminal = torch.empty(1, height, channels, device=values.device)
    counts = torch.zeros(3, device=values.device, dtype=torch.int64)
    modes, means, seconds = [], [], []
    for block, start in enumerate(range(0, height, block_rows)):
        stop = min(start + block_rows, height)
        if freeze_modes:
            coefficient = old_modes[block]
        elif policy == "live":
            coefficient = select_coefficient(mean, second, metadata)
        elif policy == "coarse":
            coefficient = (metadata["rho"].clamp(-0.5, 0.5) * 2).round() / 2
        else:
            coefficient = torch.zeros_like(mean)
        modes.append(coefficient.clone())
        means.append(mean.clone())
        seconds.append(second.clone())
        current = quantize_source(values[:, :, start:stop], coefficient, bits=bits,
                                  fixed_exponents=fixed[:, start:stop] if fixed is not None else None)
        output[:, :, start:stop] = current.reconstructed
        terminal[:, start:stop] = current.terminal
        exponents[:, start:stop] = current.exponents
        counts += current.counts
        if policy == "live":
            samples = current.terminal[0]
            mean = (1 - smoothing) * mean + smoothing * samples.mean(0)
            second = (1 - smoothing) * second + smoothing * samples.square().mean(0)
    result = SourceFeedback(output, terminal, exponents, counts)
    statistics = result.statistics()  # Audit invalid events before accepting replay.
    decision = Decision(layer, tuple(values.shape), bits, block_rows, exponents.cpu().short(),
                        torch.stack(modes).cpu(), torch.stack(means).cpu(), torch.stack(seconds).cpu())
    comparison = {"scale_groups": decision.exponents.numel(), "mode_decisions": decision.coefficients.numel(),
                  "blocks": len(modes), "mode_switches": int(torch.count_nonzero(
                      decision.coefficients[1:] != decision.coefficients[:-1]))}
    row_counts = torch.tensor([min(block_rows, height - start) for start in range(0, height, block_rows)])
    comparison["mode_channel_rows"] = {
        str(mode): int(((decision.coefficients == mode).sum(-1) * row_counts).sum())
        for mode in (-0.5, 0.0, 0.5)
    }
    comparison["nonzero_feedback_positions"] = width * (
        comparison["mode_channel_rows"]["-0.5"] + comparison["mode_channel_rows"]["0.5"])
    if reference is not None:
        comparison.update({
            "scale_mismatches": int(torch.count_nonzero(decision.exponents != reference.exponents)),
            "mode_mismatches": int(torch.count_nonzero(decision.coefficients != reference.coefficients)),
            "maximum_mean_state_difference": float((decision.means_before - reference.means_before).abs().max()),
            "maximum_second_state_difference": float((decision.seconds_before - reference.seconds_before).abs().max()),
        })
    return result, decision, {**statistics, **comparison}
