"""Known nearest-neighbor resize/conv folding, with explicit output phases.

Colbert et al., arXiv2107.07647v1, Section4.2, establish coefficient folding.
This numerical reference does not claim a new decomposition or hardware cost.
"""

import torch
from torch.nn import functional as F


def phase_bank(weight: torch.Tensor) -> torch.Tensor:
    """Fold a dense3x3 kernel into four phase-major dense2x2 kernels.

    For phase(p,q), source offsets are floor((p+r-1)/2), floor((q+s-1)/2).
    Sum taps sharing an offset. Accumulate BF16 inputs in FP32; double inputs
    remain double for real-arithmetic reference checks. Bias stays per output.
    """
    if weight.ndim != 4 or weight.shape[-2:] != (3, 3):
        raise ValueError("expected dense [output,input,3,3] weights")
    values = weight if weight.dtype == torch.float64 else weight.float()
    outputs, inputs = values.shape[:2]
    bank = torch.zeros(4, outputs, inputs, 2, 2, device=values.device, dtype=values.dtype)
    for phase_row in range(2):
        for phase_column in range(2):
            phase = phase_row * 2 + phase_column
            for row in range(3):
                for column in range(3):
                    target_row = (phase_row + row - 1) // 2 + (phase_row == 0)
                    target_column = (phase_column + column - 1) // 2 + (phase_column == 0)
                    bank[phase, :, :, target_row, target_column] += values[:, :, row, column]
    return bank.flatten(0, 1).contiguous()


def convolve_phases(values: torch.Tensor, bank: torch.Tensor,
                    bias: torch.Tensor | None = None) -> torch.Tensor:
    """Compute four low-resolution phase planes and interleave high-res output.

    The reference computes an extra border row/column, then crops each phase
    with its own offset. Count this work; it is not an ideal zero-overhead crop.
    """
    if (values.ndim != 4 or bank.ndim != 4 or bank.shape[0] % 4
            or bank.shape[1] != values.shape[1] or bank.shape[-2:] != (2, 2)):
        raise ValueError("expected NCHW source and four-phase dense2x2 bank")
    batch, _, height, width = values.shape
    outputs = bank.shape[0] // 4
    planes = F.conv2d(F.pad(values, (1, 1, 1, 1)), bank,
                      bias.repeat(4) if bias is not None else None)
    result = torch.empty(batch, outputs, height * 2, width * 2,
                         device=values.device, dtype=planes.dtype)
    for phase_row in range(2):
        for phase_column in range(2):
            phase = phase_row * 2 + phase_column
            result[:, :, phase_row::2, phase_column::2] = planes[
                :, phase * outputs:(phase + 1) * outputs,
                phase_row:phase_row + height, phase_column:phase_column + width]
    return result


def arithmetic_counts(batch: int, channels: int, outputs: int, height: int, width: int) -> dict:
    """Dense arithmetic and representation sizes; not time, traffic or energy."""
    if min(batch, channels, outputs, height, width) < 1:
        raise ValueError("positive dimensions required")
    old = batch * (2 * height) * (2 * width) * outputs * channels * 9
    ideal = batch * height * width * 4 * outputs * channels * 4
    reference = batch * (height + 1) * (width + 1) * 4 * outputs * channels * 4
    return {"expanded_macs": old, "ideal_folded_macs": ideal,
            "reference_folded_macs": reference,
            "original_weight_scalars": outputs * channels * 9,
            "folded_weight_scalars": outputs * channels * 16,
            "low_source_scalars": batch * channels * height * width,
            "expanded_source_scalars": batch * channels * height * width * 4}
