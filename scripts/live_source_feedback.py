"""Causal block-row feedback selection using already available terminal residuals.

This is an experiment-side controller, not a hardware latency/energy model.
Its bias/variance objective is approximate and does not certify consumer error.
"""

import torch

from source_feedback_backend import SourceFeedback, quantize_source


def weight_metadata(weight: torch.Tensor) -> dict:
    """Per-input-channel lag correlation and spatial DC gain, normalized by energy."""
    values = weight.detach().float()
    if values.ndim != 4:
        raise ValueError("dense convolution weights required")
    energy = values.square().sum((0, 2, 3))
    correlation = (values[..., 1:] * values[..., :-1]).sum((0, 2, 3))
    dc_energy = values.sum((2, 3)).square().sum(0)
    denominator = torch.where(energy > 0, energy, 1)
    return {
        "rho": torch.where(energy > 0, correlation / denominator, 0),
        "dc_gain": torch.where(energy > 0, dc_energy / denominator, 0),
    }


def select_coefficient(mean: torch.Tensor, second: torch.Tensor, metadata: dict) -> torch.Tensor:
    """Minimize a white-residual scalar surrogate on the half-step mode grid.

    J(a) = variance*(1+a*a-2*a*rho) + mean*mean*(a-1)**2*dc_gain.
    This omits cross-channel bias, residual correlation and dependence on a.
    """
    variance = (second - mean.square()).clamp_min(0)
    bias = mean.square() * metadata["dc_gain"]
    denominator = variance + bias
    continuous = torch.where(denominator > 0,
                             (variance * metadata["rho"] + bias) / denominator.clamp_min(1e-30),
                             metadata["rho"])
    return (continuous.clamp(-0.5, 0.5) * 2).round() / 2


def quantize_live_rows(values: torch.Tensor, metadata: dict, *, bits: int = 4,
                       block_rows: int = 32, smoothing: float = 0.25) -> tuple[SourceFeedback, dict]:
    """Use preceding block terminal samples to select the NEXT block's mode.

    Each row still starts with zero carry. State resets for every consumer call
    and batch item; there is no input calibration, label access or extra GEMM.
    Terminal samples need not represent all pixels or be independent, so their
    moment estimates and the resulting decisions require empirical evaluation.
    """
    if (values.ndim != 4 or block_rows < 1 or not 0 < smoothing <= 1
            or not values.shape[2] or not values.shape[3]):
        raise ValueError("nonempty NCHW source, positive row block and smoothing in (0,1] required")
    batch, channels, height, width = values.shape
    output = torch.empty_like(values)
    terminal = torch.empty(batch, height, channels, device=values.device, dtype=torch.float32)
    exponents = torch.empty(batch, height, width, channels // 32,
                            device=values.device, dtype=torch.int32)
    counts = torch.zeros(3, device=values.device, dtype=torch.int64)
    modes = torch.zeros(3, device=values.device, dtype=torch.int64)
    switches = torch.zeros((), device=values.device, dtype=torch.int64)
    blocks = 0
    for item in range(batch):
        mean = torch.zeros(channels, device=values.device)
        second = torch.zeros_like(mean)
        coefficient = select_coefficient(mean, second, metadata)
        for start in range(0, height, block_rows):
            end = min(start + block_rows, height)
            modes += torch.stack([(coefficient == value).sum() * (end - start)
                                  for value in (-0.5, 0, 0.5)])
            current = quantize_source(values[item:item + 1, :, start:end], coefficient, bits=bits)
            output[item:item + 1, :, start:end] = current.reconstructed
            terminal[item:item + 1, start:end] = current.terminal
            exponents[item:item + 1, start:end] = current.exponents
            counts += current.counts
            samples = current.terminal[0]
            mean = (1 - smoothing) * mean + smoothing * samples.mean(0)
            second = (1 - smoothing) * second + smoothing * samples.square().mean(0)
            following = select_coefficient(mean, second, metadata)
            if end < height:
                switches += (following != coefficient).sum()
            coefficient = following
            blocks += 1
    dynamic = {"block_rows": block_rows, "smoothing": smoothing, "blocks": blocks,
               "terminal_samples": batch * height * channels,
               "mode_channel_rows": dict(zip(("-0.5", "0", "0.5"), modes.cpu().tolist())),
               "coefficient_switches": int(switches),
               "sensor": "terminal residuals from preceding row blocks only"}
    return SourceFeedback(output, terminal, exponents, counts), dynamic
