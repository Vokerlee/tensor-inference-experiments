"""NCHW experiment adapters for the engine's causal source-group quantizer."""

from dataclasses import dataclass

import torch

from integer_segment_backend import load_native


@dataclass
class SourceFeedback:
    reconstructed: torch.Tensor
    terminal: torch.Tensor
    exponents: torch.Tensor
    counts: torch.Tensor

    def statistics(self) -> dict:
        """Audit deferred failures before exposing any accepted numerical result."""
        clipped, introduced, invalid = self.counts.cpu().tolist()
        if invalid:
            raise ValueError(f"source feedback reported {invalid} invalid events")
        # BF16 finite source exponents fit this fixed histogram; do not sort a
        # potentially multi-million-element metadata tensor for every layer.
        shifted = self.exponents.flatten().long() + 160
        if shifted.numel() and (int(shifted.min()) < 0 or int(shifted.max()) >= 320):
            raise ValueError("unexpected source exponent outside BF16 range")
        histogram = torch.bincount(shifted, minlength=320).cpu().tolist()
        state = self.terminal.double()
        return {
            "source_scalars": self.reconstructed.numel(),
            "source_groups": self.exponents.numel(),
            "clipped_values": clipped,
            "introduced_nonzeros": introduced,
            "invalid_events": invalid,
            "exponent_histogram": {str(i - 160): n for i, n in enumerate(histogram) if n},
            "terminal_scalars": state.numel(),
            "terminal_error_squared": float(state.square().sum()),
            "terminal_maximum_absolute": float(state.abs().amax()) if state.numel() else 0.0,
        }


def quantize_source(values: torch.Tensor, coefficient: torch.Tensor, *, bits: int = 4,
                    preserve_zeros: bool = False) -> SourceFeedback:
    """Keep graph consumers independent: return a new reconstructed source.

    Coefficients are caller-prepared FP32. The native primitive scans width,
    resetting at every batch/spatial row. Layout conversion and reconstruction
    are experiment costs, not a proposed hardware dataflow or packed integer op.
    Call ``statistics`` before accepting results to audit deferred invalid events.
    """
    if values.ndim != 4:
        raise ValueError("expected a four-dimensional NCHW source")
    batch, channels, height, width = values.shape
    source = values.permute(0, 2, 3, 1).contiguous().reshape(batch * height, width, channels)
    load_native()
    output, terminal, exponents, counts = torch.ops.tensor_inference_engine.source_feedback(
        source, coefficient, bits, preserve_zeros, False)
    return SourceFeedback(
        output.reshape(batch, height, width, channels).permute(0, 3, 1, 2),
        terminal.reshape(batch, height, channels),
        exponents.reshape(batch, height, width, channels // 32), counts)


def quantize_weights(weight: torch.Tensor, *, bits: int = 8) -> torch.Tensor:
    """POT groups of32 INPUT channels at each spatial kernel position.

    This matches source-before-im2col grouping; groups never span two kernel
    positions. It reconstructs BF16 values and does not implement integer conv.
    """
    if (weight.ndim != 4 or weight.shape[1] % 32 or bits not in (4, 8)
            or weight.dtype != torch.bfloat16):
        raise ValueError("expected aligned dense BF16 convolution weights and bits4/8")
    if not torch.isfinite(weight).all():
        raise ValueError("finite weights required")
    columns, channels, rows, width = weight.shape
    grouped = weight.float().permute(0, 2, 3, 1).reshape(columns, rows, width, channels // 32, 32)
    maximum = grouped.abs().amax(-1, keepdim=True)
    limit = (1 << (bits - 1)) - 1
    exponent = torch.where(maximum > 0, (maximum / limit).log2().ceil(), 0).int()
    scale = torch.ldexp(torch.ones_like(maximum), exponent)
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("weight scale outside FP32 range")
    restored = ((grouped / scale).round().clamp(-limit, limit) * scale).bfloat16()
    if not torch.isfinite(restored).all():
        raise ValueError("reconstructed weights outside BF16 range")
    return restored.reshape(columns, rows, width, channels).permute(0, 3, 1, 2).contiguous()
