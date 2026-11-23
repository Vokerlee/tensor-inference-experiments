"""Bounded online row partitioning; capacities are not measured SRAM traffic."""

from dataclasses import dataclass

import torch


@dataclass
class ScheduleCounters:
    resident_phases: int = 0
    windows: int = 0
    mixed_windows: int = 0
    natural_mode_changes: int = 0
    packed_mode_changes: int = 0
    bf16_rows: int = 0
    fp8_rows: int = 0
    reference_weight_capacity_bytes: int = 0
    selected_weight_capacity_bytes: int = 0
    selected_weight_capacity_bytes_without_residency: int = 0
    peak_argument_payload_bytes: int = 0
    peak_result_payload_bytes: int = 0
    peak_row_index_bytes: int = 0


def ordered_modes(available, previous):
    """Online choice: retain the previous mode first if available; no future peek."""
    if previous in available:
        return [previous, *[mode for mode in available if mode != previous]]
    return list(available)


def account(selected_rows, width, counters, *, buffer_rows=32, resident_rows=1024,
            candidate_prepared=True):
    """Map [M,Nwindows] decisions to a declared two-format tile-residency option.

    M is streamed. A receiver tile is N128; BF16 K128 and FP8 K256. Both weight
    formats can remain in the argument buffer through a resident M group.
    Accumulator residency, mode-switch cost and physical memory actions require
    implementation evidence. Counts use padded interface capacities.
    """
    if min(buffer_rows, resident_rows, width) < 1 or resident_rows % buffer_rows:
        raise ValueError("positive sizes and whole lookahead windows per resident group required")
    modes = selected_rows.detach().cpu().numpy().astype(bool)
    high_calls = (width + 127) // 128
    high_bytes, low_bytes = high_calls * 128 * 128 * 2, 256 * 128
    for first in range(0, len(modes), resident_rows):
        group = modes[first:first + resident_rows]
        counters.peak_result_payload_bytes = max(counters.peak_result_payload_bytes, len(group) * 128 * 4)
        for column in range(group.shape[1]):
            counters.resident_phases += 1
            current = group[:, column]
            any_low, any_high = bool(current.any()), bool((~current).any())
            weight_bytes = high_bytes * any_high + low_bytes * any_low
            counters.reference_weight_capacity_bytes += high_bytes
            counters.selected_weight_capacity_bytes += weight_bytes
            counters.fp8_rows += int(current.sum())
            counters.bf16_rows += int((~current).sum())
            counters.natural_mode_changes += int((current[1:] != current[:-1]).sum())
            previous = None
            for begin in range(0, len(current), buffer_rows):
                window = current[begin:begin + buffer_rows]
                low, high = bool(window.any()), bool((~window).any())
                counters.windows += 1
                counters.mixed_windows += low and high
                counters.selected_weight_capacity_bytes_without_residency += high_bytes * high + low_bytes * low
                counters.peak_argument_payload_bytes = max(counters.peak_argument_payload_bytes,
                    weight_bytes + len(window) * width * (2 + candidate_prepared))
                counters.peak_row_index_bytes = max(counters.peak_row_index_bytes, len(window) * 4)
                first_mode = bool(window[0])
                available = [first_mode, not first_mode] if high and low else [first_mode]
                for mode in ordered_modes(available, previous):
                    if previous is not None:
                        counters.packed_mode_changes += mode != previous
                    previous = mode


def execute(original, quantized, weight, quantized_weight, prior, selected_rows, product, *,
            split_low=False, buffer_rows=32, resident_rows=1024):
    """Execute only selected row products, restoring IDs without regrouping K.

    This is an optional numerical oracle, not a packed-FP8 hardware kernel.
    Inputs quantized/weights are reconstructed BF16; outputs stay FP32.
    """
    result = prior.clone()
    for first in range(0, len(original), resident_rows):
        for column in range(selected_rows.shape[1]):
            start, stop = column * 128, min((column + 1) * 128, weight.shape[0])
            previous = None
            for begin in range(first, min(first + resident_rows, len(original)), buffer_rows):
                end = min(begin + buffer_rows, first + resident_rows, len(original))
                mask = selected_rows[begin:end, column]
                first_mode = bool(mask[0])
                available = ([first_mode, not first_mode] if bool((mask != first_mode).any())
                             else [first_mode])
                for mode in ordered_modes(available, previous):
                    indices = torch.where(mask == mode)[0] + begin
                    a = quantized[indices] if mode else original[indices]
                    w = quantized_weight[start:stop] if mode else weight[start:stop]
                    current = result[indices, start:stop]
                    step = 128 if not mode or split_low else 256
                    for offset in range(0, original.shape[1], step):
                        current = current + product(a[:, offset:offset + step], w[:, offset:offset + step])
                    result[indices, start:stop] = current
                    previous = mode
    return result
