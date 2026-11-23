"""Numerical K128/K256 precision oracle using the engine's BF16/FP32 GEMM.

FP8 operands are reconstructed exactly into BF16 where representable. This
executes changed arithmetic but is not a packed FP8 kernel or a hardware model.
"""

import math
from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from integer_segment_backend import load_native
import precision_moments
import row_precision_schedule


POLICIES = ("bf16", "fp8-paired", "fp8-split", "adaptive-paired", "adaptive-split",
            "moment-paired", "moment-split", "diagonal-paired", "diagonal-split")


@dataclass
class PreparedWeight:
    reconstructed: torch.Tensor
    l1: torch.Tensor
    error_l1: torch.Tensor
    exponent_histogram: dict
    moments: torch.Tensor | None = None


@dataclass
class Counters:
    useful_products: int = 0
    fp8_products: int = 0
    reference_vectors: int = 0
    selected_vectors: int = 0
    mode_groups: int = 0
    fp8_mode_groups: int = 0
    within_pair_mode_transitions: int = 0
    reference_gemm_calls: int = 0
    candidate_activation_rows: int = 0
    activation_scale_exponents: dict = field(default_factory=dict)
    normalized_bound_histogram: dict = field(default_factory=dict)
    normalized_moment_score_histogram: dict = field(default_factory=dict)
    schedule: row_precision_schedule.ScheduleCounters = field(default_factory=row_precision_schedule.ScheduleCounters)


def add_histogram(destination: dict, values: torch.Tensor) -> None:
    keys, counts = torch.unique(values, return_counts=True)
    for key, count in zip(keys.cpu().tolist(), counts.cpu().tolist(), strict=True):
        key = str(key)
        destination[key] = destination.get(key, 0) + count


def encode_fp8(values: torch.Tensor):
    """One current POT scale per row of a nonempty K<=256 source slice."""
    if values.ndim != 2 or not 0 < values.shape[1] <= 256:
        raise ValueError("expected nonempty row slices of at most256 values")
    original = values.float()
    maximum = original.abs().amax(-1, keepdim=True)
    if not torch.isfinite(maximum).all():
        raise ValueError("FP8 reference requires finite values")
    exponent = torch.where(maximum > 0, (maximum / 448).log2().ceil(), 0).int()
    scale = torch.ldexp(torch.ones_like(maximum), exponent)
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("FP8 source scale must remain finite and positive")
    normalized = original / scale
    reconstructed = (normalized.to(torch.float8_e4m3fn).float() * scale).bfloat16()
    if not torch.isfinite(reconstructed).all():
        raise ValueError("reconstructed FP8 values exceed the BF16 oracle range")
    return reconstructed, exponent.squeeze(-1)


def window_maximum(values: torch.Tensor):
    return F.pad(values, (0, (-values.numel()) % 128)).reshape(-1, 128).amax(-1)


def prepare_weight(weight: torch.Tensor, *, with_moments=False) -> PreparedWeight:
    if weight.ndim != 2 or weight.dtype != torch.bfloat16 or min(weight.shape) < 1:
        raise ValueError("expected nonempty BF16 [outputs,reduction] weights")
    reconstructed = torch.empty_like(weight)
    norms, errors, histogram, moments = [], [], {}, []
    for begin in range(0, weight.shape[1], 256):
        original = weight[:, begin:begin + 256]
        current, exponent = encode_fp8(original)
        reconstructed[:, begin:begin + 256] = current
        norms.append(window_maximum(original.float().abs().sum(-1)))
        errors.append(window_maximum((original.float() - current.float()).abs().sum(-1)))
        if with_moments:
            moments.append(precision_moments.weight_statistics(original, current))
        add_histogram(histogram, exponent)
    return PreparedWeight(reconstructed, torch.stack(norms), torch.stack(errors), histogram,
                          torch.stack(moments) if with_moments else None)


def select_cohorts(values, reconstructed, weight_l1, weight_error_l1, *, tolerance, cohort):
    """Cheap exact-arithmetic error-bound formula; FP32 here is not a certificate.

    |(A-Aq)W + Aq(W-Wq)| <= max|A-Aq| * sum|W| + max|Aq| * sum|W-Wq|.
    Normalize the criterion by max|A|*sum|W|, an operand upper scale rather than
    the actual output magnitude. No receiving matrix product selects a mode.
    """
    a = values.float()
    q = reconstructed.float()
    maximum = a.abs().amax(-1)
    error = (a - q).abs().amax(-1)
    quantized_maximum = q.abs().amax(-1)
    bound = error[:, None] * weight_l1 + quantized_maximum[:, None] * weight_error_l1
    scale = maximum[:, None] * weight_l1
    fits = bound <= tolerance * scale
    padded = F.pad(fits, (0, 0, 0, (-len(values)) % cohort), value=True)
    selected = padded.reshape(-1, cohort, len(weight_l1)).all(1)
    relative = torch.where(scale > 0, bound / scale, torch.where(bound == 0, 0, torch.inf))
    return selected, relative


def multiply(values, weight, prepared=None, *, policy="adaptive-paired", tolerance=0.04,
             cohort=32, counters=None, row_buffer=0, resident_rows=1024, execution="oracle"):
    if (policy not in POLICIES or values.ndim != 2 or weight.ndim != 2
            or values.shape[1] != weight.shape[1] or min(*values.shape, *weight.shape) < 1
            or values.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or not math.isfinite(tolerance) or tolerance < 0 or cohort < 1):
        raise ValueError("invalid BF16 matrix, policy, tolerance or cohort")
    if (execution not in ("oracle", "packed") or row_buffer < 0 or resident_rows < 1
            or (row_buffer and resident_rows % row_buffer) or (execution == "packed" and not row_buffer)):
        raise ValueError("invalid row-buffer residency or execution contract")
    if prepared is None and policy != "bf16":
        raise ValueError("quantized policies require prepared immutable-weight metadata")
    if prepared is not None and (prepared.reconstructed.shape != weight.shape
                                 or prepared.reconstructed.device != weight.device):
        raise ValueError("prepared weight shape/device differs")
    moment_policy = policy.startswith(("moment", "diagonal"))
    adaptive_policy = moment_policy or policy.startswith("adaptive")
    if moment_policy and prepared.moments is None:
        raise ValueError("moment policies require prepared receiver statistics")
    load_native()
    counters = counters if counters is not None else Counters()
    rows, reduction = values.shape
    columns = weight.shape[0]
    windows = (columns + 127) // 128
    output = torch.zeros(rows, columns, device=values.device, dtype=torch.float32)
    row_counts = torch.arange(math.ceil(rows / cohort), device=values.device)
    row_counts = (rows - row_counts * cohort).clamp_max(cohort)

    def product(a, w):
        counters.reference_gemm_calls += 1
        return torch.ops.tensor_inference_engine.gemm(a, w.T, True)

    for pair, begin in enumerate(range(0, reduction, 256)):
        stop = min(begin + 256, reduction)
        width = stop - begin
        high_calls = math.ceil(width / 128)
        original = values[:, begin:stop]
        original_weight = weight[:, begin:stop]
        shape = (len(row_counts), windows)
        quantized = quantized_weight = None
        if policy == "bf16" or (adaptive_policy and width <= 128):
            selected = torch.zeros(shape, device=values.device, dtype=torch.bool)
        else:
            quantized, exponent = encode_fp8(original)
            quantized_weight = prepared.reconstructed[:, begin:stop]
            counters.candidate_activation_rows += rows
            add_histogram(counters.activation_scale_exponents, exponent)
            if moment_policy:
                selected, relative = precision_moments.select_cohorts(
                    original, quantized, prepared.moments[pair], tolerance=tolerance,
                    cohort=cohort, centered=policy.startswith("moment"))
                bins = torch.bucketize(relative.flatten(), torch.tensor(
                    [0, 0.01, 0.02, 0.04, 0.08, 0.16, 1], device=values.device))
                add_histogram(counters.normalized_moment_score_histogram, bins)
            elif policy.startswith("adaptive"):
                selected, relative = select_cohorts(
                    original, quantized, prepared.l1[pair], prepared.error_l1[pair],
                    tolerance=tolerance, cohort=cohort)
                # Diagnostic bins only; no division is needed by the selector.
                bins = torch.bucketize(relative.flatten(), torch.tensor(
                    [0, 0.01, 0.02, 0.04, 0.08, 0.16, 1], device=values.device))
                add_histogram(counters.normalized_bound_histogram, bins)
            else:
                selected = torch.ones(shape, device=values.device, dtype=torch.bool)
        selected_rows = selected.repeat_interleave(cohort, 0)[:rows]
        mask = selected_rows.repeat_interleave(128, 1)[:, :columns]
        any_low, all_low = bool(selected.any()), bool(selected.all())
        low_vectors = int((selected * row_counts[:, None]).sum())
        counters.reference_vectors += rows * windows * high_calls
        low_calls = high_calls if policy.endswith("split") else 1
        counters.selected_vectors += (rows * windows - low_vectors) * high_calls + low_vectors * low_calls
        counters.mode_groups += selected.numel()
        counters.fp8_mode_groups += int(selected.sum())
        counters.within_pair_mode_transitions += int((selected[1:] != selected[:-1]).sum())
        counters.useful_products += rows * columns * width
        counters.fp8_products += int(mask.sum()) * width
        if row_buffer:
            row_precision_schedule.account(selected_rows, width, counters.schedule,
                                           buffer_rows=row_buffer, resident_rows=resident_rows,
                                           candidate_prepared=quantized is not None)
        if execution == "packed":
            output = row_precision_schedule.execute(
                original, quantized, original_weight, quantized_weight, output, selected_rows, product,
                split_low=policy.endswith("split"), buffer_rows=row_buffer, resident_rows=resident_rows)
            continue
        high = low = None
        if not all_low:
            high = output
            for offset in range(0, width, 128):
                high = high + product(original[:, offset:offset + 128],
                                      original_weight[:, offset:offset + 128])
        if any_low:
            low = output
            step = 128 if policy.endswith("split") else 256
            for offset in range(0, width, step):
                low = low + product(quantized[:, offset:offset + step],
                                    quantized_weight[:, offset:offset + step])
        output = low if all_low else high if not any_low else torch.where(mask, low, high)
    return output
