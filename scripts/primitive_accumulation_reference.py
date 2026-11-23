"""Numerical oracles for grouped codes with a mandatory FP32 vector output.

INT8/E2M1 codes and K=128/256 are explicit research contracts, not a claim that
the target FP8 array implements these integer modes. No GPU timing here models
the cost of a hardware implementation.
"""

from dataclasses import dataclass

import torch

from accumulation_scale_diagnostics import guarded_rows


@dataclass
class Counters:
    vector_groups: int = 0
    accepted_vectors: int = 0
    useful_products: int = 0
    accepted_products: int = 0
    reference_group_merges: int = 0
    replaced_group_merges: int = 0
    output_scalars: int = 0
    changed_output_scalars: int = 0
    squared_output_difference: float = 0.0
    squared_reference_output: float = 0.0
    max_output_difference: float = 0.0
    source_groups_quantized: int = 0
    lowered_group_slots: int = 0
    valid_lowered_group_uses: int = 0


def multiply(a_codes: torch.Tensor, a_exponents: torch.Tensor,
             w_codes: torch.Tensor, w_exponents: torch.Tensor, *,
             reduction_width: int = 128, accumulator_width: int | None = None,
             counters: Counters | None = None, a_statistics=None,
             bound_kind: str = "min") -> torch.Tensor:
    """Reduce represented operands, emitting FP32 after every K slice.

    Inputs are [rows, groups, 32] activation and [columns, groups, 32] weight
    codes, with one power-of-two exponent per group. Group integer products
    fit FP32 exactly for these code ranges. The baseline scales and sequentially
    merges group sums in FP32 within a primitive. The guarded variant instead
    left-aligns integer sums and converts once, using a float64 exact-integer
    oracle only when a sufficient signed-width guard accepts all 128 columns.
    Emitted FP32 slices are merged in FP32 on BOTH paths; no integer state is
    retained across matrix primitive outputs.
    """
    if reduction_width not in (128, 256):
        raise ValueError("this reference tests explicit K=128 or K=256 contracts")
    if accumulator_width is not None and not 2 <= accumulator_width <= 48:
        raise ValueError("exact oracle supports accumulator widths 2..48")
    if (a_codes.ndim != 3 or w_codes.ndim != 3 or a_codes.shape[-1] != 32
            or w_codes.shape[1:] != a_codes.shape[1:]
            or a_exponents.shape != a_codes.shape[:-1]
            or w_exponents.shape != w_codes.shape[:-1]):
        raise ValueError("expected matching group32 codes and per-group exponents")
    if a_codes.is_floating_point() or w_codes.is_floating_point():
        raise ValueError("operand codes must be integers")
    if torch.any((a_codes < -127) | (a_codes > 127)) or torch.any((w_codes < -127) | (w_codes > 127)):
        raise ValueError("group integer dot oracle requires code magnitudes <=127")
    rows, groups, _ = a_codes.shape
    columns = w_codes.shape[0]
    if rows == 0 or columns == 0 or groups == 0:
        raise ValueError("empty reductions/outputs are outside this pilot contract")
    output = torch.zeros(rows, columns, dtype=torch.float32, device=a_codes.device)
    groups_per_slice = reduction_width // 32
    for begin in range(0, groups, groups_per_slice):
        end = min(begin + groups_per_slice, groups)
        a = a_codes[:, begin:end]
        ae = a_exponents[:, begin:end]
        w = w_codes[:, begin:end]
        we = w_exponents[:, begin:end]
        if accumulator_width is not None:
            statistics = None if a_statistics is None else (
                a_statistics[0][:, begin:end],
                None if a_statistics[1] is None else a_statistics[1][:, begin:end])
            fits, anchors = guarded_rows(a, ae, w, we, accumulator_width, 128,
                                        a_statistics=statistics, bound_kind=bound_kind)
        for column in range(0, columns, 128):
            stop = min(column + 128, columns)
            reference = torch.zeros(rows, stop - column, dtype=torch.float32, device=a.device)
            integer = torch.zeros_like(reference, dtype=torch.float64)
            if accumulator_width is not None:
                accepted = fits[:, column // 128]
                anchor = anchors[:, column // 128]
            for group in range(end - begin):
                # |sum of any prefix| <=32*127^2 <2^24: FP32 is exact here.
                dot = a[:, group].float() @ w[column:stop, group].float().T
                exponent = ae[:, group, None] + we[None, column:stop, group]
                term = torch.ldexp(dot, exponent)
                reference = reference + term
                if accumulator_width is not None:
                    active = (a[:, group].abs().amax(-1)[:, None] > 0) & (
                        w[column:stop, group].abs().amax(-1)[None] > 0)
                    shift = torch.where(active & accepted[:, None],
                                        exponent - anchor[:, None], 0)
                    contribution = torch.where(accepted[:, None], dot.double(), 0)
                    integer = integer + torch.ldexp(contribution, shift)
            if accumulator_width is not None:
                exact_output = torch.ldexp(integer, anchor[:, None]).float()
                selected = torch.where(accepted[:, None], exact_output, reference)
                if counters is not None:
                    difference = selected.double() - reference.double()
                    counters.output_scalars += reference.numel()
                    counters.changed_output_scalars += int(torch.count_nonzero(difference))
                    counters.squared_output_difference += float(difference.square().sum())
                    counters.squared_reference_output += float(reference.double().square().sum())
                    counters.max_output_difference = max(counters.max_output_difference,
                                                          float(difference.abs().amax()))
                reference = selected
            # Deliberately preserve the fixed FP32 boundary between K slices.
            output[:, column:stop] += reference
            if counters is not None:
                accepted_rows = int(accepted.sum()) if accumulator_width is not None else 0
                useful_k = (end - begin) * 32
                counters.vector_groups += rows
                counters.accepted_vectors += accepted_rows
                counters.useful_products += rows * (stop - column) * useful_k
                counters.accepted_products += accepted_rows * (stop - column) * useful_k
                merges = max(end - begin - 1, 0)
                counters.reference_group_merges += rows * (stop - column) * merges
                counters.replaced_group_merges += accepted_rows * (stop - column) * merges
    return output
