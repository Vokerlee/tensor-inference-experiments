"""Operand-bounded selective emulation at an explicit BF16 observation boundary.

The reference is ordered RN FP32 K128 primitives, not the exact real dot product.
Only the four-chain RN candidate has an audited operation graph here. Tensor
Core candidates use an explicitly conditional FP32 rounding model.
"""

import math
from dataclasses import dataclass

import torch

from integer_segment_backend import load_native


U32 = 2.0 ** -24
ETA32 = 2.0 ** -149


def gamma(count, precision=24):
    product = count * 2.0 ** -precision
    if not 0 <= product < 1:
        raise ValueError("rounding depth outside the gamma bound domain")
    return math.nextafter(product / (1 - product), math.inf) if count else 0.0


def up(values):
    return torch.where(values == 0, values, torch.nextafter(values, torch.full_like(values, math.inf)))


def add_up(first, second):
    return up(first + second)


def multiply_up(first, second):
    return up(first * second)


def positive_sum_upper(values, dimension):
    depth = values.shape[dimension]
    denominator = math.nextafter(1 - gamma(depth, 53), 0.0)
    return up(values.sum(dimension) / denominator)


def bad_source(values, dimension):
    words = values.view(torch.int16).int() & 0x7fff
    # Exclude nonfinite inputs and BF16 subnormals that a fast device might flush.
    return (((words > 0) & (words < 0x80)) | (words >= 0x7f80)).any(dimension)


@dataclass
class WeightBound:
    values: torch.Tensor
    l1: torch.Tensor
    maximum: torch.Tensor
    l2: torch.Tensor
    bad: torch.Tensor


def prepare(weight):
    if weight.ndim != 2 or weight.dtype != torch.bfloat16 or not weight.is_contiguous():
        raise ValueError("contiguous BF16 [K,N] weight required")
    values = weight.double().abs()
    squares = positive_sum_upper(values.square(), 0)
    return WeightBound(weight, positive_sum_upper(values, 0), values.amax(0),
                       up(squares.sqrt()), bad_source(weight, 0))


def norm_bound(source, weight):
    values = source.double().abs()
    maximum = values.amax(1)
    l1 = positive_sum_upper(values, 1)
    l2 = up(positive_sum_upper(values.square(), 1).sqrt())
    return torch.minimum(torch.minimum(multiply_up(maximum[:, None], weight.l1[None]),
                                       multiply_up(l1[:, None], weight.maximum[None])),
                         multiply_up(l2[:, None], weight.l2[None]))


def rounding_cell_accepts(candidate, error):
    """Strict exact BF16 cell boundaries; never cast interval endpoints to BF16."""
    rounded = candidate.bfloat16()
    word = rounded.view(torch.int16).int() & 0xffff
    exponent = word & 0x7f80
    negative = (word & 0x8000) != 0
    previous = torch.where(negative, word + 1, word - 1).to(torch.int16).view(torch.bfloat16).double()
    following = torch.where(negative, word - 1, word + 1).to(torch.int16).view(torch.bfloat16).double()
    center = rounded.double()
    lower = (previous + center) * 0.5
    upper = (following + center) * 0.5
    value = candidate.double()
    low = torch.nextafter(value - error, torch.full_like(error, -math.inf))
    high = torch.nextafter(value + error, torch.full_like(error, math.inf))
    low = torch.where(error == 0, value, low)
    high = torch.where(error == 0, value, high)
    return ((exponent > 0) & (exponent < 0x7f00) & torch.isfinite(error)
            & (error >= 0) & (low > lower) & (high < upper))


def candidate_product(source, weight, primitive, backend):
    if backend == "chains4":
        return torch.ops.tensor_inference_engine.ordered_gemm(source, weight, primitive, 4)[0]
    if backend != "tensorcore":
        raise ValueError("candidate requires chains4 or tensorcore")
    result = torch.zeros(source.shape[0], weight.shape[1], device=source.device, dtype=torch.float32)
    for begin in range(0, source.shape[1], primitive):
        stop = min(begin + primitive, source.shape[1])
        result = result + torch.ops.tensor_inference_engine.gemm(
            source[:, begin:stop], weight[begin:stop], True)
    return result


@dataclass
class Observation:
    output: torch.Tensor
    accepted: torch.Tensor
    error_bound: torch.Tensor
    fallback_outputs: int
    invalid_indices: torch.Tensor


def full_reference(source, weight, bias=None, *, primitive=128, tiled=False):
    load_native()
    output = torch.ops.tensor_inference_engine.ordered_gemm(source, weight, primitive, 1, None, None, True, tiled)[0]
    if bias is not None:
        output = output + bias.float()
    return output.bfloat16()


def hybrid(source, weight, bias=None, *, primitive=128, backend="chains4", filter_backend="native"):
    load_native()
    if (source.ndim != 2 or source.dtype != torch.bfloat16 or not source.is_contiguous()
            or source.shape[1] != weight.values.shape[0] or not source.shape[1]):
        raise ValueError("nonempty K and contiguous BF16 source with matching weight required")
    if primitive < 1 or backend not in ("chains4", "tensorcore") or filter_backend not in ("native", "eager", "compiled"):
        raise ValueError("invalid primitive, candidate or filter")
    if bias is not None and (bias.shape != (weight.values.shape[1],)
            or bias.dtype not in (torch.bfloat16, torch.float32) or bias.device != source.device):
        raise ValueError("matching BF16/FP32 bias vector required")
    reduction = source.shape[1]
    pieces = math.ceil(reduction / primitive)
    width = min(primitive, reduction)
    # Both contracts reset primitive accumulators and merge their FP32 results.
    reference_depth = width + pieces
    fast_depth = math.ceil(width / 4) + 2 + pieces if backend == "chains4" else width + pieces
    precision = 24 if backend == "chains4" else 23
    reference_gamma = gamma(reference_depth)
    fast_gamma = gamma(fast_depth, precision)
    coefficient = math.nextafter(reference_gamma + fast_gamma, math.inf)
    denominator = math.nextafter(1 - max(reference_gamma, fast_gamma), 0.0)
    underflow = math.nextafter(4 * (reduction + 4 * pieces) * ETA32 / denominator, math.inf)
    if backend == "tensorcore":
        underflow *= 2.0 ** 23  # Conditional model also allows flushing tiny intermediate values.
    fast = candidate_product(source, weight.values, primitive, backend)
    if filter_backend == "native":
        norms = torch.ops.tensor_inference_engine.bf16_norms(source)
        projected, accepted, error = torch.ops.tensor_inference_engine.bf16_observe(
            fast, norms, weight.norms, coefficient, underflow,
            bias.float().contiguous() if bias is not None else None)
    else:
        fn = compiled_filter if filter_backend == "compiled" else filter_candidate
        projected, accepted, error = fn(source, weight, fast, bias, coefficient, underflow)
    indices = (~accepted).flatten().nonzero().flatten()
    count = indices.numel()
    invalid = torch.zeros(1, device=source.device, dtype=torch.int64)
    if count == projected.numel():
        repaired, invalid = torch.ops.tensor_inference_engine.ordered_gemm(source, weight.values, primitive, 1)
    elif count:
        repaired, invalid = torch.ops.tensor_inference_engine.ordered_gemm(
            source, weight.values, primitive, 1, indices, fast, False)
    else:
        return Observation(projected, accepted, error, 0, invalid)
    if bias is not None:
        repaired = repaired + bias.float()
    return Observation(torch.where(accepted, projected, repaired.bfloat16()), accepted, error, count, invalid)


def filter_candidate(source, weight, fast, bias, coefficient, underflow):
    scale = norm_bound(source, weight)
    error = add_up(multiply_up(scale, coefficient), underflow)
    zero = scale == 0
    fast = torch.where(zero, torch.zeros_like(fast), fast)
    observed = fast
    if bias is not None:
        observed = fast + bias.float()
        error = add_up(multiply_up(error, 1 + U32),
            add_up(multiply_up(add_up(fast.double().abs(), bias.double().abs()), 2 * U32), 2 * ETA32))
    error = torch.where(zero, torch.zeros_like(error), error)
    domain = (scale < torch.finfo(torch.float32).max / 4)
    domain &= ~bad_source(source, 1)[:, None] & ~weight.bad[None]
    if bias is not None:
        domain &= torch.isfinite(bias)[None]
        domain &= scale + bias.double().abs() < torch.finfo(torch.float32).max / 4
    accepted = domain & (rounding_cell_accepts(observed, error) | (zero & torch.isfinite(observed)))
    return observed.bfloat16(), accepted, error


compiled_filter = torch.compile(filter_candidate, dynamic=False, fullgraph=True,
                                options={"emulate_precision_casts": True})


@dataclass
class NativeWeight:
    values: torch.Tensor
    norms: torch.Tensor


def prepare_native(weight):
    if weight.ndim != 2 or weight.dtype != torch.bfloat16 or not weight.is_contiguous():
        raise ValueError("contiguous BF16 [K,N] weight required")
    load_native()
    return NativeWeight(weight, torch.ops.tensor_inference_engine.bf16_norms(weight, True))


def unsafe_compiled_filter_for_audit():
    """Test-only baseline: fusion may remove the rounding used by the predicate."""
    return torch.compile(filter_candidate, dynamic=False, fullgraph=True,
                         options={"emulate_precision_casts": False})
