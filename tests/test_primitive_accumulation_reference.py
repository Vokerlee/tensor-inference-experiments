"""Check the primitive's exact-integer path, fallback and fixed FP32 boundary."""

import sys
from fractions import Fraction
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from primitive_accumulation_reference import Counters, multiply


def test_guarded_result_matches_rational_oracle_with_column_tail():
    generator = torch.Generator().manual_seed(147)
    a = torch.randint(-12, 13, (3, 4, 32), generator=generator)
    w = torch.randint(-12, 13, (131, 4, 32), generator=generator)
    ae = torch.randint(-5, 1, (3, 4), generator=generator, dtype=torch.int32)
    we = torch.randint(-5, 1, (131, 4), generator=generator, dtype=torch.int32)
    counts = Counters()
    actual = multiply(a, ae, w, we, accumulator_width=32, counters=counts)
    for row in range(3):
        for column in range(131):
            exact = sum(Fraction(int((a[row, group] * w[column, group]).sum()))
                        * Fraction(2) ** int(ae[row, group] + we[column, group])
                        for group in range(4))
            assert actual[row, column] == torch.tensor(float(exact), dtype=torch.float32)
    assert counts.accepted_vectors == counts.vector_groups == 6


def test_rejected_width_uses_the_identical_group_float_baseline():
    codes = torch.full((2, 4, 32), 127, dtype=torch.int64)
    exponents = torch.zeros(2, 4, dtype=torch.int32)
    counts = Counters()
    baseline = multiply(codes, exponents, codes, exponents)
    fallback = multiply(codes, exponents, codes, exponents, accumulator_width=8, counters=counts)
    assert torch.equal(fallback, baseline)
    assert counts.accepted_vectors == 0


def test_each_primitive_emits_fp32_instead_of_retaining_exact_whole_k_state():
    # The first primitive emits 2^24 after rounding 2^24+1; the second emits -2^24.
    a = torch.zeros(1, 8, 32, dtype=torch.int64)
    w = torch.zeros_like(a)
    a[0, 0, 0] = w[0, 0, 0] = 1
    a[0, 1, 0] = w[0, 1, 0] = 1
    a[0, 4, 0] = 1
    w[0, 4, 0] = -1
    ae = torch.tensor([[24, 0, 0, 0, 24, 0, 0, 0]], dtype=torch.int32)
    we = torch.zeros_like(ae)
    sliced = multiply(a, ae, w, we, accumulator_width=32, reduction_width=128)
    whole = multiply(a, ae, w, we, accumulator_width=32, reduction_width=256)
    assert sliced.item() == 0
    assert whole.item() == 1


def test_invalid_integer_code_range_is_rejected():
    codes = torch.full((1, 4, 32), 128, dtype=torch.int64)
    exponent = torch.zeros(1, 4, dtype=torch.int32)
    with pytest.raises(ValueError, match="magnitudes"):
        multiply(codes, exponent, codes, exponent)


def test_cached_source_statistics_preserve_the_guarded_result():
    generator = torch.Generator().manual_seed(91)
    a = torch.randint(-127, 128, (3, 4, 32), generator=generator)
    w = torch.randint(-127, 128, (5, 4, 32), generator=generator)
    ae = torch.randint(-10, 0, (3, 4), generator=generator, dtype=torch.int32)
    we = torch.randint(-10, 0, (5, 4), generator=generator, dtype=torch.int32)
    stats = (a.abs().amax(-1), a.abs().sum(-1))
    actual = multiply(a, ae, w, we, accumulator_width=24, a_statistics=stats)
    reference = multiply(a, ae, w, we, accumulator_width=24)
    assert torch.equal(actual, reference)
    maximum_only = multiply(a, ae, w, we, accumulator_width=32,
                            a_statistics=(stats[0], None), bound_kind="max")
    assert torch.equal(maximum_only, multiply(a, ae, w, we, accumulator_width=32, bound_kind="max"))
