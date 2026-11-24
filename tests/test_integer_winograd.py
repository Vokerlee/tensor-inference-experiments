"""Independent integer accumulation, K tails and group-scale boundaries."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from integer_winograd import IntegerWeights, multiply, prepare
from winograd_rounding import prepare as prepare_geometry


def test_worst_case_integer_partial_is_exact_with_bf16_operands_fp32_sum():
    source = torch.full((2, 3, 256), 127.)
    weights = IntegerWeights(torch.full((2, 256, 7), 127, dtype=torch.int8),
                             torch.zeros((2, 1, 7), dtype=torch.int32), 256, 256)
    actual, _ = multiply(source, weights)
    expected = source.long() @ weights.codes.long()
    torch.testing.assert_close(actual.double(), expected.double(), rtol=0, atol=0)


def test_each_k_block_has_its_own_postproduct_scale_and_tail_mask():
    torch.manual_seed(771)
    source = torch.randn(2, 3, 300)
    coefficients = torch.randn(2, 300, 5)
    weights = prepare(coefficients)
    actual, stats = multiply(source, weights)
    grouped = torch.nn.functional.pad(source, (0, 212)).reshape(2, 3, 2, 256)
    maximum = grouped.abs().amax(-1)
    exponents = (maximum / 127).log2().ceil().int()
    scales = torch.ldexp(torch.ones_like(maximum), exponents)
    codes = (grouped / scales[..., None]).round().clamp(-127, 127).long()
    expected = torch.zeros(2, 3, 5)
    weight_scales = torch.ldexp(torch.ones_like(weights.exponents, dtype=torch.float32), weights.exponents)
    for block in range(2):
        integer = codes[:, :, block] @ weights.codes[:, block * 256:(block + 1) * 256].long()
        expected += integer.float() * scales[:, :, block, None] * weight_scales[:, block, None]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.count_nonzero(weights.codes[:, 300:]) == 0
    assert stats["padded_channels"] == 512


def test_cross_plane_rounding_uses_the_same_native_block_scale_contract():
    torch.manual_seed(86)
    weights = prepare(torch.randn(16, 37, 3))
    geometry = prepare_geometry(weights.reconstructed(), rank=1)
    result, stats = multiply(torch.randn(16, 2, 37), weights, bits=4, geometry=geometry)
    assert result.shape == (16, 2, 3)
    assert stats["activation_groups"] == 32
    assert stats["rounding"]["source_zeros_changed"] == 0
    zeros, _ = multiply(torch.zeros(16, 2, 37), weights, bits=8)
    assert torch.count_nonzero(zeros) == 0


def test_complete_integer_convolution_consumes_bounded_tiles_and_exact_k_tails():
    from integer_winograd import convolution
    from winograd_reference import transform_weights, transforms

    torch.manual_seed(13)
    source = torch.randint(-2, 3, (1, 32, 5, 7)).float()
    weight = torch.zeros(3, 32, 3, 3)
    weight[:, :, 1, 1] = 1
    weights = prepare(transform_weights(weight, transforms(dtype=torch.float32)))
    expected = torch.nn.functional.conv2d(source, weight, padding=1)
    for streaming in (False, True):
        actual, stats = convolution(source, weights, tile_batch=3, streaming_inverse=streaming)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert stats["tile_batches"] > 1
        assert stats["activation_groups"] > 0


def test_invalid_backend_codes_are_rejected_before_a_product(monkeypatch):
    import integer_winograd
    import pytest
    from plane_encoding import PlaneEncoding

    weights = prepare(torch.ones(16, 256, 3))
    source = torch.zeros(16, 1, 256)
    invalid = PlaneEncoding(torch.full_like(source, 128), torch.zeros(16, 1, 1, dtype=torch.int32), {})
    monkeypatch.setattr(integer_winograd, "encode_planes", lambda *args, **kwargs: invalid)
    monkeypatch.setattr(integer_winograd, "product", lambda *args: pytest.fail("invalid codes reached product"))
    with pytest.raises(ValueError, match="code range/grid"):
        multiply(source, weights, geometry=object())


def test_native_grid_boundary_retains_the_encoders_actual_exponent():
    import pytest
    from winograd_shift_rounding import ShiftGeometry

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    source = torch.zeros(16, 1, 256, device="cuda")
    source[:, 0, 0] = torch.nextafter(torch.tensor(127., device="cuda"), torch.tensor(float("inf"), device="cuda"))
    weights = IntegerWeights(torch.ones(16, 256, 3, dtype=torch.int8, device="cuda"),
                             torch.zeros(16, 1, 3, dtype=torch.int32, device="cuda"), 256, 256)
    geometry = ShiftGeometry(torch.zeros(256, 16, device="cuda", dtype=torch.int16), "cuda", rank=0)
    actual, stats = multiply(source, weights, geometry=geometry)
    torch.testing.assert_close(actual, torch.full_like(actual, 128), rtol=0, atol=0)
    assert stats["exponent_histogram"] == {"1": 16}
