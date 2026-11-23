"""Check fused native metadata planning against independent tensor bounds."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from accumulation_scale_diagnostics import conservative_group_bounds, encode, fits_bound
from accumulation_segments import greedy_ends
from integer_segment_backend import decode_ends, load_native, plan, weight_statistics

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module", autouse=True)
def native():
    load_native()


def metadata(groups=9, columns=131):
    torch.manual_seed(731)
    a, ae = encode(torch.randn(17, groups * 32, device="cuda"), "int8")
    w, we = encode(torch.randn(columns, groups * 32, device="cuda"), "int8")
    a[0].zero_()
    w[0].zero_()
    return a, ae, w, we


@pytest.mark.parametrize("groups_per_slice", [1, 4, 8])
@pytest.mark.parametrize("bits", [8, 20, 24, 32, 48])
@pytest.mark.parametrize("bound_kind", ["max", "min"])
def test_native_partitions(groups_per_slice, bits, bound_kind):
    a, ae, w, we = metadata()
    maximum, l1 = a.abs().amax(-1).int(), a.abs().sum(-1).int()
    weights = weight_statistics(w, we)
    masks, oversized, valid = plan(maximum, ae, weights, groups_per_slice=groups_per_slice,
                                   bits=bits, l1=l1 if bound_kind == "min" else None)
    assert valid.all()
    decoded = decode_ends(masks, groups_per_slice, a.shape[1])
    for begin in range(0, a.shape[1], groups_per_slice):
        end = min(begin + groups_per_slice, a.shape[1])
        bounds, lo, hi = conservative_group_bounds(a[:, begin:end], ae[:, begin:end],
                                                  w[:, begin:end], we[:, begin:end],
                                                  bound_kind=bound_kind)
        expected = greedy_ends(bounds, lo, hi, bits)
        assert torch.equal(decoded[..., begin:end], expected)
        single_fits = fits_bound(bounds[..., None], (hi - lo)[..., None], bits)
        indices = torch.arange(end - begin, device="cuda")
        expected_oversized = ((~single_fits).long() << indices).sum(-1).byte()
        assert torch.equal(oversized[..., begin // groups_per_slice], expected_oversized)


def test_stream_graph_meta_and_invalid_metadata():
    a, ae, w, we = metadata(groups=8)
    maximum = a.abs().amax(-1).int()
    weights = weight_statistics(w, we)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(100000)
        copied = maximum.clone()
        result = plan(copied, ae, weights)
    stream.synchronize()
    expected = plan(maximum, ae, weights)
    for actual, reference in zip(result, expected, strict=True):
        assert torch.equal(actual, reference)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = plan(maximum, ae, weights)
    graph.replay()
    assert torch.equal(captured[0], expected[0])
    torch.library.opcheck(torch.ops.tensor_inference_engine.integer_segment_plan.default,
                         (maximum, ae, weights["l1"], weights["low"], weights["high"]))
    bad = maximum.clone()
    bad[0, 0] = -1
    masks, oversized, valid = plan(bad, ae, weights)
    assert not valid[0, :, 0].any()
    assert (masks[0, :, 0] == 15).all() and (oversized[0, :, 0] == 15).all()
    assert all(t.numel() == 0 for t in plan(maximum[:0], ae[:0], weights))
    with pytest.raises(RuntimeError, match="int32"):
        plan(maximum.long(), ae, weights)
