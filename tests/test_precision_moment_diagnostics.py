"""Local oracle sampling and immutable model arithmetic checks."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from paired_precision_reference import encode_fp8
from precision_moment_diagnostics import MomentDiagnostics


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA engine required")
@torch.inference_mode()
def test_diagnostics_keep_original_forward_and_record_independent_local_errors():
    torch.manual_seed(802)
    module = torch.nn.Linear(257, 129, device="cuda", dtype=torch.bfloat16)
    values = torch.randn(6, 257, device="cuda").bfloat16()
    expected = module(values).clone()
    collector = MomentDiagnostics(3)
    collector.observe("layer", module, values)
    collector.observe("layer", module, values)
    assert torch.equal(module(values), expected)
    assert len(collector.records["layer"]) == 1
    record = collector.records["layer"][0]
    assert record["sampled_row_indices"] == [0, 2, 5]
    assert len(record["slices"]) == 1
    a = values[[0, 2, 5], :256]
    w = module.weight[:, :256]
    z = encode_fp8(a)[0]
    q = encode_fp8(w)[0]
    high = a.cpu().double() @ w.cpu().double().T
    low = z.cpu().double() @ q.cpu().double().T
    error = (high - low).square()
    means = torch.stack([error[:, :128].mean(-1), error[:, 128:].mean(-1)], -1)
    actual = torch.tensor(record["slices"][0]["actual_error_energy"], dtype=torch.float64)
    torch.testing.assert_close(actual, means, rtol=5e-4, atol=1e-5)
