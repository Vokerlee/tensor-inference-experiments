import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from selective_reference_models import compare_controls, instrumented, prepare_layers


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
@pytest.mark.parametrize("policy", ["bf16", "dynamic8", "live4"])
def test_branched_model_preserves_observed_outputs_control_and_original_sources(policy):
    torch.manual_seed(826)
    first = torch.nn.Linear(32, 64).cuda().bfloat16()
    second = torch.nn.Linear(64, 32).cuda().bfloat16()
    layers = {"first": first, "second": second}
    cache = prepare_layers(layers)
    source = torch.randn(1, 64, 5, 32, device="cuda").bfloat16()
    before = source.clone()

    def execute():
        return second(torch.nn.functional.gelu(first(source))) + source

    with instrumented(layers, cache, "full-direct", policy, record=True) as reference:
        expected = execute()
    for method in ("full-tiled", "hybrid-chains4", "hybrid-tensorcore"):
        with instrumented(layers, cache, method, policy, record=True,
                          reference_outputs=reference["outputs"]) as audit:
            actual = execute()
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        assert all(row["output_mismatches"] == 0 for row in audit["calls"])
        assert all(not any(row["mismatches"].values())
                   for row in compare_controls(audit["controllers"], reference["controllers"]))
        assert all(value.item() == 0 for value in audit["invalid"])
    assert torch.equal(source.view(torch.int16), before.view(torch.int16))
