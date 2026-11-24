#!/usr/bin/env python3
"""Record whether Inductor preserves the rounding needed by a BF16-cell guard."""

import argparse
from pathlib import Path

import torch

from activation_pilot import ENGINE, ROOT, digest, revision, write_json
from selective_bf16_reference import rounding_cell_accepts


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    candidate = torch.tensor([1 + 2.0 ** -8 - 2.0 ** -23, 1 + 2.0 ** -8 + 2.0 ** -23,
                              -1 - 2.0 ** -8 - 2.0 ** -23], device="cuda")
    error = torch.full_like(candidate, 3 * 2.0 ** -23, dtype=torch.float64)
    expected = rounding_cell_accepts(candidate, error)
    results = {}
    for preserve in (False, True):
        torch._dynamo.reset()
        compiled = torch.compile(rounding_cell_accepts, dynamic=False, fullgraph=True,
                                 options={"emulate_precision_casts": preserve})
        actual = compiled(candidate, error)
        results[str(preserve)] = {"accepted": actual.tolist(),
                                  "differences_from_eager": int((actual != expected).sum())}
    report = {"schema": 1, "status": "complete", "experiments": revision(ROOT),
        "engine": revision(ENGINE), "torch": torch.__version__, "torch_git": torch.version.git_version,
        "gpu": torch.cuda.get_device_name(), "candidate_fp32": candidate.tolist(),
        "candidate_words_int32": candidate.view(torch.int32).tolist(),
        "error_fp64": error.tolist(), "eager_accepted": expected.tolist(), "compiled": results,
        "source_sha256": {name: digest(ROOT / "scripts" / name) for name in (
            "precision_cast_audit.py", "selective_bf16_reference.py")},
        "limitations": ["Constructed guard witness, not a network quality or timing result",
            "Default cast elimination is documented compiler behavior, not a newly established compiler bug",
            "A correctly rounded BF16 observation is a semantic requirement of this guard"]}
    if expected.any() or results["True"]["differences_from_eager"]:
        report["status"] = "invalid"
    write_json(args.output, report)
    print(results)


if __name__ == "__main__":
    main()
