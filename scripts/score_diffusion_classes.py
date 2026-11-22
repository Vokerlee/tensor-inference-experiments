#!/usr/bin/env python3
"""Measure conditional-class consistency of generated images with fixed classifiers."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from activation_pilot import MODELS, digest, write_json


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    report = {"schema": 1, "script_sha256": digest(Path(__file__)), "classifiers": {},
              "limitations": ["Conditional-class diagnostic, not FID or aesthetic quality",
                              "Small pilots do not establish a general method ranking"]}
    for name, (factory, weights) in MODELS.items():
        model = factory(weights=weights).eval().to(device="cuda", dtype=torch.bfloat16)
        classifier = {"weights": weights.url, "runs": []}
        for directory in args.run:
            source = json.loads((directory / "report.json").read_text())
            labels = torch.tensor(source["settings"]["classes"], device="cuda")
            records = []
            for case in source["cases"]:
                filename = directory / f"{case['case'].replace(':', '-')}.npz"
                with np.load(filename) as data:
                    images = torch.from_numpy(data["images"]).permute(0, 3, 1, 2)
                inputs = weights.transforms()(images).to(device="cuda", dtype=torch.bfloat16)
                logits = model(inputs).float()
                probabilities = logits.softmax(-1)
                expected = probabilities.gather(1, labels[:, None]).squeeze(1)
                predicted = logits.argmax(-1)
                record = {"case": case["case"], "count": labels.numel(),
                          "top1": (predicted == labels).float().mean().item(),
                          "top5": (logits.topk(5).indices == labels[:, None]).any(-1).float().mean().item(),
                          "mean_target_probability": expected.mean().item(),
                          "target_probabilities": expected.cpu().tolist(),
                          "predictions": predicted.cpu().tolist(),
                          "generated_array_sha256": digest(filename)}
                records.append(record)
            classifier["runs"].append({"directory": directory.name,
                                        "source_report_sha256": digest(directory / "report.json"),
                                        "cases": records})
        report["classifiers"][name] = classifier
        del model
        torch.cuda.empty_cache()
    write_json(args.output, report)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
