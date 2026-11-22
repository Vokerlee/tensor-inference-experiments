#!/usr/bin/env python3
"""Reproduce the main classifier, diffusion and action-accounting comparisons."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["0", "217", "482", "491", "497", "566", "569", "571"] * 5


def commands(output: Path, stage: str):
    def command(script, *arguments):
        return [sys.executable, str(ROOT / "scripts" / script), *map(str, arguments)]

    def cases(values):
        return [argument for value in values for argument in ("--case", value)]

    if stage in {"all", "classifiers"}:
        for prefix, model in [("vit", "vit_b_16"), ("swin", "swin_t")]:
            common = ["--model", model, "--phase", "quality", "--samples-per-class", "100",
                      "--holdout-offset", "20", "--batch-size", "16", "--low-format", "e2m1"]
            for name, backend, selected in [
                ("format-holdout", "cuda", ["baseline", "affine:4", "fp4:4",
                    "precision-diagonal:0.01", "precision-gram:0.01",
                    "adaptive-diagonal:0.01", "adaptive-gram:0.01"]),
                ("native-policy", "cuda-tile", ["baseline", "precision-diagonal:0.01",
                                                "precision-gram:0.01"]),
                ("geometry-ablation", "cuda-tile", ["baseline", "precision-gram-numerator:0.01",
                                                    "precision-gram-denominator:0.01"]),
            ]:
                yield command("activation_pilot.py", *common, "--selector-backend", backend,
                              *cases(selected), "--output", output / f"{prefix}-{name}.json")
            yield command("activation_pilot.py", *common, "--selector-backend", "cuda",
                          *cases(["baseline", "common-adaptive-gram:0.01", "common-adaptive-gram:0.03"]),
                          "--output", output / f"{prefix}-common-calls.json")
            yield command("activation_pilot.py", *common, "--selector-backend", "cuda",
                          *cases(["baseline", "gram:16"]),
                          "--output", output / f"{prefix}-forced-sparse.json")
    if stage in {"all", "diffusion"}:
        common = ["--seed", "20261125", "--steps", "50", "--batch-size", "2",
                  "--classes", *CLASSES, "--low-format", "e2m1"]
        yield command("diffusion_activation_pilot.py", *common,
                      *cases(["baseline", "dense:8", "affine:4", "fp4:4",
                              "precision-diagonal:0.01", "precision-gram:0.01", "adaptive-gram:0.01"]),
                      "--output", output / "dit-40")
        yield command("diffusion_activation_pilot.py", *common, "--selector-backend", "cuda-tile",
                      "--reference-run", output / "dit-40",
                      *cases(["precision-gram-numerator:0.01", "sparse-residual:4",
                              "double-sparse:4", "double-sparse-p2:4"]),
                      "--output", output / "dit-residual-40")
        yield command("score_diffusion_classes.py", "--run", output / "dit-40", "--run",
                      output / "dit-residual-40", "--output", output / "dit-class-consistency-main.json")
    if stage in {"all", "accounting"}:
        for prefix, model in [("vit", "vit_b_16"), ("swin", "swin_t"), ("dit", "dit")]:
            yield command("profile_matrix_work.py", "--model", model,
                          "--output", output / f"{prefix}-work.json")
        yield command("systolic_actions.py", "--shape", "128", "768", "3072",
                      "--retained-k", "1536", "--output", output / "systolic-actions.json")
    if stage in {"all", "diagnostics"}:
        for prefix, model in [("vit", "vit_b_16"), ("swin", "swin_t")]:
            yield command("activation_pilot.py", "--model", model, "--phase", "operator",
                          "--output", output / f"{prefix}-operator.json")
            yield command("activation_stream_diagnostics.py", "--model", model,
                          "--output", output / f"{prefix}-streams-refined.json")
    if stage in {"all", "timing"}:
        yield command("benchmark_selection.py", "--output", output / "final-selection-timing.json")
    yield command("summarize_activation_pilots.py", "--input", output,
                  "--output", output / "summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["all", "classifiers", "diffusion", "accounting",
                                           "diagnostics", "timing"],
                        default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for command in commands(args.output.resolve(), args.stage):
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
