#!/usr/bin/env python3
"""Replay recorded arithmetic runs from their exact Git revisions in worktrees."""

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FLAGS = ("model", "activation_format", "k_slice", "bound_kind", "strategy",
         "samples_per_class", "holdout_offset", "batch_size")


def command(interpreter: Path, checkout: Path, settings: dict, output: Path) -> list[str]:
    result = [str(interpreter), str(checkout / "scripts/primitive_accumulation_pilot.py")]
    for key in FLAGS:
        if key in settings:
            result.extend(("--" + key.replace("_", "-"), str(settings[key])))
    for width in settings["widths"]:
        result.extend(("--width", str(width)))
    return result + ["--output", str(output)]


def checkout_revision(commit: str, directory: Path) -> None:
    if directory.exists():
        current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=directory, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=directory, text=True).strip()
        if current != commit or dirty:
            raise ValueError(f"existing replay checkout differs or is dirty: {directory}")
    else:
        subprocess.run(["git", "worktree", "add", "--detach", str(directory), commit], cwd=ROOT, check=True)
    subprocess.run(["git", "submodule", "update", "--init", "--recursive"], cwd=directory, check=True)


def validate(report: dict, expected: dict, manifest: list) -> dict:
    if report["checkpoint"] != expected["checkpoint"] or report["manifest"] != manifest:
        raise ValueError("replay checkpoint or image manifest differs from the record")
    if (report["experiments"] != expected["experiments"] or report["engine"] != expected["engine"]
            or report["source_sha256"] != expected["source_sha256"]):
        raise ValueError("replay source revision, hashes or engine differ")
    changes = {}
    for case in report["cases"]:
        reference = expected["predictions"][case["name"]]
        if len(reference) != len(case["predictions"]):
            raise ValueError("replay prediction count differs")
        changes[case["name"]] = sum(a != b for a, b in zip(reference, case["predictions"], strict=True))
    return {"checkpoint_match": True, "image_manifest_match": True, "sources_match": True,
            "prediction_differences": changes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="append")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    record = json.loads(args.provenance.read_text())
    selected = args.run or sorted(record["runs"])
    if any(name not in record["runs"] for name in selected):
        parser.error("unknown run name")
    output = args.output.resolve()
    interpreter = args.python.absolute()  # Keep a venv's executable symlink intact.
    validations = {}
    for name in selected:
        expected = record["runs"][name]
        commit = expected["experiments"]["commit"]
        if not re.fullmatch(r"[0-9a-f]{40}", commit) or Path(name).name != name:
            parser.error("invalid recorded revision or run name")
        checkout = output / ".source-checkouts" / commit
        result_path = output / f"{name}.json"
        invocation = command(interpreter, checkout, expected["settings"], result_path)
        print(shlex.join(invocation), flush=True)
        if args.dry_run:
            continue
        import torch
        import torchvision
        if (torch.__version__ != expected["environment"]["torch"]
                or torchvision.__version__ != expected["environment"]["torchvision"]):
            raise ValueError("install the recorded pinned Torch/Torchvision environment before replay")
        output.mkdir(parents=True, exist_ok=True)
        checkout.parent.mkdir(parents=True, exist_ok=True)
        checkout_revision(commit, checkout)
        with (output / f"{name}.log").open("w") as log:
            subprocess.run(invocation, cwd=checkout, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads(result_path.read_text())
        manifest = record["image_manifests"][expected["image_manifest"]]
        validations[name] = validate(report, expected, manifest)
        validations[name]["raw_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
        (output / "replay-validation.json").write_text(json.dumps(validations, indent=2) + "\n")
        print(name, validations[name], flush=True)


if __name__ == "__main__":
    main()
