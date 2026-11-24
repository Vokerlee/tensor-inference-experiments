#!/usr/bin/env python3
"""Preserve selective-reference JSON, referenced arrays and explicit supplements."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from archive_statistics import archive, verify


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_references(value):
    if isinstance(value, dict):
        for key in ("trace", "audit", "arrays", "geometry"):
            source = value.get(key)
            if isinstance(source, str) and source.endswith(".npz"):
                yield Path(source), value[key + "_sha256"]
        for child in value.values():
            yield from array_references(child)
    elif isinstance(value, list):
        for child in value:
            yield from array_references(child)


def relative_path(path: str) -> Path:
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("evidence path escapes archive")
    return relative


def verify_all(destination: Path) -> tuple[int, int]:
    records = verify(destination)
    evidence = json.loads((destination / "evidence.json").read_text())
    for entry in evidence["files"]:
        path = destination / relative_path(entry["archive"])
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError(f"evidence changed: {path}")
    return len(records["files"]), len(evidence["files"])


def create(root: Path, reports: list[Path], supplements: list[Path], destination: Path) -> None:
    root = root.resolve()
    prepared = {}
    for report in reports:
        for path, expected in array_references(json.loads(report.read_text())):
            path = path.resolve()
            path.relative_to(root)
            if digest(path) != expected:
                raise ValueError(f"referenced array changed: {path}")
            prepared[path] = {"kind": "referenced-array", "sha256": expected}
    for path in supplements:
        path = path.resolve()
        path.relative_to(root)
        if path in prepared:
            raise ValueError("supplement duplicates an array")
        prepared[path] = {"kind": "supplement", "sha256": digest(path)}
    archive(root, destination, reports)
    evidence = {"schema": 1, "files": [], "limitations": [
        "Array containers and explicit supplements are preserved byte for byte",
        "Archiving does not establish scientific validity or recover missing historical source versions",
        "Weight checkpoints and captured operand tensors remain regenerable inputs, not archived statistics"]}
    for path, descriptor in sorted(prepared.items()):
        relative = path.relative_to(root)
        target = destination / relative
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        evidence["files"].append({"source": relative.as_posix(), "archive": relative.as_posix(),
                                  "bytes": path.stat().st_size, **descriptor})
    (destination / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    verify_all(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument("--input", type=Path, nargs="+")
    parser.add_argument("--supplement", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        if args.input or args.supplement:
            parser.error("verification accepts no input files")
    else:
        if not args.input:
            parser.error("archive creation requires explicit reports")
        create(args.root, args.input, args.supplement, args.output)
    records, attachments = verify_all(args.output)
    print(f"Verified {records} complete JSON records and {attachments} byte-exact attachments")


if __name__ == "__main__":
    main()
