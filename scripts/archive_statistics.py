#!/usr/bin/env python3
"""Preserve complete JSON statistics with deterministic compression and hashes."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


LOCAL_FIELDS = {("settings", "library"), ("settings", "output"), ("snapshot",)}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def portable_payload(raw: bytes) -> tuple[bytes, list[str]]:
    """Normalize only declared machine-local paths; retain every numerical value."""
    document = json.loads(raw)
    changed = []

    def visit(value, location=()):
        if isinstance(value, dict):
            return {key: visit(item, (*location, key)) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item, (*location, str(index))) for index, item in enumerate(value)]
        if isinstance(value, str) and "/home/" in value:
            if location not in LOCAL_FIELDS or not value.startswith("/home/"):
                raise ValueError(f"unreviewed machine-local path at {'/'.join(location)}")
            changed.append("/" + "/".join(location))
            return "<local-path>/" + Path(value).name
        return value

    normalized = visit(document)
    if not changed:
        return raw, changed
    return (json.dumps(normalized, indent=2, allow_nan=False) + "\n").encode(), changed


def archive(root: Path, destination: Path, inputs: list[Path]) -> dict:
    root = root.resolve()
    paths = sorted((path.resolve() for path in inputs), key=str)
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate inputs")
    # Validate the complete batch before creating an immutable destination.
    prepared = []
    for path in paths:
        relative = path.relative_to(root)
        raw = path.read_bytes()
        payload, normalized = portable_payload(raw)
        report = json.loads(payload)
        identity = {key: report[key] for key in ("model", "experiments", "engine")
                    if isinstance(report, dict) and key in report}
        compressed = gzip.compress(payload, compresslevel=9, mtime=0)
        prepared.append((relative, raw, payload, compressed, normalized, identity))
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": 1, "files": [], "limitations": [
        "Archiving preserves recorded evidence; it does not verify scientific validity or source cleanliness",
        "Missing/dirty source metadata and preliminary runs remain as recorded",
        "Only listed machine-local path fields are normalized; scientific numerical values are unchanged",
        "Checkpoints, generated images and numerical output arrays are separate from these JSON statistics",
    ]}
    for relative, raw, payload, compressed, normalized, identity in prepared:
        target = Path(str(relative) + ".gz")
        output = destination / target
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(compressed)
        manifest["files"].append({
            "source": relative.as_posix(), "archive": target.as_posix(),
            "source_sha256": digest(raw), "payload_sha256": digest(payload),
            "archive_sha256": digest(compressed), "source_bytes": len(raw),
            "payload_bytes": len(payload), "archive_bytes": len(compressed),
            "normalized_fields": normalized, **identity,
        })
    (destination / "index.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def verify(destination: Path) -> dict:
    manifest = json.loads((destination / "index.json").read_bytes())
    for entry in manifest["files"]:
        relative = Path(entry["archive"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("archive path escapes destination")
        compressed = (destination / relative).read_bytes()
        payload = gzip.decompress(compressed)
        if (digest(compressed) != entry["archive_sha256"]
                or digest(payload) != entry["payload_sha256"]
                or len(compressed) != entry["archive_bytes"]
                or len(payload) != entry["payload_bytes"]):
            raise ValueError(f"archive integrity mismatch: {relative}")
        json.loads(payload)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument("--input", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        if args.input:
            parser.error("verification does not accept inputs")
        manifest = verify(args.output)
    else:
        if not args.input:
            parser.error("archive creation requires explicit --input files")
        manifest = archive(args.root, args.output, args.input)
        verify(args.output)
    print(f"Verified {len(manifest['files'])} complete statistical records")


if __name__ == "__main__":
    main()
