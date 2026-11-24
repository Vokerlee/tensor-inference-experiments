#!/usr/bin/env python3
"""Preserve completed task runs, including every gzip query, in bounded ZIP shards."""

import argparse
import gzip
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from archive_selective_reference_evidence import digest, relative_path


MANIFEST = "task-evidence.json"


def query_bytes(path: Path, expected: dict) -> bytes:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected["archive_sha256"]:
        raise ValueError(f"query container changed: {path}")
    raw = gzip.decompress(payload)
    if hashlib.sha256(raw).hexdigest() != expected["payload_sha256"]:
        raise ValueError(f"query payload changed: {path}")
    return payload


def create(source: Path, destination: Path, shard_size: int = 500,
           supplements: tuple[Path, ...] = ()) -> None:
    if shard_size < 1:
        raise ValueError("positive shard size required")
    original = source / "report.json"
    report = json.loads(original.read_text())
    if report["status"] != "complete":
        raise ValueError("only completed runs can be archived")
    names = [row["file"] for row in report["queries"]]
    if len(names) != len(set(names)):
        raise ValueError("duplicate query paths")
    for name in names:
        relative_path(name)
    destination.mkdir(parents=True, exist_ok=False)
    raw = original.read_bytes()
    (destination / "report.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    gallery = Path(report["gallery"])
    if digest(gallery) != report["gallery_sha256"]:
        raise ValueError("gallery changed")
    shutil.copyfile(gallery, destination / "gallery.npz")
    files = []
    for first in range(0, len(names), shard_size):
        name = f"queries-{first // shard_size:03d}.zip"
        target = destination / name
        members = []
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as shard:
            for row in report["queries"][first:first + shard_size]:
                payload = query_bytes(source / row["file"], row)
                entry = zipfile.ZipInfo(row["file"], date_time=(1980, 1, 1, 0, 0, 0))
                entry.external_attr = 0o100644 << 16
                shard.writestr(entry, payload)
                members.append({"file": row["file"], "bytes": len(payload),
                                "sha256": row["archive_sha256"]})
        files.append({"file": name, "bytes": target.stat().st_size,
                      "sha256": digest(target), "members": members})
    extras = []
    for index, supplement in enumerate(supplements):
        name = f"supplement-{index:02d}-{supplement.name}"
        shutil.copyfile(supplement, destination / name)
        extras.append({"file": name, "bytes": supplement.stat().st_size,
                       "sha256": digest(supplement)})
    manifest = {"schema": 1, "source": str(source.resolve()),
                "report_payload_sha256": hashlib.sha256(raw).hexdigest(),
                "report_archive_sha256": digest(destination / "report.json.gz"),
                "gallery_sha256": report["gallery_sha256"], "queries": len(names),
                "shards": files, "supplements": extras}
    (destination / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    verify(destination)


def load_report(directory: Path) -> dict:
    manifest = json.loads((directory / MANIFEST).read_text())
    path = directory / "report.json.gz"
    if digest(path) != manifest["report_archive_sha256"]:
        raise ValueError("raw report container changed")
    raw = gzip.decompress(path.read_bytes())
    if hashlib.sha256(raw).hexdigest() != manifest["report_payload_sha256"]:
        raise ValueError("raw report payload changed")
    return json.loads(raw)


def verify(directory: Path) -> int:
    manifest = json.loads((directory / MANIFEST).read_text())
    report = load_report(directory)
    if digest(directory / "gallery.npz") != manifest["gallery_sha256"]:
        raise ValueError("archived gallery changed")
    expected = {row["file"]: row for row in report["queries"]}
    seen = set()
    for row in manifest["shards"]:
        path = directory / relative_path(row["file"])
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("query shard changed")
        with zipfile.ZipFile(path) as shard:
            names = shard.namelist()
            if names != [entry["file"] for entry in row["members"]] or len(names) != len(set(names)):
                raise ValueError("query shard membership changed")
            for entry in row["members"]:
                name = entry["file"]
                relative_path(name)
                if name in seen or name not in expected:
                    raise ValueError("duplicate or unexpected query")
                raw = shard.read(name)
                if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                    raise ValueError("query bytes changed")
                if entry["sha256"] != expected[name]["archive_sha256"]:
                    raise ValueError("query differs from raw report")
                payload = gzip.decompress(raw)
                if hashlib.sha256(payload).hexdigest() != expected[name]["payload_sha256"]:
                    raise ValueError("query JSON payload changed")
                seen.add(name)
    for row in manifest["supplements"]:
        path = directory / relative_path(row["file"])
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("supplement changed")
    if seen != set(expected) or len(seen) != manifest["queries"]:
        raise ValueError("incomplete query archive")
    return len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--supplement", type=Path, action="append", default=[])
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        if args.source or args.supplement:
            parser.error("verification accepts no source/supplement")
    elif args.source is None:
        parser.error("source required")
    else:
        create(args.source, args.output, args.shard_size, tuple(args.supplement))
    print(f"Verified {verify(args.output)} complete queries and the reference gallery")


if __name__ == "__main__":
    main()
