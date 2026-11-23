#!/usr/bin/env python3
"""Losslessly archive complete decision arrays with deduplication and XOR deltas."""

import argparse
import gzip
import hashlib
import io
import json
import shutil
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def relative_path(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("archive path escapes its root")
    return path


def xor(first, second):
    if len(first) != len(second):
        raise ValueError("XOR inputs have different lengths")
    return np.bitwise_xor(np.frombuffer(first, dtype=np.uint8),
                         np.frombuffer(second, dtype=np.uint8)).tobytes()


def archive(root, reports, destination):
    root = root.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    # Check report completeness and declared source identities before writing.
    sources, identities = [], set()
    for path in reports:
        report = json.loads(path.read_text())
        if report.get("status") != "complete":
            raise ValueError("only complete trace reports accepted")
        report_source = path.resolve().relative_to(root).as_posix()
        for row in report["runs"]:
            source = Path(row["trace"]).resolve()
            relative = source.relative_to(root).as_posix()
            if relative in identities:
                raise ValueError("duplicate trace source")
            identities.add(relative)
            image = (row["bundle"], row["bundle_input_index"])
            sources.append((source, relative, image, row["trace_sha256"], report_source))
    destination.mkdir(parents=True)
    (destination / "objects").mkdir()
    manifest = {"schema": 1, "format": "npy-gzip-xor", "objects": {}, "files": [],
                "limitations": [
                    "Every original NPY member is restored byte for byte, including dtype, shape and array bit patterns",
                    "Original NPZ container hashes are provenance; regenerated ZIP packaging may differ",
                    "This archive stores decisions and sensor state, not model weights or generated output tensors",
                ]}
    anchors = {}
    for source, relative, image, expected, report_source in sources:
        packed = source.read_bytes()
        if sha256(packed) != expected:
            raise ValueError(f"trace hash changed: {relative}")
        entry = {"source": relative, "source_sha256": expected, "source_bytes": len(packed),
                 "report": report_source, "members": []}
        with zipfile.ZipFile(io.BytesIO(packed)) as container:
            names = container.namelist()
            if len(names) != len(set(names)):
                raise ValueError("duplicate NPY member")
            for name in names:
                relative_path(name)
                raw = container.read(name)
                array = np.load(io.BytesIO(raw), allow_pickle=False)
                if array.dtype.hasobject:
                    raise ValueError("object arrays cannot be archived")
                identity = sha256(raw)
                entry["members"].append({"name": name, "object": identity})
                if identity in manifest["objects"]:
                    continue
                compressed = gzip.compress(raw, compresslevel=6, mtime=0)
                descriptor = {"encoding": "raw", "raw_bytes": len(raw)}
                key = (*image, name, len(raw))
                anchor = anchors.get(key)
                if anchor is not None:
                    delta = gzip.compress(xor(raw, anchor[1]), compresslevel=6, mtime=0)
                    if len(delta) < len(compressed):
                        compressed = delta
                        descriptor.update(encoding="xor", base=anchor[0])
                else:
                    anchors[key] = (identity, raw)
                descriptor.update(archive_sha256=sha256(compressed), archive_bytes=len(compressed))
                (destination / "objects" / f"{identity}.gz").write_bytes(compressed)
                manifest["objects"][identity] = descriptor
        manifest["files"].append(entry)
        print(f"archived {len(manifest['files'])}/{len(sources)} traces", flush=True)
    (destination / "index.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def reader(destination):
    manifest = json.loads((destination / "index.json").read_text())
    if manifest.get("schema") != 1 or manifest.get("format") != "npy-gzip-xor":
        raise ValueError("unsupported trace archive")
    active = set()

    @lru_cache(maxsize=64)
    def read(identity):
        if len(identity) != 64 or any(character not in "0123456789abcdef" for character in identity):
            raise ValueError("invalid object identity")
        if identity in active:
            raise ValueError("cyclic XOR dependency")
        active.add(identity)
        try:
            descriptor = manifest["objects"][identity]
            packed = (destination / "objects" / f"{identity}.gz").read_bytes()
            if (sha256(packed) != descriptor["archive_sha256"]
                    or len(packed) != descriptor["archive_bytes"]):
                raise ValueError("trace archive integrity mismatch")
            raw = gzip.decompress(packed)
            if descriptor["encoding"] == "xor":
                raw = xor(raw, read(descriptor["base"]))
            elif descriptor["encoding"] != "raw":
                raise ValueError("unknown object encoding")
            if sha256(raw) != identity or len(raw) != descriptor["raw_bytes"]:
                raise ValueError("trace payload integrity mismatch")
            return raw
        finally:
            active.remove(identity)
    return manifest, read


def verify(destination, restore=None):
    manifest, read = reader(destination)
    if restore is not None:
        restore.mkdir(parents=True, exist_ok=False)
    for entry in manifest["files"]:
        relative = relative_path(entry["source"])
        members = [(relative_path(item["name"]).as_posix(), read(item["object"]))
                   for item in entry["members"]]
        for _, raw in members:
            np.load(io.BytesIO(raw), allow_pickle=False)
        if restore is not None:
            path = restore / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as container:
                for name, raw in members:
                    container.writestr(name, raw)
            with zipfile.ZipFile(path) as container:
                for item in entry["members"]:
                    if sha256(container.read(item["name"])) != item["object"]:
                        raise ValueError("restored NPY integrity mismatch")
    return manifest


def split_archive(root, reports, source, destination):
    """Make independently restorable input shards for atomic evidence commits."""
    root = root.resolve()
    groups = {}
    for path in reports:
        report = json.loads(path.read_text())
        if report.get("status") != "complete":
            raise ValueError("complete reports required for input sharding")
        for row in report["runs"]:
            identity = (row["bundle"], row["bundle_input_index"])
            relative = Path(row["trace"]).resolve().relative_to(root).as_posix()
            groups.setdefault(identity, set()).add(relative)
    original = json.loads((source / "index.json").read_text())
    files = {entry["source"]: entry for entry in original["files"]}
    if set(files) != set().union(*groups.values()) or sum(map(len, groups.values())) != len(files):
        raise ValueError("input partition does not match complete trace archive")
    destination.mkdir(parents=True, exist_ok=False)
    summary = []
    for index, (identity, paths) in enumerate(sorted(groups.items())):
        needed = set()

        def collect(key):
            if key in needed:
                return
            needed.add(key)
            entry = original["objects"][key]
            if entry["encoding"] == "xor":
                collect(entry["base"])

        entries = [entry for entry in original["files"] if entry["source"] in paths]
        for entry in entries:
            for member in entry["members"]:
                collect(member["object"])
        directory = destination / f"input-{index:03d}"
        (directory / "objects").mkdir(parents=True)
        for key in sorted(needed):
            shutil.copyfile(source / "objects" / f"{key}.gz", directory / "objects" / f"{key}.gz")
        manifest = {**original, "files": entries,
                    "objects": {key: original["objects"][key] for key in sorted(needed)},
                    "input_identity": list(identity),
                    "parent_index_sha256": sha256((source / "index.json").read_bytes())}
        (directory / "index.json").write_text(json.dumps(manifest, indent=2) + "\n")
        verify(directory)
        summary.append({"directory": directory.name, "input_identity": list(identity),
                        "traces": len(entries), "objects": len(needed)})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument("--report", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--split", type=Path)
    parser.add_argument("--restore", type=Path)
    args = parser.parse_args()
    if args.split is not None:
        if not args.report or args.verify or args.restore:
            parser.error("splitting requires reports and excludes verify/restore")
        shards = split_archive(args.root, args.report, args.split, args.output)
        print(f"Verified {len(shards)} independently restorable input shards")
        return
    if args.verify:
        if args.report:
            parser.error("verification does not accept source reports")
    else:
        if not args.report or args.restore:
            parser.error("creation requires reports and does not accept --restore")
        archive(args.root, args.report, args.output)
    manifest = verify(args.output, args.restore)
    print(f"Verified {len(manifest['files'])} traces / {len(manifest['objects'])} unique arrays")


if __name__ == "__main__":
    main()
