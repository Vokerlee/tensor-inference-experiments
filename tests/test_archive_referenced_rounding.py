"""Scientific arrays remain recoverable across deduplication and corruption."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from archive_selective_reference_evidence import attachment_paths, create, restore, verify_all


def test_geometry_and_statistical_array_references_are_preserved_once(tmp_path):
    source = tmp_path / "results"
    source.mkdir()
    arrays = source / "payload.npz"
    arrays.write_bytes(b"opaque statistical container")
    checksum = hashlib.sha256(arrays.read_bytes()).hexdigest()
    report = source / "report.json"
    report.write_text(json.dumps({"geometry": str(arrays), "geometry_sha256": checksum,
                                  "records": [{"arrays": str(arrays), "arrays_sha256": checksum}]}))
    target = tmp_path / "archive"
    create(source, [report], [], target)
    assert verify_all(target) == (1, 1)
    assert (target / "payload.npz").read_bytes() == arrays.read_bytes()
    (target / "payload.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        verify_all(target)


def test_wrong_referenced_hash_is_rejected_before_creating_an_archive(tmp_path):
    source = tmp_path / "results"
    source.mkdir()
    payload = source / "geometry.npz"
    payload.write_bytes(b"metadata")
    report = source / "report.json"
    report.write_text(json.dumps({"geometry": str(payload), "geometry_sha256": "wrong"}))
    target = tmp_path / "archive"
    with pytest.raises(ValueError):
        create(source, [report], [], target)
    assert not target.exists()


def test_content_aliases_restore_every_original_reference_without_losing_bytes(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    first, second = root / "left.npz", root / "right.npz"
    first.write_bytes(b"same recorded numerical payload")
    second.write_bytes(first.read_bytes())
    checksum = hashlib.sha256(first.read_bytes()).hexdigest()
    report = root / "report.json"
    report.write_text(json.dumps({"rows": [{"arrays": str(path), "arrays_sha256": checksum}
                                          for path in (first, second)]}))
    archive = tmp_path / "archive"
    create(root, [report], [], archive, deduplicate=True)
    assert verify_all(archive) == (1, 2)
    paths = attachment_paths(archive)
    assert paths["left.npz"] == paths["right.npz"]
    assert len(list(archive.glob("*.npz"))) == 1
    target = tmp_path / "restored"
    restore(archive, target)
    for name in ("left.npz", "right.npz", "report.json"):
        assert (target / name).read_bytes() == (root / name).read_bytes()
    with pytest.raises(FileExistsError):
        restore(archive, target)
