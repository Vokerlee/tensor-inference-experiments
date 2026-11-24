"""Scientific arrays remain recoverable across deduplication and corruption."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from archive_selective_reference_evidence import create, verify_all


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
