"""Protect full scientific records and immutable archive provenance."""

import gzip
import json

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from archive_statistics import archive, verify


def test_archive_retains_large_integers_histograms_and_dirty_status(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    source = root / "run.json"
    report = {"model": "example", "experiments": {"commit": "abc", "dirty": True},
              "counter": 2**63 + 37, "histogram": {"-17": 4}, "errors": [0.0, 1e-34, None],
              "settings": {"output": "/home/person/experiment/results/run.json"}}
    source.write_text(json.dumps(report))
    destinations = [tmp_path / "first", tmp_path / "second"]
    for destination in destinations:
        manifest = archive(root, destination, [source])
        verify(destination)
        entry = manifest["files"][0]
        restored = json.loads(gzip.decompress((destination / entry["archive"]).read_bytes()))
        assert restored == {**report, "settings": {"output": "<local-path>/run.json"}}
        assert entry["normalized_fields"] == ["/settings/output"]
        assert entry["experiments"]["dirty"] is True
    assert (destinations[0] / "run.json.gz").read_bytes() == (destinations[1] / "run.json.gz").read_bytes()
    with pytest.raises(FileExistsError):
        archive(root, destinations[0], [source])
    (destinations[0] / "run.json.gz").write_bytes(gzip.compress(b'{"counter": 0}'))
    with pytest.raises(ValueError, match="integrity mismatch"):
        verify(destinations[0])


def test_unreviewed_path_rejected_before_creating_archive(tmp_path):
    source = tmp_path / "source.json"
    source.write_text('{"unreviewed": "/home/person/private"}')
    destination = tmp_path / "archive"
    with pytest.raises(ValueError, match="unreviewed"):
        archive(tmp_path, destination, [source])
    assert not destination.exists()


def test_declared_profile_path_normalization_preserves_numerical_values():
    from archive_statistics import portable_payload
    import json

    raw = json.dumps({"profile": "/home/research/models/geometry.json", "macs": 123456,
                      "profile_sha256": "fixed-profile-identity"}).encode()
    payload, changed = portable_payload(raw)
    value = json.loads(payload)
    assert value["profile"] == "<local-path>/geometry.json"
    assert value["macs"] == 123456 and value["profile_sha256"] == "fixed-profile-identity"
    assert changed == ["/profile"]
