"""All query bytes survive sharding; corruption and incomplete runs fail closed."""

import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from archive_feature_task import create, load_report, verify


def run_fixture(tmp_path):
    source = tmp_path / "run"
    (source / "queries").mkdir(parents=True)
    gallery = source / "gallery.npz"
    gallery.write_bytes(b"gallery fixture")
    rows = []
    for index in range(3):
        raw = json.dumps({"index": index, "full": [1, 2, 3]}).encode()
        compressed = gzip.compress(raw, mtime=0)
        path = f"queries/{index:05d}.json.gz"
        (source / path).write_bytes(compressed)
        rows.append({"file": path, "archive_sha256": hashlib.sha256(compressed).hexdigest(),
                     "payload_sha256": hashlib.sha256(raw).hexdigest()})
    report = {"status": "complete", "gallery": str(gallery), "queries": rows,
              "gallery_sha256": hashlib.sha256(gallery.read_bytes()).hexdigest()}
    (source / "report.json").write_text(json.dumps(report))
    return source, report


def test_all_shards_and_original_report_survive(tmp_path):
    source, report = run_fixture(tmp_path)
    destination = tmp_path / "archive"
    create(source, destination, shard_size=2)
    assert verify(destination) == 3
    assert load_report(destination) == report
    assert len(list(destination.glob("queries-*.zip"))) == 2


def test_corrupted_last_shard_is_rejected(tmp_path):
    source, _ = run_fixture(tmp_path)
    destination = tmp_path / "archive"
    create(source, destination, shard_size=2)
    (destination / "queries-001.zip").write_bytes(b"changed")
    with pytest.raises(ValueError, match="shard changed"):
        verify(destination)


def test_incomplete_run_cannot_be_published_as_complete(tmp_path):
    source, report = run_fixture(tmp_path)
    report["status"] = "running"
    (source / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="completed"):
        create(source, tmp_path / "archive")


def test_archiving_from_another_checkout_uses_the_local_verified_gallery(tmp_path):
    source, report = run_fixture(tmp_path)
    report["gallery"] = "results/historical-checkout/gallery.npz"
    (source / "report.json").write_text(json.dumps(report))
    create(source, tmp_path / "archive")
    assert verify(tmp_path / "archive") == 3
