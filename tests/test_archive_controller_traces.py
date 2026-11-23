import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from archive_controller_traces import archive, split_archive, verify


def sources(tmp_path):
    root = tmp_path / 'results'
    root.mkdir()
    runs = []
    patterns = np.zeros(10000, dtype=np.uint32)
    patterns[:5] = [0, 0x80000000, 0x7fc01234, 0x7f800000, 0xff800000]
    for index in range(3):
        path = root / f'{index}.npz'
        values = patterns.copy()
        values[index + 10] = index
        np.savez_compressed(path, exponents=np.arange(10, dtype=np.int16),
                            state=values.view(np.float32), metadata=np.array([index], dtype=np.uint8))
        runs.append({'trace': str(path), 'trace_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'bundle': 'same-input', 'bundle_input_index': 0})
    report = root / 'report.json'
    report.write_text(json.dumps({'status': 'complete', 'runs': runs}))
    return root, report


def test_full_trace_arrays_survive_archive_restore_and_dedup(tmp_path):
    root, report = sources(tmp_path)
    destination = tmp_path / 'archive'
    manifest = archive(root, [report], destination)
    assert len(manifest['objects']) < 3 * 3
    assert any(row['encoding'] == 'xor' for row in manifest['objects'].values())
    verify(destination, tmp_path / 'restored')
    for original in root.glob('*.npz'):
        with np.load(original, allow_pickle=False) as first, np.load(tmp_path / 'restored' / original.name) as second:
            assert first.files == second.files
            for key in first.files:
                assert first[key].dtype == second[key].dtype
                assert first[key].shape == second[key].shape
                assert first[key].tobytes() == second[key].tobytes()
    with pytest.raises(FileExistsError):
        archive(root, [report], destination)
    identity = next(iter(manifest['objects']))
    (destination / 'objects' / f'{identity}.gz').write_bytes(b'broken')
    with pytest.raises(ValueError, match='integrity'):
        verify(destination)


def test_changed_trace_or_incomplete_report_is_rejected(tmp_path):
    root, report = sources(tmp_path)
    document = json.loads(report.read_text())
    document['status'] = 'running'
    report.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='complete'):
        archive(root, [report], tmp_path / 'incomplete')
    assert not (tmp_path / 'incomplete').exists()
    document['status'] = 'complete'
    document['runs'][0]['trace_sha256'] = 'changed'
    report.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='hash changed'):
        archive(root, [report], tmp_path / 'modified')


def test_input_shards_are_independently_restorable(tmp_path):
    root, report = sources(tmp_path)
    document = json.loads(report.read_text())
    document["runs"][-1]["bundle_input_index"] = 1
    report.write_text(json.dumps(document))
    destination = tmp_path / "archive"
    archive(root, [report], destination)
    shards = split_archive(root, [report], destination, tmp_path / "shards")
    assert [row["traces"] for row in shards] == [2, 1]
    for row in shards:
        verified = verify(tmp_path / "shards" / row["directory"], tmp_path / row["directory"])
        assert len(verified["files"]) == row["traces"]
