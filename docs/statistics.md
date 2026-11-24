# Complete statistical records

Keep a short model card in `models/<name>/README.md`, immutable condition-specific
architecture profiles, and links to numerical reports and full statistics. Do
not replace layer records or histograms with headline averages. Keep unfavorable
and preliminary runs, with their original missing/dirty source status.

`scripts/archive_statistics.py` preserves explicitly selected complete JSON
records in deterministic gzip files, with an `index.json` containing original,
archived-payload and compressed-file SHA256 hashes. It refuses to overwrite an
archive directory. Only declared machine-local library/output/snapshot/profile paths
are normalized; the index identifies each changed field. Numerical values,
large integer counters, histograms and recorded source status remain unchanged.
Archiving alone does not certify a run's scientific validity or provenance.

```sh
.venv/bin/python scripts/archive_statistics.py --root results \
    --input results/run/report.json --output reports/statistics/run
.venv/bin/python scripts/archive_statistics.py --verify \
    --output reports/statistics/run
```

Read a record with Python's standard library:

```python
import gzip
import json

with gzip.open("reports/statistics/run/run/report.json.gz", "rt") as stream:
    report = json.load(stream)
```

Keep experiment source/checkpoint identities, input manifests, settings and
coverage alongside ranges, moments, error metrics, selection counters and
logical work statistics. Keep large checkpoints and regenerable input
materializations in ignored result/cache directories, with replay commands and
fingerprints. Complete measured outputs, error arrays, controller tapes and
prepared metadata can be scientific evidence and belong in indexed compressed
archives when needed to audit a result. Preserve unfavorable variants too;
rendered PNG previews can be regenerated from archived image arrays.

## Referenced arrays and exact restoration

`scripts/archive_selective_reference_evidence.py` combines portable JSON
payloads with byte-exact referenced NPZ containers and explicit supplements.
Its `evidence.json` lists each logical source path, stored path, size and SHA256.
With `--deduplicate`, identical containers share storage while all logical paths
remain restorable. Keep a local `!*.log` ignore exception and verify that every
indexed container and log is tracked before publishing.

```sh
python scripts/archive_selective_reference_evidence.py --root results \
    --input results/run/report.json --supplement results/run.log \
    --deduplicate --output reports/statistics/run
python scripts/archive_selective_reference_evidence.py --verify \
    --output reports/statistics/run
python scripts/archive_selective_reference_evidence.py \
    --output reports/statistics/run --restore restored-run
```

Restoration requires a new directory and recreates every logical attachment.
JSON with normalized paths restores its portable payload; include the original
JSON as a byte-exact supplement when original path bytes must also be retained.
A complete JSON payload is distinct from a completed experiment: interrupted
status and dirty-source flags must remain unchanged. Archive checks establish
integrity, not scientific validity.

Shard oversized evidence losslessly rather than silently dropping arrays.
`scripts/archive_feature_task.py` preserves complete compressed query payloads
in bounded ZIP shards, alongside gallery and provenance files. Dataset-specific
analyzers must independently recompute statistics from preserved output arrays;
checksums alone do not establish that the recorded metrics are correct.

Do not infer device energy, SRAM transactions or target controller cost from
CUDA timings or dense logical vector counts. Keep architecture-only estimates,
modified-model numerical results and hardware measurements separate.
