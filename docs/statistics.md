# Complete statistical records

Keep a short model card in `models/<name>/README.md`, immutable condition-specific
architecture profiles, and links to numerical reports and full statistics. Do
not replace layer records or histograms with headline averages. Keep unfavorable
and preliminary runs, with their original missing/dirty source status.

`scripts/archive_statistics.py` preserves explicitly selected complete JSON
records in deterministic gzip files, with an `index.json` containing original,
archived-payload and compressed-file SHA256 hashes. It refuses to overwrite an
archive directory. Only declared machine-local library/output/snapshot paths
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
logical work statistics. Scientific JSON records belong in Git; large model
weights, generated images and full numerical arrays stay in ignored result/cache
directories with replay commands and recorded fingerprints. Those local arrays
are not part of the compressed-statistics archive.

Do not infer device energy, SRAM transactions or target controller cost from
CUDA timings or dense logical vector counts. Keep architecture-only estimates,
modified-model numerical results and hardware measurements separate.
