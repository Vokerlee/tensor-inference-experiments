# Complete statistics — 2026-11-23

117 full JSON scientific records, 50,889,715 source bytes and
4,845,657 compressed bytes. [Model index](models.md) groups them by recorded
model or an explicitly described filename navigation convention.
[Integrity manifest](index.json) retains file hashes, source status where recorded,
sizes and declared path normalization.

The snapshot retains all JSON scientific reports present in the experiment
results tree at export: exploratory and final runs, geometry profiles, range/
operator statistics, arithmetic and model diagnostics, GPU timing, replay checks
and earlier report exports. This includes unfavorable and superseded runs.
Clean original modern runs have 24 cases; the single-row ablation adds three.
Earlier smoke reports retain their dirty/missing source status. Preserving
evidence does not certify its validity; use the consolidated reports for
interpretation and the recorded source revision for reproduction.

Excluded: source-checkout copies and dependency files, package-install resolver
plans (not measurements), and the intentionally corrupted source-hash test
fixture. No measured statistical field is omitted from an included report.
Only machine-local library/output/snapshot path fields are normalized, as listed
per file in the manifest. Original files remain under ignored `results/`; their
SHA256 hashes are retained even when the exported path-normalized payload differs.

Generated images, checkpoints and large output/trajectory arrays remain local
under ignored results/cache directories. They are not contained in these JSON
archives. Numerical reports retain their fingerprints and the consolidated
reports provide exact source revisions and replay commands.

```sh
.venv/bin/python scripts/archive_statistics.py --verify \
    --output reports/statistics/2026-11-23
```

Read any `.json.gz` file with Python `gzip.open` and `json.load`; see
[archiving conventions](../../../docs/statistics.md).
