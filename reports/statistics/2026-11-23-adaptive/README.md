# Adaptive precision and complete request statistics — 2026-11-23

19 additional complete JSON records (2,539,691,021 source bytes;
39,338,012 compressed bytes). The earlier
[117-record archive](../2026-11-23/README.md) remains immutable.
[Integrity manifest](index.json) identifies every file and its original recorded
source status. These gzip files preserve the full reports, including per-call
activation summaries, prepared coefficients and complete resident-phase traces.

- `moment-precision-20261123/`: centered/diagonal selector comparisons and
  sampled same-input receiving-output diagnostics, three modern models.
- `row-precision-20261123/`: cohort1/cohort32 comparisons and declared bounded
  partition/residency counters, six complete runs.
- `precision-phase-traces-20261123/`: full window-level traces, including the
  separate1000-image ConvNeXt-V2 holdout (offset100).
- `format-residency-replay-20261123.json.gz`: hypothetical fixed-trace setup
  cost-grid aggregates/witnesses; not energy, cycles or changed-network execution.
- `sana-full-pipeline-final-clean.json.gz`: canonical complete-request shape
  profile with three independent dispatch FLOP checks.

Precommit/untracked-script shape probes are retained with their recorded status
and are not substitutes for the canonical clean profile.
`sana-full-pipeline-final.json.gz` records the actual clean pre-squash source
958c0d4; its immutable `snapshot/sana-pipeline-pre-squash-2026-11-23` tag preserves
that source. The canonical profile uses the master-lineage cb0a432 revision.
All records preserve unfavorable results. The separate1000-image classifier
comparison does not establish an adaptive-precision advantage over uniform FP8.
MAC proportions and interface capacities do not establish energy proportions.

```sh
.venv/bin/python scripts/archive_statistics.py --verify \
    --output reports/statistics/2026-11-23-adaptive
```

Large checkpoints, generated images and full output arrays remain under ignored
result/cache paths with fingerprints/replay conditions. Full phase traces are
scientific statistics and are included here, even when raw JSON exceeds1GB.
