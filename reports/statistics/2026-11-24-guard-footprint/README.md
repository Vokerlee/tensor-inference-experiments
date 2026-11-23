# Complete convolution uncertainty-footprint records — 2026-11-24

Two complete derived JSON records retain the preliminary one-layout study and
the canonical two-layout analysis. Canonical data include1152 patch/criterion/
cohort records, every K-slice source-group coordinate and flagged-row count,
and48 aggregate alternatives. Original masks remain in the parent
[scale-filter archive](../2026-11-24-controller-replay/README.md).

The [manifest](index.json) preserves all original/payload/archive hashes and
lengths without normalization. Counts are conditional footprints of fixed
sampled masks, not independent model observations or executed reference work.
[Interpretation and reproduction](../../2026-11-24-controller-guard-footprint.md).

```sh
.venv/bin/python scripts/archive_statistics.py --verify \
    --output reports/statistics/2026-11-24-guard-footprint
```
