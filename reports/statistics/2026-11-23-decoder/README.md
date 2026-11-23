# Complete decoder statistics — 2026-11-23

Eight complete JSON records preserve two exact-input capture manifests and six
fixed/live feedback campaign reports. Original source payloads total9,816,463
bytes; deterministic compressed payloads total1,375,766 bytes. All source and
payload hashes match: no normalization or scientific data removal occurred.
The [manifest](index.json) records hashes, sizes and every original path.

All40 cases,160 per-input comparisons and2880 quantized calls are retained.
Reports include original/weight-only controls, coefficient arrays, complete
exponent histograms, clipping/zero counts, terminal statistics, live mode counts,
block/switch counts and measurement/library/checkpoint provenance. Unfavorable
controller results are preserved. Transient tensors and uncollected internal
state histories are not claimed as recorded statistics; exact inputs/source
allow deterministic replay. Large latent/image arrays remain ignored locally,
with SHA256 fingerprints in these manifests/reports.

[Interpretation and readable tables](../../2026-11-23-decoder-feedback.md).

```sh
.venv/bin/python scripts/archive_statistics.py --verify \
    --output reports/statistics/2026-11-23-decoder
```
