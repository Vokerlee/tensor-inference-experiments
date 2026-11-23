# Complete resize-folding statistics — 2026-11-23

Both complete twelve-case reports are retained, including96 per-input
comparisons and480 branch records. Each record preserves dense MAC/source/
weight counts, quantizer histograms/counters, coefficients and fingerprints.
The [manifest](index.json) identifies original paths, hashes and byte sizes.
Source/payload hashes match: raw JSON is preserved byte for byte.

The exact-input manifests remain in the separate
[decoder archive](../2026-11-23-decoder/README.md). Large output/latent arrays
are ignored locally, with SHA256 fingerprints and exact-source replay.
[Readable results](../../2026-11-23-resize-folding.md) distinguish known
arithmetic transformations from unmeasured hardware and quality properties.

```sh
.venv/bin/python scripts/archive_statistics.py --verify \
    --output reports/statistics/2026-11-23-resize
```
