# Complete controller replay evidence — 2026-11-24

Seven complete JSON records preserve112 full-decoder cases per precision,
all per-call histograms/decisions, paired pixel/descriptor analyses, full DINOv3
vectors, exhaustive finite-BF16 scale contracts and every sampled guard outcome.
The [manifest](index.json) records hashes and lengths. All source/payload hashes
match: no path or numerical normalization occurred.

Both preliminary and canonical guard/analysis records remain distinguishable.
The canonical guard rounds observed perturbation bounds outward; the preliminary
version and all original numbers are retained. Canonical analysis includes the
additional descriptor observer and exact dyadic permutation comparisons.

Source identities: full decoder runs8950a98e16ad448af058f6e89ab9d6f49a17b7ea;
descriptor observer0e1ed11e592c62a2c0d19cb8b9ebc9e77e71309c;
canonical guard96eca307368d611c8e5cb92e909799c6fb6eb534;
analysis script hash is recorded in its JSON. All neural producer snapshots
were clean. Engineaba8a998291b6129f6e39bffc3bad09dd0097d3e and the native-binary
hash are recorded. A binary hash identifies tested behavior, not a guarantee of
identical compilation on another device/toolchain.

Complete decision/sensor arrays are stored in independently restorable input
shards alongside this index. Each contains all28 INT4/INT8 control traces for
one input, with complete lossless delta dependencies. Original NPZ container
hashes are provenance; restored NPY member bytes are exact. Images, latent arrays
and model weights remain separate reproducible inputs/outputs with recorded
identities. These curated inputs are reused across precisions, not independent
samples per variant or decision.

```sh
.venv/bin/python scripts/archive_statistics.py --verify \
    --output reports/statistics/2026-11-24-controller-replay
```

## Complete input trace archives

| Input | Original bundle index | Traces | Archive |
|---|---|---:|---|
| 000 | `results/decoder-inputs-20261123`, 0 | 28 | [Restore/verify](traces/input-000/README.md) |
| 001 | `results/decoder-inputs-20261123`, 1 | 28 | [Restore/verify](traces/input-001/README.md) |
| 002 | `results/decoder-inputs-20261123`, 2 | 28 | [Restore/verify](traces/input-002/README.md) |
| 003 | `results/decoder-inputs-20261123`, 3 | 28 | [Restore/verify](traces/input-003/README.md) |
| 004 | `results/decoder-inputs-holdout-20261123`, 0 | 28 | [Restore/verify](traces/input-004/README.md) |
| 005 | `results/decoder-inputs-holdout-20261123`, 1 | 28 | [Restore/verify](traces/input-005/README.md) |
| 006 | `results/decoder-inputs-holdout-20261123`, 2 | 28 | [Restore/verify](traces/input-006/README.md) |
| 007 | `results/decoder-inputs-holdout-20261123`, 3 | 28 | [Restore/verify](traces/input-007/README.md) |
