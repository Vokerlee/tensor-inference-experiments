# Complete decision traces: input 006

Input bundle: `results/decoder-inputs-holdout-20261123`, bundle index `2`.
All 28 INT4/INT8 full-decoder control tapes are retained, including
original replay, changed-backend live/frozen control and mode/scale ablations.
NPY decision and sensor members restore byte for byte; original NPZ hashes
identify the original containers. All delta dependencies are local to this shard.

See the [complete statistics and conditions](../../README.md).

```sh
.venv/bin/python scripts/archive_controller_traces.py --verify \
    --output reports/statistics/2026-11-24-controller-replay/traces/input-006
```
