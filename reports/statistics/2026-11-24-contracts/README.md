# Complete cost and arithmetic-contract records — 2026-11-24

Four complete JSON records retain the full conditional cost grid,34 constructed
arithmetic witnesses plus one controller witness, and preliminary/canonical
sampled decoder diagnostics. All scenario and unfavorable results are preserved.
The [manifest](index.json) records original/payload/archive hashes and sizes.
All original and payload hashes match; no normalization occurred.

Cost ratios are hypothetical, not hardware characterization. The preliminary
real-trajectory record remains dirty as originally recorded; the clean rerun
reproduced all48 sampled records exactly. Constructed cases and patch samples
are not natural-input failure rates or additional independent networks.

[Interpretation, conditions and tables](../../2026-11-24-contract-validation.md).

```sh
.venv/bin/python scripts/archive_statistics.py --verify \
    --output reports/statistics/2026-11-24-contracts
```
