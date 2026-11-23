# Restorable controller decision traces

`scripts/archive_controller_traces.py` preserves every member of each recorded
`decisions.npz` as exact NPY bytes, including integer exponents, FP32 coefficient
choices, sensor moments, dtype, shape, order and metadata. Identical members share
one content-addressed object. Where smaller, a gzip-compressed bytewise XOR
against an earlier same-input/member anchor replaces a raw compressed object.
This is lossless storage, not reduced-precision statistics.

The index retains original NPZ hashes and source report identities. Restoration
reconstructs NPY members byte for byte; regenerated ZIP headers/compression may
change the NPZ container hash. Thus verify member identities rather than claiming
container-byte identity. The decoder's `read_tape` accepts the restored archive.
Original generated image arrays and checkpoints remain separate artifacts.

```sh
python scripts/archive_controller_traces.py --root results \
    --report results/controller-replay-a4-20261124/report.json \
             results/controller-replay-a8-20261124/report.json \
    --output results/controller-trace-archive-20261124
python scripts/archive_controller_traces.py \
    --output results/controller-trace-archive-20261124 --verify \
    --restore results/restored-controller-traces
```

Creation and restoration require new destinations. Verification checks object
hashes, lengths, delta dependencies and every referenced NPY member without model
weights or CUDA. Keep the full archive, its index and complete producer reports
together when publishing results. Restored counterfactual tapes remain diagnostic
interventions; storage does not establish numerical or scientific validity.

A completed archive can be split with `--split SOURCE_ARCHIVE --report ...
--output NEW_DIRECTORY`. The original reports define input identities across
precisions. Every generated `input-NNN` shard includes all required delta anchors
and can be verified/restored independently. This supports one complete input per
data commit while preserving the parent index hash and every original member.
