# Preserved Winograd f4-model evidence

Complete native F(4,3) nearest, rank1/rank4 completed-proxy and causal packed-shift comparisons.

Two complete clean-source reports retain every policy output, per-input metric,
actual layer record, prepared metadata and available execution log. Discovery
reuses one previously captured input and all25 eligible convolutions. Follow-up
reuses four other previously captured inputs and24 convolutions with at least128
output channels. Software tile batch256 is not a target-hardware memory schedule.
These are reused conditions, not an untouched population-level holdout.

`index.json` preserves report payloads and source identities; `evidence.json`
maps all24 logical attachments to byte-exact containers. Duplicate containers
share storage while every logical path remains restorable. PNG previews can be
regenerated from archived image arrays. Full checkpoint/input fingerprints and
capture conditions remain in the reports.

```sh
python scripts/archive_selective_reference_evidence.py --verify --output <archive>
python scripts/archive_selective_reference_evidence.py --output <archive> --restore <new-directory>
```

`scripts/report_winograd.py` independently recomputes aggregate and per-input
raw/image metrics. Local receiving error, full-decoder pixel fidelity, task
quality, interface work and hardware energy remain distinct observations.
