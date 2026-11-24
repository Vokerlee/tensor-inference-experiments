# Complete joint precision/transform controls

Two complete clean-source reports retain the shape-only mixed controls:
BF16 F(4,3) in eligible narrow layers, with INT8 or FP8 F(2,3) elsewhere.
Every selected input is propagated through the actual complete decoder.
Discovery uses one reused input and25 layers; follow-up uses four reused inputs
and24 layers with at least128 output channels. Spatial divisibility by4 is
checked. Software batch256 does not assert a physical buffer schedule.

All raw/image outputs, per-input/layer records, floating coefficient hashes,
prepared wide-branch scales, source/binary/checkpoint identities and logs are
preserved. Sum floating, integer and FP8 vector fields to count both branches;
omitting the BF16 branch produces an invalid workload comparison. Reduced
packet counts do not establish energy savings or new dynamic adaptation.

`index.json` preserves two report payloads and `evidence.json` maps all12 logical
attachments to byte-exact containers. Restore every logical path with:

```sh
python scripts/archive_selective_reference_evidence.py --verify --output <archive>
python scripts/archive_selective_reference_evidence.py --output <archive> --restore <new-directory>
```

The independent `scripts/report_winograd.py` analyzer recomputes aggregate and
per-input metrics and checks repeated canonical-policy output fingerprints.
