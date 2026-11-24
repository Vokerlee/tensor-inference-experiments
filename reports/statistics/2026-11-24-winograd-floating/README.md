# Preserved Winograd evidence: floating

`index.json` preserves complete report payloads and source identities.
`evidence.json` maps every logical attachment to a byte-exact content container.
Duplicate containers share storage. Full raw/image outputs, operator outputs,
prepared metadata and available execution logs remain available, including
unfavorable variants. PNG previews can be regenerated from stored image arrays.

The first mapping report remains marked source-dirty. Its clean repeat is
separate. Machine-local mapping-profile paths alone are normalized in portable
JSON; each corresponding original JSON is a byte-exact supplement.

Verify or restore the complete logical tree:

```sh
python scripts/archive_selective_reference_evidence.py --verify --output <archive>
python scripts/archive_selective_reference_evidence.py --output <archive> --restore <new-directory>
```

`scripts/report_winograd.py` independently checks output metrics and regenerates
tables from these archives. Restoration includes every deduplicated path.
A recursive checkout restores code versions; checkpoints and original captured
inputs remain separately reproducible, hash-identified experiment inputs.
