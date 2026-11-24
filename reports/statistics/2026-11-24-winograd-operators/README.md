# Preserved Winograd evidence: operators

`index.json` preserves complete report payloads and source identities.
`evidence.json` maps every logical attachment to a byte-exact content container.
Duplicate containers share storage. Full raw/image outputs, operator outputs,
prepared metadata and available execution logs remain available, including
unfavorable variants. PNG previews can be regenerated from stored image arrays.

All three operator campaigns completed from clean source snapshots. The F(2,3)
rounding probe uses32 sampled spatial tiles per layer; F(4,3) uses8. Both use
group32 reconstructed BF16 operands. The separate matched-axis probe uses32
tiles and a proposed K256/W8/native-grid integer contract. Keep these numerical
contracts and sample allocations separate; do not pool their error totals.

Verify or restore every logical report, array and log:

```sh
python scripts/archive_selective_reference_evidence.py --verify --output <archive>
python scripts/archive_selective_reference_evidence.py --output <archive> --restore <new-directory>
```

`scripts/report_winograd.py` independently checks all550 output containers and
regenerates tables. Restoration includes every deduplicated logical path.
A recursive checkout restores code versions; checkpoints and original captured
inputs remain separately reproducible, hash-identified experiment inputs.
