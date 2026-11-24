# Complete producer-support geometry comparisons

Two complete clean-source reports compare support-conditioned causal rounding
on one reused discovery input and four reused follow-up inputs. The F(2,3) INT4
and F(4,3) INT8 policies use producer-declared9/16 and25/36 supports respectively.
Every actual inactive plane is checked before encoding; a violated precondition
is rejected. Original replay matches the stored capture.

All final raw/image outputs, prepared factors, packed coefficients, active masks,
input/source/binary identities and logs are preserved. Actual weights and product
layout remain unchanged: this numerical study does not claim sparse hardware
execution or energy savings. Discovery covers25 convolutions; follow-up covers24
with at least128 output channels. Both use software batch256. These five inputs
were captured in earlier studies, not a new independent population-level test.

`index.json` stores the two complete report payloads; `evidence.json` maps all12
logical attachments to byte-exact deduplicated containers. Verify and restore:

```sh
python scripts/archive_selective_reference_evidence.py --verify --output <archive>
python scripts/archive_selective_reference_evidence.py --output <archive> --restore <new-directory>
```

The independent `scripts/report_winograd.py` analyzer recomputes all aggregate
and per-input raw/image metrics. Preserve mixed and unfavorable comparisons;
source-support identities alone do not establish an improved rounding policy.
