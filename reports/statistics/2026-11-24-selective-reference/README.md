# Complete selective-reference evidence — 2026-11-24

Nine canonical clean-source records preserve12 real operator cases,72 complete
model audits,18 repeated model-timing cases,24 operator/backend component
profiles and one constructed precision-cast witness. A tenth JSON record retains
an earlier private prototype measurement without retrospective source claims.

The [JSON manifest](index.json) preserves original, payload and archive hashes;
no fields were normalized. The [attachment manifest](evidence.json) preserves
24 complete per-output operator audits,18 model controller containers and16
explicit supplemental files byte for byte. Controller containers shared by
multiple methods remain referenced by all corresponding report rows.

The operator arrays retain accepted masks, error bounds, candidate/reference
FP32 values and repaired BF16 output words. Controller containers retain every
recorded exponent, coefficient, preceding/terminal moment, counter and event
identity. BF16 policies have empty controller containers by design. Inputs,
weights and original images are regenerable from the pinned producers and their
hashes; they are not statistical evidence stored in this archive.

Preliminary logs include unsuccessful development and unsafe compiled-filter
results. The preserved prototype source is its final surviving version; source
versions corresponding to every earlier log were not captured. It must not be
used to reconstruct missing provenance or mixed into the canonical timings.
The early JSON's `complete` status describes that recorded run, not validation
under the later protocol. Canonical evidence and interpretation are in the
[study report](../../2026-11-24-selective-reference.md).

```sh
.venv/bin/python scripts/archive_selective_reference_evidence.py --verify \
    --output reports/statistics/2026-11-24-selective-reference
.venv/bin/python scripts/selective_reference_analysis.py \
    --root reports/statistics/2026-11-24-selective-reference \
    --output /tmp/selective-reference
```

The second command regenerates all four CSV tables and the JSON summary directly
from the committed compressed records. Compare these with the dated report
files. Neither the ignored producer outputs nor the original workstation are
needed to regenerate the tables. Statistical integrity does not prove universal
arithmetic correctness, network accuracy or target-device performance.
