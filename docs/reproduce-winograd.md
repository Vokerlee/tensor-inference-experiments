# Reproducing transform/precision comparisons

Use the recursively pinned repositories and the CPython3.12 environment from
`requirements-modern-models.txt`. Build the engine's optional Torch adapter as
documented in `deps/tensor-inference-engine/docs/torch.md`. The recorded platform
uses Torch2.11/CUDA12.8 and an RTX4070. Historical source commits, checkpoint
revision, compiler metadata and native-library SHA256 remain in each raw report.
Matching a protocol is distinct from reproducing historical bits on another
GPU or arithmetic backend.

## Audit preserved evidence without model inference

```sh
python scripts/report_winograd.py --output /tmp/winograd-tables
python scripts/archive_selective_reference_evidence.py --verify \
    --output reports/statistics/2026-11-24-winograd-mixed-model
python scripts/archive_selective_reference_evidence.py \
    --output reports/statistics/2026-11-24-winograd-mixed-model \
    --restore /tmp/restored-winograd-mixed
```

The first command independently audits complete raw/image and local operator
outputs, checks repeated canonical-policy fingerprints and regenerates all CSV,
condition and summary files. Archives retain every unfavorable policy. Restoring
creates a new tree and every logical attachment, including shared containers.
The superseded native adapter and initial dirty mapping remain preliminary.

## Regenerate original pipeline inputs

The raw reports embed the complete original capture manifests, including all
four discovery-capture prompts even though the comparison uses only its first
input. The following executes the existing full-pipeline capture workflow from
those exact recorded prompts/seeds; it does not synthesize new evaluation data.
Check out the recorded capture source/environment for historical reproduction.
Current-source replay should also record and compare its new fingerprints.

```sh
python - <<'PY'
import gzip
import json
import subprocess
import sys
from pathlib import Path

archive = Path('reports/statistics/2026-11-24-winograd-native')
for name, campaign in [
    ('discovery', 'native-matched-winograd-discovery-20261124'),
    ('followup', 'native-matched-winograd-holdout-20261124'),
]:
    with gzip.open(archive / campaign / 'report.json.gz', 'rt') as stream:
        capture = json.load(stream)['inputs']
    command = [sys.executable, 'scripts/sana_decoder_feedback.py', 'prepare',
               '--count', str(len(capture['prompts'])), '--seed', str(capture['seed']),
               '--steps', str(capture['steps']), '--output', f'results/recreated-{name}']
    for prompt in capture['prompts']:
        command.extend(['--prompt', prompt])
    subprocess.run(command, check=True)
PY
```

Generation requires the public pinned SANA checkpoint and its normal download
cache. The capture records the actual scaled argument of `vae.decode`, original
raw/image outputs and their fingerprints. Do not substitute scheduler callback
latents. A failed identity/replay check must be investigated rather than omitted.

## Execute complete modified decoder policies

```sh
python scripts/winograd_decoder_pilot.py \
    --inputs results/recreated-followup --count 4 --minimum-outputs 128 \
    --tile-batch 256 --case int8-k256-a8-native --case int8-k256-a8-shift \
    --case mixed-bf16-f4-int8-f2 --case fp8-k256-f2 \
    --case mixed-bf16-f4-fp8-f2 --output results/replayed-winograd
```

The output directory must be new. Each run first checks the unmodified decoder
against the capture, then propagates actual modified tensors through every
remaining layer and records all output arrays and layer statistics. Use
`--count 1 --minimum-outputs 0` with the discovery capture for its original scope.
Software batch256 accelerates the oracle; the separate hardware capacity ledger
uses appropriately smaller streamed batches. It does not measure ASIC cycles.

Local probes are `winograd_rounding_pilot.py` (`--tile 2 --tiles 32`, or
`--tile 4 --tiles 8`) and `winograd_axis_pilot.py --tiles 32`, using the same
capture. `winograd_mapping_pilot.py` regenerates the immutable-profile interface
and encoded-retention ledgers. Preserve complete JSON, referenced NPZ metadata
and logs through the indexed archive workflow before deriving a report.

INT4 code range, FP8 format, prepared weight representation, reduction grouping,
original-RGB bypass, producer support and inverse-transform order are distinct
conditions. Use the matching control and source snapshot; never pool these
contracts or treat repeated curated inputs as a new independent holdout.
