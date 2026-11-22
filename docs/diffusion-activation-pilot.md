# Paired diffusion activation pilot

Install the optional, fully pinned extension to the existing environment:

```sh
.venv/bin/python -m pip install -r requirements-diffusion.txt
.venv/bin/python -m pip check
.venv/bin/python scripts/diffusion_activation_pilot.py \
    --output results/dit-trajectories
```

Shared requirements are inherited, not duplicated. The script pins
`facebook/DiT-XL-2-256` revision
`eab87f77abd5aef071a632f08807fbaab0b704d0`. Weights and generated data stay in
the user cache and ignored results. Model files are downloaded from the
official repository and their SHA256 hashes are recorded. No checkpoint is
redistributed by this repository.

The pipeline uses its checkpoint's DDIM scheduler,50 steps, guidance4 and eight
declared ImageNet class indices. Initial-noise seeds and batching are fixed for
each paired comparison. `--seed` selects an independent pilot. The transformer
and VAE are in evaluation mode with BF16 computation. Only the56 feed-forward
linear input edges are transformed; conditioning, attention and VAE are intact.

Examples of cases, repeatable with `--case`:

- `baseline`, `dense:8`, `dense:4`, `affine:4`, `fp4:4`;
- `diagonal:4`, `gram:4` for fixed2:4;
- `precision-diagonal:0.01`, `precision-gram:0.01`;
- `adaptive-diagonal:0.01`, `adaptive-gram:0.01`.

Baseline must be first. `--low-format e2m1` changes the four-bit candidates of
the adaptive/precision menu. E2M1 uses FP32 scales, not the complete vendor
NVFP4 recipe. Fixed symmetric cases use fused CUDA preparation; other modes
use the numerical policy reference. All products remain BF16, not native
low-bit execution. Graph and buffering assumptions are the same as in the
[classification pilot](activation-pilot.md).

Reports retain per-step input-latent discrepancy and final pixel MSE/PSNR
against the same-seed baseline. NPZ files retain full observed trajectories and
images; PNGs permit inspection. These are numerical fidelity diagnostics.
Different valid images can have low PSNR; high PSNR does not establish improved
generation quality. Eight samples do not support FID, a distribution-quality
claim, or a general ranking of methods. GPU timings and ASIC energy are not
measured by this script. Thresholds are global and not fitted using labels.

## Conditional-class diagnostic

```sh
.venv/bin/python scripts/score_diffusion_classes.py --run results/dit-trajectories \
    --output results/dit-class-consistency.json
```

Two fixed, unmodified ImageNet classifiers evaluate generated float images
against their conditioning class. Reports include top1/top5, target-class
probabilities, predictions and input-array hashes. This supplements visual and
trajectory inspection; it measures class consistency, not FID, perceptual
quality or diversity. Repeated `--run` arguments score multiple completed runs.

## Reusing a paired reference

`--reference-run results/an-existing-run` loads its baseline images/states and
skips baseline generation. Omit `--case baseline` in this mode. The checkpoint
revision and bytes, seed, class list/order, steps, batch size, environment and
scheduler settings must match. Private scheduler provenance keys are excluded
from the semantic configuration comparison. The baseline archive is copied,
and source report/archive hashes are retained. Its arrays are diagnostic
references only: no selector receives baseline outputs or trajectories.

Cases `precision-gram-numerator:0.01` and
`precision-gram-denominator:0.01` separate the two uses of geometry as described
in the classification guide. The native tile backend supports both.

## Known-family residual baselines

`sparse-residual:4` uses magnitude2:4 coarse FP4 plus dense FP4 residual;
`double-sparse:4` uses two2:4 streams selected by quantization-error gain;
`double-sparse-p2:4` additionally restricts the residual/coarse scale ratio to
powers of two, shifts0..8, with current-value scaling. These are explicit
adaptations, not full SharQ/VR-SparseQuant reproductions and not novel-algorithm
claims. Their implementation belongs to the generic engine reference.

The model hook sums the two reconstructions and casts once to BF16 before the
ordinary product. This isolates representation error and does not reproduce
rounding of separately accumulated products or measure sparse execution.
Mode counters are not populated for these fixed two-stream policies. A real
sparse+dense implementation has1.5 dense FP4 nonzero-product count; two2:4
streams have1.0 before format/metadata/combination overhead. Without sparse
support both may require two dense passes. Numerical improvement alone does
not establish an energy improvement; retain that cost in comparisons.
