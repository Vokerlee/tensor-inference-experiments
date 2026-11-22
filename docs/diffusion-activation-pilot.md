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
