# Reproducing activation representation pilots

Initialize recursive submodules, install the pinned base and optional diffusion
requirements, then build the engine's Torch library using its instructions.
The scripts require CUDA. No model weights, images or generated archives are
committed; official downloads use the local user cache.

```sh
.venv/bin/python scripts/reproduce_activation_pilots.py --output results/reproduced --dry-run
.venv/bin/python scripts/reproduce_activation_pilots.py --output results/reproduced
```

The first command prints the exact commands. `--stage classifiers`, `diffusion`,
`accounting`, `diagnostics` or `timing` runs a subset. Run timing with other GPU jobs stopped.
The composed pipeline ends by generating a compact `summary.json`; individual
reports and their original provenance remain unchanged. Reuse a fresh output
directory for a new replication and compare source hashes as well as metrics.

Main classifier comparisons use1000 Imagenette validation images, skipping the
reserved diagnostics and20/class screening allocation, with batch16, official
checkpoint preprocessing and the full1000-class classifier. Runtime decisions
cover128x128 input blocks with32-element scales. Batching is recorded because
it can change which token rows share a decision block. The low menu is E2M1,
with INT8 and BF16 alternatives. Known affine INT4 is a separate strong baseline.

Main diffusion comparisons use40 images: eight conditioning classes repeated
five times, seed20261125, batch2,50 DDIM steps and guidance4. The residual run
reuses the identically configured baseline only for diagnostics. A selector
never sees reference predictions, baseline trajectories or labels. Generation
quality needs broader evaluation than same-seed PSNR or class consistency.

The summary retains execution commits, dirty flags and source/library hashes
verbatim. An exploratory dirty checkout is not retrospectively described as a
clean execution of a later commit. The snapshot containing the reproduction
scripts can be newer than those exploratory executions. Source report and
sample-manifest hashes allow the individual cases to be distinguished.

These are numerical activation-reconstruction experiments with BF16 products.
No packed low-bit matrix kernel, ASIC energy measurement or automatic hardware
mode mapping is supplied. FP4/INT8 arithmetic with static BF16 weights still
needs a defined datapath; integer weights, block scaling, accumulation, format
coalescing and sparse execution introduce additional numerical/physical checks.
A format-dependent larger array cannot freely combine inconsistent128-block
choices. Action counts in the separate ledger make their assumptions explicit.
