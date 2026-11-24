# Tensor Inference Experiments

Reproducible inference experiments: model adapters, data pipelines, evaluation,
and automated report generation.

The [model knowledge base](models/README.md) indexes short model cards,
checkpoint identities, complete condition-specific profiles and numerical
experiment reports. Shape statistics and quality measurements remain separate.

## Engine dependency

The execution engine is included as a submodule at `deps/tensor-inference-engine`.
Each commit of this repository records the exact engine revision used by its
experiments.

Clone the repository with its submodules:

```sh
git clone --recurse-submodules https://github.com/Vokerlee/tensor-inference-experiments.git
```

For an existing checkout, or after checking out a historical revision:

```sh
git submodule update --init --recursive
```

## Python environment

Use CPython 3.12 on Linux x86_64. Initialize the engine submodule before
installing the environment:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

The requirements include the engine's pinned CUDA 12.8 runtime/build/test
environment through `-r deps/tensor-inference-engine/requirements.txt`.
Only torchvision, NumPy, and Pillow are added here, with exact versions.
Shared package pins and the official CUDA wheel index remain in the engine's
file. The recorded engine revision contains that file, so a recursive checkout
restores both parts of the environment.

CUDA runtime wheels require a compatible NVIDIA driver. Building custom CUDA
operators additionally needs a separate native compiler/toolkit setup; the
planned baseline uses CUDA 12.8. Model checkpoints, datasets, and the native
toolchain are versioned separately from Python dependencies. No inference
experiment is implemented by this environment setup alone.

For the optional timm vision backbones and text-conditioned diffusion pilots,
install `requirements-modern-models.txt`. It includes the existing shared and
diffusion pins, then fixes the additional model/tokenizer packages and their
dependencies; it does not duplicate the engine environment.

## CUDA GEMM baseline

See [build, validation and benchmark instructions](docs/benchmark-gemm.md) for
running the custom engine operator against Torch with matched arithmetic.

## Activation representation pilots

The [modern paired-precision report](reports/2026-11-23-paired-precision.md)
compares live BF16/FP8 selection in DINOv3, ConvNeXt-V2 and SANA-Sprint,
including a single-row conditioning ablation. The
[complete statistical archive](reports/statistics/2026-11-23/README.md) preserves
117 full JSON records, indexed by model, with integrity checks.

The [spatial feedback report](reports/2026-11-23-spatial-feedback.md) separates
same-input convolution error from complete-model quality for source-group
quantization with fixed and kernel-derived residual feedback.

The [source-quantization and primitive-local accumulation report](reports/2026-11-23-primitive-accumulation.md)
adds paired arithmetic execution, convolution geometry, exact-source replay and
native planner timing. Action-count reductions and numerical quality are
reported separately from implementation overhead and unmeasured ASIC energy.

See the [measured results](reports/2026-11-22-activation-representations.md),
[reproduction pipeline](docs/reproduce-activation-pilots.md) and
[systolic action assumptions](docs/systolic-actions.md). Numerical quality,
CUDA preparation cost and prospective hardware actions are evaluated separately.

The [modern decoder feedback report](reports/2026-11-23-decoder-feedback.md)
adds exact final-latent replay, fixed/live feedback comparisons and the complete
statistical archive for eight SANA-Sprint outputs.

The [limitations and negative-results register](reports/limitations-and-negative-results.md)
keeps unfavorable variants, supporting evidence and concrete revisit conditions.

The [cost and arithmetic-contract study](reports/2026-11-24-contract-validation.md)
separates hypothetical energy break-even conditions, constructed arithmetic
witnesses and sampled live-controller divergence.

The [full controller replay study](reports/2026-11-24-controller-replay.md)
adds complete-model live/frozen interventions, paired pixel/descriptor comparisons,
exhaustive BF16 scale contracts and conditional stability filtering. All decision
and sensor tapes are preserved as independently restorable input archives.

The [selective-reference study](reports/2026-11-24-selective-reference.md)
measures independently bounded exact BF16 observations against complete native
reference execution. Full-model/control fidelity holds on the tested inputs,
but both selective implementations are slower. Complete per-output audits,
controller arrays and repeated software-emulation timings are preserved.

[Shared activation rounding report](reports/2026-11-24-activation-rounding.md)
links complete receiving, descriptor and preliminary evidence with regenerable
comparison tables.

[Complete attention recognition replication](reports/2026-11-24-attention-task.md)
retains all3100 evaluated queries, paired uncertainty and independent full-output
audits, including failure of the declared accuracy-preservation screens.

[Joint transform/precision study](reports/2026-11-24-winograd.md) compares
full-decoder Winograd, native bounded rounding, FP8, structural support and mixed
precision/lowering controls. Complete arrays and paired-work tables are preserved;
[reproduction instructions](docs/reproduce-winograd.md) distinguish numerical
oracles, actual interface geometry and unmeasured hardware energy.
