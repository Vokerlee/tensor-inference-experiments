# Tensor Inference Experiments

Reproducible inference experiments: model adapters, data pipelines, evaluation,
and automated report generation.

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

## CUDA GEMM baseline

See [build, validation and benchmark instructions](docs/benchmark-gemm.md) for
running the custom engine operator against Torch with matched arithmetic.

## Activation representation pilots

See the [measured results](reports/2026-11-22-activation-representations.md),
[reproduction pipeline](docs/reproduce-activation-pilots.md) and
[systolic action assumptions](docs/systolic-actions.md). Numerical quality,
CUDA preparation cost and prospective hardware actions are evaluated separately.
