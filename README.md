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
