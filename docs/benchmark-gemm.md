# CUDA GEMM baseline

Initialize the pinned engine submodule and install this repository's inherited
requirements using the root README instructions. Build its native Torch adapter
using the same Python environment:

```sh
python3.12 deps/tensor-inference-engine/scripts/fetch_cuda_toolkit.py
cmake -S deps/tensor-inference-engine -B deps/tensor-inference-engine/build \
  -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_MAKE_PROGRAM="$PWD/.venv/bin/ninja" \
  -DTIE_BUILD_TORCH=ON -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build deps/tensor-inference-engine/build --parallel 2
ctest --test-dir deps/tensor-inference-engine/build --output-on-failure
.venv/bin/python -m pytest deps/tensor-inference-engine/tests/test_torch.py
.venv/bin/python scripts/benchmark_gemm.py
```

Override CUDA architectures at configure time for another supported GPU; the
default is SM89. `--library` can select a different engine build. The script's
default shapes include square products and rectangular projection-like products,
with row-major and transposed/column-major right operands. Specify a smaller run:

```sh
.venv/bin/python scripts/benchmark_gemm.py --shape 128x128x128 \
  --shape 1024x1024x1024 --trials 3 --iterations 20
```

BF16 inputs/output and FP32 accumulation are compared with Torch while TF32 and
BF16 reduced-precision reduction are disabled. A numerical check precedes each
case. GPU timings use CUDA events with warmup and repeated samples; graph mode
captures multiple calls per replay to amortize host launch overhead. Each graph
retains its outputs, so graph batching also consumes memory. `--mode eager`
measures repeated allocating calls instead. Both backends use the same mode,
shapes, layouts and settings; reference timings include Torch's selected kernels.

The JSON records source revisions/dirty state, package/build fingerprints,
settings, per-trial times, throughput and numerical differences. Keep generated
runs in ignored `results/`; retain selected raw reports under `reports/` with a
short interpretation. Revision identifiers preserve source identity; rebuilding
in another directory need not reproduce the binary hash or exact timings.

These measurements describe a GPU numerical baseline, including its framework
adapter. They do not measure a target accelerator's energy or emulate all of its
rounding/dataflow boundaries. The first backend supports 2D BF16 GEMM only;
batched operators, convolution, quantization and trained-model task evaluation
are separate coverage additions. Record shared-GPU noise and repeat measurements
before drawing broad performance conclusions.
