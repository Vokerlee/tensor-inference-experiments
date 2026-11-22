# Grouped activation pilot

Use the pinned environment and engine submodule. No additional Python package
is required. The first download retrieves official Torchvision checkpoints and
Imagenette2-160 into the user cache, outside Git.

```sh
.venv/bin/python scripts/activation_pilot.py --model vit_b_16 --phase operator \
    --output results/vit-operator.json
.venv/bin/python scripts/activation_pilot.py --model swin_t --phase operator \
    --output results/swin-operator.json
```

`--phase geometry` only reads weights. Operator mode captures eight token rows
per image at every MLP linear input on four validation images per class. A
fixed shuffle reserves the first eight images/class for diagnostics; subsequent
task evaluations use a disjoint allocation. Reports contain checkpoint/source
SHA256 hashes, numerical settings, exact image paths and geometry statistics.

All comparisons keep represented BF16 weights and max-derived quantization
scales fixed. The engine reconstructs symmetric INT4/INT8 values with group
size 32, then the model-input BF16 cast is applied. `--bits 16` means unchanged
BF16 values before masking. Every sparse policy retains two slots per quartet;
retained zero values need not become nonzero. Compare magnitude, receiving-row
norm, quantization-aware diagonal gain, full local Gram score and gated score.
This is not a reproduction of any complete published method with correction,
rotation, calibration or a different numerical format.

Reported full-product errors include interactions between quartets. Sum of
group losses is also recorded, explicitly as a different quantity. Per-row
errors enable paired inspection. The scripts issue diagnostic FP32 products;
those products are never proposed as a deployed selector.

The input images originate at 160-pixel resolution and are transformed by each
checkpoint's official preprocessing. This pilot is neither full ImageNet nor
evidence for diffusion quality. It measures numerical effects, not native
INT4/INT8 or 2:4 execution, target-chip throughput, or target-chip energy.
