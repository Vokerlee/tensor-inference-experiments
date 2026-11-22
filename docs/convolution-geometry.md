# Convolution geometry inventory

Use the [matrix-unit interface](matrix-unit-interface.md): K=128 for BF16,
K=256 for FP8, and 128 FP32 outputs in both cases.

```sh
.venv/bin/python scripts/convolution_geometry.py --model convnext_tiny \
    --output results/convnext-geometry.json
```

The script executes shapes on Torch's meta device, with no weights download
or numerical inference. It supports ResNet-50, ConvNeXt-Tiny, MobileNet-V3-Large,
ViT-B/16 and Swin-T. An optional `unet` mode uses a saved Diffusers configuration
and `--image-size` for the latent spatial dimension. Record its repository
revision alongside the configuration hash; this mode needs the separately
pinned diffusion requirements.

A grouped convolution has K = kernel_height * kernel_width * input_channels /
groups and N = output_channels / groups per independent group. A spatial
3 by 3 window contains nine values per input channel. It contains nine values
in the full reduction only when that group has one input channel.

Two conditional schedules are inventoried. Concatenated-channel lowering uses
consecutive K slices of the complete reduction. Separate-channel-window
lowering issues each channel's spatial window independently. Both need a
declared mechanism for merging partial results. They have different counts of
issued vector groups even when their useful MAC count is identical. Independent
groups use separate output vectors here: a faster grouped/depthwise packing
requires a different mapping and cannot be assumed free.

The report separates boundary padding from array capacity padding. Its byte
counts describe scheduled interface capacity, not SRAM transactions or measured
energy. Output volume per K slice is explicitly conditional on emitting every
slice. MAC shares cover Conv2d, Linear and observed SDPA QK/AV products; other
operators are excluded. A small MAC share can still have poor array utilization; neither
metric alone establishes the fraction of system energy.

