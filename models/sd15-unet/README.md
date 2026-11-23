# Stable Diffusion1.5 UNet

Historical convolutional diffusion-denoiser shape control; configuration-only, no quality result.

This card indexes retained evidence; these older models are controls rather
than the primary modern-workload set.

- Profile input size: 64; inspect the full profile for scope.
- Matrix/convolution MACs: 401,636,720,640.
- Convolution MAC share: 55.2671%.
- Spatial-convolution MAC share: 49.8367%.
- [Complete preserved profile](geometry-2026-11-23.json) includes operator geometry, conditional schedules, source revision and script hash.

[Configuration identity](configuration-source.json) pins the external UNet configuration.

Measurements: [Primitive arithmetic report](../../reports/2026-11-23-primitive-accumulation.md). A report can cover only some models/operators; inspect
its conditions instead of transferring a result to every network in a table.
MAC shares are not energy. Meta execution does not evaluate model quality.
