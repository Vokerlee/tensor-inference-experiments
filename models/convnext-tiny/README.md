# ConvNeXt Tiny

Original ConvNeXt architectural control; spatial work includes depthwise layers.

This card indexes retained evidence; these older models are controls rather
than the primary modern-workload set.

- Profile input size: 224; inspect the full profile for scope.
- Matrix/convolution MACs: 4,455,531,264.
- Convolution MAC share: 6.5753%.
- Spatial-convolution MAC share: 6.5753%.
- [Complete preserved profile](geometry-2026-11-23.json) includes operator geometry, conditional schedules, source revision and script hash.

Measurements: [Primitive arithmetic report](../../reports/2026-11-23-primitive-accumulation.md). A report can cover only some models/operators; inspect
its conditions instead of transferring a result to every network in a table.
MAC shares are not energy. Meta execution does not evaluate model quality.
