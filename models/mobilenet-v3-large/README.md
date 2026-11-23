# MobileNet-V3 Large

Mobile depthwise/pointwise architecture control; not a dense-convolution substitute.

This card indexes retained evidence; these older models are controls rather
than the primary modern-workload set.

- Profile input size: 224; inspect the full profile for scope.
- Matrix/convolution MACs: 216,589,760.
- Convolution MAC share: 98.8417%.
- Spatial-convolution MAC share: 10.9939%.
- [Complete preserved profile](geometry-2026-11-23.json) includes operator geometry, conditional schedules, source revision and script hash.

Measurements: [Primitive arithmetic report](../../reports/2026-11-23-primitive-accumulation.md). A report can cover only some models/operators; inspect
its conditions instead of transferring a result to every network in a table.
MAC shares are not energy. Meta execution does not evaluate model quality.

[Preserved scientific statistics](../../reports/statistics/2026-11-23/models.md#mobilenet-v3-large)
include the complete dated records available at this snapshot, including earlier
exploratory runs. Consult their original conditions and source status.
