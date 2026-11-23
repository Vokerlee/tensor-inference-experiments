# Source quantization and primitive-local accumulation — 2026-11-23

The tested combination reuses source activation codes/statistics through
convolution lowering, then selects bounded integer accumulation chunks within
each matrix primitive. It preserves the FP32 output boundary. Numerical pilots
found useful partitions without changing predictions on the sampled images.
Standalone CUDA planning has material overhead; neither ASIC energy savings
nor a new mechanism relative to all prior work has been established.

## Executed method and controls

Activations use online power-of-two scales for channel groups of 32 values;
weights use static grouped INT8 encoding. The reported runs use INT8 activation
codes. They perform changed arithmetic during inference, rather than merely
collecting statistics from an unchanged model. There is no activation
calibration, retraining or label-fitted decision threshold.

For aligned dense convolutions, each source spatial coordinate is encoded once
along its channels. Codes, scales and maximum/L1 statistics are gathered in
matching R,S,C order for each receiving window. This is a MicroQonv-inspired
inference variant, not its complete square-block training method. The Torch
oracle materializes lowering; no streaming feeder or reduced SRAM traffic has
been implemented by this reference. ResNet18's unaligned first convolution
and final classifier remain unchanged. ViT and Swin experiments transform MLP
products only, not their entire matrix workload.

Each primitive reduces at most K=128 or K=256 values and emits 128 FP32 output
scalars per full output window. The row dimension M is streamed. Native BF16
K128 and FP8 K256 geometry does not imply native INT8/E2M1 support; these code
modes are explicit datapath research contracts. FP32 outputs from successive
K slices merge in FP32 on both compared paths. No integer state persists across
primitive calls.

The arithmetic baseline already uses exact integer group dot products and
FP32 scaling/merging at group boundaries. It is not an FP32 FMA for every input
product. The candidate replaces some of these **group-boundary** operations:

1. Form a conservative absolute product bound for every group from cached
   activation and weight metadata. The maximum-only version needs no source
   L1 reduction. The tighter version takes the minimum of maximum-times-L1
   bounds in the two directions.
2. Share the selection across a 128-column output window. Account for weight
   exponent extrema and align to the minimum active exponent.
3. Either accept the complete slice, or greedily extend contiguous chunks
   until their absolute bounds no longer fit. Oversized single groups use
   fallback arithmetic. Zero groups are bypassable on either path.
4. Convert each accepted chunk into the FP32 carry, then emit the mandatory
   FP32 vector at the original primitive boundary.

The guard bounds intermediate prefixes, not just a final sum after
cancellation. A signed 24-bit selection uses the conservative positive limit
2^23−1. The numerical oracle uses FP64 solely to represent bounded integer
sums exactly; a deployed controller neither performs FP64 operations nor
evaluates an additional receiving GEMM. The separate reference products are
verification work, not proposed online controller work.

See [primitive arithmetic](../docs/primitive-accumulation.md),
[source lowering](../docs/source-convolution-accumulation.md),
[chunk selection](../docs/accumulation-segments.md) and
[hardware interface](../docs/matrix-unit-interface.md).

## Paired numerical results

The 100-image runs use ten Imagenette validation examples per class, offset 20,
official pretrained checkpoints/preprocessing and the original 1000-class
classifier outputs. The ten-image diagnostics use one example per class at
the same offset and overlap the larger set. These exploratory subsets are
not ImageNet accuracy measurements or an independent confirmatory holdout.

On the 100-image runs, original and quantized group-FP32 predictions were the
same: ResNet18 57/100, ViT-B/16 82/100 and Swin-T 74/100. All tested guarded
variants also retained those predictions. Equal predictions do not imply
identical logits or identical intermediate tensors.

Complete-slice I24 accepted useful-product coverage:

| Model / bound | K128, 100 images | K256 |
|---|---:|---:|
| ResNet18, maximum only | 80.37% | not run |
| ResNet18, maximum/L1 | 97.01% | 79.09%, 10 images |
| ViT-B/16, maximum/L1 | 84.49% | 60.55%, 100 images |
| Swin-T, maximum/L1 | 88.07% | 71.47%, 100 images |

These complete-slice I24 runs introduced no local FP32 differences from their
group-FP32 reference and no final-logit differences on the tested data. This
observation is specific to the arithmetic and values tested.

For greedy I24, the table compares remaining FP32 group merges with the
group-FP32 baseline after bypassing zero group outputs. Both the complete-slice
comparator and the greedy selection use **the same represented input tensors**
and maximum-only metadata. Different input trajectories are not compared as
though they were identical.

| Model | K | Images | Merges removed, complete slice | Merges removed, chunks | Changed local FP32 scalars | Changed final logits |
|---|---:|---:|---:|---:|---:|---:|
| ResNet18 | 128 | 100 | 78.40% | 92.52% | 6 | 0 |
| ViT-B/16 | 128 | 100 | 68.00% | 82.48% | 0 | 0 |
| Swin-T | 128 | 100 | 72.61% | 84.27% | 438257 | 0 |
| ResNet18 | 256 | 10 | 13.03% | 83.25% | 0 | 0 |
| Swin-T | 256 | 10 | 33.11% | 79.45% | 70845 | 501 / 10000 |

Swin K256's 501 changed logits have maximum absolute difference 0.0078125 and
root-mean-square difference about 0.000363; none changed the predicted class.
These counts are changes from the paired quantized reference, not errors
against a ground-truth tensor. They concern different tensor populations:
local counts sum primitive outputs; final counts concern classifier logits.

Greedy grouping changes FP32 parentheses, so a range guarantee alone does not
guarantee FP32 equivalence. Wider integer accumulation is not automatically
closer to a sequential FP32 reference either. For example, complete-slice I32
on Swin K256/100 images changed 329282 local scalars and 582/100000 final logits,
with maximum logit difference 0.0078125, while retaining predictions. The I24
and I32 results must therefore remain separate.

The tighter maximum/L1 whole-slice guard can outperform maximum-only greedy
selection; the latter is not claimed to dominate every strategy. Its attraction
is avoiding an additional source L1 reduction. Timing and physical costs,
rather than merge count alone, must decide between them.

Zero-output bypass is a comparison convention, not free prediction of a
future zero sum. Its detection/clock-gating implementation remains a cost on
both paths. Raw counts without that bypass are also preserved in the data.
No result here saves multiplier products or matrix calls.

## Source reuse and workload relevance

In the ten-image ResNet source diagnostic, 635040 encoded source groups serve
3833100 valid lowered occurrences: approximately 6.036 uses per group. The
4014080 expanded group slots also include padding. This reuse is a logical
count, not an SRAM-transaction ratio or an original contribution beyond
quantize-before-lowering work such as MicroQonv.

The separate meta-device geometry inventory counts Conv2d and explicit or
composite matrix products, including QK/AV work without double counting.
These percentages describe MACs, not energy:

| Model | Conv2d MAC share | Spatial-kernel Conv2d MAC share |
|---|---:|---:|
| ResNet18 | 99.97% | 98.91% |
| ResNet50 | 99.95% | 48.12% |
| ConvNeXt-Tiny | 6.58% | 6.58% |
| MobileNet-V3-Large | 98.84% | 10.99% |
| ViT-B/16 | 0.66% | 0.66% |
| Swin-T | 0.32% | 0.32% |
| Stable Diffusion 1.5 UNet | 55.27% | 49.84% |

The UNet inventory executes shapes from a pinned configuration at latent
64×64; it does not load weights or test generation quality. Its complete
convolution reductions are not nine-element dot products: nine is the spatial
tap count **per channel**. The inventory compares concatenated-channel and
separate-channel-window schedules conditionally, without asserting the actual
feeder's mapping. Depthwise/group-separated scheduling is not an optimized
packed-depthwise baseline. Swin functional matrix products appear partly in
the exported `other_explicit_matrix_macs`, so the `attention_macs` field alone
is not its complete attention workload.

## Native controller cost

A generic C++/CUDA implementation constructs bounded partitions directly from
row and output-window statistics, without materializing per-pair bound tensors.
Its 31 experiment-side tests match an independent tensor implementation; native
CPU partition tests also compare against exhaustive choices.

The [timing record](2026-11-23-integer-segment-timing.json) measures cached
metadata planning on synthetic operands, not model inference. Seven interleaved
trials use CUDA graphs with 16 calls/replay and 20 timed replays/trial on an
RTX 4070. BF16 GEMM comparators emit FP32 and disable reduced BF16 reduction.
The faster of the custom and Torch comparator medians is shown below; it is
not an exhaustive search for the fastest possible GEMM.

| Shape label (M,K,N) | Faster measured GEMM, µs | K128 max planner, µs | K256 max planner, µs |
|---|---:|---:|---:|
| ViT up (197,768,3072) | 27.965 | 5.520 | 6.557 |
| ViT down (197,3072,768) | 30.842 | 5.213 | 6.003 |
| Swin up (3136,96,384) | 10.246 | 2.405 | 2.424 |
| Swin down (3136,384,96) | 8.781 | 2.816 | 3.942 |
| ResNet conv (3136,576,64) | 8.679 | 3.760 | 5.005 |
| UNet conv (4096,2880,320) | 163.571 | 32.602 | 37.498 |

Planning alone costs approximately 17–43% of that GEMM comparator for K128
and 19–58% for K256. Source quantization, source-statistic generation, im2col,
fallback and execution of the chosen partition are **excluded**. These results
do not demonstrate a useful standalone GPU inference optimization. They also
do not predict an ASIC implementation's cycle count or energy. Fusing or
pipelining control with source preparation is an untested implementation
possibility, not an assumed removal of this overhead.

## Energy and prior-work boundaries

The relevant condition is that saved operations cost more energy than added
operations. For action types j, a bookkeeping expression is

`Delta E = sum_j ((N_baseline,j - N_candidate,j) * energy_per_action_j)`.

Here positive Delta E means a saving; N counts executed actions with matched
storage/datapath boundaries. Include alignment shifts, integer additions,
FP32 conversion/merging, source statistics, metadata loads/broadcasts, register
state, control, clocks and any changed SRAM activity. Integer chunks and FP32
carry may be simultaneously live. Fallback arithmetic remains present.
The deployment's internal accumulation format and stationarity must be known
before attributing any of these savings to it.

Against a source-reuse baseline, its existing quantization savings cancel in
this comparison: do not count them a second time as accumulation gains. An
affected component's energy fraction also caps whole-system savings; removing
90% of one action type cannot establish 90% inference-energy savings. No
universal coefficient is taken from a different chip or technology node.

MicroQonv already supplies quantize-once-before-lowering. Bucket Getter already
supplies integer buckets, carries, adaptive bucket management and shared
floating accumulation. Intel's US20250284460A1 describes dynamic exponent
caching, shifts and overflow-driven floating finalization. MIAO supplies an
optimized, calibrated microscaling/accumulator hardware comparator. These
mechanisms cannot be claimed as new merely by combining their names.

A remaining comparison concerns whether reuse of source-group metadata can
make a shared, pre-issue, primitive-local partition controller cheaper than
existing bucket/control organizations under the same FP32 output contract.
This is an unresolved mechanism/cost comparison, not a novelty conclusion;
Bucket Getter's complete text was not available during this review. Sparsity
gating beyond the explicit zero-group comparator is not evaluated here.

Primary sources: [MicroQonv](https://arxiv.org/html/2609.28358v1),
[Bucket Getter](https://ieeexplore.ieee.org/document/10411385/),
[Intel application](https://patents.google.com/patent/US20250284460A1/en),
[MIAO](https://arxiv.org/html/2611.19722v1).

## Data and reproduction

- [Aggregate cases](2026-11-23-primitive-accumulation.csv): 42 cases from 12 runs.
- [Per-layer counters](2026-11-23-primitive-accumulation.layers.csv): 660 rows.
- [Geometry inventory](2026-11-23-primitive-accumulation.geometry.csv): seven models.
- [Execution provenance](2026-11-23-primitive-accumulation.provenance.json): exact
  executed revisions/source hashes, engine gitlinks, settings, checkpoints,
  image manifests, paired predictions and raw report hashes.

Clean historical sources remain reachable through pushed snapshot tags:
`snapshot/primitive-accumulation-2026-11-23`,
`snapshot/source-convolution-2026-11-23`,
`snapshot/segmented-accumulation-2026-11-23` and
`snapshot/convolution-geometry-2026-11-23-e734203`.

```sh
.venv/bin/python scripts/reproduce_primitive_pilots.py \
    --provenance reports/2026-11-23-primitive-accumulation.provenance.json \
    --output results/primitive-replay --dry-run
.venv/bin/python scripts/reproduce_primitive_pilots.py \
    --provenance reports/2026-11-23-primitive-accumulation.provenance.json \
    --output results/primitive-replay --run resnet18-k256-min-10
.venv/bin/python scripts/benchmark_integer_segments.py \
    --output results/integer-segment-timing.json
```

An actual detached-source replay of `resnet18-k256-min-10` verified checkpoint,
manifest and source identity and reproduced all four case prediction lists.
This is a replay check, not an independent statistical replication. The
arithmetic replayer does not regenerate the geometry inventory; use the
recorded geometry revision/configuration and its separate instructions.
Raw run JSON, datasets and checkpoints remain outside Git and can be rebuilt.
