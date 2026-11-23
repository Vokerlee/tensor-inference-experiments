# Source-group convolution arithmetic reference

`scripts/convolution_accumulation_reference.py` quantizes channel groups of32
at each original image coordinate. Codes, scales and optional maximum/L1
metadata belong to that source group. Lowering moves the existing codes and
metadata into R,S,C order; it does not choose new scales for each occurrence.
The weight encoding follows that same reduction order.

This is an inference-only baseline inspired by
[MicroQonv](https://arxiv.org/abs/2609.28358v1)'s pre-lowering representation.
It does not reproduce MicroQonv's square training blocks or backward pass.
Source representation reuse is established prior work; no novelty is claimed
for moving quantization before im2col.

The reference supports dense, zero-padded Conv2d with input channels divisible
by32. It rejects unaligned and depthwise cases instead of inventing free
padding/partitioning. The encoded activation group and its scale therefore
remain contiguous after lowering. Boundary padding generates zero codes and
zero metadata; it is distinct from a valid source group whose values are zero.

The consumer uses the [primitive arithmetic reference](primitive-accumulation.md),
including a live prefix guard and an FP32 output after each K slice. Cached
statistics match the current represented source values. Maximum-only guards
omit L1 collection; the tighter alternative collects it once per source group.
Neither policy uses another matrix product to decide whether the range fits.

The Torch implementation materializes expanded operands to verify numerical
semantics. A streaming implementation must instead gather corresponding source
groups/metadata with an explicitly budgeted line buffer and operand feeder.
Code packing, scale delivery, metadata reads, source retention and weight
alignment are not measured or assumed free here. This reference supplies no
native CUDA throughput or target-energy estimate.

Counters distinguish encoded source groups, expanded group slots and valid
source-group occurrences. Boundary zeros are included only in expanded slots;
they are not counted as repeated source quantization. Such occurrence counts
are not physical SRAM transaction counts.

Tests verify that cached statistics agree with lowered codes for strided,
dilated and boundary windows, that reconstructed source values survive
lowering, and that both K contracts reproduce direct integer-valued Torch
convolution including a partial output-column vector.

## Paired model execution

```sh
.venv/bin/python scripts/primitive_accumulation_pilot.py --model resnet18 \
    --batch-size 1 --samples-per-class 10 --k-slice 128 --width 24 --width 32 \
    --bound-kind max --output results/resnet18-source-accumulation.json
```

Use the pinned base environment. The first execution downloads the official
Torchvision checkpoint. The first convolution has three input channels and
remains unchanged. All selected dense convolutions use fixed encoded weights
and online source-group scales. The final classifier and remaining operations
retain the original model behavior.

The group-FP32 and guarded cases share the same source quantization. Their
comparison isolates additional effects of changing accumulation. The original
BF16 case measures their combined quantization/arithmetic difference. Reports
retain the exact target layer names, image manifest, checkpoint and source
hashes, and per-layer counters. `--bound-kind min` is a separate guard-cost
ablation, not a precision threshold fitted from evaluation labels.
