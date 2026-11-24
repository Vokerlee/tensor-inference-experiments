# Complete recognition check of attention activation rounding

None of the three predeclared comparisons passes its replication screen.
INT4 for query or key projections alone loses about2 percentage points against
uniform INT8 on this fixed task; the allowed loss was1 point. Role-balanced
rank4 rounding improves the nearest INT4 point estimate without establishing
superiority under the declared paired comparison. These tested configurations
fail the acceptance rule for accuracy-preserving deployment on this benchmark.

## Task, controls and independence

The pinned DINOv3 ViT-S checkpoint produces384-dimensional descriptors rather
than classifier logits. Evaluation therefore uses a fixed cosine5-nearest-neighbor
classifier with2000 BF16 descriptors from CIFAR100 training images,20 per class.
The discovery set has100 test images, one per class. Replication uses the next30
selected test images per class,3000 total, disjoint from discovery.

Gallery labels belong to the evaluator. No activation examples, labels, gallery
features or evaluation outcomes prepare the quantizer. Prepared geometry uses
immutable receiving weights; scales and code choices use the current activation.
Checkpoint, preprocessing, deterministic sample allocations, selected-image hashes
and the official CIFAR100 archive identity remain in the raw reports.

Q, K and V denote query, key and value projections in the attention mechanism.
All policies execute the same three output-column partitions, including the
BF16 control. This changes no matrix MAC count. The independent audit finds
matched BF16 descriptors identical to the original fused projection control on
all3100 images. Other dense linear inputs use current group32 INT8 encoding,
except the explicitly uniform-BF16 and uniform-INT4 policies. Weights stay BF16.

## Replication results

| Policy | Correct /3000 | Accuracy |
|---|---:|---:|
| BF16 control |1941|64.700%|
| Uniform INT8 |1931|64.367%|
| Uniform INT4 |1425|47.500%|
| INT4 QKV, nearest |1715|57.167%|
| INT4 QKV, raw rank1 |1762|58.733%|
| INT4 QKV, role-balanced rank4 |1750|58.333%|
| INT4 Q/K, INT8 V |1732|57.733%|
| INT4 Q only, INT8 K/V |1871|62.367%|
| INT4 K only, INT8 Q/V |1868|62.267%|

The three primary comparisons were fixed before replication. Paired bootstrap
resampling preserves each class's30-query allocation; the family uses a
one-sided lower quantile at0.05/3. Percentile intervals are approximate benchmark
comparisons. McNemar tests equality on discordant pairs, rather than testing a
noninferiority margin.

| Candidate vs comparator | Accuracy difference | Descriptive paired95% interval | Family lower bound | Required margin | Pass |
|---|---:|---:|---:|---:|---|
| Balanced rank4 vs nearest QKV4 |+1.167pp|[-0.100,+2.400]pp|-0.200pp|0pp|No|
| Q4/K8/V8 vs uniform INT8 |-2.000pp|[-3.100,-0.900]pp|-3.233pp|-1pp|No|
| Q8/K4/V8 vs uniform INT8 |-2.100pp|[-3.167,-1.033]pp|-3.267pp|-1pp|No|

Raw rank1 has a higher task point estimate than balanced rank4 despite the latter's
lower descriptor MSE. This is an exploratory ordering, not an additional declared
superiority result. The [ten-image descriptor study](2026-11-24-activation-rounding.md)
and this task answer different questions; a better local or descriptor metric
alone is insufficient as an acceptance criterion.

## Cost and transfer limits

The role-selective cases need two distinct encodings per QKV projection, versus
one shared encoding for uniform INT8. On this model the audit counts24 versus12
such encodings per image. It also checks that all policies retain the same
5,541,986,304 selected linear MACs per forward. Q-only or K-only precision changes
cover one third of QKV work, not the entire network.

These are BF16-reconstructed INT codes, not a native FP8 or INT4 throughput test.
No CUDA duration, bit count or logical MAC count here establishes target-device
energy. The task is a constructed fixed-gallery CIFAR100 evaluation, not the
published DINOv3/ImageNet benchmark or a guarantee for other applications.

## Complete evidence and independent checks

The [byte-exact archive](statistics/2026-11-24-attention-task/README.md) contains
all3100 query records and27900 policy outputs, per-call observations, both
original/matched references, fixed galleries and logs. An independent NumPy
reconstruction checks every complete-gallery neighbor order, class vote and
full-descriptor error. All match; maximum cosine discrepancies are below2.4e-15.

[Derived tables and audits](2026-11-24-attention-task/) preserve both allocations,
all per-class confusion matrices and every query. Large tables are deterministic
gzip files. Discovery has one observation per class, so within-class bootstrap
intervals are explicitly unavailable. Its point estimates cannot pass a
replication screen.

See the [reproduction protocol](../docs/attention-rounding-task.md) for regenerating
tables, auditing all outputs and rerunning the frozen producer version. The
unfavorable results are retained as complete numerical evidence.
