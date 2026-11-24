# Complete frozen-feature attention task evidence

`discovery` preserves100 queries (one per CIFAR100 class); `replication`
preserves3000 disjoint queries (30 per class). Each query retains every policy's
full descriptor, neighbor decisions, current activation/call statistics and
matched/original BF16 controls. The two fixed2000-image training galleries,
producer logs and original report bytes are also preserved.

Query gzip containers are stored byte for byte inside bounded ZIP shards.
Each directory's `task-evidence.json` indexes every member, size, container hash
and original payload hash. Original path strings remain historical provenance;
the archive's relative member paths resolve the preserved bytes independently.

Verify with `scripts/archive_feature_task.py --verify --output DIRECTORY`.
`scripts/audit_feature_task.py` independently reconstructs complete-gallery
NumPy neighbor order, class votes, descriptor errors and matched matrix work.
The complete audit covers3100 queries and27900 policy outputs.

This is a constructed fixed-gallery CIFAR100 recognition task using pretrained
DINOv3 descriptors, not the published DINOv3/ImageNet benchmark. Gallery labels
are used by the evaluator only; no activation calibration or labels enter the
quantizer. Both query allocations, gallery identities, checkpoint and producer
source hashes remain in the raw reports. The one-query-per-class discovery
cannot estimate a within-class bootstrap interval; inferential comparisons use
the separately declared replication allocation.
