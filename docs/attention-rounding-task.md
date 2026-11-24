# Attention activation rounding task protocol

Use the pinned modern-model environment and initialize the engine submodule.
The recorded producer version is `8a89047a676e283887215ec679120ad1ae4e7f0c`.
Its checkpoint, runtime versions, source-file hashes and image manifests are
preserved inside each raw report; checking out that version restores the exact
producer code. Original paths are provenance; archive members use relative paths.

The task uses the pretrained DINOv3-small descriptor with official preprocessing,
CIFAR100, a fixed20-training-images/class gallery and cosine5-neighbor voting.
Exact vote ties choose the nearest member among tied classes. Quantizers receive
neither labels nor gallery features. The model's Q/K/V products are partitioned
identically for every policy, including BF16.

The first test image/class is discovery. The next30/class, offset1, form the
separate replication. The three declared comparisons are balanced rank4 versus
nearest QKV4, and Q-only/K-only INT4 versus uniform INT8 with a1-percentage-point
noninferiority margin. Fixed-class paired bootstrap uses20000 draws and the
recorded deterministic seed. The three-way primary family uses lower alpha0.05/3.
These approximate intervals do not provide universal quality guarantees.

Rerun the producer on its recorded version:

```sh
.venv/bin/python scripts/attention_rounding_task_pilot.py \
    --gallery-per-class 20 --test-per-class 1 \
    --output results/attention-discovery
.venv/bin/python scripts/attention_rounding_task_pilot.py \
    --gallery-per-class 20 --test-per-class 30 --test-offset 1 \
    --output results/attention-replication
```

The current report generator and auditor operate directly on the public archive:

```sh
.venv/bin/python scripts/report_feature_task.py \
    --input discovery reports/statistics/2026-11-24-attention-task/discovery \
    --input replication reports/statistics/2026-11-24-attention-task/replication \
    --output reports/2026-11-24-attention-task
OPENBLAS_NUM_THREADS=8 .venv/bin/python scripts/audit_feature_task.py \
    --input reports/statistics/2026-11-24-attention-task/replication \
    --output reports/2026-11-24-attention-task/audit-replication.json
```

The generator emits readable `policies.csv` and `paired.csv`, plus complete
`queries.csv.gz` and `summary.json.gz`. Compression is deterministic. The summary
retains all confusion matrices and per-class correct counts. The NumPy audit
reconstructs predictions independently from every preserved full descriptor;
it is data verification, not a new network inference or hardware measurement.
