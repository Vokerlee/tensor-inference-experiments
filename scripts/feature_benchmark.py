"""Pinned-dataset frozen-feature evaluation helpers; no quantizer calibration."""

import hashlib
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100


SEED = 20261124


def balanced_indices(targets: list[int], per_class: int, offset: int = 0) -> list[int]:
    if per_class < 1 or offset < 0:
        raise ValueError("positive class count and nonnegative offset required")
    generator = random.Random(SEED)
    indices = []
    for label in sorted(set(targets)):
        candidates = [index for index, target in enumerate(targets) if target == label]
        generator.shuffle(candidates)
        selected = candidates[offset:offset + per_class]
        if len(selected) != per_class:
            raise ValueError("allocation exceeds a class population")
        indices.extend(selected)
    generator.shuffle(indices)
    return indices


def dataset(data, *, train: bool, per_class: int, offset: int = 0, batch_size: int = 1):
    cache = Path.home() / ".cache/tensor-inference-experiments/datasets"
    source = CIFAR100(cache, train=train, download=True, transform=data.transforms())
    indices = balanced_indices(source.targets, per_class, offset)
    selection = source.data[indices].copy()
    labels = torch.tensor([source.targets[index] for index in indices], dtype=torch.int64)
    raw_hash = hashlib.sha256(selection.tobytes()).hexdigest()
    record = {"dataset": "CIFAR100", "split": "train" if train else "test", "seed": SEED,
              "per_class": per_class, "offset": offset, "indices": indices,
              "targets": labels.tolist(), "selected_rgb_sha256": raw_hash,
              "archive_url": source.url, "archive_md5": source.tgz_md5,
              "class_names": source.classes}
    loader = DataLoader(Subset(source, indices), batch_size=batch_size,
                        num_workers=2, pin_memory=True, shuffle=False)
    return loader, record


def nearest(features: torch.Tensor, gallery: torch.Tensor, labels: torch.Tensor, k: int = 5):
    if (features.ndim != 2 or gallery.ndim != 2 or features.shape[1] != gallery.shape[1]
            or labels.shape != (gallery.shape[0],) or not 1 <= k <= gallery.shape[0]):
        raise ValueError("compatible feature/gallery matrices and valid k required")
    queries = torch.nn.functional.normalize(features.double(), dim=1)
    references = torch.nn.functional.normalize(gallery.double(), dim=1)
    similarities = queries @ references.t()
    # Stable order makes exact similarity ties deterministic by gallery index.
    neighbors = torch.argsort(similarities, descending=True, stable=True)[:, :k]
    neighbor_labels = labels[neighbors]
    predictions = []
    for row in neighbor_labels:
        counts = torch.bincount(row, minlength=int(labels.max()) + 1)
        winners = counts == counts.max()
        # Equal votes are resolved by the nearest member among tied classes.
        predictions.append(next(int(label) for label in row if winners[label]))
    return torch.tensor(predictions, dtype=torch.int64), neighbors, similarities.gather(1, neighbors)
