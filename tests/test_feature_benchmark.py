"""Balanced allocation, split offsets and deterministic frozen-feature voting."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from feature_benchmark import balanced_indices, nearest


def test_balanced_allocations_are_deterministic_and_offsets_are_disjoint():
    targets = [0] * 12 + [1] * 12 + [2] * 12
    first = balanced_indices(targets, 3)
    second = balanced_indices(targets, 3, 3)
    assert first == balanced_indices(targets, 3)
    assert not set(first) & set(second)
    assert [sum(targets[index] == label for index in first) for label in range(3)] == [3, 3, 3]
    with pytest.raises(ValueError):
        balanced_indices(targets, 13)


def test_similarity_and_vote_ties_have_a_declared_resolution():
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    gallery = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    labels = torch.tensor([2, 1, 0, 3])
    predictions, indices, _ = nearest(features, gallery, labels, k=2)
    assert indices.tolist() == [[0, 1], [2, 3]]
    assert predictions.tolist() == [2, 0]


def test_feature_rescaling_preserves_cosine_neighbors():
    features = torch.tensor([[1.0, 2.0]])
    gallery = torch.tensor([[1.0, 2.0], [-2.0, 1.0]])
    labels = torch.tensor([4, 9])
    assert torch.equal(nearest(features, gallery, labels, 1)[0], nearest(features * 8, gallery * 2, labels, 1)[0])
