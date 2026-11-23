"""Independent permutation oracle and coherent-bias/cancellation regressions."""

import itertools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from precision_moments import predicted_energy, select_cohorts, weight_statistics


def test_centered_prediction_matches_exhaustive_common_permutation_average():
    generator = torch.Generator().manual_seed(706)
    a, z = torch.randn(2, 3, 5, generator=generator, dtype=torch.float64)
    w, q = torch.randn(2, 131, 5, generator=generator, dtype=torch.float64)
    statistics = weight_statistics(w, q)
    actual_errors, actual_outputs = [], []
    for permutation in itertools.permutations(range(5)):
        output = a[:, permutation] @ w.T
        low = z[:, permutation] @ q.T
        actual_errors.append(torch.stack([(output - low)[:, begin:begin + 128].square().mean(-1)
                                          for begin in (0, 128)], -1))
        actual_outputs.append(torch.stack([output[:, begin:begin + 128].square().mean(-1)
                                          for begin in (0, 128)], -1))
    error, reference = predicted_energy(a, z, statistics)
    torch.testing.assert_close(error, torch.stack(actual_errors).mean(0), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(reference, torch.stack(actual_outputs).mean(0), rtol=1e-12, atol=1e-12)


def test_coherent_bias_and_zero_sum_receiver_are_not_independent_noise():
    a = torch.full((1, 256), 1.125, dtype=torch.float64)
    z = torch.ones_like(a)
    w = torch.ones(2, 256, dtype=torch.float64)
    statistics = weight_statistics(w, w)
    error, _ = predicted_energy(a, z, statistics)
    diagonal, _ = predicted_energy(a, z, statistics, centered=False)
    assert error.item() == 32**2
    assert diagonal.item() == 4
    w[:, ::2] *= -1
    error, reference = predicted_energy(a, z, weight_statistics(w, w))
    assert error.item() == reference.item() == 0


def test_zero_energy_cohorts_accept_and_padded_rows_do_not_reject():
    a = torch.zeros(3, 256)
    w = torch.ones(129, 256)
    selected, relative = select_cohorts(a, a, weight_statistics(w, w), tolerance=0, cohort=2)
    assert selected.shape == (2, 2)
    assert selected.all() and (relative == 0).all()
