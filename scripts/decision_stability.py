"""Conservative scale-decision filtering for an exhaustively measured BF16 table.

This is an interval guard, not a new verification principle. A certificate is
conditional on a supplied valid perturbation bound and the declared decision
contract. It says nothing about output codes, feedback state or model quality.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ScaleTable:
    maxima: np.ndarray
    exponents: np.ndarray

    def __post_init__(self):
        self.maxima = np.asarray(self.maxima, dtype=np.float64)
        self.exponents = np.asarray(self.exponents, dtype=np.int32)
        if (self.maxima.ndim != 1 or len(self.maxima) < 2
                or self.exponents.shape != self.maxima.shape or self.maxima[0] != 0
                or not np.isfinite(self.maxima).all() or not (np.diff(self.maxima) > 0).all()):
            raise ValueError("expected sorted finite nonnegative maxima and matching exponents")
        changes = np.r_[True, self.exponents[1:] != self.exponents[:-1]]
        self.regions = np.cumsum(changes) - 1

    def stable(self, maximum, bound):
        """True only when all table maxima in the enclosing interval share a scale.

        The caller guarantees that both trajectories have BF16 finite sources,
        all groupwise maxima are represented by this complete table, and their
        elementwise absolute perturbation is bounded by ``bound``. An observed
        pairwise maximum is a diagnostic bound for that pair, not unseen inputs.
        """
        maximum, bound = np.broadcast_arrays(np.asarray(maximum, dtype=np.float64),
                                             np.asarray(bound, dtype=np.float64))
        if (not np.isfinite(maximum).all() or not np.isfinite(bound).all()
                or (bound < 0).any() or (maximum < 0).any()):
            raise ValueError("finite nonnegative maxima and bounds required")
        indices = np.searchsorted(self.maxima, maximum)
        if (indices >= len(self.maxima)).any() or not np.array_equal(self.maxima[indices], maximum):
            raise ValueError("reference maximum absent from decision contract")
        with np.errstate(over="ignore"):
            lower = np.maximum(0, np.nextafter(maximum - bound, -np.inf))
            upper = np.nextafter(maximum + bound, np.inf)
        first = np.searchsorted(self.maxima, lower, side="left")
        last = np.searchsorted(self.maxima, upper, side="right") - 1
        last = np.minimum(last, len(self.maxima) - 1)
        first = np.minimum(first, len(self.maxima) - 1)
        # Zero uncertainty is exact, including a reference exactly on a boundary.
        return (bound == 0) | (self.regions[first] == self.regions[last])


def grouped_sources(values):
    """CPU NCHW BF16 values converted losslessly to FP64 [...,C/32,32]."""
    if values.ndim != 4 or values.shape[1] % 32:
        raise ValueError("expected group32-aligned NCHW source")
    source = values.double().numpy().transpose(0, 2, 3, 1)
    return source.reshape(*source.shape[:-1], source.shape[-1] // 32, 32)
