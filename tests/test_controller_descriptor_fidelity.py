import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from controller_descriptor_fidelity import fidelity, image_from_values


def test_observer_preserves_rgb_rounding_and_reference_identity():
    pixels = np.array([[[-1, 0.5, 2]]], dtype=np.float32)
    assert np.asarray(image_from_values(pixels)).tolist() == [[[0, 128, 255]]]
    assert fidelity([1, 2, 3], [1, 2, 3]) == {'cosine_distance': 0.0, 'relative_mse': 0.0}
    assert fidelity([-1, 0], [1, 0]) == {'cosine_distance': 2.0, 'relative_mse': 4.0}
    assert fidelity([2, 0], [1, 0]) == {'cosine_distance': 0.0, 'relative_mse': 1.0}


def test_observer_rejects_degenerate_descriptors():
    with pytest.raises(ValueError):
        fidelity([0, 0], [1, 0])
    with pytest.raises(ValueError):
        fidelity([1, 2], [1])
