"""Compare all spatial outputs before adding quantization and array tails."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from winograd_axis_pilot import representations


def test_all_representations_preserve_tile_output_coordinates():
    torch.manual_seed(331)
    tiles, weights = torch.randn(3, 5, 4, 4), torch.randn(7, 5, 3, 3)
    reference = torch.nn.functional.conv2d(tiles, weights)
    for method, source, coefficients, inverse in representations(tiles, weights):
        actual = inverse(source @ coefficients)
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5, msg=method)
