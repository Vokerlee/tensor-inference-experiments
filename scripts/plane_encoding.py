"""An encoded tensor and its authoritative scale metadata travel together."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PlaneEncoding:
    reconstructed: torch.Tensor
    exponents: torch.Tensor
    statistics: dict
