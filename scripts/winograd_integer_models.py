"""Model-side adapters for proposed K256 integer Winograd precision modes."""

from contextlib import contextmanager
from dataclasses import dataclass, replace as replace_dataclass
from pathlib import Path
from math import ceil

import numpy as np
import torch

from activation_pilot import digest
from integer_winograd import convolution, prepare
from integer_winograd_one_axis import convolution as convolution_one
from winograd_one_axis import transform_weights as transform_one
from winograd_reference import transform_weights, transforms
from winograd_rounding import prepare as prepare_geometry
from winograd_shift_rounding import ShiftGeometry, prepare as prepare_shift


@dataclass(frozen=True)
class Policy:
    bits: int
    rank: int = 0
    streaming: bool = False
    backend: str = "torch"
    tile: int = 2


POLICIES = {
    "int8-k256-a8": Policy(8),
    "int8-k256-a4": Policy(4),
    "int8-k256-a4-r1": Policy(4, rank=1),
    "int8-k256-a4-r4": Policy(4, rank=4),
    "int8-k256-a8-r1": Policy(8, rank=1),
    "int8-k256-a8-r4": Policy(8, rank=4),
    "int8-k256-a8-stream": Policy(8, streaming=True),
    "int8-k256-a4-r1-native": Policy(4, rank=1, backend='cuda'),
    "int8-k256-a8-r1-native": Policy(8, rank=1, backend='cuda'),
    "int8-k256-a4-r4-native": Policy(4, rank=4, backend='cuda'),
    "int8-k256-a8-r4-native": Policy(8, rank=4, backend='cuda'),
    "int8-k256-a4-shift": Policy(4, rank=1, backend='shift-cuda'),
    "int8-k256-a8-shift": Policy(8, rank=1, backend='shift-cuda'),
    "int8-k256-a4-native": Policy(4, backend='nearest-cuda'),
    "int8-k256-a8-native": Policy(8, backend='nearest-cuda'),
    "int8-k256-a4-axis-fit": Policy(4, backend='axis-fit'),
    "int8-k256-a8-axis-fit": Policy(8, backend='axis-fit'),
    "int8-k256-a4-axis-all": Policy(4, backend='axis-all'),
    "int8-k256-a8-axis-all": Policy(8, backend='axis-all'),
    "int8-k256-f4-a8-native": Policy(8, backend='nearest-cuda', tile=4),
    "int8-k256-f4-a4-native": Policy(4, backend='nearest-cuda', tile=4),
    "int8-k256-f4-a8-r1-native": Policy(8, rank=1, backend='cuda', tile=4),
    "int8-k256-f4-a4-r1-native": Policy(4, rank=1, backend='cuda', tile=4),
    "int8-k256-f4-a8-r4-native": Policy(8, rank=4, backend='cuda', tile=4),
    "int8-k256-f4-a8-shift": Policy(8, rank=1, backend='shift-cuda', tile=4),
    "int8-k256-f4-a4-shift": Policy(4, rank=1, backend='shift-cuda', tile=4),
}


@contextmanager
def modified(layers: dict, case: str, records: list, tile_batch: int,
             evidence: Path | None = None):
    policy = POLICIES[case]
    bits, rank, streaming = policy.bits, policy.rank, policy.streaming
    backend, tile = policy.backend, policy.tile
    originals = {name: layer.forward for name, layer in layers.items()}
    weights, geometries, metadata, arrays, one_axis = {}, {}, [], {}, {}
    for index, (name, layer) in enumerate(layers.items()):
        scheme = transforms(tile, dtype=torch.float32, device=layer.weight.device)
        one_axis[name] = backend == "axis-all" or (backend == "axis-fit" and
                         ceil(3 * layer.in_channels / 256) <= 2 * ceil(layer.in_channels / 256))
        coefficients = (transform_one(layer.weight.float()) if one_axis[name]
                        else transform_weights(layer.weight, scheme))
        weights[name] = prepare(coefficients, 256)
        reconstructed = weights[name].reconstructed()
        geometries[name] = (prepare_geometry(reconstructed, tile=tile, rank=rank).to(layer.weight.device)
                            if rank else None)
        exponents = weights[name].exponents.cpu().numpy()
        arrays[f"layer{index}.weight_exponents"] = exponents
        codes, counts = torch.unique(weights[name].codes, return_counts=True)
        row = {"layer": name, "array_prefix": f"layer{index}", "rank": rank,
               "rounding_backend": backend, "tile": tile, "lowering": "one-axis" if one_axis[name] else "two-axis",
               "original_channels": layer.in_channels, "lowered_reduction": weights[name].channels,
               "padded_channels": weights[name].codes.shape[1],
               "weight_groups": weights[name].exponents.numel(),
               "weight_code_histogram": {str(int(code)): int(count) for code, count in zip(codes, counts)},
               "transformed_weight_sse": float((reconstructed[:, :coefficients.shape[1]] - coefficients).double().square().sum())}
        geometry = geometries[name]
        if geometry is not None:
            for key in ("factor", "diagonal", "normalizer", "eigenvalues", "metric"):
                arrays[f"layer{index}.{key}"] = getattr(geometry, key).cpu().numpy()
            eigenvalues = geometry.eigenvalues[:weights[name].channels]
            energy = eigenvalues.sum(-1)
            shares = eigenvalues[:, -rank:].sum(-1) / energy.clamp_min(1e-30)
            row["top_rank_energy_mean_nonzero_actual_channels"] = float(
                shares.sum() / (energy > 0).sum().clamp_min(1))
        if geometry is not None and backend == "cuda":
            geometries[name] = replace_dataclass(geometry, backend="cuda")
        elif geometry is not None and backend == "shift-cuda":
            geometries[name] = prepare_shift(geometry, backend="cuda")
            arrays[f"layer{index}.packed_metadata"] = geometries[name].metadata.cpu().numpy()
        elif geometry is None and backend in ("nearest-cuda", "axis-fit", "axis-all"):
            packed = torch.zeros(reconstructed.shape[1], reconstructed.shape[0],
                                 dtype=torch.int16, device=layer.weight.device)
            geometries[name] = ShiftGeometry(packed, "cuda", rank=0)
            arrays[f"layer{index}.packed_metadata"] = packed.cpu().numpy()
        metadata.append(row)
    prepared = {"weight_preparation": metadata}
    if evidence is not None:
        evidence.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(evidence, **arrays)
        prepared.update({"geometry": str(evidence), "geometry_sha256": digest(evidence)})

    def replace(name, layer):
        def forward(values):
            if one_axis[name]:
                result, stats = convolution_one(values, weights[name], geometries[name], layer.bias,
                                                padding=layer.padding[0], tile_batch=tile_batch, bits=bits)
            else:
                result, stats = convolution(values, weights[name], layer.bias, tile=tile, padding=layer.padding[0],
                                            tile_batch=tile_batch, bits=bits, geometry=geometries[name],
                                            streaming_inverse=streaming)
                stats["lowering"] = "two-axis"
            records.append({"layer": name, "input_shape": list(values.shape),
                            "output_shape": list(result.shape), **stats})
            return result
        return forward

    try:
        for name, layer in layers.items():
            layer.forward = replace(name, layer)
        yield prepared
    finally:
        for name, layer in layers.items():
            layer.forward = originals[name]
