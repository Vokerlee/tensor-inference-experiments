"""Model-side adapters for proposed K256 integer Winograd precision modes."""

from contextlib import contextmanager
from dataclasses import replace as replace_dataclass
from pathlib import Path

import numpy as np
import torch

from activation_pilot import digest
from integer_winograd import convolution, prepare
from winograd_reference import transform_weights, transforms
from winograd_rounding import prepare as prepare_geometry
from winograd_shift_rounding import ShiftGeometry, prepare as prepare_shift


POLICIES = {
    "int8-k256-a8": (8, 0, False),
    "int8-k256-a4": (4, 0, False),
    "int8-k256-a4-r1": (4, 1, False),
    "int8-k256-a4-r4": (4, 4, False),
    "int8-k256-a8-r1": (8, 1, False),
    "int8-k256-a8-r4": (8, 4, False),
    "int8-k256-a8-stream": (8, 0, True),
    "int8-k256-a4-r1-native": (4, 1, False, "cuda"),
    "int8-k256-a8-r1-native": (8, 1, False, "cuda"),
    "int8-k256-a4-r4-native": (4, 4, False, "cuda"),
    "int8-k256-a8-r4-native": (8, 4, False, "cuda"),
    "int8-k256-a4-shift": (4, 1, False, "shift-cuda"),
    "int8-k256-a8-shift": (8, 1, False, "shift-cuda"),
    "int8-k256-a4-native": (4, 0, False, "nearest-cuda"),
    "int8-k256-a8-native": (8, 0, False, "nearest-cuda"),
}


@contextmanager
def modified(layers: dict, case: str, records: list, tile_batch: int,
             evidence: Path | None = None):
    bits, rank, streaming = POLICIES[case][:3]
    backend = POLICIES[case][3] if len(POLICIES[case]) > 3 else "torch"
    originals = {name: layer.forward for name, layer in layers.items()}
    weights, geometries, metadata, arrays = {}, {}, [], {}
    for index, (name, layer) in enumerate(layers.items()):
        scheme = transforms(2, dtype=torch.float32, device=layer.weight.device)
        coefficients = transform_weights(layer.weight, scheme)
        weights[name] = prepare(coefficients, 256)
        reconstructed = weights[name].reconstructed()
        geometries[name] = (prepare_geometry(reconstructed, rank=rank).to(layer.weight.device)
                            if rank else None)
        exponents = weights[name].exponents.cpu().numpy()
        arrays[f"layer{index}.weight_exponents"] = exponents
        codes, counts = torch.unique(weights[name].codes, return_counts=True)
        row = {"layer": name, "array_prefix": f"layer{index}", "rank": rank,
               "rounding_backend": backend,
               "original_channels": weights[name].channels,
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
        elif geometry is None and backend == "nearest-cuda":
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
            result, stats = convolution(values, weights[name], layer.bias, padding=layer.padding[0],
                                        tile_batch=tile_batch, bits=bits, geometry=geometries[name],
                                        streaming_inverse=streaming)
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
