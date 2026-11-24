"""Model-side wrappers for current-input activation rounding experiments."""

from contextlib import contextmanager

import torch

from activation_rounding import group_grid, prepare_geometry, round_groups


POLICIES = ("bf16", "uniform8", "uniform4", "qkv4-nearest", "qkv4-r1", "qkv4-r4", "qkv8-r1")


def policy_parameters(name: str, policy: str) -> tuple[int, int]:
    if policy not in POLICIES:
        raise ValueError("unknown rounding policy")
    if policy == "bf16":
        return 16, 0
    if policy.startswith("uniform"):
        return int(policy.removeprefix("uniform")), 0
    if not name.endswith(".attn.qkv"):
        return 8, 0
    precision, method = policy.split("-")
    rank = 0 if method == "nearest" else int(method.removeprefix("r"))
    return int(precision.removeprefix("qkv")), rank


def prepare_layers(layers: dict) -> dict:
    """Only receiving weights prepare geometry; no activation samples are read."""
    prepared = {}
    for name, layer in layers.items():
        if name.endswith(".attn.qkv"):
            values = layer.weight.detach().reshape(layer.weight.shape[0], -1).t().contiguous()
            prepared[name] = {rank: prepare_geometry(values, rank).to(values.device) for rank in (1, 4)}
    return prepared


def reconstruct(values: torch.Tensor, bits: int, geometry=None):
    if values.dtype != torch.bfloat16:
        raise ValueError("BF16 model source expected")
    matrix = values.reshape(-1, values.shape[-1]).contiguous()
    if geometry is not None:
        result = round_groups(matrix, geometry, bits)
        return result.reconstructed.reshape_as(values), result
    grouped, _, scale, limit = group_grid(matrix, 32, bits)
    restored = (grouped.round().clamp(-limit, limit) * scale[:, :, None]).reshape_as(values).bfloat16()
    if not torch.isfinite(restored).all():
        raise ValueError("nonfinite model reconstruction")
    return restored, None


@contextmanager
def modified(layers: dict, prepared: dict, policy: str, *, record: bool = True):
    originals = {name: layer.forward for name, layer in layers.items()}
    calls = []

    def replace(name, layer):
        def forward(values):
            if not isinstance(layer, torch.nn.Linear):
                raise ValueError("rounding model wrapper currently supports Linear only")
            bits, rank = policy_parameters(name, policy)
            output_source, result = values, None
            if bits != 16:
                geometry = prepared[name][rank] if rank else None
                output_source, result = reconstruct(values, bits, geometry)
            output = originals[name](output_source)
            if record:
                row = {"layer": name, "source_shape": list(values.shape), "bits": bits, "rank": rank,
                       "macs": values.numel() * layer.out_features,
                       "source_scalars": values.numel(), "source_group_count": values.numel() // 32,
                       "source_sse": float((output_source.float() - values.float()).square().sum()),
                       "changed_source_scalars": int((output_source != values).sum()),
                       "exact_zeros_changed": int(((values == 0) & (output_source != 0)).sum())}
                if result is not None:
                    row.update(selected_groups=int(result.candidate_selected.sum()),
                               selected_group_count=result.candidate_selected.numel(),
                               code_histogram=torch.bincount((result.codes.int() + 128).flatten(), minlength=256).cpu().tolist())
                calls.append(row)
            return output
        return forward

    try:
        for name, layer in layers.items():
            layer.forward = replace(name, layer)
        yield calls
    finally:
        for name, layer in layers.items():
            layer.forward = originals[name]
