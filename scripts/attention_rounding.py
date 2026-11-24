"""Attention-role comparisons with shared source encodings and matched N splits."""

from contextlib import contextmanager

import torch
import torch.nn.functional as functional

from activation_rounding import prepare_geometry
from activation_rounding_models import reconstruct


POLICIES = ("bf16", "uniform8", "qkv4-nearest", "qk4-v8", "q4-k8-v8", "q8-k4-v8",
            "qk4-r1-v8", "qkv4-raw-r1", "qkv4-balanced-r1", "qkv4-balanced-r4", "qkv8-balanced-r4")


def component_plan(policy: str):
    if policy not in POLICIES:
        raise ValueError("unknown attention rounding policy")
    if policy == "bf16":
        return [(16, 0, "raw")] * 3
    if policy.startswith("uniform"):
        return [(int(policy.removeprefix("uniform")), 0, "raw")] * 3
    if policy == "qkv4-nearest":
        return [(4, 0, "raw")] * 3
    if policy == "qk4-v8":
        return [(4, 0, "raw"), (4, 0, "raw"), (8, 0, "raw")]
    if policy == "q4-k8-v8":
        return [(4, 0, "raw"), (8, 0, "raw"), (8, 0, "raw")]
    if policy == "q8-k4-v8":
        return [(8, 0, "raw"), (4, 0, "raw"), (8, 0, "raw")]
    if policy == "qk4-r1-v8":
        return [(4, 1, "qk"), (4, 1, "qk"), (8, 0, "raw")]
    precision, metric, rank = policy.split("-")
    return [(int(precision.removeprefix("qkv")), int(rank.removeprefix("r")), metric)] * 3


def prepare(layers: dict) -> dict:
    geometries = {}
    for name, layer in layers.items():
        if not name.endswith(".attn.qkv"):
            continue
        weight = layer.weight.detach().t().contiguous()
        if weight.shape[1] % 3:
            raise ValueError("equal-width Q/K/V projections required")
        components = weight.chunk(3, 1)
        balanced = torch.cat([part.float() / part.float().norm().clamp_min(1e-30)
                              for part in components], 1)
        qk = weight[:, :2 * (weight.shape[1] // 3)].contiguous()
        geometries[name] = {
            ("raw", 1): prepare_geometry(weight, 1).to(weight.device),
            ("balanced", 1): prepare_geometry(balanced, 1).to(weight.device),
            ("balanced", 4): prepare_geometry(balanced, 4).to(weight.device),
            ("qk", 1): prepare_geometry(qk, 1).to(weight.device),
        }
    return geometries


@contextmanager
def split_attention(layers: dict, geometries: dict, policy: str, *, record: bool = True):
    """Every policy executes the same three projection products, including BF16.

    Encoding is shared between roles with identical settings. Splitting output
    columns does not add original matrix MACs; software launches and actual
    hardware scheduling costs are separate from that arithmetic count.
    """
    originals = {name: layer.forward for name, layer in layers.items() if name.endswith(".attn.qkv")}
    calls = []

    def replacement(name, layer):
        def forward(values):
            plan = component_plan(policy)
            weights = layer.weight.chunk(3, 0)
            biases = layer.bias.chunk(3, 0) if layer.bias is not None else [None] * 3
            encodings, results, output = {}, {}, []
            for role, settings, weight, bias in zip(("q", "k", "v"), plan, weights, biases, strict=True):
                bits, rank, metric = settings
                if settings not in encodings:
                    if bits == 16:
                        encodings[settings], results[settings] = values, None
                    else:
                        geometry = geometries[name][(metric, rank)] if rank else None
                        encodings[settings], results[settings] = reconstruct(values, bits, geometry)
                quantized, result = encodings[settings], results[settings]
                output.append(functional.linear(quantized, weight, bias))
                if record:
                    row = {"layer": name, "role": role, "bits": bits, "rank": rank, "metric": metric,
                           "source_shape": list(values.shape), "outputs": weight.shape[0],
                           "macs": values.numel() * weight.shape[0],
                           "source_sse": float((quantized.float() - values.float()).square().sum()),
                           "source_zeros_changed": int(((values == 0) & (quantized != 0)).sum()),
                           "distinct_encodings_for_projection": len(set(plan))}
                    if result is not None:
                        row.update(selected_groups=int(result.candidate_selected.sum()),
                                   groups=result.candidate_selected.numel())
                    calls.append(row)
            return torch.cat(output, -1)
        return forward

    try:
        for name in originals:
            layers[name].forward = replacement(name, layers[name])
        yield calls
    finally:
        for name, original in originals.items():
            layers[name].forward = original
