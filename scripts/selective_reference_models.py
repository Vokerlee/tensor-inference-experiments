"""Model-side reference wrappers and causal source-control audit payloads."""

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from live_source_feedback import select_coefficient, weight_metadata
from selective_bf16_reference import full_reference, hybrid, prepare_native
from source_feedback_backend import quantize_source


@dataclass
class Weight:
    prepared: object
    bias: torch.Tensor | None
    controller: dict


def prepare_layers(layers):
    cache = {}
    for name, layer in layers.items():
        values = layer.weight.detach().reshape(layer.weight.shape[0], -1)
        matrix = values.t().contiguous()
        cache[name] = Weight(prepare_native(matrix),
            layer.bias.detach().float().contiguous() if layer.bias is not None else None,
            weight_metadata(values[:, :, None, None]))
    return cache


def grid_view(values, convolution):
    if values.shape[0] != 1:
        raise ValueError("this model audit uses batch1 causal state")
    if convolution:
        return values
    if values.ndim == 4:
        return values.permute(0, 3, 1, 2)
    if values.ndim == 3:
        return values.transpose(1, 2).unsqueeze(2)
    if values.ndim == 2:
        return values[:, :, None, None]
    raise ValueError("unsupported linear source shape")


def restore_grid(grid, shape, convolution):
    if convolution:
        return grid
    if len(shape) == 4:
        return grid.permute(0, 2, 3, 1)
    if len(shape) == 3:
        return grid.squeeze(2).transpose(1, 2)
    return grid[:, :, 0, 0]


def source_control(values, metadata, policy, record):
    if policy == "bf16":
        return values, None, []
    if policy not in ("dynamic8", "live4"):
        raise ValueError("expected bf16/dynamic8/live4")
    _, channels, height, width = values.shape
    if channels % 32:
        raise ValueError("group32 controller requires aligned channels")
    bits = 4 if policy == "live4" else 8
    block_rows = 32 if policy == "live4" else height
    mean = torch.zeros(channels, device=values.device, dtype=torch.float32)
    second = torch.zeros_like(mean)
    output = torch.empty_like(values)
    counts, exponents, coefficients, means, seconds, terminals = [], [], [], [], [], []
    for start in range(0, height, block_rows):
        stop = min(start + block_rows, height)
        alpha = select_coefficient(mean, second, metadata) if policy == "live4" else torch.zeros_like(mean)
        if record:
            coefficients.append(alpha.clone())
            means.append(mean.clone())
            seconds.append(second.clone())
        current = quantize_source(values[:, :, start:stop], alpha, bits=bits)
        output[:, :, start:stop] = current.reconstructed
        counts.append(current.counts)
        if record:
            exponents.append(current.exponents)
            terminals.append(current.terminal)
        if policy == "live4":
            samples = current.terminal[0]
            mean = 0.75 * mean + 0.25 * samples.mean(0)
            second = 0.75 * second + 0.25 * samples.square().mean(0)
    trace = None
    if record:
        trace = {"shape": list(values.shape), "bits": bits, "block_rows": block_rows,
                 "exponents": torch.cat(exponents, 1).cpu(),
                 "coefficients": torch.stack(coefficients).cpu(),
                 "means_before": torch.stack(means).cpu(), "seconds_before": torch.stack(seconds).cpu(),
                 "terminal": torch.cat(terminals, 1).cpu(),
                 "mean_after": mean.cpu(), "second_after": second.cpu(),
                 "counts": torch.stack(counts).sum(0).cpu()}
    return output, trace, [count[2] for count in counts]


def write_controls(path, traces):
    metadata, arrays = [], {}
    for index, trace in enumerate(traces):
        metadata.append({key: value for key, value in trace.items() if not isinstance(value, torch.Tensor)})
        for key, value in trace.items():
            if isinstance(value, torch.Tensor):
                arrays[f"{index}_{key}"] = value.numpy()
    arrays["metadata"] = np.frombuffer(json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8)
    np.savez_compressed(path, **arrays)


def compare_controls(actual, expected):
    if len(actual) != len(expected):
        raise ValueError("controller event count changed")
    comparisons = []
    for current, reference in zip(actual, expected, strict=True):
        differences = {}
        for key, value in reference.items():
            if isinstance(value, torch.Tensor):
                if current[key].shape != value.shape or current[key].dtype != value.dtype:
                    raise ValueError("controller tensor metadata changed")
                differences[key] = int((current[key] != value).sum())
            elif current[key] != value:
                raise ValueError("controller event identity changed")
        comparisons.append({"layer": current["layer"], "mismatches": differences})
    return comparisons


@contextmanager
def instrumented(layers, weights, method, policy="bf16", *, record=False, reference_outputs=None):
    if method not in ("full-direct", "full-tiled", "hybrid-chains4", "hybrid-tensorcore"):
        raise ValueError("unknown selective model method")
    originals = {name: module.forward for name, module in layers.items()}
    audit = {"calls": [], "controllers": [], "outputs": [], "invalid": []}

    def replacement(name, module):
        def forward(values):
            if values.dtype != torch.bfloat16:
                raise ValueError("selected model boundaries must be BF16")
            convolution = isinstance(module, torch.nn.Conv2d)
            source, controller, invalid = source_control(grid_view(values, convolution),
                                                       weights[name].controller, policy, record)
            audit["invalid"].extend(invalid)
            source = restore_grid(source, values.shape, convolution)
            matrix = (source.permute(0, 2, 3, 1) if convolution else source).reshape(-1, module.weight.shape[1]).contiguous()
            prepared, bias = weights[name].prepared, weights[name].bias
            result = None
            if method.startswith("full"):
                output = full_reference(matrix, prepared.values, bias, tiled=method == "full-tiled")
            else:
                result = hybrid(matrix, prepared, bias, backend=method.removeprefix("hybrid-"))
                output = result.output
                audit["invalid"].append(result.invalid_indices[0])
            if convolution:
                output = output.reshape(values.shape[0], values.shape[2], values.shape[3], -1).permute(0, 3, 1, 2)
            else:
                output = output.reshape(*values.shape[:-1], module.weight.shape[0])
            if record:
                position = len(audit["outputs"])
                current = output.detach().cpu().contiguous()
                mismatches = None
                if reference_outputs is not None:
                    expected = reference_outputs[position]
                    if expected.shape != current.shape:
                        raise ValueError("model event output shape changed")
                    mismatches = int((current.view(torch.int16) != expected.view(torch.int16)).sum())
                audit["outputs"].append(current)
                audit["calls"].append({"layer": name, "source_shape": list(values.shape),
                    "shape_mnk": [matrix.shape[0], prepared.values.shape[1], matrix.shape[1]],
                    "outputs": current.numel(), "fallback_outputs": result.fallback_outputs if result else None,
                    "output_mismatches": mismatches,
                    "output_sha256": hashlib.sha256(current.view(torch.uint16).numpy().tobytes()).hexdigest()})
                if controller is not None:
                    controller["layer"] = name
                    audit["controllers"].append(controller)
            return output
        return forward

    try:
        for name, module in layers.items():
            module.forward = replacement(name, module)
        yield audit
    finally:
        for name, module in layers.items():
            module.forward = originals[name]
