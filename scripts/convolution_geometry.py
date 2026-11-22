#!/usr/bin/env python3
"""Inventory convolution geometry without checkpoint downloads or GPU execution."""

import argparse
import hashlib
import json
import math
import subprocess
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torchvision
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode


@dataclass(frozen=True)
class Convolution:
    name: str
    batch: int
    input_channels: int
    output_channels: int
    groups: int
    input_height: int
    input_width: int
    output_height: int
    output_width: int
    kernel_height: int
    kernel_width: int
    stride_height: int
    stride_width: int
    dilation_height: int
    dilation_width: int
    pad_top: int
    pad_left: int

    @property
    def reduction(self) -> int:
        return self.kernel_height * self.kernel_width * self.input_channels // self.groups

    @property
    def rows(self) -> int:
        return self.batch * self.output_height * self.output_width

    @property
    def macs(self) -> int:
        return self.rows * self.output_channels * self.reduction

    def valid_macs(self) -> int:
        """Count valid image coordinates, excluding mathematical boundary padding."""
        height = valid_axis(self.input_height, self.output_height, self.kernel_height,
                            self.stride_height, self.dilation_height, self.pad_top)
        width = valid_axis(self.input_width, self.output_width, self.kernel_width,
                           self.stride_width, self.dilation_width, self.pad_left)
        return self.batch * self.output_channels * (self.input_channels // self.groups) * height * width

    def schedule(self, reduction_width: int, channel_windows: bool = False) -> dict:
        """Count issued vector slots under an explicitly group-separated mapping.

        M rows stream independently. No square output tile or free array partition
        is assumed. Channel-window mapping reduces each input channel separately;
        it needs a subsequent merge of its partial results.
        """
        if reduction_width <= 0:
            raise ValueError("reduction width must be positive")
        channels = self.input_channels // self.groups
        spatial = self.kernel_height * self.kernel_width
        segments = (channels * math.ceil(spatial / reduction_width) if channel_windows
                    else math.ceil(self.reduction / reduction_width))
        output_tiles = math.ceil((self.output_channels // self.groups) / 128)
        vectors = self.rows * self.groups * segments * output_tiles
        capacity = vectors * reduction_width * 128
        return {
            "reduction_width": reduction_width,
            "output_width": 128,
            "mapping": "separate-channel-windows" if channel_windows else "concatenated-channels",
            "k_segments_per_group": segments,
            "vector_issue_groups": vectors,
            "scheduled_product_slots": capacity,
            "dense_product_utilization": self.macs / capacity,
            "valid_product_utilization": self.valid_macs() / capacity,
            "lhs_capacity_bytes": vectors * 256,
            "fp32_output_capacity_bytes_if_emitted_each_segment": vectors * 128 * 4,
        }


def valid_axis(input_size: int, output_size: int, kernel: int,
               stride: int, dilation: int, padding: int) -> int:
    return sum(0 <= position * stride + tap * dilation - padding < input_size
               for position in range(output_size) for tap in range(kernel))


@contextmanager
def attention_counter(matrix=None):
    """Count QK and AV products in calls through Torch's SDPA interface."""
    original = F.scaled_dot_product_attention
    counts = {"macs": 0, "calls": 0}
    previous_fastpath = torch.backends.mha.get_fastpath_enabled()

    def count(query, key, value, *args, **kwargs):
        # These workloads have ordinary multi-head attention, without GQA.
        batches = math.prod(torch.broadcast_shapes(query.shape[:-2], key.shape[:-2], value.shape[:-2]))
        counts["macs"] += batches * query.shape[-2] * key.shape[-2] * (query.shape[-1] + value.shape[-1])
        counts["calls"] += 1
        if matrix is not None:
            matrix.inside_sdpa += 1
        try:
            return original(query, key, value, *args, **kwargs)
        finally:
            if matrix is not None:
                matrix.inside_sdpa -= 1

    F.scaled_dot_product_attention = count
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield counts
    finally:
        F.scaled_dot_product_attention = original
        torch.backends.mha.set_fastpath_enabled(previous_fastpath)


class MatrixCounter(TorchDispatchMode):
    """Count explicit matrix products, excluding products already charged to SDPA."""

    def __init__(self):
        super().__init__()
        self.macs = 0
        self.inside_sdpa = 0

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        if not self.inside_sdpa:
            if function == torch.ops.aten.addmm.default:
                left, right = args[1:3]
                self.macs += left.shape[0] * left.shape[1] * right.shape[1]
            elif function in (torch.ops.aten.mm.default, torch.ops.aten.bmm.default):
                left, right = args[:2]
                self.macs += math.prod(left.shape[:-2]) * left.shape[-2] * left.shape[-1] * right.shape[-1]
        return function(*args, **(kwargs or {}))


def inventory(model, execute) -> dict:
    convolutions = []
    linear_macs = 0

    def capture(name):
        def hook(module, inputs, output):
            nonlocal linear_macs
            if isinstance(module, torch.nn.Linear):
                linear_macs += output.numel() * module.in_features
                return
            x = inputs[0]
            left, _, top, _ = module._reversed_padding_repeated_twice
            convolutions.append(Convolution(
                name, x.shape[0], module.in_channels, module.out_channels, module.groups,
                x.shape[2], x.shape[3], output.shape[2], output.shape[3],
                *module.kernel_size, *module.stride, *module.dilation, top, left))
        return hook

    handles = [module.register_forward_hook(capture(name))
               for name, module in model.named_modules()
               if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))]
    try:
        with MatrixCounter() as matrix, attention_counter(matrix) as attention:
            execute()
    finally:
        for handle in handles:
            handle.remove()
    conv_macs = sum(layer.macs for layer in convolutions)
    if matrix.macs < linear_macs:
        raise RuntimeError("dispatch inventory missed a linear matrix product")
    other_matrix_macs = matrix.macs - linear_macs
    denominator = conv_macs + matrix.macs + attention["macs"]
    return {
        "conv_macs": conv_macs,
        "linear_macs": linear_macs,
        "attention_macs": attention["macs"],
        "attention_calls": attention["calls"],
        "other_explicit_matrix_macs": other_matrix_macs,
        "matrix_and_conv_macs": denominator,
        "conv_fraction_of_matrix_and_conv_macs": conv_macs / denominator,
        "spatial_conv_macs": sum(layer.macs for layer in convolutions
                                 if layer.kernel_height * layer.kernel_width > 1),
        "depthwise_conv_macs": sum(layer.macs for layer in convolutions
                                   if layer.groups == layer.input_channels > 1),
        "short_full_reduction_macs_k_le_25": sum(layer.macs for layer in convolutions
                                                 if layer.reduction <= 25),
        "layers": [
            {**asdict(layer), "K": layer.reduction, "M": layer.rows,
             "N_per_group": layer.output_channels // layer.groups,
             "macs": layer.macs, "valid_macs": layer.valid_macs(),
             "schedules": [layer.schedule(k, separate)
                           for k in (128, 256) for separate in (False, True)]}
            for layer in convolutions
        ],
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=(
        "resnet50", "convnext_tiny", "mobilenet_v3_large", "vit_b_16", "swin_t", "unet"))
    parser.add_argument("--unet-config", type=Path)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.image_size <= 0:
        parser.error("image size must be positive")
    config_hash = None
    with torch.device("meta"):
        if args.model == "unet":
            if args.unet_config is None:
                parser.error("unet requires a locally recorded --unet-config")
            from diffusers import UNet2DConditionModel
            config_hash = hashlib.sha256(args.unet_config.read_bytes()).hexdigest()
            model = UNet2DConditionModel.from_config(json.loads(args.unet_config.read_text())).eval()
            sample = torch.empty(1, model.config.in_channels, args.image_size, args.image_size)
            context = torch.empty(1, 77, model.config.cross_attention_dim)
            execute = lambda: model(sample, torch.tensor([1]), encoder_hidden_states=context)
        else:
            model = torchvision.models.get_model(args.model, weights=None).eval()
            sample = torch.empty(1, 3, args.image_size, args.image_size)
            execute = lambda: model(sample)
        result = inventory(model, execute)
    root = Path(__file__).resolve().parents[1]
    result.update({
        "schema": 1, "model": args.model, "image_size": args.image_size,
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "unet_config_sha256": config_hash,
        "limitations": [
            "Shape-only meta execution: no model quality, latency or energy measurement",
            "MAC denominator covers Conv2d, explicit mm/addmm/bmm and SDPA QK/AV, excluding other operators",
            "K=128 BF16 and K=256 FP8 refer to LHS input length; output width is always 128 FP32",
            "INT8 and FP4 physical geometry is unspecified",
            "Mappings are alternatives, not a statement of current hardware stationarity or grouping",
            "Capacity bytes are interface slots, not measured SRAM reads or output cadence",
            "Per-segment FP32 output volume is conditional on emitting every K segment",
            "Concatenated K contributions require a legal partial-sum merge; packing independent groups needs another mapping",
        ],
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "layers"}, indent=2))


if __name__ == "__main__":
    main()
