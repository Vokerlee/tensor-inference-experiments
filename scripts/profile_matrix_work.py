#!/usr/bin/env python3
"""Inventory matrix/convolution work and the share inside targeted MLP linears."""

import argparse
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile, record_function

from activation_pilot import load_model, write_json


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["vit_b_16", "swin_t", "dit"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.model == "dit":
        from diffusers import DiTPipeline
        from huggingface_hub import snapshot_download
        from diffusion_activation_pilot import REVISION
        snapshot = snapshot_download("facebook/DiT-XL-2-256", revision=REVISION,
                                     allow_patterns=["*.json", "*.bin", "*.safetensors"])
        pipe = DiTPipeline.from_pretrained(snapshot, torch_dtype=torch.bfloat16)
        model = pipe.transformer.eval().to("cuda")
        layers = {name: layer for name, layer in model.named_modules()
                  if isinstance(layer, torch.nn.Linear) and ".ff." in name}
        latent = torch.randn(1, 4, 32, 32, device="cuda", dtype=torch.bfloat16)
        def execute():
            return model(latent, timestep=torch.tensor([999], device="cuda"),
                         class_labels=torch.tensor([0], device="cuda"))
    else:
        model, _, layers = load_model(args.model)
        inputs = torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
        def execute():
            return model(inputs)
    contexts = {}

    def enter(name):
        def hook(module, inputs):
            contexts[name] = record_function(f"target_mlp:{name}")
            contexts[name].__enter__()
        return hook

    def leave(name):
        def hook(module, inputs, output):
            contexts.pop(name).__exit__(None, None, None)
        return hook

    handles = []
    for name, layer in layers.items():
        handles.extend([layer.register_forward_pre_hook(enter(name)),
                        layer.register_forward_hook(leave(name), always_call=True)])
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with sdpa_kernel(SDPBackend.MATH), profile(activities=[ProfilerActivity.CPU],
                                                  record_shapes=True, with_flops=True) as trace:
            execute()
            torch.cuda.synchronize()
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
        for handle in handles:
            handle.remove()
    total, targeted, operations, excluded = 0, 0, {}, {}
    for event in trace.events():
        if not event.flops:
            continue
        if event.name not in {"aten::mm", "aten::addmm", "aten::bmm", "aten::conv2d"}:
            excluded[event.name] = excluded.get(event.name, 0) + event.flops
            continue
        total += event.flops
        operations[event.name] = operations.get(event.name, 0) + event.flops
        ancestor = event.cpu_parent
        while ancestor is not None:
            if ancestor.name.startswith("target_mlp:"):
                targeted += event.flops
                break
            ancestor = ancestor.cpu_parent
    report = {"schema": 1, "model": args.model, "torch": torch.__version__,
              "matrix_convolution_flops": total, "target_mlp_flops": targeted,
              "target_fraction": targeted / total, "operations": operations,
              "excluded_flop_categories": excluded,
              "limitations": ["Matrix/convolution FLOP inventory, not complete FLOPs or energy",
                              "Math attention and disabled MHA fast path expose matrix work",
                              "Synthetic inputs determine shapes, not quality or data-dependent switching"]}
    write_json(args.output, report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
