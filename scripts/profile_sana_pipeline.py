#!/usr/bin/env python3
"""Profile complete SANA text-to-image component shapes without loaded weights."""

import argparse
import json
from pathlib import Path

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils.flop_counter import FlopCounterMode

from activation_pilot import ENGINE, ROOT, digest, revision, write_json
from modern_model_geometry import profile_matrices
from modern_models import SANA, sana_transformer_meta, snapshot


def component(model, execute):
    model.requires_grad_(False)
    geometry, matrices = profile_matrices(model, execute)
    with FlopCounterMode(display=False) as counter:
        output = execute()
    expected = 2 * geometry["matrix_and_conv_macs"]
    if counter.get_total_flops() != expected:
        raise ValueError("independent dispatch FLOP count differs from matrix/convolution inventory")
    return {"parameter_elements": sum(p.numel() for p in model.parameters()),
            "output_shape": list(output.shape), "shape_execution_dtype": str(output.dtype),
            "geometry": geometry, "matrix_shapes": matrices,
            "independent_torch_dispatch_flops": counter.get_total_flops()}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--text-tokens", type=int, default=300)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    if min(args.image_size, args.text_tokens, args.steps) < 1 or args.image_size % 32:
        parser.error("positive sizes and image dimensions divisible by32 required")
    from diffusers import AutoencoderDC
    from transformers import Gemma2Config, Gemma2Model

    directory = snapshot(*SANA, config_only=True)
    size = args.image_size // 32
    with torch.device("meta"):
        vae = AutoencoderDC.from_config(json.loads((directory / "vae/config.json").read_text())).eval().to(dtype=torch.bfloat16)
        latent = torch.empty(1, 32, size, size, dtype=torch.bfloat16)
        decoder = component(vae.decoder, lambda: vae.decode(latent, return_dict=False)[0])
    denoiser, _ = sana_transformer_meta()
    denoiser = denoiser.to(dtype=torch.bfloat16)
    denoising = component(denoiser, lambda: denoiser(
        latent, encoder_hidden_states=torch.empty(1, args.text_tokens, 2304, device="meta", dtype=torch.bfloat16),
        timestep=torch.empty(1, device="meta"), guidance=torch.empty(1, device="meta"),
        return_dict=False)[0])
    with FakeTensorMode(allow_non_fake_inputs=True):
        config = Gemma2Config.from_dict(json.loads((directory / "text_encoder/config.json").read_text()))
        config._attn_implementation = "eager"
        text = Gemma2Model(config).eval()
        embeds = torch.empty(1, args.text_tokens, config.hidden_size)
        mask = {name: torch.empty(1, 1, args.text_tokens, args.text_tokens)
                for name in ("full_attention", "sliding_attention")}
        encoding = component(text, lambda: text(inputs_embeds=embeds, attention_mask=mask,
                                                use_cache=False).last_hidden_state)
    components = {"text_encoder": encoding, "denoiser": denoising, "image_decoder": decoder}
    calls = {"text_encoder": 1, "denoiser": args.steps, "image_decoder": 1}
    totals = {key: sum(calls[name] * values["geometry"][key] for name, values in components.items())
              for key in ("matrix_and_conv_macs", "conv_macs", "spatial_conv_macs", "depthwise_conv_macs",
                          "linear_macs", "attention_macs", "other_explicit_matrix_macs")}
    report = {
        "schema": 1, "model": "sana-sprint", "scope": "one text-to-image request; text encoder, denoiser and decoder",
        "checkpoint": {"repository": SANA[0], "revision": SANA[1], "weights_loaded": False,
                       "config_sha256": {name: digest(directory / name) for name in (
                           "model_index.json", "text_encoder/config.json", "transformer/config.json", "vae/config.json")}},
        "experiments": revision(ROOT), "engine": revision(ENGINE), "torch": torch.__version__,
        "settings": {"image_size": args.image_size, "text_tokens": args.text_tokens, "steps": args.steps, "batch": 1},
        "source_sha256": {name: digest(ROOT / "scripts" / name) for name in (
            "profile_sana_pipeline.py", "modern_model_geometry.py", "modern_models.py", "convolution_geometry.py")},
        "component_calls": calls, "components": components, "totals": totals,
        "limitations": [
            "Shape-only meta/fake execution, not numerical model quality or target energy",
            "Text path uses CPU-fake FP32 eager shape execution; decoder/denoiser use BF16 meta execution",
            "Dtype-independent dense MAC counts do not verify numerical equivalence or mixed-precision behavior",
            "Text path uses explicit dense eager attention shapes with prepared mask shapes",
            "Padding/causal/sliding-mask kernel elision may change actual implementation work",
            "Embedding lookup, nonlinear/vector work, tokenization, scheduler and memory actions are excluded",
            "Parameter elements do not imply all parameters are read per request; no LM head is executed",
            "No classifier-free double denoiser call: this checkpoint uses guidance embeddings",
            "Decoder encoder-path parameters/compute are excluded from text-to-image decoding",
            "No physical inference latency, SRAM transactions or component energy is inferred from MACs",
        ],
    }
    write_json(args.output, report)
    for name, values in components.items():
        macs = values["geometry"]["matrix_and_conv_macs"] * calls[name]
        print(name, "request MACs", macs, "share", macs / totals["matrix_and_conv_macs"])
    print("spatial convolution request MAC share", totals["spatial_conv_macs"] / totals["matrix_and_conv_macs"])


if __name__ == "__main__":
    main()
