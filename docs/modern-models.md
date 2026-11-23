# Pinned modern models

Install `requirements-modern-models.txt` for the optional model adapters.
`scripts/modern_models.py` loads exact Hugging Face revisions and records
checkpoint/configuration hashes. Vision state dictionaries are loaded strictly
from Safetensors; the adapters do not execute remote model code. Dependencies
remain pinned separately from checkpoints.

The initial set contains DINOv3 ViT-S, a feature backbone with no classification
head; ConvNeXt-V2 Tiny, a1000-class classifier; and SANA-Sprint0.6B, a
text-conditioned diffusion model. These support different numerical/task
comparisons. DINOv3 embeddings must not be reported as1000-class predictions.
ConvNeXt-V2 is an architectural control, not claimed to be the newest model.

The vision adapter reuses checkpoint preprocessing. SANA-Sprint loads the
official tokenizer, text encoder, denoiser, VAE and scheduler. CPU offload is
available for memory capacity; its latency has no role in target-hardware
energy or throughput estimates. Text encoder and VAE work must be separated
from denoiser-only optimizations.

Meta adapters execute shapes with no model weights or numerical quality
evaluation. Checkpoint revisions are still pinned for configuration identity.
Numerical pilots must explicitly use pretrained weights and retain exact
image/prompt/noise settings, model outputs and changed operator coverage.

```sh
.venv/bin/python scripts/modern_model_geometry.py --model dinov3-small \
    --output results/dinov3-geometry.json
.venv/bin/python scripts/modern_model_geometry.py --model convnextv2-tiny \
    --output results/convnextv2-geometry.json
.venv/bin/python scripts/modern_model_geometry.py --model sana-sprint \
    --output results/sana-sprint-geometry.json
```

The meta inventory records all dense linear and1×1-convolution matrix shapes,
plus the complete matrix/convolution MAC denominator. SANA counts cover one
denoiser invocation at1024×1024 image resolution with300 text tokens, excluding
text encoding and VAE decoding. Interface capacities compare BF16 K128 with
FP8 K256, both producing128 FP32 outputs from256-byte LHS vectors. The FP8
counts assume every targeted product can use that mode; they neither establish
its numerical validity nor measure SRAM traffic, cycles or energy.
