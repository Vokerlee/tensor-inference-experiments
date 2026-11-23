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
