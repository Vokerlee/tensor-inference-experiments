# Model knowledge base

Short per-model descriptions and preserved measurement evidence. Keep static
architecture/checkpoint facts separate from input-dependent statistics,
shape-only profiles, numerical quality, software-backend timing and hardware
estimates. An architecture name alone does not define a reproducible run.

| Model | Conditions / role | Card |
|---|---|---|
| DINOv3 ViT-S | 256×256 image, batch1 | [Card](dinov3-small/README.md) |
| ConvNeXt-V2 Tiny | 224×224 image, batch1 | [Card](convnextv2-tiny/README.md) |
| SANA-Sprint0.6B | 1024×1024 image, batch1;32×32 latent;300 text tokens | [Card](sana-sprint-06b/README.md) |
| ResNet18 | retained historical profile | [Card](resnet18/README.md) |
| ResNet50 | retained historical profile | [Card](resnet50/README.md) |
| ConvNeXt Tiny | retained historical profile | [Card](convnext-tiny/README.md) |
| MobileNet-V3 Large | retained historical profile | [Card](mobilenet-v3-large/README.md) |
| ViT-B/16 | retained historical profile | [Card](vit-b16/README.md) |
| Swin-T | retained historical profile | [Card](swin-t/README.md) |
| Stable Diffusion1.5 UNet | retained historical profile | [Card](sd15-unet/README.md) |
| DiT-XL/2,256px | retained historical denoiser profile | [Card](dit-xl2-256/README.md) |

Profiles are immutable records: add a new date/condition-specific filename
when input size, batch, precision, coverage or implementation changes. Keep
complete JSON/CSV statistics, source/checkpoint hashes and unfavorable results;
link report-level aggregates instead of replacing detailed evidence with a
single headline. Existing historical fields are preserved as originally
recorded, including missing provenance or dirty flags. Do not retrospectively
claim stronger provenance for an old run.

Large checkpoints, images and raw generated arrays remain outside Git under
ignored results/cache paths; record manifests, fingerprints and exact replay
commands so numerical data can be regenerated. Keep the compact scientific
statistics and their interpretation in this repository. PhD hypotheses and
private planning do not belong in model cards.
