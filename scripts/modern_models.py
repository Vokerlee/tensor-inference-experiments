"""Pinned model/checkpoint adapters for current vision and diffusion pilots."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from activation_pilot import digest


VISION = {
    "dinov3-small": ("timm/vit_small_patch16_dinov3.lvd1689m",
                     "3bf4720a82ec2066db88137180ff1f83a675cef0"),
    "convnextv2-tiny": ("timm/convnextv2_tiny.fcmae_ft_in22k_in1k",
                        "b1dd46230e80bf4cc3fa0c3c905db2c3ec53a817"),
}
SANA = ("Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
        "aa76e7f4f4928f378716b6716a2130fba3caf5b1")


def snapshot(repository: str, revision: str, *, config_only: bool = False) -> Path:
    patterns = ["*.json"] if config_only else ["*.json", "*.safetensors", "*.model", "*.txt"]
    return Path(snapshot_download(repository, revision=revision, allow_patterns=patterns))


def provenance(repository: str, revision: str, directory: Path) -> dict:
    return {"repository": repository, "revision": revision,
            "files_sha256": {str(path.relative_to(directory)): digest(path)
                             for path in sorted(directory.rglob("*")) if path.is_file()}}


@dataclass
class VisionData:
    """Adapt official preprocessing to the existing paired Imagenette loader."""
    transform: object
    configuration: dict

    @property
    def meta(self):
        from torchvision.models import ViT_B_16_Weights
        return {"categories": ViT_B_16_Weights.IMAGENET1K_V1.meta["categories"]}

    def transforms(self):
        return self.transform


def vision(name: str, *, device="cuda", pretrained: bool = True,
           dtype=torch.bfloat16):
    import timm
    from safetensors.torch import load_file

    if torch.device(device).type == "meta" and pretrained:
        raise ValueError("meta inventory must explicitly disable pretrained loading")
    repository, revision = VISION[name]
    directory = snapshot(repository, revision, config_only=not pretrained)
    configuration = json.loads((directory / "config.json").read_text())
    options = {"num_classes": configuration["num_classes"],
               "pretrained_cfg": configuration["pretrained_cfg"]}
    if "global_pool" in configuration:
        options["global_pool"] = configuration["global_pool"]
    # Meta inventories allocate no real parameter storage and execute no values.
    with torch.device("meta" if device == "meta" else "cpu"):
        model = timm.create_model(configuration["architecture"], pretrained=False, **options)
    if pretrained:
        model.load_state_dict(load_file(directory / "model.safetensors"), strict=True)
    model.eval().to(device=device, dtype=dtype)
    data_configuration = timm.data.resolve_model_data_config(model)
    data = VisionData(timm.data.create_transform(**data_configuration, is_training=False),
                      data_configuration)
    record = provenance(repository, revision, directory) if pretrained else {
        "repository": repository, "revision": revision,
        "config_sha256": digest(directory / "config.json"), "weights_loaded": False}
    record.update({"timm": timm.__version__, "num_classes": configuration["num_classes"],
                   "preprocessing": data_configuration, "weights_loaded": pretrained})
    return model, data, record


def sana_transformer_meta():
    from diffusers import SanaTransformer2DModel

    directory = snapshot(*SANA, config_only=True)
    configuration = json.loads((directory / "transformer/config.json").read_text())
    with torch.device("meta"):
        model = SanaTransformer2DModel.from_config(configuration)
    record = {"repository": SANA[0], "revision": SANA[1], "weights_loaded": False,
              "config_sha256": digest(directory / "transformer/config.json")}
    return model.eval(), record


def sana_pipeline(*, offload: bool = True):
    from diffusers import SanaSprintPipeline

    directory = snapshot(*SANA)
    pipeline = SanaSprintPipeline.from_pretrained(directory, torch_dtype=torch.bfloat16)
    pipeline.set_progress_bar_config(disable=True)
    if offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to("cuda")
    return pipeline, provenance(*SANA, directory)
