"""InstrAct adapter for an unmodified CLIP-ViP checkout.

Expected checkout:
  models/upstream/clip_vip  (https://github.com/microsoft/XPretrain)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn
from transformers.models.clip.configuration_clip import CLIPConfig


UPSTREAM = Path(__file__).resolve().parents[1] / "upstream/clip_vip"


def _upstream_model():
    package_root = UPSTREAM / "CLIP-ViP"
    source = package_root / "src/modeling/CLIP_ViP.py"
    if not source.is_file():
        raise FileNotFoundError(
            "CLIP-ViP is missing. Follow README.md and clone XPretrain into "
            f"{UPSTREAM}"
        )
    sys.path.insert(0, str(package_root))
    try:
        from src.modeling.CLIP_ViP import CLIPModel
    finally:
        sys.path.pop(0)
    return CLIPModel


class CLIPViPVisionAdapter(nn.Module):
    def __init__(self, vision_model, projection, num_frames, patch_size):
        super().__init__()
        self.vision_model = vision_model
        self.projection = projection
        self.num_frames = int(num_frames)
        self.patch_size = int(patch_size)

    def forward(self, video):
        video = video.permute(0, 2, 1, 3, 4).contiguous()
        batch, frames, _, height, width = video.shape
        if frames != self.num_frames:
            raise ValueError(f"expected {self.num_frames} frames, got {frames}")
        output = self.vision_model(video, output_hidden_states=True, return_dict=True)
        patches = (height // self.patch_size) * (width // self.patch_size)
        spatial = output.last_hidden_state[:, -(frames * patches):]
        spatial = spatial.reshape(batch, frames, patches, -1)
        return {
            "global": self.projection(output.pooler_output),
            "feat_3d": self.projection(spatial),
        }


class CLIPViPTextAdapter(nn.Module):
    def __init__(self, text_model, projection):
        super().__init__()
        self.text_model = text_model
        self.projection = projection

    def forward(self, token_ids, return_embed=True):
        output = self.text_model(
            input_ids=token_ids,
            attention_mask=token_ids.ne(0).long(),
            output_hidden_states=True,
            return_dict=True,
        )
        result = (
            self.projection(output.pooler_output),
            self.projection(output.last_hidden_state),
        )
        return result if return_embed else result[0]


def build_clip_vip(config, checkpoint_state):
    model_cls = _upstream_model()
    source = Path(config.backbone.clip_vip.hf_config)
    if not source.is_absolute():
        source = Path(__file__).resolve().parents[2] / source
    clip_config = CLIPConfig.from_pretrained(source, local_files_only=True)
    vip = config.backbone.clip_vip
    clip_config.vision_additional_config = OmegaConf.create({
        "type": "ViP",
        "temporal_size": int(vip.temporal_size),
        "if_use_temporal_embed": 1,
        "logit_scale_init_value": 4.60,
        "add_cls_num": int(vip.add_cls_num),
    })
    model = model_cls(clip_config)
    state = {
        key.removeprefix("clipmodel."): value
        for key, value in checkpoint_state.items()
        if key.startswith("clipmodel.")
    }
    model.load_state_dict(state, strict=False)
    return (
        CLIPViPVisionAdapter(
            model.vision_model,
            model.visual_projection,
            config.data.num_frames,
            clip_config.vision_config.patch_size,
        ),
        CLIPViPTextAdapter(model.text_model, model.text_projection),
    )
