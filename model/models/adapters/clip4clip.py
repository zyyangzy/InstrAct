"""InstrAct adapter for an unmodified Clip4Clip checkout.

Expected checkout:
  models/upstream/clip4clip  (https://github.com/ArrowLuo/CLIP4Clip)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from torch import nn


UPSTREAM = Path(__file__).resolve().parents[1] / "upstream/clip4clip"


def _build_upstream(state):
    source = UPSTREAM / "modules/module_clip.py"
    if not source.is_file():
        raise FileNotFoundError(
            "Clip4Clip is missing. Follow README.md and clone it into "
            f"{UPSTREAM}"
        )
    spec = importlib.util.spec_from_file_location(
        "instract_upstream_clip4clip_module_clip", source
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_model(state).float()


class Clip4ClipVisionAdapter(nn.Module):
    def __init__(self, model, num_frames):
        super().__init__()
        self.model = model
        self.num_frames = int(num_frames)

    def forward(self, video):
        batch, _, frames, height, width = video.shape
        if frames != self.num_frames:
            raise ValueError(f"expected {self.num_frames} frames, got {frames}")
        images = video.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, 3, height, width
        )
        global_frames, hidden = self.model.encode_image(images, return_hidden=True)
        # Upstream returns [CLS, spatial patches]; the adapter removes CLS.
        spatial = hidden[:, 1:]
        return {
            "global": global_frames.reshape(batch, frames, -1).mean(1),
            "feat_3d": spatial.reshape(batch, frames, spatial.shape[1], -1),
        }


class Clip4ClipTextAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, token_ids, return_embed=True):
        result = self.model.encode_text(token_ids, return_hidden=True)
        return result if return_embed else result[0]


def build_clip4clip(config, checkpoint_state):
    model = _build_upstream(checkpoint_state)
    return (
        Clip4ClipVisionAdapter(model, config.data.num_frames),
        Clip4ClipTextAdapter(model),
    )
