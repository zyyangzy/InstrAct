"""InternVideo checkpoint adaptation and fine-tuning policy for InstrAct."""

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class LoadReport:
    loaded: list
    missing: list
    unexpected: list
    mismatched: list


def _set_trainable(module, trainable):
    if module is None:
        return
    if isinstance(module, nn.Parameter):
        module.requires_grad = trainable
        return
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def _unfreeze_last_blocks(transformer, count):
    if transformer is None or count <= 0:
        return
    blocks = getattr(transformer, "resblocks", None)
    if blocks is None:
        return
    for block in blocks[-min(count, len(blocks)):]:
        _set_trainable(block, True)


def _resize_temporal_embedding(tensor, num_frames):
    """Resize a ``[1,T,C]`` temporal embedding by linear interpolation."""
    if tensor.ndim != 3 or tensor.shape[1] == num_frames:
        return tensor
    return F.interpolate(
        tensor.transpose(1, 2).float(),
        size=num_frames,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2).to(tensor.dtype).contiguous()


def _extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a state-dict mapping")
    for key in ("model", "state_dict", "model_state_dict"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]
    return checkpoint


def _release_key(key):
    """Map experimental checkpoint names to the release model."""
    while key.startswith("module."):
        key = key[len("module."):]
    for prefix in ("instr_action.", "instract."):
        if key.startswith(prefix):
            key = key[len(prefix):]

    replacements = {
        "temp": "temperature",
        "alpha_sidetune": "action_mix",
        "alpha_distill": "distill_alpha",
        "text_action_proj.": "text_action_projection.",
    }
    if key in replacements:
        return replacements[key]
    for old, new in replacements.items():
        if old.endswith(".") and key.startswith(old):
            return new + key[len(old):]
    return key


class InternVideoAdapter:
    """Keep all InternVideo-specific loading and fine-tuning outside InstrAct."""

    @staticmethod
    def load_checkpoint(model, checkpoint_path, num_frames, strict=False):
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"InternVideo checkpoint not found: {path}")

        # Official InternVideo checkpoints contain an EasyDict object and
        # therefore cannot be opened by PyTorch 2.6+'s weights-only default.
        # Only pass checkpoints obtained from a trusted source here.
        try:
            checkpoint = torch.load(
                path, map_location="cpu", weights_only=False
            )
        except TypeError:  # PyTorch versions predating weights_only
            checkpoint = torch.load(path, map_location="cpu")
        source = _extract_state_dict(checkpoint)
        adapted = {}
        for source_key, value in source.items():
            key = _release_key(source_key)
            if key == "vision_encoder.temporal_positional_embedding":
                value = _resize_temporal_embedding(value, num_frames)
            adapted[key] = value

        destination = model.state_dict()
        loadable = {}
        mismatched = []
        unexpected = []
        for key, value in adapted.items():
            if key not in destination:
                unexpected.append(key)
            elif destination[key].shape != value.shape:
                mismatched.append(
                    (key, tuple(value.shape), tuple(destination[key].shape))
                )
            else:
                loadable[key] = value

        missing = sorted(set(destination) - set(loadable))
        if strict and (missing or unexpected or mismatched):
            raise RuntimeError(
                "checkpoint mismatch: "
                f"{len(missing)} missing, {len(unexpected)} unexpected, "
                f"{len(mismatched)} shape mismatches"
            )
        model.load_state_dict(loadable, strict=False)
        return LoadReport(
            loaded=sorted(loadable),
            missing=missing,
            unexpected=sorted(unexpected),
            mismatched=sorted(mismatched),
        )

    @staticmethod
    def resize_model_temporal_embedding(vision_encoder, num_frames):
        embedding = getattr(
            vision_encoder, "temporal_positional_embedding", None
        )
        if embedding is None or embedding.shape[1] == num_frames:
            return
        resized = _resize_temporal_embedding(embedding.detach(), num_frames)
        vision_encoder.temporal_positional_embedding = nn.Parameter(resized)

    @staticmethod
    def configure_finetuning(model, config, freeze_backbone=False):
        vision = model.vision_encoder
        text = model.text_encoder
        _set_trainable(vision, False)
        _set_trainable(text, False)

        if freeze_backbone:
            return

        vision_config = config.finetune.vision
        for name in ("ln_pre", "ln_post", "proj"):
            _set_trainable(getattr(vision, name, None), True)

        temporal = getattr(vision, "temporal_positional_embedding", None)
        if temporal is not None:
            temporal.requires_grad = not vision_config.freeze_temporal_position

        spatial = getattr(vision, "positional_embedding", None)
        if spatial is not None:
            spatial.requires_grad = not vision_config.freeze_spatial_position

        _set_trainable(
            getattr(vision, "conv1", None),
            not vision_config.freeze_patch_embedding,
        )
        _unfreeze_last_blocks(
            getattr(vision, "transformer", None),
            vision_config.unfreeze_last_blocks,
        )

        text_config = config.finetune.text
        if not text_config.freeze:
            _set_trainable(getattr(text, "ln_final", None), True)
            projection = getattr(text, "text_projection", None)
            if projection is not None:
                projection.requires_grad = True
            if text_config.unfreeze_token_embedding:
                token_embedding = getattr(text, "token_embedding", None)
                _set_trainable(token_embedding, True)
            _unfreeze_last_blocks(
                getattr(text, "transformer", None),
                text_config.unfreeze_last_blocks,
            )

    @staticmethod
    def parameter_groups(model, config, base_lr, weight_decay):
        """Create optimizer groups using the release fine-tuning multipliers."""
        no_decay = model.no_weight_decay()
        multipliers = config.finetune.lr_multiplier
        groups = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith(("action_perceiver", "text_decoder")) or name in {
                "action_mix", "distill_alpha"
            }:
                multiplier = multipliers.action_modules
            elif any(token in name for token in (
                "positional_embedding", "token_embedding", "latents"
            )):
                multiplier = multipliers.embeddings
            else:
                multiplier = multipliers.backbone
            decay = 0.0 if name in no_decay or name.endswith(".bias") else weight_decay
            groups.append({
                "params": [parameter],
                "lr": base_lr * multiplier,
                "weight_decay": decay,
            })
        return groups
