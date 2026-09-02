"""Backbone-independent InstrAct model.

The action-centric components are kept separate from the video-text backbone so
that users can attach InstrAct to InternVideo or another compatible encoder.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from modules.contrastive_loss import VTC_VTM_Loss
from modules.dtw_loss import compute_dtw_loss
from modules.hard_negative_loss import verb_hard_neg_nce_torch
from modules.mask_action_modeling import MultimodalTextDecoder
from modules.perceiver import Perceiver


def _feature_enabled(args, name):
    return bool(getattr(args, "use_instract", False) or getattr(args, name, False))


def _valid_step_mask(lengths, max_steps, device):
    lengths = torch.as_tensor(lengths, device=device)
    positions = torch.arange(max_steps, device=device)
    return positions[None] < lengths[:, None]


def _step_indices(lengths, max_steps, device):
    valid = _valid_step_mask(lengths, max_steps, device)
    positions = torch.arange(max_steps, device=device).expand_as(valid)
    return positions.masked_fill(~valid, -1)


class InstrAct(nn.Module):
    """Action-centric pretraining wrapper around external video/text encoders.

    Required backbone interfaces:

    * ``vision_encoder(video) -> {"global": [B,D], "feat_3d": [B,F,P,D]}``
    * ``text_encoder(ids, return_embed=True) -> (global, token_features)``
    """

    def __init__(self, config, args, vision_encoder, text_encoder):
        super().__init__()
        model_config = config.model
        data_config = config.data

        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.max_negatives = data_config.max_negatives
        self.max_verbs = data_config.max_verbs
        self.max_text = 2 + self.max_negatives + self.max_verbs

        self.use_hard_negatives = bool(
            getattr(args, "use_instract", False)
            or getattr(args, "use_hard_negatives", False)
        )
        self.use_dtw = _feature_enabled(args, "use_dtw")
        self.use_mam = _feature_enabled(args, "use_mam")
        self.use_action_perceiver = (
            _feature_enabled(args, "use_action_perceiver")
            or self.use_dtw
            or self.use_mam
        )

        self.temperature = nn.Parameter(torch.tensor(float(model_config.temperature)))
        self.temperature_min = float(model_config.temperature_min)
        self.contrastive_loss = VTC_VTM_Loss(False)
        self.loss_weights = model_config.loss_weights

        text_dim = model_config.text_feature_dim
        if self.use_action_perceiver:
            action_config = dict(model_config.action_perceiver)
            self.action_perceiver = Perceiver(**action_config)
            action_dim = action_config["dim"]
            self.text_action_projection = (
                nn.Identity()
                if text_dim == action_dim
                else nn.Linear(text_dim, action_dim, bias=False)
            )
            self.action_mix = nn.Parameter(
                torch.tensor(float(model_config.action_mix_init))
            )
            self.distill_alpha = nn.Parameter(
                torch.tensor(float(model_config.distill_alpha_init))
            )
        else:
            self.action_perceiver = None
            self.text_action_projection = nn.Identity()
            self.register_parameter("action_mix", None)
            self.register_parameter("distill_alpha", None)

        self.text_decoder = (
            MultimodalTextDecoder(**dict(model_config.multimodal_decoder))
            if self.use_mam else None
        )
        self.dtw_config = model_config.dtw

    @torch.no_grad()
    def clamp_temperature(self):
        self.temperature.clamp_(min=self.temperature_min)

    def encode_vision(self, video):
        output = self.vision_encoder(video)
        if not isinstance(output, dict) or not {"global", "feat_3d"} <= output.keys():
            raise TypeError(
                "vision_encoder must return a dict containing global and feat_3d"
            )
        return output

    def encode_text(self, token_ids):
        output = self.text_encoder(token_ids, return_embed=True)
        if not isinstance(output, tuple) or len(output) != 2:
            raise TypeError(
                "text_encoder must return (global_features, token_features)"
            )
        return output

    def no_weight_decay(self):
        names = {"temperature", "action_mix", "distill_alpha"}
        if self.text_decoder is not None:
            names.add("text_decoder.mask_action_embedding")
        for encoder_name in ("vision_encoder", "text_encoder"):
            encoder = getattr(self, encoder_name)
            if hasattr(encoder, "no_weight_decay"):
                names.update(
                    f"{encoder_name}.{name}"
                    for name in encoder.no_weight_decay()
                )
        return names

    def _action_features(self, visual_tokens):
        action_tokens = self.action_perceiver(visual_tokens)
        return action_tokens, F.normalize(action_tokens.mean(1), dim=-1)

    def _mix_global_and_action(self, global_features, action_features):
        weight = torch.sigmoid(self.action_mix)
        return (
            weight * F.normalize(global_features, dim=-1)
            + (1 - weight) * action_features
        )

    def encode(self, video, token_ids):
        """Encode one video/text pair for downstream retrieval."""
        vision = self.encode_vision(video)
        text_global, _ = self.encode_text(token_ids)
        if not self.use_action_perceiver:
            return vision["global"], text_global

        action_tokens, action_global = self._action_features(vision["feat_3d"])
        del action_tokens
        return (
            self._mix_global_and_action(vision["global"], action_global),
            self.text_action_projection(text_global),
        )

    def forward(
        self,
        video,
        text,
        *,
        hard_negative_mask=None,
        num_verbs=None,
        action_mask=None,
        idx=None,
        return_loss_dict=False,
    ):
        self.clamp_temperature()
        batch_size = video.shape[0]

        vision = self.encode_vision(video)
        if text.ndim == 3:
            if text.shape[:2] != (batch_size, self.max_text):
                raise ValueError(
                    "batched text must have shape "
                    f"[B, {self.max_text}, N], got {tuple(text.shape)}"
                )
            flat_text = text.reshape(-1, text.shape[-1])
            text_ids = text
        elif text.ndim == 2:
            if text.shape[0] != batch_size * self.max_text:
                raise ValueError(
                    f"expected {batch_size * self.max_text} flattened text "
                    f"rows, got {text.shape[0]}"
                )
            flat_text = text
            text_ids = text.reshape(batch_size, self.max_text, text.shape[-1])
        else:
            raise ValueError(
                "text must have shape [B, max_text, N] or "
                f"[B*max_text, N], got {tuple(text.shape)}"
            )

        text_global, text_tokens = self.encode_text(flat_text)
        if text_global.shape[0] != batch_size * self.max_text:
            raise ValueError(
                f"expected {batch_size * self.max_text} text rows, "
                f"got {text_global.shape[0]}"
            )

        text_global = text_global.reshape(batch_size, self.max_text, -1)
        text_tokens = text_tokens.reshape(
            batch_size, self.max_text, text_tokens.shape[1], text_tokens.shape[2]
        )

        positive_text = text_global[:, 0]
        hard_negative_text = text_global[:, :1 + self.max_negatives]
        verb_text = text_global[:, -self.max_verbs:]
        mam_text_tokens = text_tokens[:, 1 + self.max_negatives]
        original_ids = text_ids[:, 0]

        visual_global = vision["global"]
        teacher_action = None
        action_tokens = None
        if self.use_action_perceiver:
            action_tokens, action_global = self._action_features(
                vision["feat_3d"]
            )
            visual_global = self._mix_global_and_action(
                vision["global"],
                action_global,
            )
            positive_text = self.text_action_projection(positive_text)
            hard_negative_text = self.text_action_projection(hard_negative_text)

            if num_verbs is None:
                raise ValueError("num_verbs is required by the Action Perceiver")
            verb_mask = _valid_step_mask(
                num_verbs, self.max_verbs, verb_text.device
            )
            # Shared cross-attention weights provide a trained verb-conditioned
            # teacher; stopping gradients prevents the target branch collapsing
            # into the student.
            with torch.no_grad():
                teacher_action = self.action_perceiver(
                    vision["feat_3d"],
                    latents=self.text_action_projection(verb_text),
                    attn_mask=verb_mask,
                ).sum(1) / verb_mask.sum(1, keepdim=True).clamp_min(1)

        vtc_loss = self.contrastive_loss.vtc_loss(
            visual_global,
            positive_text,
            idx,
            self.temperature,
            all_gather=True,
            alpha=(
                torch.sigmoid(self.distill_alpha)
                if self.distill_alpha is not None else 0
            ),
            vision_distill=teacher_action,
        )
        loss_terms = {"vtc": vtc_loss}
        loss = self.loss_weights.vtc * vtc_loss

        if self.use_hard_negatives:
            if hard_negative_mask is None:
                raise ValueError("hard_negative_mask is required")
            hard_negative_loss = verb_hard_neg_nce_torch(
                encoded_video=visual_global[:batch_size],
                encoded_text=hard_negative_text.reshape(
                    -1, hard_negative_text.shape[-1]
                ),
                mask_text=hard_negative_mask.reshape(-1, 1),
            )
            loss_terms["hard_negative"] = hard_negative_loss
            loss = loss + self.loss_weights.hard_negative * hard_negative_loss

        if self.use_dtw:
            indices = _step_indices(
                num_verbs, self.max_verbs, action_tokens.device
            )
            dtw_loss, _ = compute_dtw_loss(
                action_tokens,
                self.text_action_projection(verb_text),
                indices,
                self.dtw_config.margin,
                self.dtw_config.order_weight,
                self.dtw_config.scale,
                "dtw_contrastive",
            )
            loss_terms["dtw"] = dtw_loss
            loss = loss + self.loss_weights.dtw * dtw_loss

        if self.use_mam:
            if action_mask is None:
                raise ValueError("action_mask is required when MAM is enabled")
            mam_loss = self.text_decoder(
                text_embeds=mam_text_tokens,
                image_tokens=action_tokens,
                input_ids=original_ids,
                action_mask=action_mask,
            )
            loss_terms["mam"] = mam_loss
            loss = loss + self.loss_weights.mam * mam_loss

        if return_loss_dict:
            return {"loss": loss, **loss_terms}
        return loss
