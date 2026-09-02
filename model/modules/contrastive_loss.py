"""Video-text contrastive loss used by InstrAct.

Code attribution:
  Adapted from the VTC implementation in OpenGVLab/InternVideo:
  https://github.com/OpenGVLab/InternVideo

  Its soft-target momentum-distillation formulation follows ALBEF,
  "Align before Fuse: Vision and Language Representation Learning with
  Momentum Distillation" (Li et al., NeurIPS 2021):
  https://proceedings.neurips.cc/paper/2021/hash/505259756244493872b7709a8a01b536-Abstract.html

InstrAct retains only the bidirectional VTC portion, adds the action-token
teacher target used by verb-guided distillation, and provides a compact
distributed all-gather implementation.
"""

from functools import lru_cache
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


def _world_size():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


class _AllGather(torch.autograd.Function):
    """All-gather equal local batches while preserving local gradients."""

    @staticmethod
    def forward(ctx, tensor, args):
        tensor = tensor.contiguous()
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        start = ctx.batch_size * ctx.rank
        return grad_output[start:start + ctx.batch_size].contiguous(), None


all_gather_with_grad = _AllGather.apply


def _similarity(vision_features, text_features, temperature):
    vision_features = F.normalize(vision_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    similarity = vision_features @ text_features.T / temperature
    return similarity, similarity.T


class VTC_VTM_Loss(nn.Module):
    """Compatibility name for the VTC loss used by ``ViCLIP``.

    The release model only calls :meth:`vtc_loss`; the unused VTM and MLM
    implementations from the training repository are intentionally omitted.
    """

    def __init__(self, vtm_hard_neg=False):
        super().__init__()
        # Retained only for constructor compatibility with the original model.
        self.vtm_hard_neg = vtm_hard_neg

    def vtc_loss(
        self,
        vision_proj,
        text_proj,
        idx,
        temp=1.0,
        all_gather=True,
        agg_method="mean",
        alpha=0,
        vision_distill=None,
    ):
        del agg_method  # Release ViCLIP passes pooled (2-D) features.

        if all_gather and _world_size() > 1:
            args = self._gather_args()
            vision_proj = all_gather_with_grad(vision_proj, args)
            text_proj = all_gather_with_grad(text_proj, args)
            if vision_distill is not None:
                vision_distill = all_gather_with_grad(vision_distill, args)
            if idx is not None:
                idx = all_gather_with_grad(idx, args)

        sim_v2t, sim_t2v = _similarity(vision_proj, text_proj, temp)
        with torch.no_grad():
            targets_v2t = self._targets(sim_v2t, idx)
            targets_t2v = targets_v2t

        if vision_distill is not None:
            teacher_v2t, teacher_t2v = _similarity(
                vision_distill, text_proj, temp
            )
            targets_v2t = (
                alpha * F.softmax(teacher_v2t, dim=1)
                + (1 - alpha) * targets_v2t
            )
            targets_t2v = (
                alpha * F.softmax(teacher_t2v, dim=1)
                + (1 - alpha) * targets_t2v
            )

        loss_v2t = -(F.log_softmax(sim_v2t, dim=1) * targets_v2t).sum(1).mean()
        loss_t2v = -(F.log_softmax(sim_t2v, dim=1) * targets_t2v).sum(1).mean()
        return 0.5 * (loss_v2t + loss_t2v)

    @staticmethod
    @torch.no_grad()
    def _targets(similarity, idx=None):
        if idx is None:
            targets = torch.eye(
                similarity.shape[0],
                device=similarity.device,
                dtype=similarity.dtype,
            )
        else:
            idx = idx.reshape(-1, 1)
            targets = idx.eq(idx.T).to(similarity.dtype)
            targets = targets / targets.sum(1, keepdim=True)
        return targets

    @staticmethod
    @lru_cache(maxsize=16)
    def _gather_args():
        return SimpleNamespace(world_size=_world_size(), rank=_rank())
