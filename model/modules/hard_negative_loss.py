"""Action-centric hard-negative NCE loss for InstrAct.

Method attribution:
  This is the release implementation of InstrAct's masked, bidirectional
  weighted HardNeg-NCE objective. The use of verb-altered captions and
  hardness-weighted NCE follows the setup studied in "Verbs in Action:
  Improving Verb Understanding in Video-Language Models" (Momeni et al.,
  ICCV 2023):
  https://openaccess.thecvf.com/content/ICCV2023/html/Momeni_Verbs_in_Action_Improving_Verb_Understanding_in_Video-Language_Models_ICCV_2023_paper.html

InstrAct extends that setup to multiple action-centric and order-swapped
caption negatives per instructional video and masks padded/foreign groups.
"""

import torch
import torch.nn.functional as F


_LARGE_NEG = -1_000_000.0


@torch.no_grad()
def _contrastive_targets(logits, text_mask, exclude_other_video_hn=True):
    batch_size, num_texts = logits.shape
    if text_mask.shape[0] % batch_size:
        raise ValueError("text_mask rows must be divisible by the video batch size")

    candidates = text_mask.shape[0] // batch_size
    labels = torch.zeros_like(logits)
    rows = torch.arange(batch_size, device=logits.device)
    labels[rows, rows * candidates] = 1

    default_mask = text_mask.T.repeat(batch_size, 1).float() * _LARGE_NEG
    if not exclude_other_video_hn:
        return labels, default_mask, torch.ones_like(default_mask)

    local_mask = text_mask.reshape(batch_size, candidates).to(logits.dtype)
    blocked = torch.ones(
        batch_size, batch_size, candidates,
        device=logits.device,
        dtype=logits.dtype,
    )
    blocked[:, :, 0] = 0
    blocked[rows, rows] = local_mask
    blocked = blocked.reshape(batch_size, num_texts)
    return labels, blocked.float() * _LARGE_NEG, 1 - blocked


def _weighted_nce(logits, labels, count=None, beta=0.0, masking=None):
    logits = logits.float()
    labels = labels.float()
    scaled = beta * logits
    weights = torch.exp(scaled - scaled.max(dim=-1, keepdim=True).values)
    weights = (1 - labels) * weights

    if masking is not None:
        weights = weights * masking.ne(_LARGE_NEG).to(weights.dtype)

    if count is None:
        count = torch.full(
            (logits.size(0), 1),
            float(logits.size(1)),
            device=logits.device,
            dtype=weights.dtype,
        )

    weights = (count - 1) * weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    weights = weights + labels
    exp_logits = torch.exp(logits - logits.max(dim=-1, keepdim=True).values)
    normalizer = (weights * exp_logits).sum(-1, keepdim=True).clamp_min(1e-12)
    return -(labels * torch.log(exp_logits / normalizer)).sum(-1)


def verb_hard_neg_nce_torch(
    encoded_video,
    encoded_text,
    mask_text,
    temperature=0.05,
    v2t_weight=1.0,
    t2v_weight=1.0,
    beta=0.0,
    loss_exclude_hn_other_vid=True,
):
    """Compute video/text NCE with each video's verb hard negatives."""
    encoded_video = F.normalize(encoded_video, dim=-1)
    encoded_text = F.normalize(encoded_text, dim=-1)
    logits = encoded_video.float() @ encoded_text.float().T
    labels, masking, inverse = _contrastive_targets(
        logits, mask_text, loss_exclude_hn_other_vid
    )
    logits = logits / float(temperature)

    count = inverse.sum(-1, keepdim=True).clamp_min(1)
    v2t_rows = _weighted_nce(logits, labels, count, beta, masking)
    # A row containing only its positive has no ranking problem.  Skip it
    # instead of dividing by log(1), which would otherwise produce NaN.
    valid_v2t = count.squeeze(-1) > 1
    if valid_v2t.any():
        v2t_loss = (
            v2t_rows[valid_v2t]
            / torch.log(count.squeeze(-1)[valid_v2t])
        ).mean()
    else:
        v2t_loss = logits.sum() * 0.0

    batch_size = logits.shape[0]
    if batch_size > 1:
        t2v_rows = _weighted_nce(logits.T, labels.T, beta=beta)
        log_batch = torch.log(
            torch.tensor(float(batch_size), device=logits.device)
        )
        t2v_loss = (t2v_rows.sum() / batch_size) / log_batch
    else:
        # With one local video there is no text-to-video ranking problem and
        # the original log(batch_size) normalization would divide by zero.
        t2v_loss = logits.sum() * 0.0
    return v2t_weight * v2t_loss + t2v_weight * t2v_loss
