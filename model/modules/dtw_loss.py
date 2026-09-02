"""Differentiable DTW alignment and order-contrastive loss for InstrAct.

Code attribution:
  The smooth-DTW dynamic-programming structure is adapted from the official
  implementation of "Representation Learning via Global Temporal Alignment
  and Cycle-Consistency" (Hadji, Derpanis, and Jepson, CVPR 2021):
  https://github.com/hadjisma/VideoAlignment

  The underlying differentiable soft-min DTW formulation was introduced
  earlier by "Soft-DTW: a Differentiable Loss Function for Time-Series"
  (Cuturi and Blondel, ICML 2017):
  https://proceedings.mlr.press/v70/cuturi17a.html

InstrAct specializes the method to video action tokens and ordered verb
phrases, and adds the forward-versus-reversed hinge regularizer described in
the InstrAct paper.
"""

import torch
import torch.nn.functional as F


def _soft_min(values, gamma):
    if gamma == 0:
        return values.min()
    logits = -values / gamma
    maximum = logits.max()
    return -gamma * (maximum + torch.logsumexp(logits - maximum, dim=-1))


def _smooth_dtw(
    video_steps,
    text_steps,
    softening="dtw_prob",
    gamma_s=0.1,
    gamma_f=0.1,
):
    video_steps = F.normalize(video_steps, dim=-1)
    text_steps = F.normalize(text_steps, dim=-1)
    similarity = video_steps @ text_steps.T
    distance = 1 - similarity
    rows, columns = distance.shape
    table = torch.zeros(
        rows + 1, columns + 1,
        dtype=torch.float32,
        device=video_steps.device,
    )

    for row in range(rows + 1):
        for column in range(columns + 1):
            if row == 0 and column == 0:
                value = table.new_tensor(0)
            elif row == 0 or column == 0:
                value = table.new_tensor(torch.finfo(torch.float32).max)
            else:
                neighbors = torch.stack((
                    table[row, column - 1],
                    table[row - 1, column - 1],
                    table[row - 1, column],
                ))
                if softening == "dtw_minGamma":
                    previous = _soft_min(neighbors, gamma_s)
                elif softening == "dtw_prob":
                    probabilities = F.softmax(-neighbors / gamma_s, dim=-1)
                    previous = (probabilities * neighbors).sum()
                elif softening == "non-diff":
                    previous = neighbors.min()
                else:
                    raise ValueError(f"Unsupported DTW softening: {softening}")
                value = distance[row - 1, column - 1] + previous

            table = table.clone()
            table[row, column] = value

    return table[-1, -1]


def compute_dtw_loss(
    embs_v,
    embs_t,
    pos_indices,
    dtw_beta,
    dtw_ratio,
    dtw_scale_factor,
    alignment_type="dtw_contrastive",
    similarity_type="cosine",
    label_smoothing=0.1,
    softning="dtw_prob",
    gamma_s=0.1,
    gamma_f=0.1,
    cyclic_action=False,
):
    """Align video actions with ordered verb phrases.

    The release model uses only ``dtw_contrastive``. Extra arguments are kept
    for call compatibility with the original training configuration.
    """
    del label_smoothing, cyclic_action
    if alignment_type != "dtw_contrastive":
        raise ValueError("Release code supports only dtw_contrastive")
    if similarity_type != "cosine":
        raise ValueError("Release code supports only cosine similarity")

    alignment_costs = []
    order_losses = []
    for video_steps, text_steps, indices in zip(embs_v, embs_t, pos_indices):
        indices = indices[indices != -1]
        if indices.numel() == 0:
            continue
        text_steps = text_steps[indices]

        forward_cost = dtw_scale_factor * _smooth_dtw(
            video_steps, text_steps, softning, gamma_s, gamma_f
        )
        reverse_cost = dtw_scale_factor * _smooth_dtw(
            video_steps.flip(0), text_steps, softning, gamma_s, gamma_f
        )
        alignment_costs.append(forward_cost)
        order_losses.append(
            torch.maximum(
                forward_cost - reverse_cost + dtw_beta,
                torch.zeros_like(forward_cost),
            )
        )

    if not alignment_costs:
        connected_zero = embs_v.sum() * 0.0 + embs_t.sum() * 0.0
        return connected_zero, connected_zero

    alignment_loss = torch.stack(alignment_costs).mean()
    order_loss = torch.stack(order_losses).mean()
    return alignment_loss + dtw_ratio * order_loss, dtw_ratio * order_loss
