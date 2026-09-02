"""Retrieval metrics for per-video hard-negative candidate groups."""

import torch


def grouped_retrieval_metrics(video_features, text_features, valid_mask):
    """Rank the positive at column zero within each video's candidate group."""
    if video_features.ndim != 2 or text_features.ndim != 3:
        raise ValueError("expected video [B,D] and text [B,C,D]")
    if text_features.shape[:2] != valid_mask.shape:
        raise ValueError("valid_mask must have shape [B,C]")
    if not valid_mask[:, 0].all():
        raise ValueError("the positive candidate at column zero must be valid")

    video_features = torch.nn.functional.normalize(video_features.float(), dim=-1)
    text_features = torch.nn.functional.normalize(text_features.float(), dim=-1)
    similarities = torch.einsum("bd,bcd->bc", video_features, text_features)
    similarities = similarities.masked_fill(~valid_mask.bool(), float("-inf"))
    ranks = similarities.argsort(dim=1, descending=True).eq(0).float().argmax(dim=1)
    candidate_counts = valid_mask.sum(dim=1)

    return {
        "R1": ranks.eq(0).float().mean().item(),
        "R5": ranks.lt(torch.minimum(candidate_counts, candidate_counts.new_full((), 5)))
        .float()
        .mean()
        .item(),
        "MR": torch.quantile(ranks.add(1).float(), 0.5).item(),
    }
