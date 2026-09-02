"""Action Perceiver used by InstrAct.

Code attribution:
  Adapted from the Knowledge Patcher Perceiver in PAXION, "Patching Action
  Knowledge in Video-Language Foundation Models" (Wang et al., NeurIPS 2023):
  https://github.com/MikeWangWZHL/Paxion

InstrAct changes the PAXION resampler into temporally ordered Action Tokens,
adds local temporal attention windows and temporal embeddings, and permits
verb embeddings to serve as externally supplied teacher queries.
"""

import torch
from torch import einsum, nn
from einops import rearrange, repeat


class FeedForward(nn.Module):
    def __init__(self, dim, mult):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult, bias=False),
            nn.GELU(),
            nn.Linear(dim * mult, dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class PerceiverAttention(nn.Module):
    def __init__(self, dim, context_dim, dim_head, heads):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm_context = nn.LayerNorm(context_dim)
        self.norm_latents = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, context, latents, context_mask=None):
        context = self.norm_context(context)
        latents = self.norm_latents(latents)
        q = rearrange(
            self.to_q(latents), "b q (h d) -> b h q d", h=self.heads
        )
        k, v = self.to_kv(context).chunk(2, dim=-1)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        similarity = einsum("b h q d, b h n d -> b h q n", q * self.scale, k)
        if context_mask is not None:
            similarity = similarity.masked_fill(
                ~context_mask[:, None],
                -torch.finfo(similarity.dtype).max,
            )
        attention = similarity.softmax(dim=-1)
        output = einsum("b h q n, b h n d -> b h q d", attention, v)
        output = rearrange(output, "b h q d -> b q (h d)")
        return self.to_out(output)


class Perceiver(nn.Module):
    """Compress dense video tokens into temporally ordered Action Tokens.

    Learned student queries attend only to their local temporal windows when
    ``local_temporal_attention`` is enabled. Externally supplied verb queries
    remain global and form the teacher stream.
    """

    def __init__(
        self,
        *,
        dim,
        k_v_dim,
        depth,
        dim_head=64,
        heads=8,
        num_latents=8,
        ff_mult=4,
        add_temporal_embedding=True,
        temporal_dropout=0.0,
        local_temporal_attention=True,
        num_frames=None,
        # Backward-compatible names from experimental configs.
        if_add_temporal_emebdding=None,
        temp_emb_drop_out=None,
    ):
        super().__init__()
        if if_add_temporal_emebdding is not None:
            add_temporal_embedding = if_add_temporal_emebdding
        if temp_emb_drop_out is not None:
            temporal_dropout = temp_emb_drop_out

        self.num_latents = num_latents
        self.num_frames = num_frames
        self.local_temporal_attention = local_temporal_attention
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.temporal_embedding = (
            nn.Parameter(torch.randn(num_latents, dim))
            if add_temporal_embedding else None
        )
        self.temporal_dropout = nn.Dropout(temporal_dropout)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PerceiverAttention(dim, k_v_dim, dim_head, heads),
                FeedForward(dim, ff_mult),
            ])
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def _local_mask(self, batch_size, token_count, device):
        if not self.local_temporal_attention or self.num_frames is None:
            return None
        if token_count % self.num_frames:
            raise ValueError(
                f"{token_count} visual tokens cannot be divided into "
                f"{self.num_frames} frames"
            )

        patches_per_frame = token_count // self.num_frames
        frame_ids = torch.arange(self.num_frames, device=device)
        starts = torch.div(
            torch.arange(self.num_latents, device=device) * self.num_frames,
            self.num_latents,
            rounding_mode="floor",
        )
        ends = torch.div(
            (torch.arange(self.num_latents, device=device) + 1) * self.num_frames
            + self.num_latents - 1,
            self.num_latents,
            rounding_mode="floor",
        )
        frame_mask = (
            (frame_ids[None] >= starts[:, None])
            & (frame_ids[None] < ends[:, None])
        )
        token_mask = repeat(
            frame_mask,
            "q f -> q (f p)",
            p=patches_per_frame,
        )
        return token_mask[None].expand(batch_size, -1, -1)

    def forward(self, x, latents=None, attn_mask=None):
        if x.ndim == 4:
            x = rearrange(x, "b f p d -> b (f p) d")
        if x.ndim != 3:
            raise ValueError("visual features must have shape [B,N,D] or [B,F,P,D]")

        learned_queries = latents is None
        if learned_queries:
            latents = repeat(self.latents, "q d -> b q d", b=x.shape[0])
            if self.temporal_embedding is not None:
                latents = latents + self.temporal_dropout(self.temporal_embedding)
            context_mask = self._local_mask(x.shape[0], x.shape[1], x.device)
        else:
            if latents.ndim != 3:
                raise ValueError("teacher verb queries must have shape [B,M,D]")
            context_mask = None

        for attention, feed_forward in self.layers:
            latents = latents + attention(x, latents, context_mask)
            latents = latents + feed_forward(latents)

        latents = self.norm(latents)
        if attn_mask is not None:
            latents = latents * attn_mask.to(latents.dtype).unsqueeze(-1)
        return latents
