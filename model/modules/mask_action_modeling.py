"""Multimodal decoder for masked action modeling.

Code attribution:
  Adapted from the open-source CoCa PyTorch implementation:
  https://github.com/lucidrains/CoCa-pytorch

  That implementation follows "Contrastive Captioners are Image-Text
  Foundation Models" (Yu et al., TMLR 2022):
  https://research.google/pubs/coca-contrastive-captioners-are-image-text-foundation-models/

InstrAct keeps the causal text-decoder and visual cross-attention pattern, but
uses it to reconstruct masked action-word BPE tokens from text context and
Action Perceiver tokens rather than to train a general image captioner.
"""

import torch
from torch import einsum, nn
import torch.nn.functional as F

from einops import rearrange


def exists(val):
    return val is not None


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


# SwiGLU follows Shazeer, "GLU Variants Improve Transformer" (2020):
# https://arxiv.org/abs/2002.05202


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


# Parallel attention/feed-forward layout follows the CoCa decoder design.


class ParallelTransformerBlock(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.norm = LayerNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, (ff_inner_dim * 2))

        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)

        self.ff_out = nn.Sequential(
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        )

        # Cache causal masks and rotary embeddings by sequence length.
        self.mask = None
        self.pos_emb = None

    def get_causal_mask(self, n, device):
        if self.mask is None or self.mask.shape[-1] < n:
            self.mask = torch.ones(
                n, n, device=device, dtype=torch.bool
            ).triu(1)
        return self.mask[:n, :n].to(device)

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n].to(device)

        pos_emb = self.rotary_emb(n, device=device)
        self.pos_emb = pos_emb
        return pos_emb

    def forward(self, x, attn_mask=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device, h = x.shape[1], x.device, self.heads

        # pre layernorm

        x = self.norm(x)

        # attention queries, keys, values, and feedforward inner

        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)

        # split heads
        # Multi-query attention follows Shazeer, "Fast Transformer Decoding:
        # One Write-Head is All You Need" (2019):
        # https://arxiv.org/abs/1911.02150

        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        # rotary embeddings

        positions = self.get_rotary_embedding(n, device)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # scale

        q = q * self.scale

        # similarity

        sim = einsum("b h i d, b j d -> b h i j", q, k)

        # InstrAct uses causal self-attention for autoregressive MAM.
        sim = sim.masked_fill(
            self.get_causal_mask(n, device),
            -torch.finfo(sim.dtype).max,
        )

        if exists(attn_mask):
            attn_mask = rearrange(attn_mask, 'b i j -> b 1 i j')
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        # attention

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.attn_out(out) + self.ff_out(ff)

# cross attention - using multi-query + one-headed key / values as in PaLM w/ optional parallel feedforward

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim,
        dim_head=64,
        heads=8,
        ff_mult=4
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        self.norm = LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether to have parallel feedforward

        ff_inner_dim = ff_mult * dim

        self.ff = nn.Sequential(
            nn.Linear(dim, ff_inner_dim * 2, bias=False),
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        )

    def forward(self, x, context):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        # pre-layernorm, for queries and context

        x = self.norm(x)

        # get queries

        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        # scale

        q = q * self.scale

        # get key / values

        k, v = self.to_kv(context).chunk(2, dim=-1)

        # query / key similarity

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        # attention

        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = sim.softmax(dim=-1)

        # aggregate

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out + self.ff(x)

class MultimodalTextDecoder(nn.Module):
    """
    Multimodal decoder for masked action modeling (MAM).

    Inputs:
      - text_embeds: token features of the original text, shape [B, N, Dt]
      - image_tokens: external image token sequence (e.g., from ViT/CLIP), shape [B, M, Di]
      - input_ids: token ids of the original, unmasked text, shape [B, N]
      - action_mask: masked action positions, shape [B, N]

    Outputs:
      - If return_loss == False: vocabulary logits with shape [B, N, V]
      - If return_loss == True: cross-entropy over masked action positions

    A learned embedding replaces every BPE piece of one action word. Causal
    textual self-attention followed by cross-attention to visual Action Tokens
    reconstructs the original token IDs at those same positions.
    """
    def __init__(
        self,
        *,
        dim,                    # decoder hidden size D
        num_tokens,             # vocab size for the output projection
        multimodal_depth=12,    # number of multimodal layers
        context_dim=None,       # image token dim; if != dim, will be linearly projected
        dim_head=64,
        heads=8,
        ff_mult=4,
        pad_id=0,
        mask_embedding_std=0.02,
    ):
        super().__init__()
        self.dim = dim
        self.pad_id = pad_id
        self.mask_action_embedding = nn.Parameter(torch.empty(dim))
        nn.init.normal_(
            self.mask_action_embedding,
            std=float(mask_embedding_std),
        )

        # Lazy input projections:
        # if text_embeds / image_tokens don't match dim/context_dim, create linear adapters on first forward
        self.text_in_proj = None   # lazily initialized based on first input
        self.img_in_proj  = None
        self.expected_context_dim = context_dim if context_dim is not None else dim

        # Multimodal stack with causal text self-attention.
        self.multimodal_layers = nn.ModuleList([
            nn.ModuleList([
                ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult),
                CrossAttention(dim=dim, context_dim=self.expected_context_dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult)
            ]) for _ in range(multimodal_depth)
        ])

        # Final LM head: layer norm then linear to vocab
        self.to_logits = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, num_tokens, bias=False)
        )

    def _maybe_project_inputs(self, text_embeds, image_tokens):
        """
        Dynamically initialize and apply input projections so that:
          text_embeds -> [B, N, dim]
          image_tokens -> [B, M, context_dim]
        """
        dt = text_embeds.shape[-1]
        di = image_tokens.shape[-1]

        # text projection (Dt -> dim) if needed
        if dt != self.dim:
            if self.text_in_proj is None:
                self.text_in_proj = nn.Linear(dt, self.dim, bias=False).to(text_embeds.device)
            text_embeds = self.text_in_proj(text_embeds)

        # image projection (Di -> context_dim) if needed
        target_c = self.expected_context_dim
        if di != target_c:
            if self.img_in_proj is None:
                self.img_in_proj = nn.Linear(di, target_c, bias=False).to(image_tokens.device)
            image_tokens = self.img_in_proj(image_tokens)

        return text_embeds, image_tokens

    @torch.no_grad()
    def _build_key_padding_mask(self, input_ids):
        """
        Build a key padding mask from input_ids:
          - True  = visible token
          - False = padding token (== pad_id)
        Returns shape [B, N, N] for masking keys in self-attention.
        Each query position shares the same key visibility row.
        """
        valid = (input_ids != self.pad_id)  # [B, N]
        N = valid.shape[1]
        attn_mask = valid[:, None, :].expand(-1, N, -1)  # [B, N, N]
        return attn_mask

    def forward(
        self,
        *,
        text_embeds,        # [B, N, Dt] external text latent sequence (e.g., CLIP token features)
        image_tokens,       # [B, M, Di] external image token sequence (e.g., ViT patches)
        input_ids=None,     # [B, N] original, unmasked token ids
        action_mask=None,   # [B, N] True at the selected action word's BPE pieces
        return_loss=True
    ):
        """
        Training:
          - all inputs are position-aligned
          - action positions are replaced with a learned mask embedding
          - only action positions contribute to the reconstruction loss

        Inference (forward only):
          - Pass your prepared text_embeds; returns per-step logits (no sampling here).
        """
        # Ensure dims match decoder/context; create adapters on first call if needed
        text_embeds, image_tokens = self._maybe_project_inputs(text_embeds, image_tokens)

        if text_embeds.shape[0] != image_tokens.shape[0]:
            raise ValueError(
                "text_embeds and image_tokens must have the same batch size, "
                f"got {text_embeds.shape[0]} and {image_tokens.shape[0]}"
            )

        if return_loss:
            if not exists(input_ids):
                raise ValueError("input_ids are required when return_loss=True")
            if not exists(action_mask):
                raise ValueError("action_mask is required when return_loss=True")

        if exists(input_ids):
            if input_ids.shape != text_embeds.shape[:2]:
                raise ValueError(
                    "input_ids must be position-aligned with text_embeds: "
                    f"got {tuple(input_ids.shape)} and {tuple(text_embeds.shape[:2])}"
                )
        if exists(action_mask):
            if action_mask.shape != text_embeds.shape[:2]:
                raise ValueError(
                    "action_mask must be position-aligned with text_embeds: "
                    f"got {tuple(action_mask.shape)} and "
                    f"{tuple(text_embeds.shape[:2])}"
                )
            action_mask = action_mask.to(
                device=text_embeds.device,
                dtype=torch.bool,
            )
            text_embeds = torch.where(
                action_mask.unsqueeze(-1),
                self.mask_action_embedding.to(text_embeds.dtype),
                text_embeds,
            )

        text_in = text_embeds
        attn_mask = (
            self._build_key_padding_mask(input_ids)
            if exists(input_ids) else None
        )

        # Pass through multimodal layers
        x = text_in
        for self_attn, cross_attn in self.multimodal_layers:
            x = x + self_attn(x, attn_mask=attn_mask)
            x = x + cross_attn(x, image_tokens)

        # Project to vocabulary logits
        logits = self.to_logits(x)                  # [B, N, V]

        if not return_loss:
            return logits

        valid_action_mask = action_mask & input_ids.to(
            device=logits.device
        ).ne(self.pad_id)
        if not valid_action_mask.any():
            # Keep the zero connected to the decoder graph for batches whose
            # annotated action was truncated or unavailable.
            return logits.sum() * 0.0

        return F.cross_entropy(
            logits[valid_action_mask],
            input_ids.to(device=logits.device, dtype=torch.long)[
                valid_action_mask
            ],
        )
