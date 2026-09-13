"""M-abs: absolute-color conditional set denoiser (handoff §4.5).

Small pre-norm transformer encoder over nodes. Per-node input = concat[normalized (L,a,b), fixed flag,
zero-noise flag, sinusoidal time embedding, broadcast n, m, optional rating token]. No positional
embeddings, no slot identifiers: the network is permutation-equivariant over valid nodes.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

Tensor = torch.Tensor


def timestep_embedding(t: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim); self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim))

    def forward(self, h: Tensor, key_padding_mask: Tensor) -> Tensor:
        x = self.n1(h)
        a, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        h = h + a
        return h + self.mlp(self.n2(h))


class AbsoluteDenoiser(nn.Module):
    def __init__(self, dim: int = 128, depth: int = 4, heads: int = 4, time_dim: int = 64,
                 rating_cond: bool = False, n_rating_bins: int = 8, edge_features: bool = False):
        super().__init__()
        self.rating_cond, self.n_rating_bins = rating_cond, n_rating_bins
        self.time_dim = time_dim
        in_dim = 3 + 2 + time_dim + 2 + (dim if rating_cond else 0)
        self.inp = nn.Sequential(nn.Linear(in_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        if rating_cond:
            self.rating_emb = nn.Embedding(n_rating_bins + 1, dim)  # last index = null token
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, 3)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def _rating_token(self, rating: Tensor | None, B: int, device) -> Tensor:
        if rating is None:
            idx = torch.full((B,), self.n_rating_bins, device=device, dtype=torch.long)
        else:
            r = rating.to(device)
            bins = ((r.clamp(1, 5) - 1) / 4 * (self.n_rating_bins - 1)).round().long()
            idx = torch.where(torch.isfinite(r), bins, torch.full_like(bins, self.n_rating_bins))
        return self.rating_emb(idx)

    def forward(self, x: Tensor, is_fixed: Tensor, mask: Tensor, t: Tensor, cond: dict | None = None) -> Tensor:
        B, N, _ = x.shape
        temb = timestep_embedding(t, self.time_dim)[:, None].expand(B, N, -1)
        n_ctx = is_fixed.sum(1, keepdim=True).float() / 5.0
        m_tgt = (mask & ~is_fixed).sum(1, keepdim=True).float() / 5.0
        feats = [x, is_fixed[..., None].float(), is_fixed[..., None].float(), temb,
                 n_ctx[..., None].expand(B, N, 1), m_tgt[..., None].expand(B, N, 1)]
        if self.rating_cond:
            feats.append(self._rating_token(None if cond is None else cond.get("rating"), B, x.device)[:, None].expand(B, N, -1))
        h = self.inp(torch.cat(feats, -1))
        kpm = ~mask
        for blk in self.blocks:
            h = blk(h, kpm)
        return self.out(self.out_norm(h))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
