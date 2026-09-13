"""M-geomlite (R1 amendment): the absolute-color denoiser plus pairwise rotation-invariant edge features
e_ij (handoff §1.5) injected as additive attention biases. No equivariance constraint on the network."""
from __future__ import annotations

import torch
import torch.nn as nn

from palette.models.absolute import AbsoluteDenoiser, timestep_embedding

Tensor = torch.Tensor


def pairwise_invariants(x: Tensor, is_fixed: Tensor) -> Tensor:
    """e_ij = [L_i, L_j, L_i-L_j, |v_i|², |v_j|², v_i·v_j, det(v_i,v_j), f_i, f_j]  → (B, N, N, 9)."""
    L = x[..., 0]; v = x[..., 1:]
    Li, Lj = L[:, :, None], L[:, None, :]
    ni = (v**2).sum(-1)
    dot = torch.einsum("bic,bjc->bij", v, v)
    det = v[:, :, None, 0] * v[:, None, :, 1] - v[:, :, None, 1] * v[:, None, :, 0]
    f = is_fixed.float()
    return torch.stack([Li.expand_as(dot), Lj.expand_as(dot), Li - Lj, ni[:, :, None].expand_as(dot),
                        ni[:, None, :].expand_as(dot), dot, det, f[:, :, None].expand_as(dot), f[:, None, :].expand_as(dot)], -1)


class EdgeBiasBlock(nn.Module):
    """Pre-norm block whose attention logits get a learned per-head bias from edge features."""

    def __init__(self, dim: int, heads: int, edge_dim: int, mlp_ratio: int = 4):
        super().__init__()
        self.heads, self.dh = heads, dim // heads
        self.n1 = nn.LayerNorm(dim); self.qkv = nn.Linear(dim, 3 * dim); self.proj = nn.Linear(dim, dim)
        self.edge = nn.Sequential(nn.Linear(edge_dim, 64), nn.SiLU(), nn.Linear(64, heads))
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim))

    def forward(self, h: Tensor, mask: Tensor, e: Tensor) -> Tensor:
        B, N, D = h.shape
        q, k, v = self.qkv(self.n1(h)).view(B, N, 3, self.heads, self.dh).unbind(2)
        logits = torch.einsum("bihd,bjhd->bhij", q, k) / self.dh**0.5 + self.edge(e).permute(0, 3, 1, 2)
        logits = logits.masked_fill(~mask[:, None, None, :], float("-inf"))
        a = torch.softmax(logits, -1)
        o = torch.einsum("bhij,bjhd->bihd", a, v).reshape(B, N, D)
        h = h + self.proj(o)
        return h + self.mlp(self.n2(h))


class GeomLiteDenoiser(AbsoluteDenoiser):
    def __init__(self, dim: int = 128, depth: int = 4, heads: int = 4, time_dim: int = 64, rating_cond: bool = False,
                 n_rating_bins: int = 8, edge_time: bool = True):
        super().__init__(dim, depth, heads, time_dim, rating_cond, n_rating_bins)
        self.edge_time = edge_time
        edge_dim = 9 + (time_dim if edge_time else 0)
        self.blocks = nn.ModuleList([EdgeBiasBlock(dim, heads, edge_dim) for _ in range(depth)])

    def forward(self, x: Tensor, is_fixed: Tensor, mask: Tensor, t: Tensor, cond: dict | None = None) -> Tensor:
        B, N, _ = x.shape
        temb = timestep_embedding(t, self.time_dim)
        n_ctx = is_fixed.sum(1, keepdim=True).float() / 5.0
        m_tgt = (mask & ~is_fixed).sum(1, keepdim=True).float() / 5.0
        feats = [x, is_fixed[..., None].float(), is_fixed[..., None].float(), temb[:, None].expand(B, N, -1),
                 n_ctx[..., None].expand(B, N, 1), m_tgt[..., None].expand(B, N, 1)]
        if self.rating_cond:
            feats.append(self._rating_token(None if cond is None else cond.get("rating"), B, x.device)[:, None].expand(B, N, -1))
        h = self.inp(torch.cat(feats, -1))
        e = pairwise_invariants(x, is_fixed)
        if self.edge_time:
            e = torch.cat([e, temb[:, None, None].expand(B, N, N, -1)], -1)
        for blk in self.blocks:
            h = blk(h, mask, e)
        return self.out(self.out_norm(h))
