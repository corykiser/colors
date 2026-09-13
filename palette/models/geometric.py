"""M-geom: exact SO(2)-equivariant geometric pathway (handoff §1.5).

Scalar channels s_i (invariant) and vector channels V_i ∈ R^{dv×2} (each rotates with the chromatic plane).
Pairwise messages are built from invariants only; vector updates are linear combinations of
V_i, V_j, J V_i, J V_j with invariant coefficients, followed by channel mixing and norm-only gating.
No componentwise nonlinearities on vectors, no directional biases, no absolute-color features.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from palette.models.absolute import timestep_embedding

Tensor = torch.Tensor


def J(v: Tensor) -> Tensor:
    """90° rotation of the last axis (…, 2): (a, b) -> (-b, a)."""
    return torch.stack([-v[..., 1], v[..., 0]], -1)


class GeomBlock(nn.Module):
    def __init__(self, ds: int, dv: int, hidden: int = 128):
        super().__init__()
        self.ds, self.dv = ds, dv
        edge_in = 2 * ds + 3 * dv          # s_i, s_j, per-channel |Vi|², Vi·Vj, det(Vi,Vj)
        self.edge = nn.Sequential(nn.Linear(edge_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.to_logit = nn.Linear(hidden, 1)
        self.to_smsg = nn.Linear(hidden, ds)
        self.to_coef = nn.Linear(hidden, 4 * dv)
        self.s_update = nn.Sequential(nn.Linear(2 * ds, hidden), nn.SiLU(), nn.Linear(hidden, ds))
        self.v_mix = nn.Linear(dv, dv, bias=False)
        self.v_gate = nn.Sequential(nn.Linear(ds + dv, hidden), nn.SiLU(), nn.Linear(hidden, dv))
        self.s_norm = nn.LayerNorm(ds)
        nn.init.zeros_(self.to_coef.weight); nn.init.zeros_(self.to_coef.bias)

    def forward(self, s: Tensor, V: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        B, N, _ = s.shape
        n2 = (V**2).sum(-1)                                                  # (B,N,dv)
        dot = torch.einsum("bicx,bjcx->bijc", V, V)                          # (B,N,N,dv)
        det = V[:, :, None, :, 0] * V[:, None, :, :, 1] - V[:, :, None, :, 1] * V[:, None, :, :, 0]
        e = torch.cat([s[:, :, None].expand(B, N, N, -1), s[:, None, :].expand(B, N, N, -1),
                       n2[:, :, None].expand(B, N, N, -1), dot, det], -1)
        h = self.edge(e)
        logit = self.to_logit(h).squeeze(-1).masked_fill(~mask[:, None, :], float("-inf"))
        a = torch.softmax(logit, -1)                                          # (B,N,N) invariant weights
        smsg = torch.einsum("bij,bijd->bid", a, self.to_smsg(h))
        s = s + self.s_update(torch.cat([self.s_norm(s), smsg], -1))
        al, be, ga, de = self.to_coef(h).chunk(4, -1)                        # (B,N,N,dv) each
        Vi = V[:, :, None]; Vj = V[:, None, :]                               # (B,N,1,dv,2), (B,1,N,dv,2)
        msg = al[..., None] * Vi + be[..., None] * Vj + ga[..., None] * J(Vi) + de[..., None] * J(Vj)
        agg = torch.einsum("bij,bijcx->bicx", a, msg)
        V = V + self.v_mix(agg.transpose(-1, -2)).transpose(-1, -2)          # channel mixing (equivariant)
        norms = (V**2).sum(-1).clamp(min=1e-8).sqrt()                        # (B,N,dv), invariant
        gate = torch.sigmoid(self.v_gate(torch.cat([s, norms], -1)))
        V = V * gate[..., None]
        rms = norms.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-3)     # norm-only normalization
        V = V / rms[..., None]
        return s, V


class GeometricDenoiser(nn.Module):
    def __init__(self, ds: int = 128, dv: int = 16, depth: int = 4, hidden: int = 128, time_dim: int = 64,
                 rating_cond: bool = False, n_rating_bins: int = 8):
        super().__init__()
        self.ds, self.dv, self.time_dim = ds, dv, time_dim
        self.rating_cond, self.n_rating_bins = rating_cond, n_rating_bins
        in_dim = 1 + 1 + 2 + time_dim + 2 + (ds if rating_cond else 0)     # L, |v|², flags, t, n, m
        self.s_in = nn.Sequential(nn.Linear(in_dim, ds), nn.SiLU(), nn.Linear(ds, ds))
        if rating_cond:
            self.rating_emb = nn.Embedding(n_rating_bins + 1, ds)
        self.v_in = nn.Linear(2, dv, bias=False)                             # V_i[c] = w_c v_i + w'_c J v_i
        self.blocks = nn.ModuleList([GeomBlock(ds, dv, hidden) for _ in range(depth)])
        self.out_L = nn.Sequential(nn.LayerNorm(ds), nn.Linear(ds, 1))
        self.out_ab = nn.Linear(dv, 1, bias=False)
        nn.init.zeros_(self.out_L[1].weight); nn.init.zeros_(self.out_L[1].bias); nn.init.zeros_(self.out_ab.weight)

    def _rating_token(self, rating, B, device):
        if rating is None:
            idx = torch.full((B,), self.n_rating_bins, device=device, dtype=torch.long)
        else:
            r = rating.to(device); bins = ((r.clamp(1, 5) - 1) / 4 * (self.n_rating_bins - 1)).round().long()
            idx = torch.where(torch.isfinite(r), bins, torch.full_like(bins, self.n_rating_bins))
        return self.rating_emb(idx)

    def forward(self, x: Tensor, is_fixed: Tensor, mask: Tensor, t: Tensor, cond: dict | None = None) -> Tensor:
        B, N, _ = x.shape
        L = x[..., :1]; v = x[..., 1:]
        temb = timestep_embedding(t, self.time_dim)[:, None].expand(B, N, -1)
        n_ctx = (is_fixed.sum(1, keepdim=True).float() / 5.0)[..., None].expand(B, N, 1)
        m_tgt = ((mask & ~is_fixed).sum(1, keepdim=True).float() / 5.0)[..., None].expand(B, N, 1)
        f = is_fixed[..., None].float()
        feats = [L, (v**2).sum(-1, keepdim=True), f, f, temb, n_ctx, m_tgt]
        if self.rating_cond:
            feats.append(self._rating_token(None if cond is None else cond.get("rating"), B, x.device)[:, None].expand(B, N, -1))
        s = self.s_in(torch.cat(feats, -1))
        w = self.v_in.weight                                                  # (dv, 2): [w_c, w'_c]
        V = w[:, 0][None, None, :, None] * v[:, :, None, :] + w[:, 1][None, None, :, None] * J(v)[:, :, None, :]
        for blk in self.blocks:
            s, V = blk(s, V, mask)
        eps_L = self.out_L(s)
        eps_ab = (self.out_ab.weight[0][None, None, :, None] * V).sum(2)     # (B,N,2)
        return torch.cat([eps_L, eps_ab], -1)
