"""M-combined: ε̂ = ε̂_geom + ε̂_abs with strictly separate pathways (handoff §1.6)."""
from __future__ import annotations

import torch
import torch.nn as nn

from palette.models.absolute import AbsoluteDenoiser
from palette.models.geometric import GeometricDenoiser


class CombinedDenoiser(nn.Module):
    def __init__(self, abs_kw: dict | None = None, geom_kw: dict | None = None, rating_cond: bool = False):
        super().__init__()
        self.abs = AbsoluteDenoiser(rating_cond=rating_cond, **(abs_kw or {}))
        self.geom = GeometricDenoiser(rating_cond=rating_cond, **(geom_kw or {}))
        self.last_stats: dict = {}

    def forward(self, x, is_fixed, mask, t, cond=None):
        eg = self.geom(x, is_fixed, mask, t, cond)
        ea = self.abs(x, is_fixed, mask, t, cond)
        with torch.no_grad():
            tgt = (mask & ~is_fixed)[..., None]
            na = (ea * tgt).norm(); ng = (eg * tgt).norm()
            self.last_stats = {"abs_over_geom_norm": float(na / (ng + 1e-8))}
        return eg + ea
