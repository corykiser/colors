"""Preference scorers on MTurk ratings (handoff §1.9, §4.4) and the R3 ordered-vs-unordered diagnostic.

Variants (feature construction differs; the head is the same small MLP):
  deepsets   permutation-invariant: sum-pool of a per-color MLP
  sortedL    colors sorted by L then concatenated (order-free but not set-symmetric)
  ordered    stored (presentation) order concatenated + adjacent differences  ← R3 diagnostic
"""
from __future__ import annotations

import torch
import torch.nn as nn

Tensor = torch.Tensor


class Scorer(nn.Module):
    def __init__(self, variant: str = "deepsets", hidden: int = 128, n: int = 5):
        super().__init__()
        self.variant, self.n = variant, n
        if variant == "deepsets":
            self.phi = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
            self.rho = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        elif variant in ("sortedL", "ordered"):
            d = n * 3 + ((n - 1) * 3 if variant == "ordered" else 0)
            self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        else:
            raise ValueError(variant)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """x: (B, n, 3) normalized Oklab in *stored order*. Returns (B,) score."""
        if self.variant == "deepsets":
            h = self.phi(x)
            if mask is not None:
                h = h * mask[..., None]
            return self.rho(h.sum(1)).squeeze(-1)
        if self.variant == "sortedL":
            idx = torch.argsort(x[..., 0], dim=1)
            x = torch.gather(x, 1, idx[..., None].expand_as(x))
            return self.mlp(x.flatten(1)).squeeze(-1)
        diffs = x[:, 1:] - x[:, :-1]
        return self.mlp(torch.cat([x.flatten(1), diffs.flatten(1)], 1)).squeeze(-1)


def pairwise_loss(s_a: Tensor, s_b: Tensor, a_pref: Tensor) -> Tensor:
    """-log σ(s_A - s_B) on pairs where A is preferred (a_pref=1) or B (a_pref=0)."""
    sign = a_pref.float() * 2 - 1
    return nn.functional.softplus(-sign * (s_a - s_b)).mean()
