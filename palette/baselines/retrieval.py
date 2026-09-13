"""Nearest-palette retrieval completion and a conditional KDE built on the same matching cost.

Cost of a context C (n colors) against a training palette P (5 colors): min over injective
assignments of C into P of the mean Oklab distance. Enumerated (5·4·3·2 ≤ 120) and vectorized.
"""
from __future__ import annotations

import itertools

import numpy as np

from palette.eval.metrics import matching_distance


class PaletteIndex:
    def __init__(self, X: np.ndarray, M: np.ndarray):
        keep = M.sum(1) == 5
        self.X = np.asarray(X[keep], dtype=np.float64)[:, :5]

    def context_cost(self, ctx: np.ndarray) -> np.ndarray:
        """(N,) min-cost injection of the n context colors into each 5-palette; 0 for empty context."""
        n = len(ctx)
        if n == 0:
            return np.zeros(len(self.X))
        D = np.linalg.norm(self.X[:, :, None, :] - ctx[None, None, :, :], axis=-1)  # (N, 5, n)
        best = np.full(len(self.X), np.inf)
        for inj in itertools.permutations(range(5), n):
            best = np.minimum(best, D[:, list(inj), np.arange(n)].sum(1))
        return best / n

    def _remaining(self, j: int, ctx: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
        P = self.X[j]
        n = len(ctx)
        if n == 0:
            used = []
        else:
            D = np.linalg.norm(P[:, None] - ctx[None], axis=-1)
            best, used = np.inf, None
            for inj in itertools.permutations(range(5), n):
                c = D[list(inj), np.arange(n)].sum()
                if c < best:
                    best, used = c, list(inj)
        rest = [i for i in range(5) if i not in used]
        if len(rest) >= m:
            pick = rng.choice(rest, m, replace=False)
        else:  # not enough remaining: pad by resampling with jitter
            pick = np.concatenate([rest, rng.choice(rest, m - len(rest))])
        return P[pick]

    def complete(self, ctx: np.ndarray, m: int, k: int, seed: int = 0, top: int | None = None) -> np.ndarray:
        """Return the remaining colors of the k nearest training palettes (distinct palettes)."""
        rng = np.random.default_rng(seed)
        ctx = np.asarray(ctx, dtype=np.float64).reshape(-1, 3)
        cost = self.context_cost(ctx)
        top = top or k
        order = np.argsort(cost)[:top]
        chosen = rng.choice(order, k, replace=len(order) < k) if top > k else order
        return np.stack([self._remaining(j, ctx, m, rng) for j in chosen])

    def kde_complete(self, ctx: np.ndarray, m: int, k: int, bandwidth: float = 0.03, jitter: float = 0.015,
                     seed: int = 0) -> np.ndarray:
        """Conditional KDE: weight training palettes by exp(-cost/bandwidth), sample palettes, jitter colors."""
        rng = np.random.default_rng(seed)
        ctx = np.asarray(ctx, dtype=np.float64).reshape(-1, 3)
        cost = self.context_cost(ctx)
        w = np.exp(-(cost - cost.min()) / bandwidth); w /= w.sum()
        chosen = rng.choice(len(self.X), k, p=w)
        out = np.stack([self._remaining(j, ctx, m, rng) for j in chosen])
        return out + rng.normal(0, jitter, out.shape)


class PipelinedBaseline:
    """Any (ctx, m, k, seed)->(k,m,3) generator wrapped in the *same* postprocessing as the neural models:
    oversample, gamut + duplicate rejection (chroma-reduction fallback), scorer rerank, greedy diverse top-k.
    Uses palette.api.Completer's selection logic so the comparison is matched on candidate budget."""

    def __init__(self, raw_fn, scorer, oversample: int = 32, dup_threshold: float = 0.03, diversity_weight: float = 1.0):
        self.raw_fn, self.scorer, self.oversample = raw_fn, scorer, oversample
        self.dup_threshold, self.diversity_weight = dup_threshold, diversity_weight

    def complete(self, ctx: np.ndarray, m: int, k: int, seed: int = 0) -> np.ndarray:
        from palette.api import Completer
        c = Completer.__new__(Completer)
        c.oversample, c.max_rounds, c.dup_threshold, c.diversity_weight = self.oversample, 1, self.dup_threshold, self.diversity_weight
        c.scorer = self.scorer
        c._raw = lambda ctx_, m_, k_, seed_, guidance=None: self.raw_fn(ctx_, m_, k_, seed_)
        return c.complete_oklab(ctx, m, k, seed)[0]
