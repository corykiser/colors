"""Source-balanced batching with on-the-fly masked palette completion and optional subset augmentation (R2)."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class PaletteArrays:
    """Stacked palettes for one split. X is *normalized* Oklab (N, nmax, 3); M is the valid mask."""
    X: np.ndarray
    M: np.ndarray
    family: np.ndarray            # (N,) str
    rating: np.ndarray            # (N,) float, nan when absent
    record_id: np.ndarray         # (N,) str

    def __len__(self) -> int:
        return len(self.X)


@dataclass
class SamplerConfig:
    batch_size: int = 256
    family_weights: dict[str, float] = field(default_factory=lambda: {"mturk2014": 1.0, "kuler": 1.0})
    subset_augment: bool = False
    subset_prob: float = 0.5                       # prob. an example is replaced by a k-subset of itself
    subset_k_probs: dict[int, float] = field(default_factory=lambda: {2: 0.2, 3: 0.4, 4: 0.4})
    empty_context_prob: float = 0.1                # unconditional examples
    min_rating: float | None = None                # filter (good-examples-only) — None keeps all
    rating_cond: bool = False                      # expose rating to the model (R6)
    rating_null_prob: float = 0.15                 # CFG null-token dropout
    seed: int = 0


class MaskedCompletionSampler:
    """Infinite iterator of training batches.

    Each item: pick a family by weight, a palette uniformly within it, optionally a k-subset
    (flagged), then a random context/target partition with a random row permutation.
    """

    def __init__(self, arrays: PaletteArrays, cfg: SamplerConfig):
        self.a, self.cfg = arrays, cfg
        self.rng = np.random.default_rng(cfg.seed)
        fams = [f for f in cfg.family_weights if (arrays.family == f).any()]
        if not fams:
            raise ValueError("no requested family present in arrays")
        self.fams = fams
        w = np.array([cfg.family_weights[f] for f in fams], float); self.w = w / w.sum()
        self.idx_by_fam = {}
        for f in fams:
            idx = np.where(arrays.family == f)[0]
            if cfg.min_rating is not None:
                r = arrays.rating[idx]
                idx = idx[np.isfinite(r) & (r >= cfg.min_rating)]
            if len(idx) == 0:
                warnings.warn(f"family {f} has no palettes after rating filter; dropped")
                continue
            self.idx_by_fam[f] = idx
        self.fams = [f for f in fams if f in self.idx_by_fam]
        if not self.fams:
            raise ValueError("no palettes left after rating filter")
        w = np.array([cfg.family_weights[f] for f in self.fams], float); self.w = w / w.sum()
        self.nmax = arrays.X.shape[1]
        ks = sorted(cfg.subset_k_probs); self.ks = np.array(ks); self.kp = np.array([cfg.subset_k_probs[k] for k in ks]); self.kp /= self.kp.sum()

    def _one(self):
        rng, cfg = self.rng, self.cfg
        f = self.fams[rng.choice(len(self.fams), p=self.w)]
        i = rng.choice(self.idx_by_fam[f])
        n = int(self.a.M[i].sum())
        rows = np.where(self.a.M[i])[0]
        subset = False
        if cfg.subset_augment and rng.random() < cfg.subset_prob:
            k = int(rng.choice(self.ks, p=self.kp))
            if k < n:
                rows = rng.choice(rows, size=k, replace=False); n = k; subset = True
        rows = rng.permutation(rows)
        c = 0 if rng.random() < cfg.empty_context_prob else int(rng.integers(1, n))  # 1..n-1 → m>=1
        x = np.zeros((self.nmax, 3), np.float32); x[:n] = self.a.X[i, rows]
        is_fixed = np.zeros(self.nmax, bool); is_fixed[:c] = True
        mask = np.zeros(self.nmax, bool); mask[:n] = True
        # shuffle slot positions so context isn't always first (no positional leakage)
        perm = rng.permutation(self.nmax)
        return x[perm], is_fixed[perm], mask[perm], f, subset, self.a.rating[i]

    def batch(self) -> dict:
        items = [self._one() for _ in range(self.cfg.batch_size)]
        x = torch.tensor(np.stack([it[0] for it in items]))
        is_fixed = torch.tensor(np.stack([it[1] for it in items]))
        mask = torch.tensor(np.stack([it[2] for it in items]))
        rating = torch.tensor(np.array([it[5] for it in items], np.float32))
        out = {"x0": x, "is_fixed": is_fixed, "mask": mask, "family": [it[3] for it in items],
               "subset_derived": torch.tensor([it[4] for it in items]), "rating": rating,
               "n_ctx": is_fixed.sum(1), "m_tgt": (mask & ~is_fixed).sum(1)}
        if self.cfg.rating_cond:
            null = torch.tensor(self.rng.random(len(items)) < self.cfg.rating_null_prob) | ~torch.isfinite(rating)
            out["cond"] = {"rating": torch.where(null, torch.full_like(rating, float("nan")), rating)}
        else:
            out["cond"] = None
        return out

    def __iter__(self):
        while True:
            yield self.batch()
