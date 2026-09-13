"""Conditional KDE baseline (R7) — thin wrapper over PaletteIndex.kde_complete for a uniform interface."""
from __future__ import annotations

import numpy as np

from palette.baselines.retrieval import PaletteIndex


class ConditionalKDE:
    def __init__(self, X: np.ndarray, M: np.ndarray, bandwidth: float = 0.03, jitter: float = 0.015):
        self.index = PaletteIndex(X, M); self.bandwidth = bandwidth; self.jitter = jitter

    def complete(self, ctx: np.ndarray, m: int, k: int, seed: int = 0) -> np.ndarray:
        return self.index.kde_complete(ctx, m, k, self.bandwidth, self.jitter, seed)
