"""Color-wheel baseline: complementary / triadic / analogous / split-complementary hues around the
context's chroma-weighted mean hue, at context-matched L and C. Works in Oklab LCh; gamut-mapped."""
from __future__ import annotations

import numpy as np

from palette.color import oklab_to_lch, lch_to_oklab, gamut_map

SCHEMES = {
    "complementary": [180.0],
    "triadic": [120.0, 240.0],
    "analogous": [30.0, -30.0, 60.0, -60.0],
    "split_complementary": [150.0, 210.0],
    "tetradic": [90.0, 180.0, 270.0],
}


def _context_stats(ctx: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    if len(ctx) == 0:  # unconditional: random hue, mid L/C
        return float(rng.uniform(-np.pi, np.pi)), 0.65, 0.12
    lch = oklab_to_lch(ctx)
    w = lch[:, 1] + 1e-6
    h = np.arctan2((w * np.sin(lch[:, 2])).sum(), (w * np.cos(lch[:, 2])).sum())
    C = float(np.average(lch[:, 1], weights=w))
    return float(h), float(lch[:, 0].mean()), max(C, 0.05)


def complete(ctx_oklab: np.ndarray, m: int, k: int, seed: int = 0) -> np.ndarray:
    """Returns (k, m, 3) Oklab completions. Cycles schemes across the k samples; jitters L and C."""
    rng = np.random.default_rng(seed)
    ctx = np.asarray(ctx_oklab, dtype=np.float64).reshape(-1, 3)
    h0, L0, C0 = _context_stats(ctx, rng)
    names = list(SCHEMES)
    out = np.zeros((k, m, 3))
    for i in range(k):
        offs = SCHEMES[names[i % len(names)]]
        hues = [h0 + np.deg2rad(offs[j % len(offs)]) for j in range(m)]
        Ls = np.clip(L0 + rng.normal(0, 0.12, m) + np.linspace(-0.15, 0.15, m)[rng.permutation(m)], 0.08, 0.95)
        Cs = np.clip(C0 * rng.uniform(0.6, 1.3, m), 0.02, 0.35)
        lab = lch_to_oklab(np.stack([Ls, Cs, np.array(hues)], -1))
        out[i], _ = gamut_map(lab)
    return out
