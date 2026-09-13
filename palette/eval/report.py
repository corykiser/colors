"""Shared generator evaluation: probe grids + held-out reconstruction diagnostic. Used by baselines and models."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch

from palette.color import srgb_to_oklab, hex_to_srgb, gamut_map, oklab_to_srgb, srgb_to_hex
from palette.data import fit_or_load_norm, ROOT
from palette.eval.metrics import matching_distance, diversity, duplicate_rate, gamut_rate
from palette.eval.probes import PROBES

CompleteFn = Callable[[np.ndarray, int, int, int], np.ndarray]   # (ctx_oklab (n,3), m, k, seed) -> (k, m, 3)


def load_scorer():
    p = ROOT / "experiments/scorer/deepsets.pt"
    if not p.exists():
        return None
    from palette.models.scorer import Scorer
    ck = torch.load(p); m = Scorer(ck["variant"]); m.load_state_dict(ck["state_dict"]); m.eval()
    norm = fit_or_load_norm()

    def score(palettes: np.ndarray) -> np.ndarray:
        x = torch.tensor(norm.normalize(palettes).astype(np.float32))
        with torch.no_grad():
            return m(x).numpy()
    return score


def probe_context(pr: dict) -> np.ndarray:
    rgb = np.array([hex_to_srgb(h) for h in pr["colors"]], dtype=np.float64).reshape(-1, 3)
    return srgb_to_oklab(rgb).reshape(-1, 3)


def hexes(lab: np.ndarray) -> list[str]:
    return [srgb_to_hex(oklab_to_srgb(c)) for c in lab]


def evaluate_generator(complete: CompleteFn, val_X: np.ndarray | None, K: int = 16, n_recon: int = 300,
                       score_fn=None, probes: list[dict] = PROBES, recon_K: int = 8) -> dict:
    t0 = time.time()
    agg = {"diversity": [], "gamut_rate_raw": [], "duplicate_rate": [], "score": []}
    probe_out = {}
    for pi, pr in enumerate(probes):
        ctx = probe_context(pr)
        comp = complete(ctx, pr["m"], K, pi)
        agg["gamut_rate_raw"].append(gamut_rate(comp))
        mapped = np.stack([gamut_map(c)[0] for c in comp])
        agg["diversity"].append(diversity(mapped))
        agg["duplicate_rate"].append(duplicate_rate(mapped))
        if score_fn is not None:
            full = np.concatenate([np.broadcast_to(ctx, (K, len(ctx), 3)), mapped], 1)
            agg["score"].append(float(score_fn(full).mean()))
        probe_out[pr["name"]] = {"context": pr["colors"], "m": pr["m"], "completions": [hexes(c) for c in mapped]}
    out = {k: float(np.mean(v)) for k, v in agg.items() if len(v)}
    if val_X is not None:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(val_X), min(n_recon, len(val_X)), replace=False)
        mins, means = [], []
        for qi, i in enumerate(sel):
            P = val_X[i].astype(np.float64); c = int(rng.integers(1, 4)); perm = rng.permutation(5)
            ctx, held = P[perm[:c]], P[perm[c:]]
            comp = complete(ctx, 5 - c, recon_K, 1000 + qi)
            d = matching_distance(comp, np.broadcast_to(held, comp.shape))
            mins.append(float(d.min())); means.append(float(d.mean()))
        out["recon_min_over_K"] = float(np.mean(mins)); out["recon_mean_over_K"] = float(np.mean(means))
    out["eval_seconds"] = round(time.time() - t0, 1)
    return {"summary": out, "probes": probe_out}
