"""Phase 5 sampling pipeline and public API.

complete(colors_srgb_hex, m, k) -> k palettes as hex lists; fixed colors returned verbatim.
Steps: oversample → gamut rejection (fallback: chroma reduction at fixed L,h) → duplicate rejection
(fallback: repulsive guidance) → rerank on the *mapped* palette → greedy diverse top-k.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from palette.color import (NormStats, srgb_to_oklab, hex_to_srgb, in_gamut, gamut_map, oklab_to_srgb, srgb_to_hex)
from palette.diffusion import VPSchedule
from palette.eval.metrics import matching_distance
from palette.eval.report import load_scorer
from palette.sampling import DiffusionCompleter


def repulsive_guidance(strength: float = 0.5, sigma: float = 0.15):
    """Additive ε correction pushing target colors apart (in normalized space). Zero effect when far apart."""
    def g(x, target, t):
        tm = target.float()[..., None]
        d = x[:, :, None] - x[:, None, :]                        # (B,N,N,3)
        r2 = (d**2).sum(-1, keepdim=True)
        w = torch.exp(-r2 / (2 * sigma**2)) * tm[:, :, None] * tm[:, None, :]
        force = (w * d / (r2.sqrt() + 1e-6)).sum(2)              # push i away from j
        return -strength * force * tm                            # ε points *toward* noise; subtracting moves x along +force
    return g


def load_model(path: str, device=None):
    from scripts.train import build_model
    device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(path, map_location="cpu")
    model = build_model(ck["config"]["model"]).to(device).eval()
    model.load_state_dict(ck["state_dict"])
    norm = NormStats.from_dict(ck["norm"])
    sched = VPSchedule(ck["config"]["diffusion"].get("T", 1000), ck["config"]["diffusion"].get("schedule", "cosine")).to(device)
    return model, sched, norm, device


@dataclass
class Completer:
    model: object; sched: VPSchedule; norm: NormStats; device: object
    steps: int = 100; method: str = "ddim"; cfg_scale: float | None = None; rating: float | None = None
    oversample: int = 32; max_rounds: int = 3; dup_threshold: float = 0.03; diversity_weight: float = 1.0
    scorer=None

    def __post_init__(self):
        if self.scorer is None:
            self.scorer = load_scorer()

    @classmethod
    def from_checkpoint(cls, path: str, **kw) -> "Completer":
        model, sched, norm, device = load_model(path)
        return cls(model, sched, norm, device, **kw)

    def _raw(self, ctx, m, k, seed, guidance=None):
        dc = DiffusionCompleter(self.model, self.sched, self.norm, self.device, self.steps, self.method,
                                self.cfg_scale, self.rating, guidance)
        return dc.complete(ctx, m, k, seed)

    def complete_oklab(self, ctx: np.ndarray, m: int, k: int, seed: int = 0) -> tuple[np.ndarray, dict]:
        """Returns (k, m, 3) Oklab completions (in gamut) plus pipeline statistics."""
        ctx = np.asarray(ctx, dtype=np.float64).reshape(-1, 3)
        pool, stats = [], {"raw": 0, "in_gamut": 0, "duplicates": 0, "rounds": 0, "mapped_fallback": 0, "guided": 0}
        need = max(k, self.oversample)

        def accept(comp, into):
            ok_g = in_gamut(comp).all(1)
            stats["raw"] += len(comp); stats["in_gamut"] += int(ok_g.sum())
            full = np.concatenate([np.broadcast_to(ctx, (len(comp), len(ctx), 3)), comp], 1)
            i, j = np.triu_indices(full.shape[1], 1)
            dmin = np.linalg.norm(full[:, i] - full[:, j], axis=-1).min(1) if len(i) else np.full(len(comp), np.inf)
            ok_d = dmin >= self.dup_threshold
            stats["duplicates"] += int((~ok_d & ok_g).sum())
            into.extend(comp[ok_g & ok_d])
            return comp[~ok_g & ok_d]

        rejected_gamut = []
        for r in range(self.max_rounds):
            stats["rounds"] += 1
            rejected_gamut.append(accept(self._raw(ctx, m, need, seed + 100 * r), pool))
            if len(pool) >= need:
                break
        if len(pool) < k:  # fallback 1: repulsive guidance for duplicates
            stats["guided"] = 1
            rejected_gamut.append(accept(self._raw(ctx, m, need, seed + 999, repulsive_guidance()), pool))
        if len(pool) < k:  # fallback 2: chroma-reduce out-of-gamut survivors
            rej = np.concatenate([r for r in rejected_gamut if len(r)]) if any(len(r) for r in rejected_gamut) else np.zeros((0, m, 3))
            mapped = np.stack([gamut_map(c)[0] for c in rej]) if len(rej) else rej
            if len(mapped):
                fullm = np.concatenate([np.broadcast_to(ctx, (len(mapped), len(ctx), 3)), mapped], 1)
                i, j = np.triu_indices(fullm.shape[1], 1)
                dmin = np.linalg.norm(fullm[:, i] - fullm[:, j], axis=-1).min(1) if len(i) else np.full(len(mapped), np.inf)
                keep = mapped[dmin >= self.dup_threshold]
                mapped = keep if len(pool) + len(keep) >= k else mapped   # drop dup-filter only if it would starve k
            stats["mapped_fallback"] = len(mapped); pool.extend(mapped)
        if len(pool) < k:
            raise RuntimeError(f"could not produce {k} valid completions (got {len(pool)})")
        pool = np.stack(pool)
        # rerank on mapped palettes
        full = np.concatenate([np.broadcast_to(ctx, (len(pool), len(ctx), 3)), pool], 1)
        scores = self.scorer(full) if self.scorer is not None else np.zeros(len(pool))
        order = np.argsort(-scores); pool, scores = pool[order], scores[order]
        # greedy diverse top-k
        chosen = [0]
        sn = (scores - scores.min()) / (np.ptp(scores) + 1e-8)
        while len(chosen) < k:
            d = np.stack([matching_distance(pool, np.broadcast_to(pool[c], pool.shape)) for c in chosen]).min(0)
            obj = sn + self.diversity_weight * d / (d.max() + 1e-8)
            obj[chosen] = -np.inf
            chosen.append(int(obj.argmax()))
        stats["pool"] = len(pool); stats["scores_top"] = [float(s) for s in scores[chosen]]
        return pool[chosen], stats

    def complete(self, colors_srgb: list[str], m: int, k: int = 4, seed: int = 0) -> list[list[str]]:
        rgb = np.array([hex_to_srgb(h) for h in colors_srgb], dtype=np.float64).reshape(-1, 3)
        ctx = srgb_to_oklab(rgb).reshape(-1, 3)
        out, _ = self.complete_oklab(ctx, m, k, seed)
        return [list(colors_srgb) + [srgb_to_hex(oklab_to_srgb(c)) for c in comp] for comp in out]


def complete(colors_srgb: list[str], m: int, k: int = 4, u=None, checkpoint: str = "experiments/mabs/model_ema.pt", seed: int = 0) -> list[list[str]]:
    """Public API (handoff §4.7). u (preference vector) is reserved; None = population model."""
    if u is not None:
        raise NotImplementedError("preference conditioning u is deferred to v2")
    return Completer.from_checkpoint(checkpoint).complete(colors_srgb, m, k, seed)
