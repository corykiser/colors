"""Wrap a trained denoiser as a completion function (raw Oklab in, raw Oklab out, fixed colors untouched)."""
from __future__ import annotations

import numpy as np
import torch

from palette.color import NormStats, oklab_to_srgb, srgb_to_oklab
from palette.diffusion import VPSchedule, FlowSchedule, sample, sample_dpmpp2m, flow_sample


class DiffusionCompleter:
    def __init__(self, model, sched: VPSchedule, norm: NormStats, device, steps: int = 100, method: str = "ddim",
                 cfg_scale: float | None = None, rating: float | None = None, guidance=None, nmax: int = 5, space: str = "oklab"):
        self.model, self.sched, self.norm, self.device, self.space = model, sched, norm, device, space
        self.steps, self.method, self.cfg_scale, self.rating, self.guidance, self.nmax = steps, method, cfg_scale, rating, guidance, nmax

    def complete(self, ctx_oklab: np.ndarray, m: int, k: int, seed: int = 0) -> np.ndarray:
        ctx = np.asarray(ctx_oklab, dtype=np.float64).reshape(-1, 3)
        n = len(ctx); N = max(n + m, 1)
        x = np.zeros((k, N, 3), np.float32)
        if n:
            x[:, :n] = self.norm.normalize(np.clip(oklab_to_srgb(ctx), 0, 1) if self.space == "srgb" else ctx)
        is_fixed = torch.zeros(k, N, dtype=torch.bool); is_fixed[:, :n] = True
        mask = torch.ones(k, N, dtype=torch.bool)
        cond = None; cond_null = None
        if self.rating is not None:
            cond = {"rating": torch.full((k,), float(self.rating))}
            cond_null = {"rating": torch.full((k,), float("nan"))}
        g = torch.Generator(device=self.device).manual_seed(seed)
        args = (self.model, self.sched, torch.tensor(x).to(self.device), is_fixed.to(self.device), mask.to(self.device))
        kw = dict(cond=cond, steps=self.steps, cfg_scale=self.cfg_scale, cond_null=cond_null, guidance=self.guidance, generator=g)
        if isinstance(self.sched, FlowSchedule):
            out = flow_sample(*args, method=self.method if self.method in ("euler", "heun") else "heun", **kw)
        elif self.method == "dpmpp2m":
            out = sample_dpmpp2m(*args, **kw)
        else:
            out = sample(*args, method=self.method, **kw)
        y = self.norm.denormalize(out[:, n:].float().cpu().numpy().astype(np.float64))
        return srgb_to_oklab(y) if self.space == "srgb" else y   # unclipped: out-of-box sRGB maps out of gamut, as it should
