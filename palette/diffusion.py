"""Variance-preserving diffusion: schedule, forward noise, masked loss, DDPM/DDIM samplers.

Only *target* nodes are noised and updated; context nodes stay clean throughout.
Chromatic noise is isotropic in (a, b) by construction (i.i.d. Gaussian per coordinate).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Protocol

import torch

Tensor = torch.Tensor


class Denoiser(Protocol):
    def __call__(self, x: Tensor, is_fixed: Tensor, mask: Tensor, t: Tensor, cond: dict | None) -> Tensor: ...


@dataclass
class VPSchedule:
    T: int = 1000
    kind: str = "cosine"  # or "linear"

    def __post_init__(self) -> None:
        if self.kind == "linear":
            betas = torch.linspace(1e-4, 0.02, self.T, dtype=torch.float64)
            ac = torch.cumprod(1 - betas, 0)
        elif self.kind == "cosine":
            s = 0.008
            steps = torch.arange(self.T + 1, dtype=torch.float64) / self.T
            f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
            ac = (f[1:] / f[0]).clamp(1e-5, 1.0)
            betas = (1 - ac[1:] / ac[:-1]).clamp(max=0.999)
            betas = torch.cat([1 - ac[:1], betas])
            ac = torch.cumprod(1 - betas, 0)
        else:
            raise ValueError(self.kind)
        self.betas = betas.float()
        self.alphas_cumprod = ac.float()

    def to(self, device) -> "VPSchedule":
        self.betas = self.betas.to(device); self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        ab = self.alphas_cumprod[t].view(-1, *([1] * (x0.dim() - 1)))
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise


def masked_eps_loss(eps_hat: Tensor, eps: Tensor, target_mask: Tensor) -> Tensor:
    """Per-palette mean over its m targets (all 3 coords), then batch mean. Padded/context slots excluded."""
    se = ((eps_hat - eps) ** 2).sum(-1) * target_mask            # (B, N)
    m = target_mask.sum(1).clamp(min=1)
    return (se.sum(1) / m).mean()


def training_step(model: Denoiser, sched: VPSchedule, x0: Tensor, is_fixed: Tensor, mask: Tensor,
                  cond: dict | None = None) -> Tensor:
    """x0: (B,N,3) normalized; is_fixed: (B,N) bool; mask: (B,N) bool valid. Returns loss."""
    B = x0.shape[0]
    target = mask & ~is_fixed
    t = torch.randint(0, sched.T, (B,), device=x0.device)
    noise = torch.randn_like(x0)
    xt = torch.where(target[..., None], sched.q_sample(x0, t, noise), x0)
    eps_hat = model(xt, is_fixed, mask, t, cond)
    return masked_eps_loss(eps_hat, noise, target)


@torch.no_grad()
def sample(model: Denoiser, sched: VPSchedule, x_ctx: Tensor, is_fixed: Tensor, mask: Tensor,
           cond: dict | None = None, steps: int = 200, method: str = "ddim", eta: float = 0.0,
           guidance: Callable[[Tensor, Tensor, int], Tensor] | None = None,
           cfg_scale: float | None = None, cond_null: dict | None = None,
           generator: torch.Generator | None = None) -> Tensor:
    """Reverse sampler updating only target slots. x_ctx holds the clean context in fixed slots.

    guidance(x, target_mask, t) may return an additive correction to eps_hat (e.g. repulsion).
    cfg: eps = eps_null + cfg_scale * (eps_cond - eps_null).
    """
    B, N, _ = x_ctx.shape
    target = mask & ~is_fixed
    x = torch.where(target[..., None], torch.randn(x_ctx.shape, device=x_ctx.device, generator=generator), x_ctx)
    ts = torch.linspace(sched.T - 1, 0, steps, device=x.device).round().long()
    ac = sched.alphas_cumprod

    def eps_fn(x, t):
        e = model(x, is_fixed, mask, t, cond)
        if cfg_scale is not None and cfg_scale != 1.0:
            e0 = model(x, is_fixed, mask, t, cond_null)
            e = e0 + cfg_scale * (e - e0)
        if guidance is not None:
            e = e + guidance(x, target, int(t[0]), mask)
        return e

    for i in range(steps):
        t = ts[i].expand(B)
        a_t = ac[ts[i]]
        a_prev = ac[ts[i + 1]] if i + 1 < steps else torch.tensor(1.0, device=x.device)
        eps = eps_fn(x, t)
        x0_hat = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        if method == "ddim":
            sigma = eta * ((1 - a_prev) / (1 - a_t)).sqrt() * (1 - a_t / a_prev).sqrt()
            dir_xt = (1 - a_prev - sigma**2).clamp(min=0).sqrt() * eps
            noise = torch.randn(x.shape, device=x.device, generator=generator) if (eta > 0 and i + 1 < steps) else 0.0
            x_new = a_prev.sqrt() * x0_hat + dir_xt + sigma * noise
        elif method == "ddpm":
            beta = 1 - a_t / a_prev
            mean = (a_prev.sqrt() * beta / (1 - a_t)) * x0_hat + ((1 - beta).sqrt() * (1 - a_prev) / (1 - a_t)) * x
            var = beta * (1 - a_prev) / (1 - a_t)
            noise = torch.randn(x.shape, device=x.device, generator=generator) if i + 1 < steps else 0.0
            x_new = mean + var.sqrt() * noise
        else:
            raise ValueError(method)
        x = torch.where(target[..., None], x_new, x)
    return x


# --- faster VP solver: DPM-Solver++ (2M), data-prediction multistep --------------------------
@torch.no_grad()
def sample_dpmpp2m(model: Denoiser, sched: VPSchedule, x_ctx: Tensor, is_fixed: Tensor, mask: Tensor,
                   cond: dict | None = None, steps: int = 20, guidance=None, cfg_scale=None, cond_null=None,
                   generator: torch.Generator | None = None) -> Tensor:
    """DPM-Solver++ 2M (Lu et al. 2022) on the VP schedule; same interface as `sample`."""
    B, N, _ = x_ctx.shape
    target = mask & ~is_fixed
    x = torch.where(target[..., None], torch.randn(x_ctx.shape, device=x_ctx.device, generator=generator), x_ctx)
    ts = torch.linspace(sched.T - 1, 0, steps, device=x.device).round().long()
    ac = sched.alphas_cumprod

    def x0_fn(x, t):
        e = model(x, is_fixed, mask, t, cond)
        if cfg_scale is not None and cfg_scale != 1.0:
            e0 = model(x, is_fixed, mask, t, cond_null); e = e0 + cfg_scale * (e - e0)
        if guidance is not None:
            e = e + guidance(x, target, int(t[0]), mask)
        a = ac[t[0]]
        return (x - (1 - a).sqrt() * e) / a.sqrt()

    def lam(i):  # log-SNR/2 at step index i (or clean state)
        a = ac[ts[i]] if i < steps else torch.tensor(1.0 - 1e-6, device=x.device)
        return 0.5 * (a.log() - (1 - a).log())

    prev_x0 = None
    for i in range(steps):
        t = ts[i].expand(B)
        x0 = x0_fn(x, t)
        l_cur, l_next = lam(i), lam(i + 1)
        h = l_next - l_cur
        a_next = ac[ts[i + 1]] if i + 1 < steps else torch.tensor(1.0, device=x.device)
        s_cur, s_next = (1 - ac[ts[i]]).sqrt(), (1 - a_next).sqrt()
        if prev_x0 is None or i + 1 == steps or i < 2:   # first-order until two *informative* predictions exist
            d = x0
        else:
            r = (l_cur - lam(i - 1)) / h
            d = (1 + 1 / (2 * r)) * x0 - (1 / (2 * r)) * prev_x0
        if i + 1 == steps:      # next state is clean (σ=0): the update reduces to x = x0_hat
            x_new = d
        else:
            x_new = (s_next / s_cur) * x - a_next.sqrt() * torch.expm1(-h) * d
        x = torch.where(target[..., None], x_new, x)
        prev_x0 = x0
    return x


# --- flow matching (rectified flow) alternative --------------------------------------------
class FlowSchedule:
    """Linear interpolation x_t = (1-t) x0 + t ε, t∈[0,1]; model predicts velocity v = ε - x0.
    Exposes T and q_sample so the training loop can treat it like VPSchedule (t is an integer index)."""

    def __init__(self, T: int = 1000):
        self.T = T
        self.alphas_cumprod = (1 - torch.arange(T, dtype=torch.float32) / T) ** 2   # for code that inspects it

    def to(self, device):
        self.alphas_cumprod = self.alphas_cumprod.to(device); return self

    def tau(self, t: Tensor) -> Tensor:
        return (t.float() + 0.5) / self.T

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        tau = self.tau(t).view(-1, *([1] * (x0.dim() - 1)))
        return (1 - tau) * x0 + tau * noise

    def target(self, x0: Tensor, noise: Tensor) -> Tensor:
        return noise - x0


def flow_training_step(model: Denoiser, sched: FlowSchedule, x0: Tensor, is_fixed: Tensor, mask: Tensor,
                       cond: dict | None = None) -> Tensor:
    B = x0.shape[0]
    target = mask & ~is_fixed
    t = torch.randint(0, sched.T, (B,), device=x0.device)
    noise = torch.randn_like(x0)
    xt = torch.where(target[..., None], sched.q_sample(x0, t, noise), x0)
    v_hat = model(xt, is_fixed, mask, t, cond)
    return masked_eps_loss(v_hat, sched.target(x0, noise), target)


@torch.no_grad()
def flow_sample(model: Denoiser, sched: FlowSchedule, x_ctx: Tensor, is_fixed: Tensor, mask: Tensor,
                cond: dict | None = None, steps: int = 20, method: str = "heun", guidance=None,
                cfg_scale=None, cond_null=None, generator: torch.Generator | None = None) -> Tensor:
    """Integrate dx/dt = -v from t=1 (noise) to t=0 with Euler or Heun; only targets move."""
    B, N, _ = x_ctx.shape
    target = mask & ~is_fixed
    x = torch.where(target[..., None], torch.randn(x_ctx.shape, device=x_ctx.device, generator=generator), x_ctx)
    taus = torch.linspace(1.0, 0.0, steps + 1, device=x.device)

    def v_fn(x, tau):
        t = (tau * sched.T - 0.5).clamp(0, sched.T - 1).round().long().expand(B)
        v = model(x, is_fixed, mask, t, cond)
        if cfg_scale is not None and cfg_scale != 1.0:
            v0 = model(x, is_fixed, mask, t, cond_null); v = v0 + cfg_scale * (v - v0)
        if guidance is not None:
            v = v + guidance(x, target, int(t[0]), mask)
        return v

    for i in range(steps):
        dt = taus[i + 1] - taus[i]          # negative
        v = v_fn(x, taus[i])
        x_e = x + dt * v
        if method == "heun" and i + 1 < steps:
            v2 = v_fn(torch.where(target[..., None], x_e, x), taus[i + 1])
            x_new = x + dt * 0.5 * (v + v2)
        else:
            x_new = x_e
        x = torch.where(target[..., None], x_new, x)
    return x
