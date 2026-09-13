"""Train a conditional set-diffusion denoiser from a yaml config. Usage:
    uv run python scripts/train.py configs/mabs.yaml [--set key.path=value ...] [--out experiments/name]
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from palette.data import load_arrays, fit_or_load_norm, ROOT
from palette.diffusion import VPSchedule, FlowSchedule, training_step, flow_training_step, masked_eps_loss
from palette.sampler import MaskedCompletionSampler, SamplerConfig, PaletteArrays
from palette.sampling import DiffusionCompleter
from palette.eval.report import evaluate_generator, load_scorer
from palette.models.absolute import AbsoluteDenoiser, count_params


def build_model(mcfg: dict):
    t = mcfg.get("type", "absolute")
    kw = {k: v for k, v in mcfg.items() if k != "type"}
    if t == "absolute":
        return AbsoluteDenoiser(**kw)
    if t == "geomlite":
        from palette.models.geomlite import GeomLiteDenoiser
        return GeomLiteDenoiser(**kw)
    if t == "geometric":
        from palette.models.geometric import GeometricDenoiser
        return GeometricDenoiser(**kw)
    if t == "combined":
        from palette.models.combined import CombinedDenoiser
        return CombinedDenoiser(**kw)
    raise ValueError(t)


def subsample(a: PaletteArrays, fraction: float, seed: int) -> PaletteArrays:
    if fraction >= 1.0:
        return a
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(a), max(64, int(round(fraction * len(a)))), replace=False))
    return PaletteArrays(a.X[idx], a.M[idx], a.family[idx], a.rating[idx], a.record_id[idx])


def fixed_val_batches(val: PaletteArrays, scfg: SamplerConfig, sched: VPSchedule, n_batches: int = 8, seed: int = 123):
    c = copy.copy(scfg); c.seed = seed; c.batch_size = 512
    s = MaskedCompletionSampler(val, c)
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_batches):
        b = s.batch()
        b["t"] = torch.randint(0, sched.T, (len(b["x0"]),), generator=g)
        b["noise"] = torch.randn(b["x0"].shape, generator=g)
        out.append(b)
    return out


@torch.no_grad()
def val_loss(model, sched, batches, device):
    model.eval(); tot = 0.0
    for b in batches:
        x0, f, m, t, noise = (b[k].to(device) for k in ("x0", "is_fixed", "mask", "t", "noise"))
        tgt = m & ~f
        xt = torch.where(tgt[..., None], sched.q_sample(x0, t, noise), x0)
        cond = None if b["cond"] is None else {k: v.to(device) for k, v in b["cond"].items()}
        y = sched.target(x0, noise) if isinstance(sched, FlowSchedule) else noise
        tot += masked_eps_loss(model(xt, f, m, t, cond), y, tgt).item()
    model.train()
    return tot / len(batches)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay; self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)


def set_nested(d, path, value):
    keys = path.split("."); cur = d
    for k in keys[:-1]: cur = cur.setdefault(k, {})
    cur[keys[-1]] = yaml.safe_load(value)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--set", nargs="*", default=[]); ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    for s in args.set: set_nested(cfg, *s.split("=", 1))
    name = args.out or f"experiments/{Path(args.config).stem}"
    out = ROOT / name; out.mkdir(parents=True, exist_ok=True)
    seed = cfg["train"].get("seed", 0); torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    space = cfg["data"].get("space", "oklab")
    norm = fit_or_load_norm(space=space)
    fams = tuple(cfg["data"]["families"])
    train = subsample(load_arrays(fams, "train", norm, space=space), cfg["data"].get("fraction", 1.0), seed)
    val = load_arrays(fams, "val", norm, space=space)
    val_raw = load_arrays(fams, "val", None)
    scfg = SamplerConfig(seed=seed, **cfg["sampler"])
    sampler = MaskedCompletionSampler(train, scfg)
    dcfg = cfg["diffusion"]; is_flow = dcfg.get("kind") == "flow"
    sched = (FlowSchedule(dcfg.get("T", 1000)) if is_flow else VPSchedule(dcfg.get("T", 1000), dcfg.get("schedule", "cosine"))).to(device)
    step_fn = flow_training_step if is_flow else training_step
    model = build_model(cfg["model"]).to(device)
    ema = EMA(model, cfg["train"].get("ema", 0.999))
    steps, lr = cfg["train"]["steps"], cfg["train"]["lr"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["train"].get("weight_decay", 0.01))
    warm = cfg["train"].get("warmup", 500)
    lr_at = lambda s: lr * min(1.0, (s + 1) / warm) * (0.5 * (1 + np.cos(np.pi * min(1.0, s / steps))) * 0.9 + 0.1)
    vb = fixed_val_batches(val, scfg, sched)
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    hist = []; t0 = time.time(); eval_every = cfg["train"].get("eval_every", 1000)
    eligible = {f: int(len(i)) for f, i in sampler.idx_by_fam.items()}
    n_eligible = sum(eligible.values())
    print(f"{name}: {count_params(model):,} params, {len(train):,} loaded / {n_eligible:,} eligible train palettes {eligible}, device={device}", flush=True)
    best_vl, best_step = float("inf"), 0
    it = iter(sampler); run = 0.0
    for step in range(1, steps + 1):
        for g in opt.param_groups: g["lr"] = lr_at(step)
        b = next(it)
        x0, f, m = b["x0"].to(device), b["is_fixed"].to(device), b["mask"].to(device)
        cond = None if b["cond"] is None else {k: v.to(device) for k, v in b["cond"].items()}
        loss = step_fn(model, sched, x0, f, m, cond)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); ema.update(model)
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        extra = getattr(model, "last_stats", None)
        if step % eval_every == 0 or step == steps:
            vl = val_loss(ema.shadow, sched, vb, device)
            rec = {"step": step, "train_loss": run, "val_loss": vl, "seconds": round(time.time() - t0, 1)}
            if extra: rec.update(extra)
            hist.append(rec); print(json.dumps(rec), flush=True)
            if vl < best_vl:
                best_vl, best_step = vl, step
                torch.save({"state_dict": ema.shadow.state_dict(), "config": cfg, "norm": norm.to_dict(), "step": step, "val_loss": vl}, out / "model_best.pt")
    train_seconds = time.time() - t0
    torch.save({"state_dict": ema.shadow.state_dict(), "config": cfg, "norm": norm.to_dict(), "step": steps, "val_loss": hist[-1]["val_loss"]}, out / "model_ema.pt")
    if cfg["train"].get("eval_checkpoint", "best") == "best" and best_step < steps:
        ema.shadow.load_state_dict(torch.load(out / "model_best.pt")["state_dict"])
        eval_ckpt = f"best@{best_step}"
    else:
        eval_ckpt = f"final@{steps}"
    # --- final evaluation with EMA weights
    scfg_s = cfg.get("sample", {})
    completer = DiffusionCompleter(ema.shadow, sched, norm, device, steps=scfg_s.get("steps", 100), method=scfg_s.get("method", "ddim"),
                                   cfg_scale=scfg_s.get("cfg_scale"), rating=scfg_s.get("rating"), space=space)
    ev = evaluate_generator(completer.complete, val_raw.X[val_raw.M.sum(1) == 5], K=scfg_s.get("K", 16), score_fn=load_scorer())
    metrics = {"name": name, "config": cfg, "git": git, "seed": seed, "params": count_params(model), "train_palettes": n_eligible,
               "train_palettes_loaded": len(train), "eligible_by_family": eligible, "family_weights": cfg["sampler"].get("family_weights"),
               "eval_checkpoint": eval_ckpt, "best_val_loss": best_vl, "best_step": best_step,
               "train_seconds": round(train_seconds, 1), "device": str(device), "history": hist,
               "final_val_loss": hist[-1]["val_loss"], **ev["summary"]}
    json.dump(metrics, open(out / "metrics.json", "w"), indent=1)
    json.dump(ev["probes"], open(out / "probes.json", "w"), indent=1)
    yaml.safe_dump(cfg, open(out / "config.yaml", "w"))
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("config", "history")}, indent=1))


if __name__ == "__main__":
    main()
