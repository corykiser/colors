"""Common held-out diffusion loss for any set of checkpoints, on identical fixed val batches
(mabs.yaml sampler config, seed 123), plus per-target-count breakdown. Writes experiments/common_val.json.
Usage: uv run python scripts/evaluate.py experiments/mabs experiments/geomlite ..."""
from __future__ import annotations

import json
import sys

import torch
import yaml

from palette.api import load_model
from palette.data import load_arrays, fit_or_load_norm, ROOT
from palette.diffusion import masked_eps_loss
from palette.sampler import SamplerConfig
from scripts.train import fixed_val_batches


@torch.no_grad()
def main(dirs):
    cfg = yaml.safe_load(open(ROOT / "configs/mabs.yaml"))
    norm = fit_or_load_norm(); val = load_arrays(("kuler", "mturk2014"), "val", norm)
    out = {}
    for d in dirs:
        ck = ROOT / d / ("model_best.pt" if (ROOT / d / "model_best.pt").exists() else "model_ema.pt")
        model, sched, _, device = load_model(str(ck))
        batches = fixed_val_batches(val, SamplerConfig(seed=0, **cfg["sampler"]), sched, n_batches=16)
        # NOTE: flow-matching and sRGB models use a different objective / representation; their common loss is
        # reported but only comparable within the same objective and space.
        tot, by_m = 0.0, {}
        for b in batches:
            x0, f, m, t, noise = (b[k].to(device) for k in ("x0", "is_fixed", "mask", "t", "noise"))
            tgt = m & ~f
            xt = torch.where(tgt[..., None], sched.q_sample(x0, t, noise), x0)
            eps = model(xt, f, m, t, None)
            y = sched.target(x0, noise) if hasattr(sched, "target") else noise   # flow models predict velocity
            tot += masked_eps_loss(eps, y, tgt).item()
            se = (((eps - y) ** 2).sum(-1) * tgt).sum(1) / tgt.sum(1).clamp(min=1)
            for mm in range(1, 6):
                sel = tgt.sum(1) == mm
                if sel.any():
                    by_m.setdefault(mm, []).append(se[sel].mean().item())
        out[d] = {"common_val_loss": tot / len(batches), "by_m": {k: sum(v) / len(v) for k, v in by_m.items()}}
        print(d, json.dumps(out[d]))
    json.dump(out, open(ROOT / "experiments/common_val.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
