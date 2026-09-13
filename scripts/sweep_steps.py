"""Item 7: sampling-step sweep on an existing checkpoint — quality, diversity, whole-completion gamut rate,
duplicate rate, acceptance rate under the API rules, and latency. DDIM vs DPM-Solver++ 2M (VP) or Euler/Heun (flow).
Usage: uv run python scripts/sweep_steps.py experiments/v2/mabs [--steps 5 10 20 50 100]"""
from __future__ import annotations

import argparse, json, time
import numpy as np
from palette.api import load_model
from palette.sampling import DiffusionCompleter
from palette.color import in_gamut, gamut_map
from palette.data import ROOT
from palette.diffusion import FlowSchedule
from palette.eval.metrics import diversity, duplicate_rate
from palette.eval.probes import PROBES
from palette.eval.report import probe_context, load_scorer


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--steps", nargs="*", type=int, default=[5, 10, 20, 50, 100]); ap.add_argument("--K", type=int, default=16)
    a = ap.parse_args()
    model, sched, norm, dev = load_model(str(ROOT / a.run / "model_best.pt"))
    methods = ("euler", "heun") if isinstance(sched, FlowSchedule) else ("ddim", "dpmpp2m")
    score = load_scorer(); rows = []
    probes = [p for p in PROBES if p["colors"]]
    for method in methods:
        for st in a.steps:
            dc = DiffusionCompleter(model, sched, norm, dev, steps=st, method=method, space=getattr(model, "space", "oklab"))
            t0 = time.time(); div, joint, per, dup, sc, acc = [], [], [], [], [], []
            for pi, pr in enumerate(probes):
                ctx = probe_context(pr); comp = dc.complete(ctx, pr["m"], a.K, pi)
                ok = in_gamut(comp); per.append(ok.mean()); joint.append(ok.all(1).mean())
                mapped = np.stack([gamut_map(c)[0] for c in comp]); div.append(diversity(mapped)); dup.append(duplicate_rate(mapped, 0.03))
                full = np.concatenate([np.broadcast_to(ctx, (a.K, len(ctx), 3)), mapped], 1); sc.append(score(full).mean())
                acc.append((ok.all(1) & (duplicate_rate(comp, 0.03) == 0)).mean() if pr["m"] > 1 else ok.all(1).mean())
            rows.append({"method": method, "steps": st, "sec_per_call": round((time.time() - t0) / len(probes), 3), "scorer": float(np.mean(sc)),
                         "diversity": float(np.mean(div)), "gamut_per_color": float(np.mean(per)), "gamut_whole": float(np.mean(joint)),
                         "dup_rate": float(np.mean(dup))})
            print(json.dumps(rows[-1]), flush=True)
    json.dump(rows, open(ROOT / a.run / "step_sweep.json", "w"), indent=1)


if __name__ == "__main__":
    main()
