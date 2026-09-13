"""Render a PNG comparison grid of probe completions. Usage: render_grid_png.py out.png dir1 dir2 ... [--n 5]"""
import argparse, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from palette.data import ROOT
from scripts.render_probes import load

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("dirs", nargs="+"); ap.add_argument("--n", type=int, default=5); ap.add_argument("--probes", type=int, default=12)
a = ap.parse_args()
runs = {}
for d in a.dirs: runs.update(load(ROOT / d))
names = list(next(iter(runs.values())).keys())[: a.probes]
rows = len(names) * len(runs)
fig, ax = plt.subplots(figsize=(11, 0.32 * rows + 0.5), dpi=110); ax.axis("off"); ax.set_xlim(0, 40); ax.set_ylim(-rows, 0)
y = 0
for pn in names:
    for rn, pr in runs.items():
        p = pr[pn]; y -= 1
        ax.text(0, y + 0.5, f"{pn[:22]} | {rn[:14]}", fontsize=6.5, va="center")
        x = 9
        for c in p["context"]:
            ax.add_patch(plt.Rectangle((x, y + 0.05), 0.9, 0.9, color=c, ec="k", lw=1.2)); x += 1
        x += 0.6
        for comp in p["completions"][: a.n]:
            for c in comp:
                ax.add_patch(plt.Rectangle((x, y + 0.05), 0.9, 0.9, color=c)); x += 1
            x += 0.5
fig.savefig(a.out, bbox_inches="tight"); print("wrote", a.out)
