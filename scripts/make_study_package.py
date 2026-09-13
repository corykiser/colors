"""Phase 6: build a blinded rating-study package (no data collection).

For each probe context: completions from colorwheel, retrieval, and the given model checkpoints.
Equal-area swatches, neutral background, randomized swatch positions; method labels are blinded
(random item ids; the key is stored separately). Repeated items are included for reliability.
Output: experiments/study/{images/*.png, items.json, key.json, README.md}
"""
from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from palette.api import Completer
from palette.baselines.colorwheel import complete as cw_complete
from palette.baselines.retrieval import PaletteIndex, PipelinedBaseline
from palette.color import gamut_map, hex_to_srgb, oklab_to_srgb, srgb_to_hex, clip_srgb
from palette.data import load_arrays, ROOT
from palette.eval.probes import PROBES
from palette.eval.report import probe_context

BG = "#c8c8c8"  # fixed neutral background (Oklab L≈0.83)


def render(hex_colors: list[str], path: Path, rng: np.random.Generator) -> list[str]:
    order = rng.permutation(len(hex_colors)).tolist()
    cols = [hex_colors[i] for i in order]
    fig, ax = plt.subplots(figsize=(1.2 * len(cols) + 0.6, 1.8), dpi=150)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis("off")
    for i, c in enumerate(cols):
        ax.add_patch(plt.Rectangle((0.3 + 1.2 * i, 0.3), 1.0, 1.0, color=c))
    ax.set_xlim(0, 1.2 * len(cols) + 0.6); ax.set_ylim(0, 1.6)
    fig.savefig(path, facecolor=BG, bbox_inches="tight", pad_inches=0.1); plt.close(fig)
    return cols


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="*", default=["experiments/mabs/model_ema.pt"])
    ap.add_argument("--per-method", type=int, default=2); ap.add_argument("--repeats", type=int, default=6); ap.add_argument("--out", default="experiments/study")
    a = ap.parse_args()
    out = ROOT / a.out; (out / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    train = load_arrays(("kuler", "mturk2014"), "train", norm=None)
    keep = np.isfinite(train.rating) & (train.rating >= 3.0)
    index = PaletteIndex(train.X[keep].astype(np.float64), train.M[keep])
    from palette.eval.report import load_scorer
    scorer = load_scorer()
    methods = {"colorwheel": lambda ctx, m, k, s: np.stack([gamut_map(c)[0] for c in cw_complete(ctx, m, k, s)]),
               "retrieval+pipeline": PipelinedBaseline(lambda ctx, m, k, s: index.complete(ctx, m, k, s), scorer).complete}
    for mp in a.models:
        c = Completer.from_checkpoint(mp)
        name = Path(mp).parent.name
        methods[name] = (lambda cc: lambda ctx, m, k, s: cc.complete_oklab(ctx, m, k, s)[0])(c)
        # R2 alternative: always complete to five colors, then keep the m best-matching... we keep a random m-subset,
        # since any preference-based subset choice would smuggle the scorer into the stimulus.
        def c2f5(ctx, m, k, s, cc=c):
            need = 5 - len(ctx)
            if need <= m:
                return cc.complete_oklab(ctx, m, k, s)[0]
            full = cc.complete_oklab(ctx, need, k, s)[0]
            rng = np.random.default_rng(s)
            return np.stack([f[rng.choice(need, m, replace=False)] for f in full])
        methods[name + ":complete5-subset"] = c2f5
    items, key = [], {}
    for pi, pr in enumerate(PROBES):
        if not pr["colors"]:
            continue  # unconditional contexts are not completion judgments
        ctx = probe_context(pr)
        for name, fn in methods.items():
            comp = fn(ctx, pr["m"], a.per_method, 7000 + pi)
            for j in range(a.per_method):
                hexes = list(pr["colors"]) + [srgb_to_hex(clip_srgb(oklab_to_srgb(c))) for c in comp[j]]
                iid = secrets.token_hex(4)
                shown = render(hexes, out / "images" / f"{iid}.png", rng)
                items.append({"item_id": iid, "image": f"images/{iid}.png", "fixed": pr["colors"], "n_fixed": len(pr["colors"]), "m": pr["m"], "shown_order": shown})
                key[iid] = {"probe": pr["name"], "method": name, "sample": j, "generated": hexes[len(pr["colors"]):]}
    rng.shuffle(items)
    rep = [dict(it, item_id=it["item_id"] + "-r", repeat_of=it["item_id"]) for it in rng.choice(items, a.repeats, replace=False)]
    items += rep
    json.dump({"background": BG, "questions": {
        "harmony": "How well do these colors go together? (1 = not at all, 7 = extremely well)",
        "liking": "How much do you like this color combination? (1 = not at all, 7 = very much)"},
        "instructions": "Swatches are equal-area on a fixed neutral background; positions are randomized. Judge the whole set.",
        "items": items}, open(out / "items.json", "w"), indent=1)
    json.dump(key, open(out / "key.json", "w"), indent=1)
    (out / "README.md").write_text("Blinded stimulus package. Show items.json to raters (never key.json). "
                                    "Ask harmony and liking as separate questions. Repeated items (suffix -r) estimate within-rater reliability.\n")
    print(f"{len(items)} items ({len(rep)} repeats) from {len(methods)} methods; key in key.json")


if __name__ == "__main__":
    main()
