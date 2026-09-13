"""CLI: uv run python scripts/sample.py --model experiments/mabs/model_ema.pt --colors "#9caf88,#6b4a2b" --m 2 --k 4"""
from __future__ import annotations

import argparse, json
from palette.api import Completer


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--colors", default="")
    ap.add_argument("--m", type=int, default=2); ap.add_argument("--k", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=100); ap.add_argument("--rating", type=float, default=None); ap.add_argument("--cfg", type=float, default=None)
    a = ap.parse_args()
    colors = [c.strip() for c in a.colors.split(",") if c.strip()]
    c = Completer.from_checkpoint(a.model, steps=a.steps, rating=a.rating, cfg_scale=a.cfg)
    for pal in c.complete(colors, a.m, a.k, a.seed):
        print(" ".join(pal))


if __name__ == "__main__":
    main()
