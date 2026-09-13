"""Phase 2 baselines (G2): colorwheel, retrieval, conditional KDE."""
from __future__ import annotations

import json

import numpy as np

from palette.baselines.colorwheel import complete as cw_complete
from palette.baselines.retrieval import PaletteIndex, PipelinedBaseline
from palette.data import load_arrays, ROOT
from palette.eval.report import evaluate_generator, load_scorer

OUT = ROOT / "experiments/baselines"; OUT.mkdir(parents=True, exist_ok=True)


def main():
    train = load_arrays(("kuler", "mturk2014"), "train", norm=None)
    val = load_arrays(("kuler", "mturk2014"), "val", norm=None)
    index = PaletteIndex(train.X.astype(np.float64), train.M)
    keep = np.isfinite(train.rating) & (train.rating >= 3.0)          # same filter as the diffusion training set
    index_f = PaletteIndex(train.X[keep].astype(np.float64), train.M[keep])
    score = load_scorer()
    retr_f = lambda ctx, m, k, s: index_f.complete(ctx, m, k, s)
    methods = {
        "colorwheel": lambda ctx, m, k, s: cw_complete(ctx, m, k, s),
        "retrieval": lambda ctx, m, k, s: index.complete(ctx, m, k, s),
        "retrieval_filtered": retr_f,
        "retrieval_filtered+pipeline": PipelinedBaseline(retr_f, score).complete,
        "kde": lambda ctx, m, k, s: index.kde_complete(ctx, m, k, seed=s),
    }
    results = {name: evaluate_generator(fn, val.X[val.M.sum(1) == 5], score_fn=score) for name, fn in methods.items()}
    json.dump({"n_index": len(index.X), "n_index_filtered": len(index_f.X), "results": results}, open(OUT / "metrics.json", "w"), indent=1)
    for name, r in results.items():
        print(name, json.dumps(r["summary"]))


if __name__ == "__main__":
    main()
