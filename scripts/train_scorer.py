"""Train population scorers on MTurk 2014 mean ratings; R3 diagnostic (ordered vs unordered features).

Regression to mean rating (MSE) on the train split, model selection on val, report on test:
Spearman vs mean rating and pairwise accuracy on all test pairs with different means.
Also reports the 'ordered' scorer's test performance when the presentation order is shuffled.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from palette.data import load_family, fit_or_load_norm, ROOT
from palette.models.scorer import Scorer

OUT = ROOT / "experiments/scorer"; OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(8)


def load():
    df = load_family("mturk2014", ["record_id", "oklab_colors", "mean_rating", "split", "extra"])
    norm = fit_or_load_norm()
    X = np.array([json.loads(s) for s in df.oklab_colors])            # stored (presentation) order
    Xn = norm.normalize(X).astype(np.float32)
    y = df.mean_rating.to_numpy(np.float32)
    var = np.array([json.loads(e)["rating_variance"] for e in df.extra], np.float32)
    return {s: (torch.tensor(Xn[df.split == s]), torch.tensor(y[df.split == s])) for s in ("train", "val", "test")}, var[df.split == "test"]


def pairwise_acc(pred, y, n_pairs=200_000, seed=0):
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(y), n_pairs); j = rng.integers(0, len(y), n_pairs)
    keep = np.abs(y[i] - y[j]) > 1e-6
    i, j = i[keep], j[keep]
    return float(((pred[i] > pred[j]) == (y[i] > y[j])).mean())


def evaluate(model, X, y, shuffle=False, seed=0):
    model.eval()
    with torch.no_grad():
        if shuffle:
            g = torch.Generator().manual_seed(seed)
            X = torch.stack([x[torch.randperm(5, generator=g)] for x in X])
        p = model(X).numpy()
    yn = y.numpy()
    return {"spearman": float(spearmanr(p, yn).correlation), "pairwise_acc": pairwise_acc(p, yn),
            "mae": float(np.abs(p - yn).mean())}


def train_one(variant, data, seed, epochs=150, lr=1e-3, wd=1e-4, bs=512):
    torch.manual_seed(seed)
    Xtr, ytr = data["train"]; Xva, yva = data["val"]
    model = Scorer(variant)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state = -1, None
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xtr))
        for s in range(0, len(Xtr), bs):
            idx = perm[s:s + bs]
            loss = ((model(Xtr[idx]) - ytr[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 5 == 4:
            v = evaluate(model, Xva, yva)["spearman"]
            if v > best:
                best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best


def main():
    data, test_var = load()
    results = {}
    t0 = time.time()
    for variant in ("deepsets", "sortedL", "ordered"):
        rows = []
        for seed in range(3):
            model, val_sp = train_one(variant, data, seed)
            te = evaluate(model, *data["test"])
            row = {"seed": seed, "val_spearman": val_sp, **{f"test_{k}": v for k, v in te.items()},
                   "params": sum(p.numel() for p in model.parameters())}
            if variant == "ordered":
                row["test_spearman_shuffled_order"] = evaluate(model, *data["test"], shuffle=True, seed=seed)["spearman"]
            rows.append(row)
            if variant == "deepsets" and seed == 0:
                torch.save({"state_dict": model.state_dict(), "variant": variant}, OUT / "deepsets.pt")
            print(variant, row, flush=True)
        df = pd.DataFrame(rows)
        results[variant] = {"mean": df.mean(numeric_only=True).to_dict(), "std": df.std(numeric_only=True).to_dict(), "runs": rows}
    # noise ceiling: split-half reliability of the mean rating is not available without raw per-user
    # re-aggregation; report mean per-theme rating variance as context instead.
    results["_context"] = {"test_mean_rating_variance": float(test_var.mean()), "n_train": len(data["train"][1]),
                           "n_val": len(data["val"][1]), "n_test": len(data["test"][1]), "seconds": round(time.time() - t0, 1)}
    json.dump(results, open(OUT / "results.json", "w"), indent=2)
    print(json.dumps({k: v["mean"] if k != "_context" else v for k, v in results.items()}, indent=1))


if __name__ == "__main__":
    main()
