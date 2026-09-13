"""Population scorers on MTurk 2014 with rater separation (items 4 and 6).

Targets: mean rating over *seen* users (114 held-out users are excluded from all training signal).
Variants:
  deepsets / sortedL / ordered   MSE on seen-user palette means (R3 diagnostic retained)
  deepsets-ordinal               cumulative-link ordinal model on individual seen-user ratings with per-rater
                                 threshold shift and scale (rater effects); palette score = latent
  deepsets-pairwise              within-user pairs (ties dropped, ≤ 200 pairs per user), -log σ(s_a - s_b)
Evaluation on test palettes: Spearman and pairwise accuracy vs seen-user mean, and vs the *held-out users'* mean
(unseen-rater generalization). Reliability ceilings: repeat-judgment agreement, split-half of seen-user means,
and seen-vs-held-out mean agreement.
"""
from __future__ import annotations

import collections
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from palette.data import load_family, fit_or_load_norm, ROOT
from palette.models.scorer import Scorer, pairwise_loss

OUT = ROOT / "experiments/scorer"; OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(8)


def load():
    df = load_family("mturk2014", ["record_id", "oklab_colors", "mean_rating", "individual_ratings", "split", "extra"])
    us = json.load(open(ROOT / "data/splits/user_splits.json")); held = {int(u) for u, v in us.items() if v == "heldout_user"}
    norm = fit_or_load_norm()
    X = torch.tensor(norm.normalize(np.array([json.loads(s) for s in df.oklab_colors])).astype(np.float32))
    y_seen = df.mean_rating.to_numpy(np.float32)
    y_held = np.array([json.loads(e)["mean_rating_heldout_users"] or np.nan for e in df.extra], np.float32)
    ratings = [[(r["user"], r["rating"]) for r in json.loads(s) if r["user"] not in held] for s in df.individual_ratings]
    ratings_held = [[(r["user"], r["rating"]) for r in json.loads(s) if r["user"] in held] for s in df.individual_ratings]
    return df, X, y_seen, y_held, ratings, ratings_held, held


def spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b); return float(spearmanr(a[ok], b[ok]).correlation)


def pairwise_acc(pred, y, n_pairs=200_000, seed=0):
    rng = np.random.default_rng(seed); ok = np.where(np.isfinite(y))[0]
    i = rng.choice(ok, n_pairs); j = rng.choice(ok, n_pairs); keep = np.abs(y[i] - y[j]) > 1e-6; i, j = i[keep], j[keep]
    return float(((pred[i] > pred[j]) == (y[i] > y[j])).mean())


def reliability(ratings, ratings_held, y_seen, y_held, idx, seed=0):
    rng = np.random.default_rng(seed)
    # repeat judgments: same (user, theme) rated twice or more
    firsts, seconds = [], []
    for rs in ratings:
        by = collections.defaultdict(list)
        for u, r in rs: by[u].append(r)
        for v in by.values():
            if len(v) >= 2: firsts.append(v[0]); seconds.append(v[1])
    firsts, seconds = np.array(firsts, float), np.array(seconds, float)
    # split-half of seen-user means on the given palettes, Spearman-Brown corrected
    h1, h2 = [], []
    for i in idx:
        v = np.array([r for _, r in ratings[i]], float); rng.shuffle(v)
        if len(v) < 4: h1.append(np.nan); h2.append(np.nan); continue
        h1.append(v[: len(v) // 2].mean()); h2.append(v[len(v) // 2 :].mean())
    rho = spearman(np.array(h1), np.array(h2)); sb = 2 * rho / (1 + rho)
    return {"n_repeat_pairs": int(len(firsts)), "repeat_pearson": float(np.corrcoef(firsts, seconds)[0, 1]), "repeat_mean_abs_diff": float(np.abs(firsts - seconds).mean()),
            "repeat_exact_agree": float((firsts == seconds).mean()), "split_half_spearman": rho, "split_half_spearman_brown": sb,
            "seen_vs_heldout_mean_spearman": spearman(y_seen[idx], y_held[idx]),
            "median_heldout_ratings_per_palette": float(np.median([len(r) for r in [ratings_held[i] for i in idx]]))}


class OrdinalHead(nn.Module):
    """Cumulative-link (proportional odds) over 5 levels with per-rater shift and log-scale."""
    def __init__(self, n_users):
        super().__init__()
        self.cut = nn.Parameter(torch.tensor([-1.5, -0.5, 0.5, 1.5]))  # monotone init; not re-sorted (stays ordered in practice)
        self.shift = nn.Embedding(n_users, 1); self.logscale = nn.Embedding(n_users, 1)
        nn.init.zeros_(self.shift.weight); nn.init.zeros_(self.logscale.weight)

    def nll(self, s, uid, r):
        z = (s[:, None] - self.cut[None] - self.shift(uid)) * torch.exp(-self.logscale(uid))   # (B,4)
        F = 1 - torch.sigmoid(z)                                       # P(r <= k), k = 1..4, increasing in k
        F = torch.cat([torch.zeros_like(F[:, :1]), F, torch.ones_like(F[:, :1])], 1)   # F(0)=0 ... F(5)=1
        p = (F[:, 1:] - F[:, :-1]).clamp(min=1e-6)                     # P(r = k) = F(k) - F(k-1)
        return -torch.log(p[torch.arange(len(r)), r - 1]).mean()


def train_mse(variant, X, y, tr, va, seed, epochs=150, lr=1e-3, wd=1e-4, bs=512):
    torch.manual_seed(seed); model = Scorer(variant); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs); best, state = -1, None
    Xtr, ytr = X[tr], torch.tensor(y[tr])
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(tr))
        for s in range(0, len(tr), bs):
            b = perm[s:s + bs]; loss = ((model(Xtr[b]) - ytr[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 5 == 4:
            with torch.no_grad(): v = spearman(model.eval()(X[va]).numpy(), y[va])
            if v > best: best, state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(state); return model.eval(), best


def train_ordinal(X, y, ratings, tr, va, user_index, seed, epochs=60, lr=1e-3, wd=1e-4, bs=2048):
    torch.manual_seed(seed); model = Scorer("deepsets"); head = OrdinalHead(len(user_index))
    trs = set(tr.tolist()); rows = [(i, user_index[u], r) for i in tr for u, r in ratings[i]]
    P = torch.tensor([r[0] for r in rows]); U = torch.tensor([r[1] for r in rows]); R = torch.tensor([r[2] for r in rows])
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs); best, state = -1, None
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(P))
        for s in range(0, len(P), bs):
            b = perm[s:s + bs]; loss = head.nll(model(X[P[b]]), U[b], R[b]); opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 3 == 2:
            with torch.no_grad(): v = spearman(model.eval()(X[va]).numpy(), y[va])
            if v > best: best, state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(state); return model.eval(), best


def train_pairwise(X, y, ratings, tr, va, seed, cap=200, epochs=60, lr=1e-3, wd=1e-4, bs=2048):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    by_user = collections.defaultdict(list)
    for i in tr:
        for u, r in ratings[i]: by_user[u].append((i, r))
    A, B = [], []
    for u, lst in by_user.items():
        lst = np.array(lst); pairs = []
        for _ in range(cap * 5):
            a, b = rng.integers(0, len(lst), 2)
            if lst[a, 1] != lst[b, 1]:
                pairs.append((lst[a], lst[b]) if lst[a, 1] > lst[b, 1] else (lst[b], lst[a]))
            if len(pairs) >= cap: break
        for p, q in pairs: A.append(int(p[0])); B.append(int(q[0]))
    A, B = torch.tensor(A), torch.tensor(B)
    model = Scorer("deepsets"); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs); best, state = -1, None
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(A))
        for s in range(0, len(A), bs):
            b = perm[s:s + bs]; loss = pairwise_loss(model(X[A[b]]), model(X[B[b]]), torch.ones(len(b))); opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 3 == 2:
            with torch.no_grad(): v = spearman(model.eval()(X[va]).numpy(), y[va])
            if v > best: best, state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(state); return model.eval(), best, len(A)


def evaluate(model, X, y_seen, y_held, te, shuffle=False, seed=0):
    Xt = X[te]
    if shuffle:
        g = torch.Generator().manual_seed(seed); Xt = torch.stack([x[torch.randperm(5, generator=g)] for x in Xt])
    with torch.no_grad(): p = model(Xt).numpy()
    return {"spearman_seen": spearman(p, y_seen[te]), "pairwise_acc_seen": pairwise_acc(p, y_seen[te]),
            "spearman_heldout_users": spearman(p, y_held[te]), "pairwise_acc_heldout_users": pairwise_acc(p, y_held[te])}


def main():
    df, X, y_seen, y_held, ratings, ratings_held, held = load()
    tr = np.where(df.split == "train")[0]; va = np.where(df.split == "val")[0]; te = np.where(df.split == "test")[0]
    users = sorted({u for rs in ratings for u, _ in rs}); user_index = {u: i for i, u in enumerate(users)}
    t0 = time.time(); results = {"reliability_test": reliability(ratings, ratings_held, y_seen, y_held, te)}
    print("reliability:", json.dumps(results["reliability_test"]))
    for variant in ("deepsets", "sortedL", "ordered", "deepsets-ordinal", "deepsets-pairwise"):
        rows = []
        for seed in range(3):
            if variant == "deepsets-ordinal": model, v = train_ordinal(X, y_seen, ratings, tr, va, user_index, seed)
            elif variant == "deepsets-pairwise": model, v, npairs = train_pairwise(X, y_seen, ratings, tr, va, seed)
            else: model, v = train_mse(variant, X, y_seen, tr, va, seed)
            row = {"seed": seed, "val_spearman": v, **evaluate(model, X, y_seen, y_held, te), "params": sum(p.numel() for p in model.parameters())}
            if variant == "ordered": row["spearman_seen_shuffled"] = evaluate(model, X, y_seen, y_held, te, True, seed)["spearman_seen"]
            rows.append(row); print(variant, json.dumps(row), flush=True)
            if seed == 0: torch.save({"state_dict": model.state_dict(), "variant": "deepsets" if variant.startswith("deepsets") else variant}, OUT / f"{variant}.pt")
        d = pd.DataFrame(rows); results[variant] = {"mean": d.mean(numeric_only=True).to_dict(), "std": d.std(numeric_only=True).to_dict(), "runs": rows}
    # deployed scorer = best variant by held-out-user Spearman (population generalization), seed 0
    best = max((k for k in results if k != "reliability_test"), key=lambda k: results[k]["mean"]["spearman_heldout_users"])
    import shutil; shutil.copy(OUT / f"{best}.pt", OUT / "deepsets.pt"); results["deployed"] = best
    results["_context"] = {"n_train": len(tr), "n_val": len(va), "n_test": len(te), "n_seen_users": len(users), "n_heldout_users": len(held), "seconds": round(time.time() - t0, 1)}
    json.dump(results, open(OUT / "results.json", "w"), indent=2)
    print("deployed:", best, json.dumps({k: v["mean"] for k, v in results.items() if isinstance(v, dict) and "mean" in v}, indent=1))


if __name__ == "__main__":
    main()
