"""Run dedup across all processed families, assign splits, write back + dataset card."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from palette.dedup import find_duplicate_groups, cross_split_near_pairs
from palette.splits import assign_group_splits, assign_user_split

ROOT = Path(__file__).resolve().parents[1]
PROC, SPL = ROOT / "data/processed", ROOT / "data/splits"
FAMILIES = ["mturk2014", "mturk2011", "kuler", "colourlovers", "wada"]


def main(seed: int = 0) -> None:
    SPL.mkdir(exist_ok=True)
    frames = {f: pd.read_parquet(PROC / f"{f}.parquet") for f in FAMILIES}
    df = pd.concat(frames.values(), ignore_index=True)
    nmax = int(df.n_colors.max())
    X = np.zeros((len(df), nmax, 3)); M = np.zeros((len(df), nmax), bool)
    for i, (s, n) in enumerate(zip(df.oklab_colors, df.n_colors)):
        X[i, :n] = json.loads(s); M[i, :n] = True
    t0 = time.time()
    group, stats = find_duplicate_groups(X, M)
    stats["seconds"] = round(time.time() - t0, 1)
    print("dedup:", stats)
    split = assign_group_splits(group, seed=seed)
    df["duplicate_group_id"] = [f"dg:{g}" for g in group]
    df["split"] = split
    # cross-family group report
    fam_by_group = df.groupby("duplicate_group_id").source_family.agg(lambda s: tuple(sorted(set(s))))
    cross = fam_by_group[fam_by_group.map(len) > 1].value_counts().head(12)
    # write back per family
    for f in FAMILIES:
        sub = df[df.source_family == f]
        sub.to_parquet(PROC / f"{f}.parquet", index=False)
    df[["record_id", "source_family", "duplicate_group_id", "split", "n_colors"]].to_parquet(SPL / "palette_splits.parquet", index=False)
    # user split for mturk2014, then recompute mean ratings *without* held-out raters (rater separation)
    users = sorted({r["user"] for s in frames["mturk2014"].individual_ratings for r in json.loads(s)})
    usplit = assign_user_split(users, seed=seed)
    json.dump(usplit, open(SPL / "user_splits.json", "w"))
    held = {u for u, v in usplit.items() if v == "heldout_user"}
    mt = df.source_family == "mturk2014"
    new_mean, new_cnt, extra = [], [], []
    for ir, ex, old in zip(df.loc[mt, "individual_ratings"], df.loc[mt, "extra"], df.loc[mt, "mean_rating"]):
        rs = json.loads(ir); seen = [r["rating"] for r in rs if r["user"] not in held]; ho = [r["rating"] for r in rs if r["user"] in held]
        e = json.loads(ex); e["mean_rating_all_users"] = old; e["mean_rating_heldout_users"] = float(np.mean(ho)) if ho else None
        e["n_seen_user_ratings"] = len(seen); e["n_heldout_user_ratings"] = len(ho)
        new_mean.append(float(np.mean(seen)) if seen else None); extra.append(json.dumps(e))
    df.loc[mt, "mean_rating"] = new_mean; df.loc[mt, "extra"] = extra
    changed = int(((df.loc[mt, "mean_rating"] >= 3) != (pd.Series([json.loads(e)["mean_rating_all_users"] for e in extra], index=df.index[mt]) >= 3)).sum())
    print(f"rater separation: {len(held)} held-out users removed from mturk2014 means; >=3 decision changed for {changed} palettes")
    # leakage audit: exact count of cross-split near-duplicate pairs (should be 0 with guaranteed-recall blocking)
    audit = {f"{a}-{b}": cross_split_near_pairs(X, M, split, a=a, b=b) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    print("cross-split near pairs:", audit)
    # dataset card
    card = {
        "seed": seed, "dedup": stats, "cross_split_near_pairs": audit, "large_groups_forced_to_train": 100,
        "rater_separation": {"heldout_users": len(held), "ge3_decision_changed": changed},
        "per_family": {f: {"records": int((df.source_family == f).sum()),
                           "groups": int(df[df.source_family == f].duplicate_group_id.nunique()),
                           "by_split": df[df.source_family == f].split.value_counts().to_dict(),
                           "by_size": df[df.source_family == f].n_colors.value_counts().sort_index().to_dict()} for f in FAMILIES},
        "cross_family_groups": {" + ".join(k): int(v) for k, v in cross.items()},
        "users_mturk2014": {"n": len(users), "heldout": sum(v == "heldout_user" for v in usplit.values())},
    }
    for f in FAMILIES:
        df[df.source_family == f].to_parquet(PROC / f"{f}.parquet", index=False)
    json.dump(card, open(ROOT / "data/manifest/dataset_card.json", "w"), indent=2)
    print(json.dumps(card, indent=1))


if __name__ == "__main__":
    main()
