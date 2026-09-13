"""Run dedup across all processed families, assign splits, write back + dataset card."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from palette.dedup import find_duplicate_groups
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
    # user split for mturk2014
    users = sorted({r["user"] for s in frames["mturk2014"].individual_ratings for r in json.loads(s)})
    usplit = assign_user_split(users, seed=seed)
    json.dump(usplit, open(SPL / "user_splits.json", "w"))
    # dataset card
    card = {
        "seed": seed, "dedup": stats,
        "per_family": {f: {"records": int((df.source_family == f).sum()),
                           "groups": int(df[df.source_family == f].duplicate_group_id.nunique()),
                           "by_split": df[df.source_family == f].split.value_counts().to_dict(),
                           "by_size": df[df.source_family == f].n_colors.value_counts().sort_index().to_dict()} for f in FAMILIES},
        "cross_family_groups": {" + ".join(k): int(v) for k, v in cross.items()},
        "users_mturk2014": {"n": len(users), "heldout": sum(v == "heldout_user" for v in usplit.values())},
    }
    json.dump(card, open(ROOT / "data/manifest/dataset_card.json", "w"), indent=2)
    print(json.dumps(card, indent=1))


if __name__ == "__main__":
    main()
