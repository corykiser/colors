"""Load processed parquet into arrays for training / evaluation. Normalization stats come from train only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from palette.color import NormStats
from palette.sampler import PaletteArrays

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"
GEN_FAMILIES = ("kuler", "mturk2014")   # default generative set (handoff §4.5); mturk2011 ⊂ mturk2014


def load_family(fam: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(PROC / f"{fam}.parquet", columns=columns)


def _stack(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    nmax = int(df.n_colors.max())
    X = np.zeros((len(df), nmax, 3)); M = np.zeros((len(df), nmax), bool)
    for i, (s, n) in enumerate(zip(df.oklab_colors, df.n_colors)):
        X[i, :n] = json.loads(s); M[i, :n] = True
    return X, M


def load_arrays(families=GEN_FAMILIES, split: str | None = "train", norm: NormStats | None = None,
                nmax: int = 5) -> PaletteArrays:
    cols = ["record_id", "source_family", "oklab_colors", "n_colors", "mean_rating", "split"]
    dfs = [load_family(f, cols) for f in families]
    df = pd.concat(dfs, ignore_index=True)
    if split is not None:
        df = df[df.split == split].reset_index(drop=True)
    X, M = _stack(df)
    if X.shape[1] < nmax:
        X = np.pad(X, ((0, 0), (0, nmax - X.shape[1]), (0, 0))); M = np.pad(M, ((0, 0), (0, nmax - M.shape[1])))
    if norm is not None:
        X = np.where(M[..., None], norm.normalize(X), 0.0)
    return PaletteArrays(X.astype(np.float32), M, df.source_family.to_numpy(),
                         df.mean_rating.to_numpy(dtype=float), df.record_id.to_numpy())


def fit_or_load_norm(families=GEN_FAMILIES, path: Path = PROC / "norm_stats.json") -> NormStats:
    if path.exists():
        return NormStats.from_dict(json.load(open(path)))
    a = load_arrays(families, "train", norm=None)
    st = NormStats.fit(a.X[a.M])
    json.dump(st.to_dict(), open(path, "w"), indent=2)
    return st


def load_ratings(user_splits_path: Path = ROOT / "data/splits/user_splits.json") -> pd.DataFrame:
    """Long table of individual MTurk 2014 ratings with palette split and user split."""
    df = load_family("mturk2014", ["record_id", "oklab_colors", "individual_ratings", "split", "duplicate_group_id", "mean_rating", "extra"])
    us = json.load(open(user_splits_path))
    rows = []
    for rid, lab, ir, sp, g in zip(df.record_id, df.oklab_colors, df.individual_ratings, df.split, df.duplicate_group_id):
        for r in json.loads(ir):
            rows.append((rid, r["user"], r["rating"], sp, us.get(str(r["user"]), "seen_user"), g))
    out = pd.DataFrame(rows, columns=["record_id", "user", "rating", "split", "user_split", "group"])
    return out
