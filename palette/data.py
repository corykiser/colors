"""Load processed parquet into arrays for training / evaluation. Normalization stats come from train only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from palette.color import NormStats, oklab_to_srgb
from palette.sampler import PaletteArrays

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"
GEN_FAMILIES = ("kuler", "mturk2014")


class SrgbNorm(NormStats):
    """All three gamma-sRGB channels mapped [0,1] -> [-1,1]. Same interface as NormStats."""
    def __init__(self):
        object.__setattr__(self, "L_mean", 0.5); object.__setattr__(self, "L_std", 0.5); object.__setattr__(self, "ab_scale", 1.0)

    def normalize(self, x):
        return (np.asarray(x, dtype=np.float64) - 0.5) / 0.5

    def denormalize(self, z):
        return np.asarray(z, dtype=np.float64) * 0.5 + 0.5

    def to_dict(self):
        return {"L_mean": 0.5, "L_std": 0.5, "ab_scale": 1.0, "space": "srgb"}   # default generative set (handoff §4.5); mturk2011 ⊂ mturk2014


def load_family(fam: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(PROC / f"{fam}.parquet", columns=columns)


def _stack(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    nmax = int(df.n_colors.max())
    X = np.zeros((len(df), nmax, 3)); M = np.zeros((len(df), nmax), bool)
    for i, (s, n) in enumerate(zip(df.oklab_colors, df.n_colors)):
        X[i, :n] = json.loads(s); M[i, :n] = True
    return X, M


def load_arrays(families=GEN_FAMILIES, split: str | None = "train", norm: NormStats | None = None,
                nmax: int = 5, space: str = "oklab") -> PaletteArrays:
    """space='srgb' returns gamma-encoded sRGB in [0,1] (clipped) instead of Oklab — the R8/item-8 representation ablation."""
    cols = ["record_id", "source_family", "oklab_colors", "n_colors", "mean_rating", "split"]
    dfs = [load_family(f, cols) for f in families]
    df = pd.concat(dfs, ignore_index=True)
    if split is not None:
        df = df[df.split == split].reset_index(drop=True)
    X, M = _stack(df)
    if space == "srgb":
        X = np.where(M[..., None], np.clip(oklab_to_srgb(X), 0, 1), 0.0)
    if X.shape[1] < nmax:
        X = np.pad(X, ((0, 0), (0, nmax - X.shape[1]), (0, 0))); M = np.pad(M, ((0, 0), (0, nmax - M.shape[1])))
    if norm is not None:
        X = np.where(M[..., None], norm.normalize(X), 0.0)
    return PaletteArrays(X.astype(np.float32), M, df.source_family.to_numpy(),
                         df.mean_rating.to_numpy(dtype=float), df.record_id.to_numpy())


def fit_or_load_norm(families=GEN_FAMILIES, path: Path | None = None, space: str = "oklab") -> NormStats:
    path = path or (PROC / ("norm_stats.json" if space == "oklab" else f"norm_stats_{space}.json"))
    if path.exists():
        return NormStats.from_dict(json.load(open(path)))
    a = load_arrays(families, "train", norm=None, space=space)
    st = NormStats.fit(a.X[a.M]) if space == "oklab" else NormStats(L_mean=0.5, L_std=0.5, ab_scale=1.0)  # srgb: map [0,1] to [-1,1] on ch0, others raw
    if space == "srgb":
        st = SrgbNorm()
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
