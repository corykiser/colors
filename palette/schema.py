"""Record schema (§2.8 of the handoff) as a validated dataclass, plus parquet I/O.

Missing information stays missing (None / empty). No reassuring defaults.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd

LABEL_TYPES = {"curated", "preference", "harmony", "popularity", "semantic"}
COLOR_SPACES = {"sRGB", "CMYK", "CIELAB", "other"}


@dataclass
class PaletteRecord:
    record_id: str
    source_family: str
    source_release: str
    source_url: str
    original_color_space: str
    original_colors: list[list[float]]
    oklab_colors: list[list[float]]
    label_type: str
    source_palette_id: str | None = None
    license_evidence: str | None = None
    commercial_reuse_status: str = "unreviewed"
    profile_or_viewing_conditions: str | None = None
    conversion_method: str | None = None
    original_order: list[int] = field(default_factory=list)
    area_weights: list[float] | None = None
    rating_question: str | None = None
    individual_ratings: list[dict[str, Any]] = field(default_factory=list)
    rating_count: int | None = None
    mean_rating: float | None = None
    rating_scale: str | None = None
    text_description: str | None = None
    duplicate_group_id: str | None = None
    quality_flags: list[str] = field(default_factory=list)
    split: str = "unassigned"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label_type not in LABEL_TYPES:
            raise ValueError(f"{self.record_id}: bad label_type {self.label_type!r}")
        if self.original_color_space not in COLOR_SPACES:
            raise ValueError(f"{self.record_id}: bad original_color_space {self.original_color_space!r}")
        n = len(self.oklab_colors)
        if n < 1:
            raise ValueError(f"{self.record_id}: empty palette")
        if any(len(c) != 3 for c in self.oklab_colors):
            raise ValueError(f"{self.record_id}: oklab colors must have 3 components")
        if len(self.original_colors) != n:
            raise ValueError(f"{self.record_id}: original/oklab length mismatch")
        if not self.original_order:
            self.original_order = list(range(n))
        if sorted(self.original_order) != list(range(n)):
            raise ValueError(f"{self.record_id}: original_order must be a permutation of 0..n-1")
        if self.area_weights is not None and len(self.area_weights) != n:
            raise ValueError(f"{self.record_id}: area_weights length mismatch")
        if self.rating_count is not None and self.individual_ratings and self.rating_count != len(self.individual_ratings):
            raise ValueError(f"{self.record_id}: rating_count disagrees with individual_ratings")
        for c in self.oklab_colors:
            if not all(np.isfinite(c)):
                raise ValueError(f"{self.record_id}: non-finite oklab color")

    @property
    def n_colors(self) -> int:
        return len(self.oklab_colors)


# --- parquet serialization -------------------------------------------------------
_JSON_COLS = ("original_colors", "oklab_colors", "original_order", "area_weights",
              "individual_ratings", "quality_flags", "extra")


def records_to_frame(records: list[PaletteRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        d = asdict(r)
        for k in _JSON_COLS:
            d[k] = json.dumps(d[k])
        d["n_colors"] = r.n_colors
        rows.append(d)
    return pd.DataFrame(rows)


def frame_to_records(df: pd.DataFrame) -> list[PaletteRecord]:
    out = []
    cols = [f for f in PaletteRecord.__dataclass_fields__]
    for row in df.to_dict("records"):
        d = {k: row[k] for k in cols if k in row}
        for k in _JSON_COLS:
            if k in d and isinstance(d[k], str):
                d[k] = json.loads(d[k])
        for k in ("source_palette_id", "license_evidence", "profile_or_viewing_conditions",
                  "conversion_method", "rating_question", "text_description", "duplicate_group_id",
                  "rating_scale"):
            if k in d and (d[k] is None or (isinstance(d[k], float) and np.isnan(d[k]))):
                d[k] = None
        for k in ("rating_count",):
            if k in d and d[k] is not None and not (isinstance(d[k], float) and np.isnan(d[k])):
                d[k] = int(d[k])
            elif k in d:
                d[k] = None
        if "mean_rating" in d and d["mean_rating"] is not None and np.isnan(d["mean_rating"]):
            d["mean_rating"] = None
        out.append(PaletteRecord(**d))
    return out


def write_parquet(records: list[PaletteRecord], path: str) -> None:
    records_to_frame(records).to_parquet(path, index=False)


def read_parquet(path: str) -> list[PaletteRecord]:
    return frame_to_records(pd.read_parquet(path))


def oklab_array(records: list[PaletteRecord], max_n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Pad to (N, max_n, 3) plus boolean mask (N, max_n)."""
    if max_n is None:
        max_n = max(r.n_colors for r in records)
    X = np.zeros((len(records), max_n, 3))
    M = np.zeros((len(records), max_n), dtype=bool)
    for i, r in enumerate(records):
        n = r.n_colors
        X[i, :n] = r.oklab_colors
        M[i, :n] = True
    return X, M
