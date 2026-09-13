"""Ingest the O'Donovan 2011 (S2) and 2014 (S1) releases.

Families produced:
  kuler          S2 kulerData.mat      preference (mean Kuler star rating, rater count unknown)
  mturk2011      S2 mturkData.mat      preference (mean of ~40 MTurk ratings; no individual ratings)
  colourlovers   S2 colorLoversData.mat popularity (hearts/views; 'targets' is an undocumented derived score)
  mturk2014      S1 themeData + allMTurkRatings   preference with individual ratings and user ids

Kuler and COLOURlovers numeric IDs collide; record_ids are site-prefixed.
"""
from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
import scipy.io as sio

from palette.schema import PaletteRecord
from palette.ingest.common import srgb_palette_to_oklab, mat_str, CONVERSION

S2_URL = "https://www.dgp.toronto.edu/~donovan/color/"
S1_URL = "https://www.dgp.toronto.edu/~donovan/cfcolor/"
S2_LICENSE = "CC BY-NC-SA 2.5 CA (colorCode/README.txt; project page)"
S1_LICENSE = ("Permissive program notice (runPrediction.m, project page); rating data treated as "
              "CC BY-NC-SA 2.5 CA — same MTurk study as the 2011 release. See AUDIT.md §4.")
MTURK_QUESTION = ("liking, 1-5 integer scale, Amazon Mechanical Turk; per O'Donovan et al. 2011 the theme was "
                  "shown as a horizontal five-swatch strip in stored order (presentation not in archive)")
ASSUMED = "colorspace_assumed_srgb"


def _common(rgb01: np.ndarray, order_n: int) -> dict:
    return dict(
        original_color_space="sRGB",
        original_colors=[[float(v) for v in row] for row in rgb01],
        oklab_colors=srgb_palette_to_oklab(rgb01),
        conversion_method=CONVERSION,
        original_order=list(range(order_n)),
        profile_or_viewing_conditions=None,
    )


def ingest_kuler(raw_dir: Path) -> list[PaletteRecord]:
    m = sio.loadmat(raw_dir / "colorCode/data/kulerData.mat")
    data, ids, targets, names = m["data"], m["ids"].ravel(), m["targets"].ravel(), m["names"].ravel()
    out = []
    for i in range(len(ids)):
        out.append(PaletteRecord(
            record_id=f"kuler:{int(ids[i])}", source_family="kuler", source_release="colorCode.zip (2011)",
            source_url=S2_URL, source_palette_id=str(int(ids[i])), license_evidence=S2_LICENSE,
            label_type="preference", rating_question="Kuler site star rating (1-5), context not in archive",
            mean_rating=float(targets[i]), rating_count=None, rating_scale="1-5",
            text_description=mat_str(names[i]), quality_flags=[ASSUMED, "rater_count_unknown", "exposure_biased"],
            **_common(data[i], 5)))
    return out


def ingest_mturk2011(raw_dir: Path) -> list[PaletteRecord]:
    m = sio.loadmat(raw_dir / "colorCode/data/mturkData.mat")
    data, ids, targets, names = m["data"], m["ids"].ravel(), m["targets"].ravel(), m["names"].ravel()
    unt = m["userNormalizedTargets"].ravel()
    seen = collections.Counter()
    out = []
    for i in range(len(ids)):
        k = int(ids[i]); seen[k] += 1
        rid = f"mturk2011:{k}" if seen[k] == 1 else f"mturk2011:{k}:b{seen[k]}"
        flags = [ASSUMED, "mean_only_no_individual_ratings"]
        if seen[k] > 1:
            flags.append("repeated_theme_second_batch")
        out.append(PaletteRecord(
            record_id=rid, source_family="mturk2011", source_release="colorCode.zip (2011)",
            source_url=S2_URL, source_palette_id=str(k), license_evidence=S2_LICENSE,
            label_type="preference", rating_question=MTURK_QUESTION, mean_rating=float(targets[i]),
            rating_count=None, rating_scale="1-5", text_description=mat_str(names[i]),
            quality_flags=flags, extra={"user_normalized_target": float(unt[i])},
            **_common(data[i], 5)))
    return out


def ingest_colourlovers(raw_dir: Path) -> list[PaletteRecord]:
    m = sio.loadmat(raw_dir / "colorCode/data/colorLoversData.mat")
    data, ids, names = m["data"], m["ids"].ravel(), m["names"].ravel()
    hearts, views, targets = m["hearts"].ravel(), m["views"].ravel(), m["targets"].ravel()
    out = []
    for i in range(len(ids)):
        rgb01 = data[i].astype(np.float64) / 255.0
        out.append(PaletteRecord(
            record_id=f"colourlovers:{int(ids[i])}", source_family="colourlovers",
            source_release="colorCode.zip (2011)", source_url=S2_URL, source_palette_id=str(int(ids[i])),
            license_evidence=S2_LICENSE, label_type="popularity",
            rating_question="COLOURlovers hearts and views (site interaction counts, exposure-dependent)",
            mean_rating=None, rating_count=None, text_description=mat_str(names[i]),
            quality_flags=[ASSUMED, "names_field_ambiguous", "derived_target_undocumented", "exposure_biased"],
            extra={"hearts": int(hearts[i]), "views": int(views[i]), "cl_target_undocumented": float(targets[i])},
            **_common(rgb01, 5)))
    return out


def ingest_mturk2014(raw_dir: Path) -> tuple[list[PaletteRecord], dict]:
    """Returns records plus a user table (demographic codes, uninterpreted)."""
    td = sio.loadmat(raw_dir / "cfcolor/release/themeData.mat", squeeze_me=True, struct_as_record=False)["datapoints"]
    r = sio.loadmat(raw_dir / "cfcolor/release/allMTurkRatings.mat")
    ids = np.asarray(td.ids).ravel(); rgb = np.asarray(td.rgb).reshape(-1, 5, 3); names = np.asarray(td.names).ravel()
    tr = r["testRatings"].astype(np.int64)  # (user, theme_index_1based, rating)
    per_theme: dict[int, list] = collections.defaultdict(list)
    for u, t, v in tr:
        per_theme[int(t)].append({"user": int(u), "rating": int(v)})
    seen = collections.Counter()
    out = []
    for i in range(len(ids)):
        k = int(ids[i]); seen[k] += 1
        ratings = per_theme[i + 1]
        vals = np.array([x["rating"] for x in ratings], dtype=float)
        rid = f"mturk2014:{i + 1}"
        flags = [ASSUMED, "individual_ratings_available"]
        if seen[k] > 1:
            flags.append("repeated_theme_second_batch")
        out.append(PaletteRecord(
            record_id=rid, source_family="mturk2014", source_release="cfcolor.zip (2014)", source_url=S1_URL,
            source_palette_id=str(k), license_evidence=S1_LICENSE, label_type="preference",
            rating_question=MTURK_QUESTION, individual_ratings=ratings, rating_count=len(ratings),
            mean_rating=float(vals.mean()) if len(vals) else None, rating_scale="1-5",
            text_description=str(names[i]) if names[i] is not None else None, quality_flags=flags,
            extra={"theme_index": i + 1, "rating_variance": float(np.asarray(td.variance).ravel()[i])},
            **_common(rgb[i], 5)))
    users = {
        "note": "demographic codes are integers whose legend is not in the archive; do not interpret",
        "user_list": r["userList"].ravel().astype(int).tolist(),
        "age": r["ageUsers"].ravel().astype(int).tolist(),
        "gender": r["genderUsers"].ravel().astype(int).tolist(),
        "country": r["countryUsers"].ravel().astype(int).tolist(),
        "experience": r["experienceUsers"].ravel().astype(int).tolist(),
    }
    return out, users
