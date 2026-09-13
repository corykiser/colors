"""Ingest the DesLauriers digitization of Sanzo Wada's *A Dictionary of Colour Combinations* (S6)."""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

from palette.schema import PaletteRecord
from palette.ingest.common import srgb_palette_to_oklab, CONVERSION

URL = "https://github.com/mattdesl/dictionary-of-colour-combinations"
LICENSE = ("MIT (DesLauriers transcription, LICENSE.md). Underlying work: Japan PD since 2018; "
           "U.S. status unresolved (handoff §2.6).")


def ingest_wada(raw_dir: Path) -> list[PaletteRecord]:
    cols = json.load(open(raw_dir / "dictionary-of-colour-combinations/colors.json"))
    by_combo: dict[int, list[int]] = collections.defaultdict(list)
    for i, c in enumerate(cols):
        for cid in c["combinations"]:
            by_combo[cid].append(i)
    out = []
    for cid in sorted(by_combo):
        idx = by_combo[cid]
        rgb01 = np.array([cols[i]["rgb"] for i in idx], dtype=np.float64) / 255.0
        out.append(PaletteRecord(
            record_id=f"wada:{cid}", source_family="wada", source_release="mattdesl v1.0.2 (2020)",
            source_url=URL, source_palette_id=str(cid), license_evidence=LICENSE,
            original_color_space="CMYK",
            original_colors=[[float(v) for v in cols[i]["cmyk"]] for i in idx],
            oklab_colors=srgb_palette_to_oklab(rgb01),
            profile_or_viewing_conditions="CMYK 'U.S. Web Coated (SWOP) v2' -> sRGB IEC61966-2.1, relative colorimetric, BPC (per repo README)",
            conversion_method="CMYK -> sRGB via ICC (repo) -> " + CONVERSION,
            original_order=list(range(len(idx))),
            label_type="curated", rating_question=None,
            text_description="; ".join(cols[i]["name"] for i in idx),
            quality_flags=["cmyk_icc_estimate", "wada_us_status_unresolved", "print_origin"],
            extra={"color_indices": idx, "srgb_from_icc": [cols[i]["rgb"] for i in idx],
                   "swatch_chapters": [cols[i]["swatch"] for i in idx]},
        ))
    return out
