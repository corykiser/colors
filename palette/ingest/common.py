from __future__ import annotations

import numpy as np

from palette import color as C

CONVERSION = "sRGB (assumed) -> linear (IEC 61966-2-1 EOTF) -> Oklab (Ottosson matrices); palette.color"


def srgb_palette_to_oklab(rgb01: np.ndarray) -> list[list[float]]:
    lab = C.srgb_to_oklab(np.asarray(rgb01, dtype=np.float64))
    return [[float(v) for v in row] for row in lab]


def mat_str(x) -> str | None:
    """MATLAB cell string -> python str (None if empty)."""
    try:
        a = np.asarray(x).ravel()
        if a.size == 0:
            return None
        s = a[0]
        if isinstance(s, np.ndarray):
            s = s.ravel()[0] if s.size else None
        return None if s is None else str(s)
    except Exception:
        return None
