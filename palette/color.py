"""Color math: sRGB <-> linear <-> Oklab, gamut tests, gamut mapping, normalization.

Implements Ottosson's published Oklab matrices directly (no library defaults).
All functions are vectorized over leading dimensions; the last axis is the channel axis.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# --- Ottosson, "A perceptual color space for image processing" -------------------
# linear sRGB -> LMS
_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
# LMS' (cube root) -> Oklab
_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)
# Oklab -> LMS'
_M2_INV = np.array(
    [
        [1.0, 0.3963377774, 0.2158037573],
        [1.0, -0.1055613458, -0.0638541728],
        [1.0, -0.0894841775, -1.2914855480],
    ]
)
# LMS -> linear sRGB
_M1_INV = np.array(
    [
        [4.0767416621, -3.3077115913, 0.2309699292],
        [-1.2684380046, 2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147, 1.7076147010],
    ]
)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """sRGB EOTF (IEC 61966-2-1). Input in [0, 1] nominally; extrapolates outside."""
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, np.sign(c) * ((np.abs(c) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """Inverse sRGB EOTF. Extrapolates outside [0, 1] (needed for gamut search)."""
    c = np.asarray(c, dtype=np.float64)
    return np.where(
        c <= 0.0031308, 12.92 * c, np.sign(c) * (1.055 * np.abs(c) ** (1 / 2.4) - 0.055)
    )


def linear_to_oklab(rgb_lin: np.ndarray) -> np.ndarray:
    lms = np.asarray(rgb_lin, dtype=np.float64) @ _M1.T
    lms_ = np.cbrt(lms)
    return lms_ @ _M2.T


def oklab_to_linear(lab: np.ndarray) -> np.ndarray:
    lms_ = np.asarray(lab, dtype=np.float64) @ _M2_INV.T
    lms = lms_**3
    return lms @ _M1_INV.T


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """sRGB in [0,1] (…, 3) -> Oklab (…, 3) with L in [0,1]."""
    return linear_to_oklab(srgb_to_linear(rgb))


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Oklab (…, 3) -> sRGB (…, 3). Not clipped; use in_gamut / gamut_map."""
    return linear_to_srgb(oklab_to_linear(lab))


def hex_to_srgb(h: str) -> np.ndarray:
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    if len(h) != 6:
        raise ValueError(f"bad hex color {h!r}")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float64) / 255.0


def srgb_to_hex(rgb: np.ndarray) -> str:
    q = np.clip(np.round(np.asarray(rgb, dtype=np.float64) * 255.0), 0, 255).astype(int)
    if q.shape != (3,):
        raise ValueError("srgb_to_hex takes a single color")
    return "#{:02x}{:02x}{:02x}".format(*q)


def oklab_to_lch(lab: np.ndarray) -> np.ndarray:
    """Cylindrical form (L, C, h[rad]) — analysis and gamut mapping only."""
    lab = np.asarray(lab, dtype=np.float64)
    C = np.hypot(lab[..., 1], lab[..., 2])
    h = np.arctan2(lab[..., 2], lab[..., 1])
    return np.stack([lab[..., 0], C, h], axis=-1)


def lch_to_oklab(lch: np.ndarray) -> np.ndarray:
    lch = np.asarray(lch, dtype=np.float64)
    return np.stack(
        [lch[..., 0], lch[..., 1] * np.cos(lch[..., 2]), lch[..., 1] * np.sin(lch[..., 2])], axis=-1
    )


def in_gamut(lab: np.ndarray, tol: float = 1e-4) -> np.ndarray:
    """Boolean (…,) — True if the Oklab color maps to sRGB within [0-tol, 1+tol].

    Checks the *linear* channels; the sign-extended transfer functions keep
    out-of-range values out of range, so either domain gives the same answer.
    """
    lin = oklab_to_linear(lab)
    return np.all((lin >= -tol) & (lin <= 1.0 + tol), axis=-1)


def gamut_map(lab: np.ndarray, tol: float = 1e-4, iters: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Map Oklab colors into the sRGB gamut by reducing chroma at fixed L and hue.

    L is first clamped to [0, 1]. Then a bisection on the chroma scale s in [0, 1]
    finds the largest s with (L, s*a, s*b) in gamut. Returns (mapped, was_mapped).
    Channel clipping is deliberately *not* used: it distorts hue.
    """
    lab = np.asarray(lab, dtype=np.float64)
    out = lab.copy()
    out[..., 0] = np.clip(out[..., 0], 0.0, 1.0)
    ok = in_gamut(out, tol)
    was_mapped = ~ok | (np.abs(out[..., 0] - lab[..., 0]) > 0)
    if np.all(ok):
        return out, was_mapped
    lo = np.zeros(lab.shape[:-1])
    hi = np.ones(lab.shape[:-1])
    ab = out[..., 1:]
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        cand = np.concatenate([out[..., :1], ab * mid[..., None]], axis=-1)
        good = in_gamut(cand, tol)
        lo = np.where(good, mid, lo)
        hi = np.where(good, hi, mid)
    s = np.where(ok, 1.0, lo)
    out[..., 1:] = ab * s[..., None]
    return out, was_mapped


def clip_srgb(rgb: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)


# --- normalization for the model (R8) ---------------------------------------------
@dataclass(frozen=True)
class NormStats:
    """L standardized by mean/std; a,b scaled by one shared std, no mean subtraction."""

    L_mean: float
    L_std: float
    ab_scale: float

    @classmethod
    def fit(cls, lab: np.ndarray) -> "NormStats":
        lab = np.asarray(lab, dtype=np.float64).reshape(-1, 3)
        ab = lab[:, 1:]
        # pooled std over both chromatic axes, about the origin (not about the mean)
        ab_scale = float(np.sqrt(np.mean(ab**2)))
        return cls(L_mean=float(lab[:, 0].mean()), L_std=float(lab[:, 0].std()), ab_scale=ab_scale)

    def normalize(self, lab: np.ndarray) -> np.ndarray:
        lab = np.asarray(lab, dtype=np.float64)
        out = np.empty_like(lab)
        out[..., 0] = (lab[..., 0] - self.L_mean) / self.L_std
        out[..., 1:] = lab[..., 1:] / self.ab_scale
        return out

    def denormalize(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        out = np.empty_like(z)
        out[..., 0] = z[..., 0] * self.L_std + self.L_mean
        out[..., 1:] = z[..., 1:] * self.ab_scale
        return out

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "NormStats":
        if d.get("space") == "srgb":
            from palette.data import SrgbNorm
            return SrgbNorm()
        return cls(**{k: float(d[k]) for k in ("L_mean", "L_std", "ab_scale")})
