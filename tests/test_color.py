import numpy as np
import pytest

from palette import color as C


def test_white_black_gray():
    lab = C.srgb_to_oklab(np.array([1.0, 1.0, 1.0]))
    assert abs(lab[0] - 1.0) < 1e-4 and abs(lab[1]) < 1e-4 and abs(lab[2]) < 1e-4
    lab = C.srgb_to_oklab(np.array([0.0, 0.0, 0.0]))
    assert np.allclose(lab, 0.0, atol=1e-6)
    lab = C.srgb_to_oklab(np.array([0.5, 0.5, 0.5]))
    assert abs(lab[1]) < 1e-5 and abs(lab[2]) < 1e-5 and 0 < lab[0] < 1


@pytest.mark.parametrize(
    "rgb, ref",
    [
        ((1, 0, 0), (0.6280, 0.2249, 0.1258)),
        ((0, 1, 0), (0.8664, -0.2339, 0.1795)),
        ((0, 0, 1), (0.4520, -0.0325, -0.3115)),
    ],
)
def test_primaries_against_published_values(rgb, ref):
    # Published sRGB primaries in Oklab (Ottosson / culori), 4 decimal places.
    lab = C.srgb_to_oklab(np.array(rgb, dtype=float))
    assert np.allclose(lab, ref, atol=1.5e-3), lab


def test_xyz_reference_rows_sum_to_white():
    # M1 rows sum to 1 → linear white maps to LMS = (1,1,1) → Oklab (1,0,0).
    assert np.allclose(C._M1.sum(1), 1.0, atol=2e-6)
    assert np.allclose(C._M2.sum(1), [1.0, 0.0, 0.0], atol=2e-6)


def test_roundtrip_precision():
    rng = np.random.default_rng(0)
    rgb = rng.random((5000, 3))
    back = C.oklab_to_srgb(C.srgb_to_oklab(rgb))
    # Ottosson's published 10-digit matrices are not exact inverses; ~1e-6 is expected.
    assert np.max(np.abs(back - rgb)) < 1e-5


def test_hex_roundtrip():
    for h in ["#ae5224", "#f9c1ce", "#000000", "#ffffff", "#123"]:
        rgb = C.hex_to_srgb(h)
        exp = h if len(h) == 7 else "#112233"
        assert C.srgb_to_hex(rgb) == exp


def test_lch_roundtrip_and_hue_sign():
    lab = np.array([[0.7, 0.1, 0.05], [0.3, -0.2, 0.1], [0.5, 0.0, 0.0]])
    assert np.allclose(C.lch_to_oklab(C.oklab_to_lch(lab)), lab)
    lch = C.oklab_to_lch(lab[0])
    assert lch[2] > 0  # b>0, a>0 → first quadrant


def test_in_gamut_and_mapping():
    # a saturated Oklab blue at high L cannot exist in sRGB
    bad = np.array([0.9, -0.05, -0.3])
    assert not C.in_gamut(bad)
    mapped, flag = C.gamut_map(bad)
    assert flag and C.in_gamut(mapped)
    # L and hue preserved, chroma reduced
    assert mapped[0] == pytest.approx(bad[0])
    h0, h1 = C.oklab_to_lch(bad)[2], C.oklab_to_lch(mapped)[2]
    assert abs(h0 - h1) < 1e-6
    assert C.oklab_to_lch(mapped)[1] < C.oklab_to_lch(bad)[1]
    # in-gamut colors untouched
    good = C.srgb_to_oklab(np.array([[0.2, 0.5, 0.7], [0.9, 0.9, 0.1]]))
    m, f = C.gamut_map(good)
    assert np.allclose(m, good) and not f.any()
    # L outside [0,1] gets clamped and flagged
    m, f = C.gamut_map(np.array([1.2, 0.0, 0.0]))
    assert f and m[0] == 1.0


def test_gamut_map_is_maximal():
    rng = np.random.default_rng(1)
    lab = np.stack([rng.uniform(0.2, 0.9, 200), rng.uniform(-0.6, 0.6, 200), rng.uniform(-0.6, 0.6, 200)], -1)
    mapped, flag = C.gamut_map(lab)
    assert C.in_gamut(mapped).all()
    # pushing chroma up by 1% should leave gamut for the mapped ones
    up = mapped.copy()
    up[..., 1:] *= 1.01
    assert not C.in_gamut(up[flag], tol=1e-4).any() or flag.sum() == 0


def test_norm_stats_no_chromatic_mean_subtraction():
    rng = np.random.default_rng(2)
    lab = C.srgb_to_oklab(rng.random((1000, 5, 3)))
    st = C.NormStats.fit(lab)
    z = st.normalize(lab)
    assert np.allclose(st.denormalize(z), lab)
    # chromatic origin is preserved: a zero-chroma color stays at zero chroma
    gray = np.array([0.5, 0.0, 0.0])
    assert np.allclose(st.normalize(gray)[1:], 0.0)
    # same scale on both axes
    assert np.allclose(st.normalize(np.array([0.5, 0.1, 0.0]))[1], st.normalize(np.array([0.5, 0.0, 0.1]))[2])
    assert C.NormStats.from_dict(st.to_dict()) == st
