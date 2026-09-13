import numpy as np
import torch

from palette.baselines.colorwheel import complete as cw_complete
from palette.baselines.retrieval import PaletteIndex
from palette.color import in_gamut, srgb_to_oklab, hex_to_srgb
from palette.eval.metrics import matching_distance, diversity, duplicate_rate
from palette.models.scorer import Scorer, pairwise_loss


def test_colorwheel_in_gamut_and_shapes():
    ctx = srgb_to_oklab(np.stack([hex_to_srgb("#9caf88"), hex_to_srgb("#6b4a2b")]))
    out = cw_complete(ctx, 2, 8)
    assert out.shape == (8, 2, 3) and in_gamut(out).all()
    assert cw_complete(np.zeros((0, 3)), 5, 3).shape == (3, 5, 3)


def test_retrieval_returns_remaining_colors_of_matching_palette():
    rng = np.random.default_rng(0)
    X = rng.uniform(-0.2, 0.2, (200, 5, 3)); X[..., 0] = rng.uniform(0.2, 0.9, (200, 5)); M = np.ones((200, 5), bool)
    idx = PaletteIndex(X, M)
    ctx = X[17, :2]
    out = idx.complete(ctx, 3, 1)
    assert matching_distance(out[0][None], X[17, 2:][None])[0] < 1e-9
    kde = idx.kde_complete(ctx, 3, 4)
    assert kde.shape == (4, 3, 3)


def test_matching_distance_perm_invariant_and_hungarian_path():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(10, 5, 3)); perm = rng.permutation(5)
    assert np.allclose(matching_distance(A, A[:, perm]), 0.0)
    B = rng.normal(size=(10, 7, 3))
    assert np.allclose(matching_distance(B, B[:, rng.permutation(7)]), 0.0)
    assert diversity(A) > 0 and 0 <= duplicate_rate(A) <= 1


def test_scorer_variants():
    x = torch.randn(4, 5, 3)
    for v in ("deepsets", "sortedL", "ordered"):
        s = Scorer(v)
        out = s(x); assert out.shape == (4,)
    ds = Scorer("deepsets"); perm = torch.randperm(5)
    assert torch.allclose(ds(x), ds(x[:, perm]), atol=1e-5)
    assert pairwise_loss(torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([1])) < pairwise_loss(torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([1]))
