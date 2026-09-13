import torch

from palette.models.geometric import GeometricDenoiser
from palette.models.combined import CombinedDenoiser
from palette.eval.invariance_tests import (context_permutation_deviation, target_permutation_deviation,
                                           rotation_equivariance_deviation)
from tests.test_invariance import _batch


def _geom(**kw):
    torch.manual_seed(0)
    m = GeometricDenoiser(ds=64, dv=8, depth=2, hidden=64, **kw).eval()
    with torch.no_grad():  # break zero-inits so outputs are nontrivial
        for blk in m.blocks:
            blk.to_coef.weight.normal_(0, 0.3); blk.to_coef.bias.normal_(0, 0.3)
        m.out_L[1].weight.normal_(0, 0.3); m.out_ab.weight.normal_(0, 0.3)
    return m


def test_geometric_symmetries():
    m = _geom(); x, f, k, t = _batch()
    out = m(x, f, k, t)
    assert torch.isfinite(out).all() and out.abs().max() > 1e-3
    assert rotation_equivariance_deviation(m, x, f, k, t) < 1e-4
    assert rotation_equivariance_deviation(m, x, f, k, t, seed=3) < 1e-4
    assert context_permutation_deviation(m, x, f, k, t) < 1e-5
    assert target_permutation_deviation(m, x, f, k, t) < 1e-5


def test_geometric_not_reflection_equivariant():
    """Reflecting b -> -b must NOT be a symmetry (the determinant term breaks it)."""
    m = _geom(); x, f, k, t = _batch()
    base = m(x, f, k, t)
    xm = x.clone(); xm[..., 2] = -x[..., 2]
    out = m(xm, f, k, t); out[..., 2] = -out[..., 2]
    tgt = k & ~f
    assert (out[tgt] - base[tgt]).abs().max() > 1e-3


def test_combined_runs_and_logs_ratio():
    torch.manual_seed(0)
    m = CombinedDenoiser(abs_kw=dict(dim=32, depth=1, heads=2), geom_kw=dict(ds=32, dv=4, depth=1, hidden=32)).eval()
    x, f, k, t = _batch()
    out = m(x, f, k, t)
    assert out.shape == x.shape and "abs_over_geom_norm" in m.last_stats
