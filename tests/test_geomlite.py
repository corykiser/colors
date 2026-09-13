import torch

from palette.models.geomlite import GeomLiteDenoiser, pairwise_invariants
from palette.eval.invariance_tests import context_permutation_deviation, target_permutation_deviation
from tests.test_invariance import _batch


def test_pairwise_invariants_are_rotation_invariant():
    x = torch.randn(3, 5, 3); f = torch.zeros(3, 5, dtype=torch.bool); f[:, :2] = True
    e = pairwise_invariants(x, f)
    th = torch.tensor(0.7); R = torch.tensor([[torch.cos(th), -torch.sin(th)], [torch.sin(th), torch.cos(th)]])
    xr = x.clone(); xr[..., 1:] = x[..., 1:] @ R.T
    assert torch.allclose(e, pairwise_invariants(xr, f), atol=1e-5)
    # reflection flips the determinant sign only
    xm = x.clone(); xm[..., 2] = -x[..., 2]
    em = pairwise_invariants(xm, f)
    assert torch.allclose(em[..., 6], -e[..., 6], atol=1e-6) and torch.allclose(em[..., :6], e[..., :6], atol=1e-6)


def test_geomlite_permutation_symmetry():
    torch.manual_seed(0)
    m = GeomLiteDenoiser(dim=64, depth=2, heads=4).eval()
    with torch.no_grad():
        m.out.weight.normal_(0, 0.1); m.out.bias.normal_(0, 0.1)
    x, f, k, t = _batch()
    assert context_permutation_deviation(m, x, f, k, t) < 1e-5
    assert target_permutation_deviation(m, x, f, k, t) < 1e-5
    assert torch.isfinite(m(x, f, k, t)).all()
