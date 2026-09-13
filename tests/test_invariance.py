import torch

from palette.models.absolute import AbsoluteDenoiser
from palette.eval.invariance_tests import context_permutation_deviation, target_permutation_deviation
from palette.diffusion import VPSchedule, sample


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(6, 5, 3, generator=g)
    is_fixed = torch.zeros(6, 5, dtype=torch.bool); is_fixed[:, :2] = True; is_fixed[3, 2] = True
    mask = torch.ones(6, 5, dtype=torch.bool); mask[4, 4] = False; mask[5, 3:] = False
    t = torch.randint(0, 1000, (6,), generator=g)
    return x, is_fixed, mask, t


def _model(**kw):
    torch.manual_seed(0)
    m = AbsoluteDenoiser(dim=64, depth=2, heads=4, **kw).eval()
    # break the zero-init output so the test is not trivially satisfied
    with torch.no_grad():
        m.out.weight.normal_(0, 0.1); m.out.bias.normal_(0, 0.1)
    return m


def test_context_and_target_permutation():
    m = _model(); x, f, k, t = _batch()
    assert context_permutation_deviation(m, x, f, k, t) < 1e-5
    assert target_permutation_deviation(m, x, f, k, t) < 1e-5


def test_permutation_with_rating_cond():
    m = _model(rating_cond=True); x, f, k, t = _batch()
    cond = {"rating": torch.tensor([3.0, float("nan"), 4.5, 1.0, 5.0, 2.2])}
    assert context_permutation_deviation(m, x, f, k, t, cond) < 1e-5
    assert target_permutation_deviation(m, x, f, k, t, cond) < 1e-5


def test_fixed_colors_bit_identical_after_sampling():
    m = _model(); x, f, k, t = _batch()
    sch = VPSchedule(1000, "cosine")
    out = sample(m, sch, x, f, k, steps=10)
    assert torch.equal(out[f], x[f])
    assert torch.isfinite(out[k]).all()
