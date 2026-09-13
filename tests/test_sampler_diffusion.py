import numpy as np
import torch

from palette.sampler import PaletteArrays, SamplerConfig, MaskedCompletionSampler
from palette.diffusion import VPSchedule, masked_eps_loss, training_step, sample


def _arrays(n=50, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5, 3)).astype(np.float32); M = np.ones((n, 5), bool)
    fam = np.array(["kuler"] * (n // 2) + ["mturk2014"] * (n - n // 2))
    rating = np.where(fam == "kuler", np.nan, rng.uniform(1, 5, n))
    return PaletteArrays(X, M, fam, rating, np.array([f"r{i}" for i in range(n)]))


def test_sampler_partition_and_flags():
    a = _arrays()
    s = MaskedCompletionSampler(a, SamplerConfig(batch_size=64, subset_augment=True, subset_prob=1.0, empty_context_prob=0.0))
    b = s.batch()
    assert b["x0"].shape == (64, 5, 3)
    assert (b["m_tgt"] >= 1).all()
    assert (b["n_ctx"] + b["m_tgt"] == b["mask"].sum(1)).all()
    assert b["subset_derived"].any()
    assert (b["mask"].sum(1)[b["subset_derived"]] < 5).all()
    assert not (b["is_fixed"] & ~b["mask"]).any()  # fixed slots are valid slots
    s2 = MaskedCompletionSampler(a, SamplerConfig(batch_size=64, subset_augment=False, empty_context_prob=1.0))
    assert (s2.batch()["n_ctx"] == 0).all()


def test_sampler_family_weights_and_rating_filter():
    a = _arrays()
    s = MaskedCompletionSampler(a, SamplerConfig(batch_size=200, family_weights={"kuler": 0.0001, "mturk2014": 1.0}, min_rating=3.0))
    b = s.batch()
    assert b["family"].count("mturk2014") > 190
    r = b["rating"][torch.tensor([f == "mturk2014" for f in b["family"]])]
    assert (r >= 3.0).all()


def test_rating_cond_null_dropout():
    a = _arrays()
    s = MaskedCompletionSampler(a, SamplerConfig(batch_size=500, rating_cond=True, rating_null_prob=0.2, family_weights={"mturk2014": 1.0}))
    b = s.batch()
    frac = torch.isnan(b["cond"]["rating"]).float().mean()
    assert 0.1 < frac < 0.35


def test_schedule_and_q_sample():
    for kind in ("linear", "cosine"):
        sch = VPSchedule(1000, kind)
        ac = sch.alphas_cumprod
        assert (ac[1:] <= ac[:-1]).all() and ac[0] > 0.99 and ac[-1] < 0.01
        x0 = torch.zeros(4096, 5, 3); noise = torch.randn_like(x0)
        xt = sch.q_sample(x0, torch.full((4096,), 999), noise)
        assert abs(xt.std() - (1 - ac[999]).sqrt()) < 0.02


def test_loss_masks_padding_and_context():
    eps_hat = torch.randn(3, 5, 3); eps = torch.randn(3, 5, 3)
    tm = torch.zeros(3, 5, dtype=torch.bool); tm[:, :2] = True
    l1 = masked_eps_loss(eps_hat, eps, tm)
    eps_hat2 = eps_hat.clone(); eps_hat2[:, 2:] += 100.0  # garbage outside targets
    assert torch.allclose(l1, masked_eps_loss(eps_hat2, eps, tm))


class Oracle:
    """Perfect denoiser when x0 is known: eps = (xt - sqrt(ab) x0)/sqrt(1-ab). Ignores inputs it shouldn't need."""
    def __init__(self, x0, sch): self.x0, self.sch = x0, sch
    def __call__(self, x, is_fixed, mask, t, cond):
        ab = self.sch.alphas_cumprod[t].view(-1, 1, 1)
        return (x - ab.sqrt() * self.x0) / (1 - ab).sqrt()


def test_sampler_recovers_x0_with_oracle_and_keeps_context():
    torch.manual_seed(0)
    sch = VPSchedule(1000, "cosine")
    x0 = torch.randn(8, 5, 3)
    is_fixed = torch.zeros(8, 5, dtype=torch.bool); is_fixed[:, :2] = True
    mask = torch.ones(8, 5, dtype=torch.bool); mask[:, 4] = False
    x_ctx = torch.where(is_fixed[..., None], x0, torch.zeros_like(x0))
    for method in ("ddim", "ddpm"):
        out = sample(Oracle(x0, sch), sch, x_ctx, is_fixed, mask, steps=50, method=method)
        assert torch.equal(out[is_fixed], x0[is_fixed])          # context bit-identical
        tgt = mask & ~is_fixed
        assert (out[tgt] - x0[tgt]).abs().max() < 5e-2, method


def test_training_step_runs_and_context_stays_clean():
    sch = VPSchedule(100, "linear")
    seen = {}
    def model(x, is_fixed, mask, t, cond):
        seen["x"] = x.clone(); return torch.zeros_like(x)
    x0 = torch.randn(4, 5, 3); is_fixed = torch.zeros(4, 5, dtype=torch.bool); is_fixed[:, 0] = True; mask = torch.ones(4, 5, dtype=torch.bool)
    loss = training_step(model, sch, x0, is_fixed, mask)
    assert loss.item() > 0
    assert torch.equal(seen["x"][:, 0], x0[:, 0])
