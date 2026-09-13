import torch
from palette.diffusion import VPSchedule, FlowSchedule, sample_dpmpp2m, flow_sample, flow_training_step
from tests.test_sampler_diffusion import Oracle


def test_dpmpp2m_recovers_x0_and_keeps_context():
    torch.manual_seed(0); sch = VPSchedule(1000, "cosine"); x0 = torch.randn(8, 5, 3)
    f = torch.zeros(8, 5, dtype=torch.bool); f[:, :2] = True; m = torch.ones(8, 5, dtype=torch.bool)
    x_ctx = torch.where(f[..., None], x0, torch.zeros_like(x0))
    out = sample_dpmpp2m(Oracle(x0, sch), sch, x_ctx, f, m, steps=12)
    assert torch.equal(out[f], x0[f]) and (out[m & ~f] - x0[m & ~f]).abs().max() < 5e-2


class FlowOracle:
    def __init__(self, x0, sch): self.x0, self.s = x0, sch
    def __call__(self, x, is_fixed, mask, t, cond):
        tau = self.s.tau(t).view(-1, 1, 1)
        return (x - (1 - tau) * self.x0) / tau - self.x0   # v = ε - x0 with ε recovered from x_t


def test_flow_sampler_recovers_x0():
    torch.manual_seed(0); sch = FlowSchedule(1000); x0 = torch.randn(8, 5, 3)
    f = torch.zeros(8, 5, dtype=torch.bool); f[:, :1] = True; m = torch.ones(8, 5, dtype=torch.bool)
    x_ctx = torch.where(f[..., None], x0, torch.zeros_like(x0))
    for method in ("euler", "heun"):
        out = flow_sample(FlowOracle(x0, sch), sch, x_ctx, f, m, steps=20, method=method)
        assert torch.equal(out[f], x0[f]) and (out[m & ~f] - x0[m & ~f]).abs().max() < 5e-2, method
    loss = flow_training_step(lambda x, f_, m_, t, c: torch.zeros_like(x), sch, x0, f, m)
    assert loss > 0
