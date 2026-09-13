import numpy as np
import torch

from palette.api import Completer, repulsive_guidance
from palette.color import NormStats, srgb_to_oklab, hex_to_srgb
from palette.diffusion import VPSchedule


class Oracle:
    """Denoiser that always predicts noise pointing to a fixed target palette (deterministic, in gamut)."""
    def __init__(self, tgt, sch): self.t, self.s = tgt, sch
    def __call__(self, x, is_fixed, mask, t, cond):
        ab = self.s.alphas_cumprod[t].view(-1, 1, 1)
        return (x - ab.sqrt() * self.t[: x.shape[1]][None]) / (1 - ab).sqrt()


def test_similar_fixed_colors_are_accepted():
    norm = NormStats(0.6, 0.2, 0.1); sch = VPSchedule(100, "cosine")
    tgt = torch.tensor(norm.normalize(srgb_to_oklab(np.array([[.5, .5, .5], [.5, .5, .5], [.2, .6, .3], [.8, .3, .2], [.1, .1, .5]]))), dtype=torch.float32)
    c = Completer(Oracle(tgt, sch), sch, norm, torch.device("cpu"), steps=10, oversample=4, scorer=lambda p: np.zeros(len(p)))
    out = c.complete(["#808080", "#818181"], 2, 2)
    assert len(out) == 2 and out[0][:2] == ["#808080", "#818181"]


def test_repulsion_pushes_targets_from_context_only_moving_targets():
    x = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.5, 0.0]]])
    target = torch.tensor([[False, True, True]]); valid = torch.ones(1, 3, dtype=torch.bool)
    g = repulsive_guidance()(x, target, 0, valid)
    assert torch.all(g[0, 0] == 0)              # context never moves
    assert g[0, 1, 1] < 0                       # ε negative in +b → x moves away from context at b=0
    assert g[0, 2].abs().max() < g[0, 1].abs().max()
