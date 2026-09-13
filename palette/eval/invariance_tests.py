"""Numerical symmetry checks (handoff §4.5 / §4.6). Each returns the max deviation (0 = exact)."""
from __future__ import annotations

import torch


def _perm_within(mask_rows: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Random permutation of the True positions of a boolean vector, identity elsewhere."""
    idx = torch.arange(len(mask_rows))
    pos = torch.where(mask_rows)[0]
    idx[pos] = pos[torch.randperm(len(pos), generator=g)]
    return idx


def context_permutation_deviation(model, x, is_fixed, mask, t, cond=None, seed=0) -> float:
    g = torch.Generator().manual_seed(seed)
    base = model(x, is_fixed, mask, t, cond)
    dev = 0.0
    for b in range(x.shape[0]):
        p = _perm_within(is_fixed[b] & mask[b], g)
        xp, fp, mp = x.clone(), is_fixed.clone(), mask.clone()
        xp[b] = x[b, p]; fp[b] = is_fixed[b, p]; mp[b] = mask[b, p]
        out = model(xp, fp, mp, t, cond)
        tgt = mask[b] & ~is_fixed[b]
        dev = max(dev, (out[b][tgt] - base[b][tgt]).abs().max().item())
    return dev


def target_permutation_deviation(model, x, is_fixed, mask, t, cond=None, seed=0) -> float:
    g = torch.Generator().manual_seed(seed)
    base = model(x, is_fixed, mask, t, cond)
    dev = 0.0
    for b in range(x.shape[0]):
        p = _perm_within(mask[b] & ~is_fixed[b], g)
        xp = x.clone(); xp[b] = x[b, p]
        out = model(xp, is_fixed, mask, t, cond)
        tgt = mask[b] & ~is_fixed[b]
        dev = max(dev, (out[b][tgt] - base[b, p][tgt]).abs().max().item())
    return dev


def rotation_equivariance_deviation(model, x, is_fixed, mask, t, cond=None, seed=0) -> float:
    """Rotate all (a,b) by θ, run, un-rotate the output, compare to the unrotated run."""
    g = torch.Generator().manual_seed(seed)
    theta = torch.rand(1, generator=g).item() * 6.283185307179586
    c, s = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
    R = torch.tensor([[c, -s], [s, c]], dtype=x.dtype)
    xr = x.clone(); xr[..., 1:] = x[..., 1:] @ R.T
    base = model(x, is_fixed, mask, t, cond)
    out = model(xr, is_fixed, mask, t, cond)
    out_un = out.clone(); out_un[..., 1:] = out[..., 1:] @ R
    tgt = mask & ~is_fixed
    return (out_un[tgt] - base[tgt]).abs().max().item()
