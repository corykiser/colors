"""Split assignment at the duplicate-group level, before any masking (handoff §4.3, R8)."""
from __future__ import annotations

import numpy as np

SPLITS = ("train", "val", "test")


def assign_group_splits(group: np.ndarray, fractions=(0.8, 0.1, 0.1), seed: int = 0) -> np.ndarray:
    """group: (N,) int group id per record → (N,) split name. All members of a group share a split."""
    rng = np.random.default_rng(seed)
    n_groups = int(group.max() + 1)
    perm = rng.permutation(n_groups)
    cuts = np.cumsum(np.array(fractions) * n_groups).astype(int)
    gsplit = np.empty(n_groups, dtype=object)
    gsplit[perm[: cuts[0]]] = "train"
    gsplit[perm[cuts[0] : cuts[1]]] = "val"
    gsplit[perm[cuts[1] :]] = "test"
    return gsplit[group]


def assign_user_split(user_ids: list[int], frac_heldout: float = 0.1, seed: int = 0) -> dict[int, str]:
    rng = np.random.default_rng(seed + 1)
    users = np.array(sorted(set(user_ids)))
    held = set(rng.choice(users, size=int(round(frac_heldout * len(users))), replace=False).tolist())
    return {int(u): ("heldout_user" if u in held else "seen_user") for u in users}
