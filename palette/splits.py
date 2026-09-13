"""Split assignment at the duplicate-group level, before any masking (handoff §4.3, R8)."""
from __future__ import annotations

import numpy as np

SPLITS = ("train", "val", "test")


def assign_group_splits(group: np.ndarray, fractions=(0.8, 0.1, 0.1), seed: int = 0,
                        large_group_to_train: int | None = 100) -> np.ndarray:
    """group: (N,) int group id per record → (N,) split name. All members of a group share a split.
    Groups larger than `large_group_to_train` (single-linkage chains) are forced into train so that one
    chain cannot dominate the composition of a held-out split (see STATUS.md G1 notes)."""
    rng = np.random.default_rng(seed)
    n_groups = int(group.max() + 1)
    sizes = np.bincount(group, minlength=n_groups)
    perm = rng.permutation(n_groups)
    cuts = np.cumsum(np.array(fractions) * n_groups).astype(int)
    gsplit = np.empty(n_groups, dtype=object)
    gsplit[perm[: cuts[0]]] = "train"
    gsplit[perm[cuts[0] : cuts[1]]] = "val"
    gsplit[perm[cuts[1] :]] = "test"
    if large_group_to_train is not None:
        gsplit[sizes > large_group_to_train] = "train"
    return gsplit[group]


def assign_user_split(user_ids: list[int], frac_heldout: float = 0.1, seed: int = 0) -> dict[int, str]:
    rng = np.random.default_rng(seed + 1)
    users = np.array(sorted(set(user_ids)))
    held = set(rng.choice(users, size=int(round(frac_heldout * len(users))), replace=False).tolist())
    return {int(u): ("heldout_user" if u in held else "seen_user") for u in users}
