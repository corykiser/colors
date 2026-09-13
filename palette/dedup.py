"""Duplicate detection across all families.

1. Exact: identical color multiset after quantizing Oklab to 1e-3 (sort rows → tuple).
2. Near: same palette size, min-cost-matching mean Oklab distance < NEAR_THRESHOLD.
   Candidate pairs come from a KD-tree on the sort-by-L flattened palette (blocking),
   then are verified with the exact matching distance.
Union-find joins both into duplicate_group_id values "dg:<int>".
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from palette.eval.metrics import matching_distance

NEAR_THRESHOLD = 0.02   # mean matched Oklab distance; handoff §4.3 starting value
BLOCK_FACTOR = 1.5      # KD-tree radius = threshold * n * factor (sort-by-L is not the optimal matching)


class _UF:
    def __init__(self, n: int):
        self.p = np.arange(n)

    def find(self, i: int) -> int:
        p = self.p
        while p[i] != i:
            p[i] = p[p[i]]
            i = p[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def exact_key(lab: np.ndarray, q: float = 1e-3) -> tuple:
    r = np.round(np.asarray(lab) / q).astype(np.int64)
    return tuple(sorted(map(tuple, r.tolist())))


def find_duplicate_groups(X: np.ndarray, M: np.ndarray, threshold: float = NEAR_THRESHOLD,
                          verbose: bool = True) -> tuple[np.ndarray, dict]:
    """X: (N, nmax, 3) Oklab, M: (N, nmax) mask. Returns group id per row (int) and stats."""
    N = X.shape[0]
    sizes = M.sum(1)
    uf = _UF(N)
    # exact
    seen: dict[tuple, int] = {}
    n_exact = 0
    for i in range(N):
        k = exact_key(X[i, : sizes[i]])
        if k in seen:
            uf.union(seen[k], i); n_exact += 1
        else:
            seen[k] = i
    # near
    n_cand = n_near = 0
    for n in np.unique(sizes):
        idx = np.where(sizes == n)[0]
        P = X[idx, :n]
        order = np.argsort(P[..., 0], axis=1)
        F = np.take_along_axis(P, order[..., None], axis=1).reshape(len(idx), -1)
        tree = cKDTree(F)
        pairs = tree.query_pairs(r=threshold * n * BLOCK_FACTOR, output_type="ndarray")
        n_cand += len(pairs)
        if len(pairs) == 0:
            continue
        for s in range(0, len(pairs), 200_000):
            chunk = pairs[s : s + 200_000]
            d = matching_distance(P[chunk[:, 0]], P[chunk[:, 1]])
            for a, b in chunk[d < threshold]:
                uf.union(int(idx[a]), int(idx[b])); n_near += 1
        if verbose:
            print(f"  size {n}: {len(idx)} palettes, {len(pairs)} candidate pairs")
    roots = np.array([uf.find(i) for i in range(N)])
    _, group = np.unique(roots, return_inverse=True)
    sizes_of_groups = np.bincount(group)
    stats = {
        "n_records": int(N), "n_groups": int(group.max() + 1),
        "exact_duplicate_rows": int(n_exact), "near_candidate_pairs": int(n_cand),
        "near_duplicate_pairs": int(n_near), "threshold": threshold,
        "largest_group": int(sizes_of_groups.max()),
        "rows_in_multi_groups": int((sizes_of_groups[group] > 1).sum()),
    }
    return group, stats
