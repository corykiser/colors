"""Duplicate detection across all families.

1. Exact: identical color multiset after quantizing Oklab to 1e-3 (sort rows → tuple).
2. Near: same palette size, min-cost-matching mean Oklab distance d < NEAR_THRESHOLD.
   Candidate pairs come from a KD-tree on the per-channel-sorted embedding F (each of L, a, b sorted
   independently, concatenated). For any permutation π, ||F_A - F_B||_2^2 <= Σ_i ||a_i - b_π(i)||^2 <= (n d)^2,
   so a query radius of n * threshold has *guaranteed* candidate recall. Pairs are then verified exactly.
Union-find joins both into duplicate_group_id values "dg:<int>".
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from palette.eval.metrics import matching_distance

NEAR_THRESHOLD = 0.02   # mean matched Oklab distance; handoff §4.3 starting value


def sorted_channel_embedding(P: np.ndarray) -> np.ndarray:
    """(N, n, 3) -> (N, 3n): each channel sorted independently. Permutation-invariant and
    1/n-Lipschitz w.r.t. the mean matching distance (see module docstring)."""
    return np.sort(P, axis=1).transpose(0, 2, 1).reshape(len(P), -1)


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
        F = sorted_channel_embedding(P)
        tree = cKDTree(F)
        pairs = tree.query_pairs(r=threshold * n * (1 + 1e-9), output_type="ndarray")
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


def cross_split_near_pairs(X: np.ndarray, M: np.ndarray, split: np.ndarray, threshold: float = NEAR_THRESHOLD,
                           a: str = "train", b: str = "val") -> int:
    """Audit: number of (a, b) record pairs of equal size with matching distance < threshold. Exact (same bound)."""
    sizes = M.sum(1); count = 0
    for n in np.unique(sizes):
        ia = np.where((sizes == n) & (split == a))[0]; ib = np.where((sizes == n) & (split == b))[0]
        if len(ia) == 0 or len(ib) == 0:
            continue
        ta = cKDTree(sorted_channel_embedding(X[ia, :n])); tb = cKDTree(sorted_channel_embedding(X[ib, :n]))
        pairs = ta.query_ball_tree(tb, r=threshold * n * (1 + 1e-9))
        cand = np.array([(i, j) for i, js in enumerate(pairs) for j in js])
        if len(cand) == 0:
            continue
        for s in range(0, len(cand), 200_000):
            c = cand[s : s + 200_000]
            d = matching_distance(X[ia[c[:, 0]], :n], X[ib[c[:, 1]], :n])
            count += int((d < threshold).sum())
    return count
