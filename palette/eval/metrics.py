"""Evaluation metrics (handoff §4.4). Set-to-set distances use min-cost matching, never row index."""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment

_PERMS: dict[int, np.ndarray] = {}


def _perms(n: int) -> np.ndarray:
    if n not in _PERMS:
        _PERMS[n] = np.array(list(itertools.permutations(range(n))))
    return _PERMS[n]


def matching_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Mean min-cost Euclidean matching distance between palettes.

    A, B: (..., n, 3) with equal n. Returns (...,). For n <= 6 all permutations are
    enumerated (vectorized); otherwise the Hungarian algorithm is used per pair.
    """
    A = np.asarray(A, dtype=np.float64); B = np.asarray(B, dtype=np.float64)
    n = A.shape[-2]
    if n <= 6:
        P = _perms(n)                                        # (n!, n)
        D = np.linalg.norm(A[..., :, None, :] - B[..., None, :, :], axis=-1)  # (..., n, n)
        rows = np.arange(n)
        costs = D[..., rows[None, :], P].sum(-1)             # (..., n!)
        return costs.min(-1) / n
    flatA = A.reshape(-1, n, 3); flatB = B.reshape(-1, n, 3)
    out = np.empty(flatA.shape[0])
    for i in range(flatA.shape[0]):
        D = np.linalg.norm(flatA[i][:, None] - flatB[i][None], axis=-1)
        r, c = linear_sum_assignment(D)
        out[i] = D[r, c].mean()
    return out.reshape(A.shape[:-2])


def fixed_preserved(returned: np.ndarray, fixed_in: np.ndarray) -> bool:
    """Exact equality of the returned fixed colors with the input (no tolerance)."""
    return np.array_equal(np.asarray(returned), np.asarray(fixed_in))


def diversity(completions: np.ndarray) -> float:
    """Mean pairwise matching distance among K completions (K, m, 3) of the *generated* part only."""
    K = completions.shape[0]
    if K < 2:
        return 0.0
    i, j = np.triu_indices(K, 1)
    return float(matching_distance(completions[i], completions[j]).mean())


def duplicate_rate(completions: np.ndarray, threshold: float = 0.02) -> float:
    """Fraction of completions (K, m, 3) whose min pairwise target distance is below threshold."""
    K, m, _ = completions.shape
    if m < 2:
        return 0.0
    i, j = np.triu_indices(m, 1)
    d = np.linalg.norm(completions[:, i] - completions[:, j], axis=-1).min(-1)
    return float((d < threshold).mean())


def gamut_rate(lab: np.ndarray) -> float:
    from palette.color import in_gamut
    return float(in_gamut(lab).mean())


def pairwise_accuracy(scores_a: np.ndarray, scores_b: np.ndarray, a_preferred: np.ndarray) -> float:
    pred = scores_a > scores_b
    return float((pred == a_preferred.astype(bool)).mean())


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).correlation)
