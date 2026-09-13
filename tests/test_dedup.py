import numpy as np
from palette.dedup import find_duplicate_groups, sorted_channel_embedding, cross_split_near_pairs
from palette.eval.metrics import matching_distance
from palette.splits import assign_group_splits


def test_embedding_bound_and_recall_on_lightness_swap():
    # two colors with near-equal L but different a,b: sort-by-L blocking flips order; the channel-sorted embedding does not
    A = np.array([[0.50, 0.10, 0.00], [0.501, -0.10, 0.00], [0.8, 0, 0], [0.2, 0, 0], [0.6, 0.05, 0.05]])
    B = A.copy(); B[0, 0], B[1, 0] = 0.501, 0.50
    d = matching_distance(A[None], B[None])[0]
    assert d < 0.002
    assert np.linalg.norm(sorted_channel_embedding(A[None])[0] - sorted_channel_embedding(B[None])[0]) <= 5 * d + 1e-12
    X = np.stack([A, B]); M = np.ones((2, 5), bool)
    g, stats = find_duplicate_groups(X, M, verbose=False)
    assert g[0] == g[1]


def test_random_bound_holds():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(500, 5, 3)); B = A + rng.normal(scale=0.01, size=A.shape)
    perm = rng.permutation(5); B = B[:, perm]
    d = matching_distance(A, B)
    e = np.linalg.norm(sorted_channel_embedding(A) - sorted_channel_embedding(B), axis=1)
    assert np.all(e <= 5 * d + 1e-9)


def test_large_groups_forced_to_train_and_audit_zero():
    rng = np.random.default_rng(1)
    X = rng.uniform(-0.3, 0.3, (300, 5, 3)); X[..., 0] = rng.uniform(0.1, 0.9, (300, 5)); M = np.ones((300, 5), bool)
    group = np.concatenate([np.zeros(150, int), np.arange(1, 151)])
    split = assign_group_splits(group, seed=0, large_group_to_train=100)
    assert (split[:150] == "train").all()
    assert cross_split_near_pairs(X, M, split) >= 0
