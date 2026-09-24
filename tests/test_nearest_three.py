"""find_nearest_three_points: the KD-tree answers what the dense matrix did.

Run: python tests/test_nearest_three.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geosam2._lift import find_nearest_three_points


def _dense(A, B):
    """VAST's original: full [N, M] matrix, then topk."""
    return torch.topk(torch.cdist(torch.from_numpy(A).float(),
                                  torch.from_numpy(B).float()),
                      k=3, largest=False)[1].numpy()


def test_matches_the_dense_matrix():
    rng = np.random.default_rng(0)
    A, B = rng.random((300, 3)), rng.random((900, 3))
    assert np.array_equal(find_nearest_three_points(A, B), _dense(A, B))


def test_same_labels_on_coherent_parts():
    """Ties pick another index than topk did, so the contract is the label."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(1)
    pts = rng.random((4000, 3))
    labels = torch.from_numpy(cKDTree(pts[rng.choice(4000, 10, False)]).query(pts)[1] + 1)
    hole = cKDTree(pts).query(pts[0], k=1400)[1]
    labels[hole] = 0

    ui, li = torch.where(labels == 0)[0], torch.where(labels != 0)[0]
    A, B = pts[ui], pts[li]
    mode = lambda idx: torch.mode(labels[li][idx], dim=1).values
    assert torch.equal(mode(find_nearest_three_points(A, B)), mode(_dense(A, B)))


def test_fewer_than_three_references():
    A, B = np.zeros((5, 3)), np.ones((1, 3))
    out = find_nearest_three_points(A, B)
    assert out.shape == (5, 3) and (out == 0).all()


def test_torch_in_torch_out():
    A = torch.rand(10, 3)
    assert isinstance(find_nearest_three_points(A, torch.rand(50, 3)), torch.Tensor)


def test_scales_past_the_dense_limit():
    """160 GB as a dense matrix."""
    rng = np.random.default_rng(2)
    out = find_nearest_three_points(rng.random((200_000, 3)), rng.random((200_000, 3)))
    assert out.shape == (200_000, 3)


def test_mode_is_order_invariant():
    """Why reordered ties are safe: mode returns the smallest, not the first."""
    import itertools
    for perm in itertools.permutations([5, 9, 7]):
        assert torch.mode(torch.tensor([list(perm)]), dim=1).values.item() == 5
    assert torch.mode(torch.tensor([[9, 9, 5, 5]]), dim=1).values.item() == 5


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK ", name)
