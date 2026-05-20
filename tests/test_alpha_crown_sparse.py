"""Unit tests for sparse_alpha=True in alpha_crown_batched and
alpha_crown_fixed_intermediate_batched: should produce identical bounds
to the dense path (modulo Adam initialisation rounding) on a small FC net.
"""

import numpy as np
import torch

from vibecheck import alpha_crown as ac


def _make_toy_gg():
    """Build a 2-layer FC graph manually: input(8) -> fc(16) -> relu -> fc(8) -> relu -> fc(4)."""
    device = torch.device("cpu")
    dtype = torch.float64

    rng = np.random.default_rng(0)
    W1 = torch.from_numpy(rng.normal(size=(16, 8))).to(dtype)
    b1 = torch.from_numpy(rng.normal(size=16,)).to(dtype) * 0.1
    W2 = torch.from_numpy(rng.normal(size=(8, 16))).to(dtype)
    b2 = torch.from_numpy(rng.normal(size=8,)).to(dtype) * 0.1
    W3 = torch.from_numpy(rng.normal(size=(4, 8))).to(dtype)
    b3 = torch.from_numpy(rng.normal(size=4,)).to(dtype) * 0.1

    ops = [
        {'name': 'fc1', 'type': 'fc', 'inputs': ['inp'], 'W': W1, 'bias': b1},
        {'name': 'relu1', 'type': 'relu', 'inputs': ['fc1'], 'layer_idx': 0},
        {'name': 'fc2', 'type': 'fc', 'inputs': ['relu1'], 'W': W2, 'bias': b2},
        {'name': 'relu2', 'type': 'relu', 'inputs': ['fc2'], 'layer_idx': 1},
        {'name': 'fc3', 'type': 'fc', 'inputs': ['relu2'], 'W': W3, 'bias': b3},
    ]
    gg = {
        'ops': ops,
        'input_name': 'inp',
        'fork_points': set(),
        'n_relu': 2,
        'input_shape': (8,),
    }
    xl = torch.full((8,), -0.5, dtype=dtype)
    xh = torch.full((8,), 0.5, dtype=dtype)
    return gg, xl, xh, dtype, device


def test_sparse_alpha_matches_dense_fixed_intermediate_batched():
    gg, xl, xh, dtype, device = _make_toy_gg()
    bbr_init = {
        0: (np.array([-0.5]*16), np.array([0.7]*16)),
        1: (np.array([-0.3]*8), np.array([0.4]*8)),
    }
    rng = np.random.default_rng(1)
    n_q = 4
    w_qs = rng.normal(size=(n_q, 4)).astype(np.float32)
    b_qs = rng.normal(size=n_q).astype(np.float32)

    lbs_dense, _, _, hist_dense = ac.run_alpha_crown_fixed_intermediate_batched(
        gg, xl, xh, bbr_init, w_qs, b_qs, device, dtype,
        n_iters=10, lr=0.1, lr_decay=1.0,
        early_stop_on_positive=False, sparse_alpha=False)
    lbs_sparse, _, _, hist_sparse = ac.run_alpha_crown_fixed_intermediate_batched(
        gg, xl, xh, bbr_init, w_qs, b_qs, device, dtype,
        n_iters=10, lr=0.1, lr_decay=1.0,
        early_stop_on_positive=False, sparse_alpha=True)

    # Per-iter histories should match within tight tolerance.
    for q in range(n_q):
        assert np.allclose(hist_dense[q], hist_sparse[q], atol=1e-6), (
            f"q={q}: dense {hist_dense[q]} vs sparse {hist_sparse[q]}")
    assert np.allclose(lbs_dense, lbs_sparse, atol=1e-6)


def test_sparse_alpha_matches_dense_batched_joint():
    gg, xl, xh, dtype, device = _make_toy_gg()
    bbr_init = {
        0: (np.array([-0.5]*16), np.array([0.7]*16)),
        1: (np.array([-0.3]*8), np.array([0.4]*8)),
    }
    rng = np.random.default_rng(1)
    n_q = 4
    w_qs = rng.normal(size=(n_q, 4)).astype(np.float32)
    b_qs = rng.normal(size=n_q).astype(np.float32)
    intermediate = [1]
    un = {1: list(np.where((bbr_init[1][0] < 0) & (bbr_init[1][1] > 0))[0])}

    lbs_dense, _, _, hist_dense = ac.run_alpha_crown_batched(
        gg, xl, xh, bbr_init, w_qs, b_qs, intermediate, un,
        device, dtype, n_iters=5, lr=0.1, lr_decay=1.0,
        early_stop_on_positive=False, sparse_alpha=False)
    lbs_sparse, _, _, hist_sparse = ac.run_alpha_crown_batched(
        gg, xl, xh, bbr_init, w_qs, b_qs, intermediate, un,
        device, dtype, n_iters=5, lr=0.1, lr_decay=1.0,
        early_stop_on_positive=False, sparse_alpha=True)

    for q in range(n_q):
        assert np.allclose(hist_dense[q], hist_sparse[q], atol=1e-6), (
            f"q={q}: dense {hist_dense[q]} vs sparse {hist_sparse[q]}")
    assert np.allclose(lbs_dense, lbs_sparse, atol=1e-6)
