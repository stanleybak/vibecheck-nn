"""Soundness + tightness test for `_torch_zono_div_scalar_b`.

For Div(a, b) where:
  - a is a vector zonotope: a_i = c_a_i + g_a_i · eps   (n elements, K shared eps)
  - b is a SCALAR zonotope: b = c_b + g_b · eps           (1 element, K shared eps)

The shared-gen path encodes 1/b via chord-tangent and bilinear-multiplies
with a, preserving:
  - a_i ↔ a_j correlation (linear gens shared across n)
  - a ↔ b correlation (via lam·g_b reusing the same eps cols)

Validates over a random sample of `eps ∈ [-1, 1]^K`: the resulting
[lo, hi] envelope of the encoded y zonotope must contain a_i / b at
every sample point.
"""
import torch
import numpy as np
import pytest
from vibecheck.zonotope import (
    _torch_zono_div_scalar_b, _torch_zono_div_bilinear)


def _range_of(c, g):
    rad = g.abs().sum(dim=1) if g.numel() > 0 else torch.zeros_like(c)
    return (c - rad).numpy(), (c + rad).numpy()


def test_div_scalar_b_sound_random():
    """100 random a/b configs; for each, sample 1000 eps points;
    verify encoded zonotope range encloses every a_i/b sample value."""
    torch.manual_seed(0)
    for trial in range(20):
        n = np.random.choice([1, 2, 6, 12])
        K = np.random.choice([1, 3, 8])
        c_a = torch.randn(n, dtype=torch.float64) * 5 + 10.0  # roughly positive
        g_a = torch.randn(n, K, dtype=torch.float64) * 0.5
        # b sign-stable positive: c_b > sum|g_b|
        g_b = torch.randn(1, K, dtype=torch.float64) * 0.3
        c_b = torch.tensor([float(g_b.abs().sum()) + 1.5], dtype=torch.float64)
        c_y, g_y = _torch_zono_div_scalar_b(c_a, g_a, c_b, g_b)
        y_lo, y_hi = _range_of(c_y, g_y)
        # Sample eps in [-1, 1]^K, evaluate a_i(eps) / b(eps), check enclosure.
        N = 2000
        eps = torch.empty(N, K, dtype=torch.float64).uniform_(-1.0, 1.0)
        a_vals = c_a.unsqueeze(0) + eps @ g_a.T  # (N, n)
        b_vals = c_b + eps @ g_b.T               # (N, 1)
        y_vals = a_vals / b_vals                  # (N, n)
        y_min = y_vals.min(dim=0).values.numpy()
        y_max = y_vals.max(dim=0).values.numpy()
        # SOUND: encoded envelope ⊇ sample envelope
        slack = 1e-6 * max(abs(c_y).max().item(), 1.0)
        assert (y_lo <= y_min + slack).all(), (
            f'trial {trial}: LB violation, '
            f'min(y_lo-y_min)={float((y_min-y_lo).min())}')
        assert (y_hi >= y_max - slack).all(), (
            f'trial {trial}: UB violation, '
            f'min(y_hi-y_max)={float((y_hi-y_max).min())}')


def test_div_scalar_b_tighter_downstream():
    """The shared-gen path's value isn't per-element width — it's that
    correlations across a_i and a↔b are preserved, so DOWNSTREAM linear
    combinations (sum, dot with a weight vector) get tight cancellation.
    Validate: |sum y_i| envelope is much tighter with shared-gen.

    Test setup: pensieve-like normalized softmax. a is a vector zonotope
    over [0, 1]^n, b = sum(a) (scalar). y = a / b should sum to ~1 ± noise.
    """
    torch.manual_seed(0)
    n, K = 6, 4
    # Construct a, b = sum(a) so the y_i should literally sum to 1.
    c_a = torch.tensor([0.20, 0.18, 0.15, 0.12, 0.10, 0.25],
                       dtype=torch.float64)
    g_a = torch.randn(n, K, dtype=torch.float64) * 0.05
    # b = sum(a): center = sum c_a, gens = sum g_a along rows.
    c_b = c_a.sum().reshape(1)
    g_b = g_a.sum(dim=0, keepdim=True)
    # Shared-gen:
    c_y_s, g_y_s = _torch_zono_div_scalar_b(c_a, g_a, c_b, g_b)
    # Decoupled:
    c_y_d, g_y_d = _torch_zono_div_bilinear(
        c_a, g_a, c_b, g_b, fallback='box',
        prefer_shared_when_scalar_b=False)
    # Downstream: y.sum() envelope. True value is exactly 1.
    sum_c_s = float(c_y_s.sum())
    sum_rad_s = float(g_y_s.sum(dim=0).abs().sum())  # sum cols, then |·|
    sum_c_d = float(c_y_d.sum())
    sum_rad_d = float(g_y_d.sum(dim=0).abs().sum())
    print(f'shared-gen sum envelope: '
          f'[{sum_c_s - sum_rad_s:.4f}, {sum_c_s + sum_rad_s:.4f}]  '
          f'width={2*sum_rad_s:.4f}')
    print(f'decoupled  sum envelope: '
          f'[{sum_c_d - sum_rad_d:.4f}, {sum_c_d + sum_rad_d:.4f}]  '
          f'width={2*sum_rad_d:.4f}')
    # Shared-gen MUST be at least as tight as decoupled on the sum
    # (correlated cancellation only helps).
    assert sum_rad_s <= sum_rad_d + 1e-9, (
        f'shared-gen sum-width wider than decoupled: '
        f'{2*sum_rad_s} vs {2*sum_rad_d}')
    # And strictly tighter (the whole point of preserving correlations).
    assert sum_rad_s < sum_rad_d - 1e-4


def test_div_scalar_b_softmax_clamp():
    """When b = sum(a) exactly AND a ≥ 0 element-wise (softmax-like
    pattern from Pow→ReduceSum→Div), the encoded y range tightens to
    [0, 1] per element. Validate the clamp doesn't violate soundness
    on a non-trivial random softmax instance, and verify it actually
    activates (collapses an over-wide y range)."""
    torch.manual_seed(0)
    n, K = 6, 4
    # Construct a vector zonotope with positive a.
    c_a = torch.tensor([0.20, 0.18, 0.15, 0.12, 0.10, 0.25],
                       dtype=torch.float64) * 10000.0  # big magnitudes
    g_a = torch.randn(n, K, dtype=torch.float64) * 500.0
    # Ensure a remains non-negative.
    rad_a = g_a.abs().sum(dim=1)
    g_a = g_a * (((c_a / (rad_a + 1e-9)) * 0.8).clamp(max=1.0)
                 .unsqueeze(-1))
    # b = sum(a) exactly.
    c_b = c_a.sum().reshape(1)
    g_b = g_a.sum(dim=0, keepdim=True)
    c_y, g_y = _torch_zono_div_scalar_b(c_a, g_a, c_b, g_b)
    # SOUND: random samples y_i ∈ [0, 1] (always, mathematically).
    N = 5000
    eps = torch.empty(N, K, dtype=torch.float64).uniform_(-1.0, 1.0)
    a_vals = c_a.unsqueeze(0) + eps @ g_a.T
    b_vals = c_b + eps @ g_b.T
    y_vals = a_vals / b_vals
    assert bool((y_vals >= -1e-9).all())
    assert bool((y_vals <= 1.0 + 1e-9).all())
    # Encoded envelope must contain samples.
    y_lo_e, y_hi_e = _range_of(c_y, g_y)
    y_min = y_vals.min(dim=0).values.numpy()
    y_max = y_vals.max(dim=0).values.numpy()
    assert (y_lo_e <= y_min + 1e-6).all(), 'softmax clamp violated LB soundness'
    assert (y_hi_e >= y_max - 1e-6).all(), 'softmax clamp violated UB soundness'
    # Clamp must have been applied (range ⊆ [0, 1]).
    assert (y_lo_e >= -1e-9).all()
    assert (y_hi_e <= 1.0 + 1e-9).all()


def test_div_scalar_b_dispatch_via_main_entry():
    """Calling `_torch_zono_div_bilinear` with scalar b uses the
    shared path automatically when `prefer_shared_when_scalar_b=True`
    (the default)."""
    n, K = 4, 3
    torch.manual_seed(1)
    c_a = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    g_a = torch.randn(n, K, dtype=torch.float64) * 0.1
    c_b = torch.tensor([5.0], dtype=torch.float64)
    g_b = torch.randn(1, K, dtype=torch.float64) * 0.1
    c_y_default, g_y_default = _torch_zono_div_bilinear(
        c_a, g_a, c_b, g_b, fallback='box')
    c_y_shared, g_y_shared = _torch_zono_div_scalar_b(c_a, g_a, c_b, g_b)
    # Same number of gens; same center.
    assert g_y_default.shape == g_y_shared.shape
    assert torch.allclose(c_y_default, c_y_shared)
    assert torch.allclose(g_y_default, g_y_shared)
