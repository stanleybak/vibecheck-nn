"""Soundness tests for `_sigmoid_tanh_chord_parallelogram`.

The parallelogram is sound iff for every (lo, hi) interval and every
x ∈ [lo, hi], σ(x) lies within [α·x + β - γ, α·x + β + γ]. We test
this empirically across many intervals and many sample points per
interval — soundness failures fail loudly.
"""
import numpy as np
import torch
import pytest
from vibecheck.verify_zono_bnb import _sigmoid_tanh_chord_parallelogram


def _check_sound(lo, hi, act_kind, n_samples=5000, atol=1e-5):
    """Verify σ(x) ∈ [α·x + β - γ, α·x + β + γ] for many x ∈ [lo, hi]."""
    lo_t = torch.as_tensor(lo, dtype=torch.float64)
    hi_t = torch.as_tensor(hi, dtype=torch.float64)
    alpha, beta, gamma = _sigmoid_tanh_chord_parallelogram(
        lo_t, hi_t, act_kind)
    act = torch.sigmoid if act_kind == 'sigmoid' else torch.tanh
    # Sample n_samples points uniformly over each cell.
    n_cells = lo_t.numel()
    lo_b = lo_t.reshape(-1)
    hi_b = hi_t.reshape(-1)
    alpha_b = alpha.reshape(-1)
    beta_b = beta.reshape(-1)
    gamma_b = gamma.reshape(-1)
    # Random points + endpoints + middle (deterministic — paranoid).
    rng = np.random.default_rng(0)
    u = rng.random((n_samples, n_cells))
    u = np.concatenate([
        u, np.zeros((1, n_cells)), np.ones((1, n_cells)),
        np.full((1, n_cells), 0.5)], axis=0)
    u_t = torch.as_tensor(u, dtype=torch.float64)
    x = lo_b + (hi_b - lo_b) * u_t  # shape (n_samples+3, n_cells)
    y = act(x)
    upper = alpha_b * x + beta_b + gamma_b
    lower = alpha_b * x + beta_b - gamma_b
    # Worst violations
    upper_viol = (y - upper).max().item()
    lower_viol = (lower - y).max().item()
    assert upper_viol <= atol, (
        f'{act_kind} upper violation {upper_viol:.4e} '
        f'(lo range {lo_b.min():.3f}..{lo_b.max():.3f}, '
        f'hi range {hi_b.min():.3f}..{hi_b.max():.3f})')
    assert lower_viol <= atol, (
        f'{act_kind} lower violation {lower_viol:.4e}')
    # Sanity: γ ≥ 0
    assert (gamma_b >= -atol).all().item()


@pytest.mark.parametrize('act', ['sigmoid', 'tanh'])
def test_pure_convex(act):
    """[lo, hi] entirely in convex region (hi ≤ 0)."""
    lo = np.linspace(-8.0, -0.5, 50)
    hi = lo + np.linspace(0.1, 5.0, 50)  # widths 0.1 to 5.0
    hi = np.minimum(hi, -0.01)  # ensure hi ≤ 0
    _check_sound(lo, hi, act)


@pytest.mark.parametrize('act', ['sigmoid', 'tanh'])
def test_pure_concave(act):
    """[lo, hi] entirely in concave region (lo ≥ 0)."""
    lo = np.linspace(0.01, 8.0, 50)
    hi = lo + np.linspace(0.1, 5.0, 50)
    _check_sound(lo, hi, act)


@pytest.mark.parametrize('act', ['sigmoid', 'tanh'])
def test_mixed_straddle_zero(act):
    """[lo, hi] straddles zero."""
    lo = np.linspace(-5.0, -0.1, 100)
    hi = -lo + np.linspace(0.01, 0.5, 100)  # symmetric and asymmetric
    _check_sound(lo, hi, act)


@pytest.mark.parametrize('act', ['sigmoid', 'tanh'])
def test_tiny_intervals(act):
    """Very narrow intervals (numerical edge)."""
    lo = np.linspace(-3.0, 3.0, 200)
    hi = lo + 1e-6
    _check_sound(lo, hi, act, n_samples=200)


@pytest.mark.parametrize('act', ['sigmoid', 'tanh'])
def test_wide_intervals(act):
    """Very wide intervals (high saturation)."""
    lo = np.full(20, -20.0)
    hi = np.full(20, 20.0)
    _check_sound(lo, hi, act)


@pytest.mark.parametrize('act', ['sigmoid', 'tanh'])
def test_random_intervals(act):
    """Many random (lo, hi) intervals across the range."""
    rng = np.random.default_rng(42)
    lo = rng.uniform(-10.0, 10.0, size=500)
    hi = lo + rng.uniform(0.01, 5.0, size=500)
    _check_sound(lo, hi, act, n_samples=2000)


@pytest.mark.parametrize('act', ['sigmoid', 'tanh'])
def test_zero_width_interval(act):
    """lo == hi (degenerate; should still be sound)."""
    lo = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    hi = lo.copy()
    _check_sound(lo, hi, act, n_samples=10)


def test_tightness_vs_box_sigmoid():
    """Parallelogram γ should be ≤ half the box width (s_hi - s_lo)/2."""
    rng = np.random.default_rng(0)
    lo = rng.uniform(-3.0, 0.0, size=100)
    hi = lo + rng.uniform(0.1, 3.0, size=100)
    lo_t = torch.as_tensor(lo); hi_t = torch.as_tensor(hi)
    _, _, gamma = _sigmoid_tanh_chord_parallelogram(lo_t, hi_t, 'sigmoid')
    s_lo = torch.sigmoid(lo_t); s_hi = torch.sigmoid(hi_t)
    box_gamma = (s_hi - s_lo) / 2
    # On average, parallelogram γ should be substantially smaller than
    # box γ — the whole point of using the chord slope.
    ratio = (gamma / box_gamma.clamp(min=1e-12)).mean().item()
    assert ratio < 0.5, (
        f'parallelogram γ not tighter than box: ratio={ratio:.3f}')


def test_critical_point_inside_sigmoid():
    """Mixed-case interval where α < 0.25 (critical points exist)."""
    # lo=-3, hi=3 → α very small, critical points well inside [-3, 3]
    lo = torch.tensor([-3.0])
    hi = torch.tensor([3.0])
    alpha, beta, gamma = _sigmoid_tanh_chord_parallelogram(lo, hi, 'sigmoid')
    # Verify pointwise
    xs = torch.linspace(-3.0, 3.0, 1000, dtype=torch.float64)
    ys = torch.sigmoid(xs)
    upper = alpha * xs + beta + gamma
    lower = alpha * xs + beta - gamma
    assert (ys <= upper + 1e-6).all().item()
    assert (ys >= lower - 1e-6).all().item()
