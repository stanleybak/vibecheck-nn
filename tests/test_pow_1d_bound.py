"""1D ground-truth Pow bound check.

Validates `_pow_two_line_coeffs` (and `_pow_chord_coeffs` parallelogram):
  - Sound on dense sample over [lo, hi] for every sign/exponent regime.
  - Reports gap vs true min/max so we know where α-CROWN tightening
    helps (and where it doesn't).

Run: .venv/bin/python -m pytest tests/test_pow_1d_bound.py -v
"""
import math
import pytest
import torch
import numpy as np
from vibecheck.verify_zono_bnb import (
    _pow_chord_coeffs, _pow_two_line_coeffs)


def _bounds_from_two_line(lo, hi, p, tangent_pos=None):
    lb_s, lb_c, ub_s, ub_c, ok, blo, bhi = _pow_two_line_coeffs(
        torch.tensor([lo]), torch.tensor([hi]), p,
        tangent_pos=(torch.tensor([tangent_pos]) if tangent_pos is not None
                     else None))
    return (float(lb_s), float(lb_c), float(ub_s), float(ub_c),
            bool(ok), float(blo), float(bhi))


# (lo, hi, exponent, regime)
POW_CASES = [
    (1.0, 2.0, 3, 'pos_odd'),
    (0.5, 5.0, 3, 'pos_odd_wide'),
    (-2.0, -1.0, 3, 'neg_odd'),
    (-2.0, -1.0, 2, 'neg_even'),
    (1.0, 2.0, 2, 'pos_even'),
    (-1.0, 1.0, 2, 'mix_even'),
    (-1.0, 1.0, 3, 'mix_odd'),
    (10.0, 20.0, 3, 'pos_big'),
    (0.0, 5.0, 3, 'zero_to_pos'),
]


@pytest.mark.parametrize('lo,hi,p,name', POW_CASES,
    ids=[c[3] for c in POW_CASES])
def test_pow_two_line_sound(lo, hi, p, name):
    lb_s, lb_c, ub_s, ub_c, ok, blo, bhi = _bounds_from_two_line(lo, hi, p)
    xs = np.linspace(lo, hi, 4001)
    ys = xs ** p
    if ok:
        lb_vals = lb_s * xs + lb_c
        ub_vals = ub_s * xs + ub_c
        # SOUND: LB ≤ y ≤ UB on the whole interval, within fp slack.
        slack = max(1e-5 * max(abs(ys).max(), 1.0), 1e-5)
        assert (lb_vals <= ys + slack).all(), (
            f'{name}: LB violation max={float((lb_vals-ys).max()):.4e}')
        assert (ub_vals >= ys - slack).all(), (
            f'{name}: UB violation max={float((ys-ub_vals).max()):.4e}')
    else:
        slack = max(1e-5 * max(abs(ys).max(), 1.0), 1e-5)
        assert (blo <= ys.min() + slack), (
            f'{name}: box_lo {blo} > true min {ys.min()}')
        assert (bhi >= ys.max() - slack), (
            f'{name}: box_hi {bhi} < true max {ys.max()}')


@pytest.mark.parametrize('lo,hi,p,name', POW_CASES,
    ids=[c[3] for c in POW_CASES])
def test_pow_chord_paralleogram_sound(lo, hi, p, name):
    lam, mu, gamma, ok, blo, bhi = _pow_chord_coeffs(
        torch.tensor([lo]), torch.tensor([hi]), p)
    xs = np.linspace(lo, hi, 4001)
    ys = xs ** p
    if bool(ok):
        # Parallelogram: y(x) ∈ [lam*x + mu - gamma, lam*x + mu + gamma]
        # for x ∈ [lo, hi]. New gen carries ±gamma.
        lb_vals = float(lam) * xs + float(mu) - float(gamma)
        ub_vals = float(lam) * xs + float(mu) + float(gamma)
        slack = max(1e-5 * max(abs(ys).max(), 1.0), 1e-5)
        assert (lb_vals <= ys + slack).all(), name
        assert (ub_vals >= ys - slack).all(), name
    else:
        slack = max(1e-5 * max(abs(ys).max(), 1.0), 1e-5)
        assert (float(blo) <= ys.min() + slack), name
        assert (float(bhi) >= ys.max() - slack), name


def test_pow_two_line_tangent_position_affects_lb():
    """Convex case: LB is tangent. Choosing tangent_pos ≠ midpoint
    should shift the LB. For convex f, tangent at p underestimates f
    on [lo, hi] uniformly; ANY p ∈ [lo, hi] is sound. Best tangent for
    a downstream maximization depends on the linear functional weight."""
    lo, hi, p = 1.0, 2.0, 3
    lb_s_mid, lb_c_mid, *_ = _bounds_from_two_line(lo, hi, p, tangent_pos=1.5)
    lb_s_lo, lb_c_lo, *_  = _bounds_from_two_line(lo, hi, p, tangent_pos=1.05)
    lb_s_hi, lb_c_hi, *_  = _bounds_from_two_line(lo, hi, p, tangent_pos=1.95)
    # At x=lo: tangent close to lo gives BEST (tightest) LB.
    val_at_lo_mid = lb_s_mid * lo + lb_c_mid
    val_at_lo_near_lo = lb_s_lo * lo + lb_c_lo
    val_at_lo_near_hi = lb_s_hi * lo + lb_c_hi
    true_lo = lo ** p  # 1.0
    assert val_at_lo_near_lo > val_at_lo_mid > val_at_lo_near_hi
    assert val_at_lo_near_lo <= true_lo + 1e-4
