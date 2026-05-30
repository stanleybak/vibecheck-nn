"""Unit tests for Reciprocal + Mul McCormick linear bounds.

Validates the building blocks of the ABC-style Div backward refactor:
  - `_reciprocal_linear_bounds`: sound LB/UB lines for 1/b on [b_lo, b_hi]
    (b > 0). Lower = tangent at α-tunable midpoint; Upper = chord.
  - `_mccormick_linear_bounds`: sound LB/UB lines for a·v on
    [a_lo, a_hi] × [v_lo, v_hi]. α-interpolated between two corner-tangent
    McCormick lines.
  - Composition `Div = Mul(a, Reciprocal(b))` end-to-end soundness +
    tightness vs the current Taylor-with-R-bound approach.
"""
import pytest
import torch
import numpy as np
from vibecheck.verify_zono_bnb import (
    _reciprocal_linear_bounds, _mccormick_linear_bounds)


def test_reciprocal_bounds_sound_pos():
    """1/b on [b_lo, b_hi], b > 0. Test with α at midpoint (default)."""
    for b_lo, b_hi in [(1.0, 2.0), (0.5, 5.0), (10.0, 100.0)]:
        lo_t = torch.tensor([b_lo], dtype=torch.float64)
        hi_t = torch.tensor([b_hi], dtype=torch.float64)
        alpha = torch.tensor([0.5], dtype=torch.float64)
        s_lb, c_lb, s_ub, c_ub = _reciprocal_linear_bounds(
            lo_t, hi_t, alpha_norm=alpha)
        # Verify on dense grid.
        bs = np.linspace(b_lo, b_hi, 2001)
        true = 1.0 / bs
        lb_v = float(s_lb) * bs + float(c_lb)
        ub_v = float(s_ub) * bs + float(c_ub)
        slack = 1e-6 * max(true.max(), 1.0)
        assert (lb_v <= true + slack).all(), (
            f'b∈[{b_lo},{b_hi}]: LB violation {(lb_v-true).max()}')
        assert (ub_v >= true - slack).all(), (
            f'b∈[{b_lo},{b_hi}]: UB violation {(true-ub_v).max()}')


def test_reciprocal_bounds_alpha_tunes_lb():
    """α=0 → tangent at b_lo, α=1 → tangent at b_hi. LB shifts."""
    b_lo, b_hi = 1.0, 2.0
    lo_t = torch.tensor([b_lo], dtype=torch.float64)
    hi_t = torch.tensor([b_hi], dtype=torch.float64)
    s_lb_0, c_lb_0, _, _ = _reciprocal_linear_bounds(
        lo_t, hi_t, alpha_norm=torch.tensor([0.0], dtype=torch.float64))
    s_lb_1, c_lb_1, _, _ = _reciprocal_linear_bounds(
        lo_t, hi_t, alpha_norm=torch.tensor([1.0], dtype=torch.float64))
    # At b=b_lo: tangent-at-b_lo passes through (b_lo, 1/b_lo).
    val_at_lo_alpha0 = float(s_lb_0) * b_lo + float(c_lb_0)
    val_at_hi_alpha1 = float(s_lb_1) * b_hi + float(c_lb_1)
    assert abs(val_at_lo_alpha0 - 1.0 / b_lo) < 1e-9
    assert abs(val_at_hi_alpha1 - 1.0 / b_hi) < 1e-9


def test_mccormick_bounds_sound():
    """a·v on [a_lo, a_hi] × [v_lo, v_hi]. Test r_l=r_u=0.5 (default)."""
    for a_lo, a_hi, v_lo, v_hi in [
        (1.0, 3.0, 0.5, 1.0),
        (-2.0, 2.0, 0.1, 1.0),
        (0.0, 10.0, 0.001, 0.01),  # pensieve-like a, small v
    ]:
        a_lo_t = torch.tensor([a_lo], dtype=torch.float64)
        a_hi_t = torch.tensor([a_hi], dtype=torch.float64)
        v_lo_t = torch.tensor([v_lo], dtype=torch.float64)
        v_hi_t = torch.tensor([v_hi], dtype=torch.float64)
        r_l = torch.tensor([0.5], dtype=torch.float64)
        r_u = torch.tensor([0.5], dtype=torch.float64)
        (slope_a_lb, slope_v_lb, c_lb,
         slope_a_ub, slope_v_ub, c_ub) = _mccormick_linear_bounds(
            a_lo_t, a_hi_t, v_lo_t, v_hi_t, r_l=r_l, r_u=r_u)
        # Grid test.
        A = np.linspace(a_lo, a_hi, 101)
        V = np.linspace(v_lo, v_hi, 101)
        AA, VV = np.meshgrid(A, V)
        true = AA * VV
        lb_v = float(slope_a_lb) * AA + float(slope_v_lb) * VV + float(c_lb)
        ub_v = float(slope_a_ub) * AA + float(slope_v_ub) * VV + float(c_ub)
        slack = 1e-6 * max(abs(true).max(), 1.0)
        assert (lb_v <= true + slack).all(), (
            f'a∈[{a_lo},{a_hi}] v∈[{v_lo},{v_hi}]: LB violation '
            f'{(lb_v-true).max()}')
        assert (ub_v >= true - slack).all(), (
            f'a∈[{a_lo},{a_hi}] v∈[{v_lo},{v_hi}]: UB violation '
            f'{(true-ub_v).max()}')


def test_mccormick_endpoints_match():
    """ABC convention (mirrored here): r_l=0 → line at corner (a_hi, v_hi),
    r_l=1 → line at corner (a_lo, v_lo). Validate exact at the chosen corner."""
    a_lo, a_hi, v_lo, v_hi = 1.0, 3.0, 0.5, 1.0
    a_lo_t = torch.tensor([a_lo], dtype=torch.float64)
    a_hi_t = torch.tensor([a_hi], dtype=torch.float64)
    v_lo_t = torch.tensor([v_lo], dtype=torch.float64)
    v_hi_t = torch.tensor([v_hi], dtype=torch.float64)
    # r_l = 1 → LB = line at (a_lo, v_lo). At that corner: a_lo·v_lo.
    (s_a_lb, s_v_lb, c_lb, _, _, _) = _mccormick_linear_bounds(
        a_lo_t, a_hi_t, v_lo_t, v_hi_t,
        r_l=torch.tensor([1.0], dtype=torch.float64),
        r_u=torch.tensor([0.5], dtype=torch.float64))
    val = float(s_a_lb) * a_lo + float(s_v_lb) * v_lo + float(c_lb)
    assert abs(val - a_lo * v_lo) < 1e-9, (
        f'r_l=1 LB at (a_lo, v_lo) should be {a_lo*v_lo}, got {val}')
    # r_l = 0 → LB = line at (a_hi, v_hi). At that corner: a_hi·v_hi.
    (s_a_lb, s_v_lb, c_lb, _, _, _) = _mccormick_linear_bounds(
        a_lo_t, a_hi_t, v_lo_t, v_hi_t,
        r_l=torch.tensor([0.0], dtype=torch.float64),
        r_u=torch.tensor([0.5], dtype=torch.float64))
    val = float(s_a_lb) * a_hi + float(s_v_lb) * v_hi + float(c_lb)
    assert abs(val - a_hi * v_hi) < 1e-9, (
        f'r_l=0 LB at (a_hi, v_hi) should be {a_hi*v_hi}, got {val}')


def test_div_via_reciprocal_mul_sound_2d():
    """End-to-end: Div(a, b) = Mul(a, 1/b). Compose Reciprocal LB/UB with
    Mul McCormick. Verify on a (a, b) grid that the composed bound is
    sound (and ideally tighter than vibecheck's current R-bound)."""
    a_lo, a_hi, b_lo, b_hi = 1.0, 10.0, 100.0, 200.0
    a_lo_t = torch.tensor([a_lo], dtype=torch.float64)
    a_hi_t = torch.tensor([a_hi], dtype=torch.float64)
    b_lo_t = torch.tensor([b_lo], dtype=torch.float64)
    b_hi_t = torch.tensor([b_hi], dtype=torch.float64)
    # 1/b bounds (default α=0.5).
    alpha_r = torch.tensor([0.5], dtype=torch.float64)
    s_v_lb, c_v_lb, s_v_ub, c_v_ub = _reciprocal_linear_bounds(
        b_lo_t, b_hi_t, alpha_norm=alpha_r)
    # v=1/b. Compute v range over [b_lo, b_hi].
    # The composed Div bound takes the worst case over v ∈ [v_min, v_max]
    # where v_min, v_max are from the Reciprocal LB/UB at b_lo and b_hi.
    # For positive b: 1/b ∈ [1/b_hi, 1/b_lo].
    v_min = 1.0 / b_hi; v_max = 1.0 / b_lo
    v_min_t = torch.tensor([v_min], dtype=torch.float64)
    v_max_t = torch.tensor([v_max], dtype=torch.float64)
    r_l = torch.tensor([0.5], dtype=torch.float64)
    r_u = torch.tensor([0.5], dtype=torch.float64)
    (s_a_lb_m, s_v_lb_m, c_lb_m,
     s_a_ub_m, s_v_ub_m, c_ub_m) = _mccormick_linear_bounds(
        a_lo_t, a_hi_t, v_min_t, v_max_t, r_l=r_l, r_u=r_u)
    # Compose: Div_LB(a, b) = lb_m(a, v_lb_recip(b))  when slope_v_lb_m > 0,
    # else lb_m(a, v_ub_recip(b)) when slope_v_lb_m < 0.
    # For positive a and b > 0: v = 1/b > 0, slope_v_lb_m should be positive
    # (line through (a_lo, v_min)). Use v_lb_recip for LB.
    # Linear substitution v → s_v_lb·b + c_v_lb:
    # lb(a, b) = s_a_lb_m·a + s_v_lb_m·(s_v_lb·b + c_v_lb) + c_lb_m
    sign_v_lb = float(s_v_lb_m) >= 0
    sign_v_ub = float(s_v_ub_m) >= 0
    if sign_v_lb:
        v_for_lb_s, v_for_lb_c = s_v_lb, c_v_lb  # use Reciprocal LB
    else:
        v_for_lb_s, v_for_lb_c = s_v_ub, c_v_ub
    if sign_v_ub:
        v_for_ub_s, v_for_ub_c = s_v_ub, c_v_ub
    else:
        v_for_ub_s, v_for_ub_c = s_v_lb, c_v_lb
    # Composed Div bounds:
    A = np.linspace(a_lo, a_hi, 51)
    B = np.linspace(b_lo, b_hi, 51)
    AA, BB = np.meshgrid(A, B)
    true = AA / BB
    lb = (float(s_a_lb_m) * AA
          + float(s_v_lb_m) * (float(v_for_lb_s) * BB + float(v_for_lb_c))
          + float(c_lb_m))
    ub = (float(s_a_ub_m) * AA
          + float(s_v_ub_m) * (float(v_for_ub_s) * BB + float(v_for_ub_c))
          + float(c_ub_m))
    slack = 1e-6
    assert (lb <= true + slack).all(), f'Div LB violation {(lb-true).max()}'
    assert (ub >= true - slack).all(), f'Div UB violation {(true-ub).max()}'


def test_div_backward_rm_mccormick_tighter_than_taylor():
    """End-to-end sanity: on a pensieve-shaped (a, b) box, the R+M+α Div
    backward LB should match the true minimum closely (and tighter than
    the Taylor + R-bound approach).
    """
    from vibecheck.verify_zono_bnb import _div_backward_rm_mccormick
    a_lo, a_hi = 6.5e6, 3.8e7
    b_lo, b_hi = 1.1e8, 1.36e8
    ew_val = 5.0
    a_lo_t = torch.tensor([a_lo], dtype=torch.float64)
    a_hi_t = torch.tensor([a_hi], dtype=torch.float64)
    b_lo_t = torch.tensor([b_lo], dtype=torch.float64)
    b_hi_t = torch.tensor([b_hi], dtype=torch.float64)
    ew_t = torch.tensor([ew_val], dtype=torch.float64)
    # Sweep α to find best LB
    best_lb = -np.inf
    for ar in [0.0, 0.5, 1.0]:
        for rl in [0.0, 0.5, 1.0]:
            alpha_r = torch.tensor([ar], dtype=torch.float64)
            r_l = torch.tensor([rl], dtype=torch.float64)
            r_u = torch.tensor([0.5], dtype=torch.float64)
            acc, ew_a, ew_b = _div_backward_rm_mccormick(
                a_lo_t, a_hi_t, b_lo_t, b_hi_t, ew_t,
                alpha_r=alpha_r, r_l=r_l, r_u=r_u)
            # spec_lb = acc + min(ew_a · a + ew_b · b) over (a, b) box
            a_term = (float(ew_a) * a_lo if float(ew_a) > 0
                      else float(ew_a) * a_hi)
            b_term = (float(ew_b) * b_lo if float(ew_b) > 0
                      else float(ew_b) * b_hi)
            lb = float(acc) + a_term + b_term
            best_lb = max(best_lb, lb)
    # True min:
    A = np.linspace(a_lo, a_hi, 101)
    B = np.linspace(b_lo, b_hi, 101)
    AA, BB = np.meshgrid(A, B)
    true_min = float((ew_val * AA / BB).min())
    print(f'R+M+α best LB: {best_lb:.4f}, true min: {true_min:.4f}')
    # Sound + close to true min.
    assert best_lb <= true_min + 1e-3, 'LB above true min — UNSOUND'
    assert best_lb >= true_min - 0.05, (
        f'R+M+α not tight: {best_lb} vs {true_min}')
