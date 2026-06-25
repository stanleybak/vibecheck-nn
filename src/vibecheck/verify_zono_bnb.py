"""Branch-and-Bound verification with zonotope forward + CROWN backward."""

import time
import numpy as np
import torch
import torch.nn.functional as F

from .settings import default_settings, resolve_torch
from .zonotope import TorchZonotope
from .patches_zonotope import PatchesZonotope
from .broadcast_util import (reconstruct_live_shape, broadcast_rows_backward,
                             assert_no_outer_broadcast)


def _bcast_ew_back(ew, op):
    """CROWN backward adjoint of an ONNX outer-broadcast on input[0]. If the op
    forward-expanded its live input (in numel != out numel, e.g. Sub (N,)-(M,)
    -> (N,M)), the effective weight at the broadcast outputs must be SUMMED back
    onto the live rows; otherwise ew passes through unchanged. Matches the
    forward broadcast layout via the shared `broadcast_util`."""
    in_shapes = op.get('in_shapes_nd')
    out_sh = op.get('out_shape_nd')
    if not in_shapes or in_shapes[0] is None or out_sh is None:
        return ew
    in_sh = tuple(int(d) for d in in_shapes[0])
    out_sh = tuple(int(d) for d in out_sh)
    n_in = int(np.prod(in_sh)); n_out = int(np.prod(out_sh))
    if n_in == n_out:
        return ew
    bias = op.get('bias')
    bt = np.asarray(bias) if bias is not None else np.zeros(())
    const_sh = tuple(bt.shape) if bt.ndim > 0 else ()
    live_sh, _ = reconstruct_live_shape(out_sh, const_sh, in_sh, n_in)
    return broadcast_rows_backward(ew, live_sh, out_sh, ew.dim() - 1)


def _sigmoid_tanh_chord_parallelogram(lo, hi, act_kind):
    """Tight sound parallelogram for σ(x) over x ∈ [lo, hi].

    Returns (alpha, beta, gamma) tensors of the same shape as lo, hi,
    such that for all x in [lo[i], hi[i]] and all e_new ∈ [-1, 1]:
        |σ(x) - (alpha[i] * x + beta[i] + gamma[i] * e_new)| <= 0

    i.e., the parallelogram { y = α·x + β + γ·e_new : x ∈ [lo, hi],
    e_new ∈ [-1,1] } strictly contains σ([lo, hi]).

    Method:
      - α = chord slope (σ(hi) - σ(lo)) / (hi - lo).
      - f(x) := σ(x) - α·x. f(lo) = f(hi) = β_chord (chord intersects
        σ at endpoints).
      - Extremes of f over [lo, hi] occur at endpoints OR at critical
        points where σ'(x) = α. For sigmoid σ'(x) = σ(x)(1-σ(x)) ≤ 0.25,
        critical points are x_± = ±atanh(sqrt(1 - 4α)) (when α ≤ 0.25);
        for tanh σ'(x) = 1 - tanh²(x) ≤ 1, critical points are
        x_± = ±atanh(sqrt(1 - α)) (when α ≤ 1).
      - β = midpoint of (min_f, max_f). γ = half-width.

    Sound by construction: σ(x) - α·x ∈ [min_f, max_f] ⇒ β ± γ wide
    enough to contain σ(x) - α·x for every x in [lo, hi].

    Used by `forward_zono_dir_adaptive` to upgrade Sigmoid/Tanh from
    box-relax (drops input correlation) to parallelogram (preserves
    correlation through the α slope, only the γ slack noise is new).
    """
    if act_kind == 'sigmoid':
        act = torch.sigmoid
    elif act_kind == 'tanh':
        act = torch.tanh
    else:
        raise ValueError(f'unknown act_kind {act_kind!r}')
    s_lo = act(lo); s_hi = act(hi)
    width = (hi - lo).clamp(min=1e-12)
    alpha = (s_hi - s_lo) / width
    beta_chord = s_lo - alpha * lo  # = s_hi - alpha * hi (chord intercept)

    # Critical points where σ'(x) = α. For sigmoid σ' ≤ 0.25, for
    # tanh σ' ≤ 1. If α exceeds the max, no critical points; f is
    # monotone, extremes only at endpoints (both = β_chord).
    #
    # Sigmoid: σ'(x) = σ(x)(1-σ(x)) = α → σ = (1±sqrt(1-4α))/2,
    #   x = logit(σ) = log(σ/(1-σ)) = 2·atanh(sqrt(1-4α)).
    # Tanh:    σ'(x) = 1 - tanh²(x) = α → tanh²(x) = 1-α,
    #   x = ±atanh(sqrt(1-α)).
    if act_kind == 'sigmoid':
        max_deriv = 0.25
        disc = (1.0 - 4.0 * alpha).clamp(min=0.0)
        sqrt_disc = torch.sqrt(disc).clamp(max=1 - 1e-9)
        x_plus = 2.0 * torch.atanh(sqrt_disc)
    else:  # tanh
        max_deriv = 1.0
        disc = (1.0 - alpha).clamp(min=0.0)
        sqrt_disc = torch.sqrt(disc).clamp(max=1 - 1e-9)
        x_plus = torch.atanh(sqrt_disc)
    x_minus = -x_plus

    # f at critical points (clamped to β_chord if outside [lo, hi]).
    in_plus = (x_plus >= lo) & (x_plus <= hi) & (alpha <= max_deriv)
    in_minus = (x_minus >= lo) & (x_minus <= hi) & (alpha <= max_deriv)
    f_plus = act(x_plus) - alpha * x_plus
    f_minus = act(x_minus) - alpha * x_minus
    f_plus_eff = torch.where(in_plus, f_plus, beta_chord)
    f_minus_eff = torch.where(in_minus, f_minus, beta_chord)

    max_f = torch.maximum(beta_chord,
                            torch.maximum(f_plus_eff, f_minus_eff))
    min_f = torch.minimum(beta_chord,
                            torch.minimum(f_plus_eff, f_minus_eff))
    beta = (max_f + min_f) / 2
    gamma = (max_f - min_f) / 2
    return alpha, beta, gamma


def _reciprocal_linear_bounds(b_lo, b_hi, alpha_norm=None,
                                skip_positivity_check=False):
    """Sound linear bounds for 1/b on [b_lo, b_hi] (b > 0).

    Returns `(slope_lb, const_lb, slope_ub, const_ub)` such that for
    b ∈ [b_lo, b_hi]:
        slope_lb·b + const_lb ≤ 1/b ≤ slope_ub·b + const_ub

    The 1/b function is convex decreasing on b > 0:
      - Upper (chord): slope = -1/(b_lo·b_hi), passes through
        (b_lo, 1/b_lo). Independent of α.
      - Lower (tangent at α-tunable mid ∈ [b_lo, b_hi]):
        slope = -1/mid², const = 2/mid.
        Mirrors α,β-CROWN's `BoundReciprocal.bound_relax`.

    `alpha_norm`: optional tensor in [0, 1] (broadcastable to b_lo).
        mid = b_lo + alpha_norm·(b_hi - b_lo). If None: midpoint.
    """
    import torch
    # `skip_positivity_check=True` skips the bool().item() GPU sync;
    # caller is responsible for graph-static b > 0 invariant (e.g., mscn
    # softmax denominators are always positive by construction). Needed
    # for jit.trace / sync-free paths.
    if not skip_positivity_check:
        assert bool((b_lo > 0).all()), (
            f'_reciprocal_linear_bounds requires b > 0; got b_lo={b_lo}')
    # Upper: chord with slope -1/(b_lo·b_hi).
    slope_ub = -1.0 / (b_lo * b_hi)
    const_ub = 1.0 / b_lo + 1.0 / b_hi  # since chord through (b_lo, 1/b_lo)
    # Lower: tangent at mid.
    if alpha_norm is None:
        mid = (b_lo + b_hi) / 2
    else:
        mid = b_lo + alpha_norm * (b_hi - b_lo)
        mid = torch.maximum(torch.minimum(mid, b_hi), b_lo)
    slope_lb = -1.0 / (mid * mid)
    const_lb = 2.0 / mid
    return slope_lb, const_lb, slope_ub, const_ub


def _mccormick_linear_bounds(a_lo, a_hi, v_lo, v_hi, r_l=None, r_u=None):
    """α-interpolated McCormick linear bounds for a·v on a box.

    Returns `(slope_a_lb, slope_v_lb, const_lb, slope_a_ub, slope_v_ub,
    const_ub)` such that for (a, v) ∈ [a_lo, a_hi] × [v_lo, v_hi]:
        slope_a_lb·a + slope_v_lb·v + const_lb ≤ a·v
        a·v ≤ slope_a_ub·a + slope_v_ub·v + const_ub

    McCormick LB lines:
      line1 (at corner (a_lo, v_lo)):
        slope_a = v_lo, slope_v = a_lo, const = -a_lo·v_lo
      line2 (at corner (a_hi, v_hi)):
        slope_a = v_hi, slope_v = a_hi, const = -a_hi·v_hi
      LB = max(line1, line2). For α-interp:
        LB_interp = r_l·line1 + (1-r_l)·line2, with r_l ∈ [0, 1].

    McCormick UB lines (mirror with crossing corners):
      line3 (at (a_hi, v_lo)): slope_a = v_lo, slope_v = a_hi,
        const = -a_hi·v_lo
      line4 (at (a_lo, v_hi)): slope_a = v_hi, slope_v = a_lo,
        const = -a_lo·v_hi
      UB = min(line3, line4). Interp:
        UB_interp = r_u·line3 + (1-r_u)·line4.

    r_l, r_u: optional tensors in [0, 1]. Default: 0.5 (midpoint).

    Soundness: each McCormick corner-line is a SOUND tangent at that
    corner; convex combinations of two sound LB lines are sound LB.
    Mirrors α,β-CROWN's `MulHelper.interpolated_relaxation`.
    """
    import torch
    if r_l is None:
        r_l = torch.full_like(a_lo, 0.5)
    if r_u is None:
        r_u = torch.full_like(a_lo, 0.5)
    # LB lines (interpolated).
    slope_a_lb = (v_lo - v_hi) * r_l + v_hi
    slope_v_lb = (a_lo - a_hi) * r_l + a_hi
    const_lb = (v_hi * a_hi - v_lo * a_lo) * r_l - v_hi * a_hi
    # UB lines.
    slope_a_ub = (v_hi - v_lo) * r_u + v_lo
    slope_v_ub = (a_lo - a_hi) * r_u + a_hi
    const_ub = (v_lo * a_hi - v_hi * a_lo) * r_u - v_lo * a_hi
    return (slope_a_lb, slope_v_lb, const_lb,
            slope_a_ub, slope_v_ub, const_ub)


def _div_backward_rm_mccormick(a_lo, a_hi, b_lo, b_hi, ew,
                                 alpha_r=None, r_l=None, r_u=None):
    """ABC-style Div backward: Mul(a, Reciprocal(b)) with α-tunable
    McCormick + α-tunable Recip tangent. Returns the backward CROWN
    contribution for `ew · y` where `y = a / b`, sound on b > 0.

    Args (all torch tensors, broadcasting across leading dims):
      a_lo, a_hi: pre-Div input `a` bounds, shape (..., n_y).
      b_lo, b_hi: pre-Div input `b` bounds, shape (..., n_y) or broadcast.
      ew: output gradient slopes, shape (..., n_y).
      alpha_r: Reciprocal tangent normalized α ∈ [0, 1], shape
        broadcastable to b_lo. Default 0.5.
      r_l, r_u: McCormick LB/UB interpolation α ∈ [0, 1]. Default 0.5.

    Returns `(acc_contrib, ew_a, ew_b)`:
      acc_contrib: scalar contribution to acc (sum over n_y).
      ew_a: backward slopes on a, shape (..., n_y).
      ew_b: backward slopes on b, shape (..., n_y).

    Soundness: for ANY α ∈ [0, 1]^*, the bound below is sound. Adam
    tunes α to maximize the spec LB. Comparison with the Taylor +
    R-bound approach (`_div_decoupled`): the R+M bound is exact at the
    corner where (a, b) minimizes `a/b`, while R-bound has slack
    proportional to the (a, b) box width and Taylor expansion error.
    """
    import torch
    # Reciprocal LB/UB for v = 1/b on [b_lo, b_hi]:
    rs_lb, rc_lb, rs_ub, rc_ub = _reciprocal_linear_bounds(
        b_lo, b_hi, alpha_norm=alpha_r)
    # v's range over [b_lo, b_hi]: [1/b_hi, 1/b_lo] (1/b is decreasing).
    v_min = 1.0 / b_hi
    v_max = 1.0 / b_lo
    # McCormick LB/UB for a·v on (a, v) box:
    (s_a_lb_m, s_v_lb_m, c_lb_m,
     s_a_ub_m, s_v_ub_m, c_ub_m) = _mccormick_linear_bounds(
        a_lo, a_hi, v_min, v_max, r_l=r_l, r_u=r_u)
    # Sign-aware substitution of v = 1/b. For LB(y), use Mul LB; for
    # UB(y), use Mul UB. Within each, choose Recip LB or UB based on
    # the sign of s_v in the Mul line (positive → use Recip LB → tighter LB).
    pos_v_lb = (s_v_lb_m >= 0).to(s_v_lb_m.dtype)
    neg_v_lb = 1.0 - pos_v_lb
    pos_v_ub = (s_v_ub_m >= 0).to(s_v_ub_m.dtype)
    neg_v_ub = 1.0 - pos_v_ub
    sv_for_lb = pos_v_lb * rs_lb + neg_v_lb * rs_ub
    cv_for_lb = pos_v_lb * rc_lb + neg_v_lb * rc_ub
    sv_for_ub = neg_v_ub * rs_lb + pos_v_ub * rs_ub  # mirrored
    cv_for_ub = neg_v_ub * rc_lb + pos_v_ub * rc_ub
    # Substituted linear bounds in (a, b):
    # LB_y(a, b) = s_a_lb_m·a + s_v_lb_m·(sv_for_lb·b + cv_for_lb) + c_lb_m
    # UB_y(a, b) = s_a_ub_m·a + s_v_ub_m·(sv_for_ub·b + cv_for_ub) + c_ub_m
    coef_a_lb = s_a_lb_m
    coef_b_lb = s_v_lb_m * sv_for_lb
    const_lb = s_v_lb_m * cv_for_lb + c_lb_m
    coef_a_ub = s_a_ub_m
    coef_b_ub = s_v_ub_m * sv_for_ub
    const_ub = s_v_ub_m * cv_for_ub + c_ub_m
    # Backward contribution: ew_pos·LB + ew_neg·UB
    ep = ew.clamp(min=0)
    en = ew.clamp(max=0)
    ew_a = ep * coef_a_lb + en * coef_a_ub
    ew_b = ep * coef_b_lb + en * coef_b_ub
    acc_contrib = (ep * const_lb + en * const_ub).sum(dim=-1)
    return acc_contrib, ew_a, ew_b


def _pow_chord_coeffs(lo, hi, p):
    """Per-element chord-tangent coefficients for x^p on [lo_i, hi_i].

    Returns `(lam, mu, gamma, use_chord, box_lo, box_hi)`:
      - lam, mu, gamma: chord-parallelogram coeffs such that
          y = lam*x + mu + gamma*ε,  ε ∈ [-1, 1]
        bounds x^p when `use_chord` is True (sign-stable interval).
      - use_chord: bool mask where chord encoding is valid.
      - box_lo, box_hi: element-wise box bounds for fallback elements
        (used when `~use_chord`).

    Soundness: ranges out of the chord-tangent band were verified empirically
    for x^2, x^3 (see soundness test in tests/test_zonotope_pow.py).
    """
    f_lo = lo ** p
    f_hi = hi ** p
    diff = (hi - lo).clamp(min=1e-30)
    lam = (f_hi - f_lo) / diff
    lam_abs = lam.abs()
    x_star_mag = (lam_abs / p).clamp(min=0).pow(1.0 / (p - 1))
    x_star = torch.where(hi <= 0, -x_star_mag, x_star_mag)
    x_star = torch.maximum(torch.minimum(x_star, hi), lo)
    f_star = x_star ** p
    chord_at_star = lam * (x_star - lo) + f_lo
    gap_at_star = (chord_at_star - f_star).abs()
    chord_intercept = f_lo - lam * lo
    tangent_intercept = f_star - lam * x_star
    mu = (chord_intercept + tangent_intercept) / 2
    gamma = gap_at_star / 2
    use_chord = (lo >= 0) | (hi <= 0)
    if p % 2 == 0:
        box_lo_v = torch.where((lo <= 0) & (hi >= 0),
            torch.zeros_like(lo), torch.minimum(f_lo, f_hi))
        box_hi_v = torch.maximum(f_lo, f_hi)
    else:
        box_lo_v = torch.minimum(f_lo, f_hi)
        box_hi_v = torch.maximum(f_lo, f_hi)
    return lam, mu, gamma, use_chord, box_lo_v, box_hi_v


def _pow_two_line_coeffs(lo, hi, p, tangent_pos=None):
    """Per-element two-linear CROWN bounds for x^p on [lo_i, hi_i].

    Mirrors α,β-CROWN BoundPow's `bound_relax_branch`: separate LB and UB
    LINES (no shared slope, no slack new-gen). Strictly tighter than
    `_pow_chord_coeffs` parallelogram in CROWN backward because LB and
    UB use independent slopes.

    Returns `(lb_slope, lb_const, ub_slope, ub_const, use_two_line,
    box_lo, box_hi)` where:
      - For x in [lo_i, hi_i], `lb_slope·x + lb_const ≤ x^p ≤
        ub_slope·x + ub_const` whenever `use_two_line[i]` is True.
      - `box_lo, box_hi` are element-wise box bounds for fallback
        elements (`~use_two_line`).

    Convex case (lo >= 0 with any p≥2, OR hi <= 0 with even p):
      UB = chord through (lo, lo^p) and (hi, hi^p)
      LB = tangent at `tangent_pos` (defaults to midpoint of [lo, hi])
    Concave case (hi <= 0 with odd p, function on neg reals is concave):
      UB = tangent at `tangent_pos`
      LB = chord
    Otherwise (sign-mixed) → fallback to box (decorrelated).

    Soundness: chord IS always on the convex side (≥ f for convex,
    ≤ f for concave). Tangent at any interior point is on the opposite
    side (≤ f for convex, ≥ f for concave). Soundness independent of
    `tangent_pos` choice within [lo, hi].
    """
    f_lo = lo ** p
    f_hi = hi ** p
    diff = (hi - lo).clamp(min=1e-30)
    chord_slope = (f_hi - f_lo) / diff
    chord_intercept = f_lo - chord_slope * lo
    # Default tangent location: midpoint of [lo, hi]. α-CROWN can
    # later optimize this per-element.
    if tangent_pos is None:
        tangent_pos = (lo + hi) / 2
    tan_pos_clipped = torch.maximum(torch.minimum(tangent_pos, hi), lo)
    tan_slope = p * tan_pos_clipped.pow(p - 1)
    tan_const = tan_pos_clipped.pow(p) - tan_slope * tan_pos_clipped
    # Even powers are convex on ALL of R (f'' = p(p-1)x^(p-2) >= 0 since
    # p-2 is even), so the tangent-LB / chord-UB two-line is valid even on a
    # sign-crossing interval — NOT just lo>=0 or hi<=0. Boxing sign-mixed
    # even powers (the old behaviour) is needlessly loose and, for self-
    # product squares over a wide [-a, a], catastrophic.
    convex_regime = (lo >= 0) | (p % 2 == 0)
    concave_regime = (hi <= 0) & (p % 2 == 1)
    use_two_line = convex_regime | concave_regime
    # Sound assignment per regime: convex → LB=tangent, UB=chord;
    # concave → LB=chord, UB=tangent.
    lb_slope = torch.where(convex_regime, tan_slope, chord_slope)
    lb_const = torch.where(convex_regime, tan_const, chord_intercept)
    ub_slope = torch.where(convex_regime, chord_slope, tan_slope)
    ub_const = torch.where(convex_regime, chord_intercept, tan_const)
    # Mixed-sign fallback (sign-crossing interval): just use box.
    if p % 2 == 0:
        box_lo_v = torch.where((lo <= 0) & (hi >= 0),
            torch.zeros_like(lo), torch.minimum(f_lo, f_hi))
        box_hi_v = torch.maximum(f_lo, f_hi)
    else:
        box_lo_v = torch.minimum(f_lo, f_hi)
        box_hi_v = torch.maximum(f_lo, f_hi)
    return (lb_slope, lb_const, ub_slope, ub_const,
            use_two_line, box_lo_v, box_hi_v)


def _sigmoid_tanh_linear_bounds(lo, hi, act_kind, n_iter=30):
    """Sound closed-form linear bounds for sigmoid/tanh on [lo, hi].

    Returns (lo_s, lo_t, up_s, up_t) such that for all x ∈ [lo, hi]:
        lo_s * x + lo_t ≤ σ(x) ≤ up_s * x + up_t

    Method (mirrors auto_LiRPA's `precompute_relaxation` in tanh.py).
    Sigmoid σ'' = σ'(1 - 2σ), so σ is **convex** on (-∞, 0) (σ < 1/2) and
    **concave** on (0, +∞) (σ > 1/2). Tanh has the same convexity sign
    pattern about 0.
      • Pure convex (hi ≤ 0): chord ABOVE σ → upper = chord.
          tangent below σ → lower = tangent at midpoint.
      • Pure concave (lo ≥ 0): chord BELOW σ → lower = chord.
          tangent above σ → upper = tangent at midpoint.
      • Mixed (lo < 0 < hi): σ convex on [lo, 0], concave on [0, hi].
          Lower: tangent at p ∈ [lo, 0] such that the line passes through
          (hi, σ(hi)). σ'(p)*(hi-p) + σ(p) = σ(hi). Binary-search the unique
          root (g(p) is monotone increasing in p on (lo, 0) since σ'' > 0
          on the convex half).
          Upper: tangent at q ∈ [0, hi] such that the line passes through
          (lo, σ(lo)). σ'(q)*(q-lo) − σ(q) + σ(lo) = 0. Mirror.

    Returns tensors with the same shape as lo/hi."""
    if act_kind == 'sigmoid':
        act = torch.sigmoid
        def dact(x):
            s = act(x); return s * (1 - s)
    elif act_kind == 'tanh':
        act = torch.tanh
        def dact(x):
            s = act(x); return 1 - s * s
    else:
        raise ValueError(f'unknown act_kind {act_kind!r}')

    s_lo = act(lo); s_hi = act(hi)
    width = (hi - lo).clamp(min=1e-12)
    chord_slope = (s_hi - s_lo) / width
    chord_b = s_lo - chord_slope * lo

    # Tangent at midpoint (used for pure cases).
    mid = (lo + hi) / 2
    s_mid = act(mid); ds_mid = dact(mid)
    tang_mid_s = ds_mid
    tang_mid_b = s_mid - ds_mid * mid

    # -- Mixed-case lower tangent at p1 ∈ [lo, 0] s.t. line(hi) == σ(hi). --
    # g(p) = σ'(p)*(hi-p) + σ(p) - σ(hi); g monotone increasing on [lo, 0].
    p_l = torch.minimum(lo, torch.zeros_like(lo))
    p_r = torch.zeros_like(lo)
    for _ in range(n_iter):
        p_m = (p_l + p_r) / 2
        g_m = dact(p_m) * (hi - p_m) + act(p_m) - s_hi
        mask = g_m > 0
        p_r = torch.where(mask, p_m, p_r)
        p_l = torch.where(mask, p_l, p_m)
    p1 = (p_l + p_r) / 2
    g_lo = dact(lo) * (hi - lo) + s_lo - s_hi
    g_at_0 = dact(torch.zeros_like(lo)) * hi + act(torch.zeros_like(lo)) - s_hi
    # Tangent point in [lo, 0] if root exists; else fall back later.
    lo_s_mixed = dact(p1)
    lo_t_mixed = act(p1) - lo_s_mixed * p1
    # If g(lo) > 0: no root in [lo, 0]; the tangent at lo would have line(hi) > σ(hi).
    # No tangent in [lo, hi] gives a sound lower bound — fall back to the
    # constant y = σ(lo) (sound since σ is monotone increasing).
    fallback_lower = g_lo > 0
    lo_s_mixed = torch.where(fallback_lower, torch.zeros_like(lo), lo_s_mixed)
    lo_t_mixed = torch.where(fallback_lower, s_lo, lo_t_mixed)
    # If g(0) ≤ 0: no root either; use tangent at 0 (slope σ'(0)).
    no_root_left = g_at_0 <= 0
    lo_s_mixed = torch.where(no_root_left, dact(torch.zeros_like(lo)), lo_s_mixed)
    lo_t_mixed = torch.where(no_root_left, act(torch.zeros_like(lo)), lo_t_mixed)

    # -- Mixed-case upper tangent at q1 ∈ [0, hi] s.t. line(lo) == σ(lo). --
    # h(q) = σ'(q)*(q-lo) - (σ(q) - σ(lo)); h monotone DECREASING on [0, hi].
    q_l = torch.zeros_like(hi)
    q_r = torch.maximum(hi, torch.zeros_like(hi))
    for _ in range(n_iter):
        q_m = (q_l + q_r) / 2
        h_m = dact(q_m) * (q_m - lo) - (act(q_m) - s_lo)
        mask = h_m > 0
        q_l = torch.where(mask, q_m, q_l)
        q_r = torch.where(mask, q_r, q_m)
    q1 = (q_l + q_r) / 2
    h_hi = dact(hi) * (hi - lo) - (s_hi - s_lo)
    h_at_0 = dact(torch.zeros_like(hi)) * (-lo) - (act(torch.zeros_like(hi)) - s_lo)
    up_s_mixed = dact(q1)
    up_t_mixed = act(q1) - up_s_mixed * q1
    # If h(hi) > 0 (σ'(hi)*(hi-lo) > σ(hi) - σ(lo)): no valid q in [0, hi].
    # Fall back to constant y = σ(hi).
    fallback_upper = h_hi > 0
    up_s_mixed = torch.where(fallback_upper, torch.zeros_like(hi), up_s_mixed)
    up_t_mixed = torch.where(fallback_upper, s_hi, up_t_mixed)
    no_root_right = h_at_0 <= 0
    up_s_mixed = torch.where(no_root_right, dact(torch.zeros_like(hi)), up_s_mixed)
    up_t_mixed = torch.where(no_root_right, act(torch.zeros_like(hi)), up_t_mixed)

    # Combine cases. Sigmoid/tanh: convex on x<0, concave on x>0.
    is_convex = hi <= 0   # entire interval in convex region
    is_concave = lo >= 0  # entire interval in concave region
    # Convex: lower = tangent at midpoint, upper = chord
    # Concave: lower = chord, upper = tangent at midpoint
    # Mixed: lower/upper from binary search
    lo_s = torch.where(is_convex, tang_mid_s,
            torch.where(is_concave, chord_slope, lo_s_mixed))
    lo_t = torch.where(is_convex, tang_mid_b,
            torch.where(is_concave, chord_b, lo_t_mixed))
    up_s = torch.where(is_convex, chord_slope,
            torch.where(is_concave, tang_mid_s, up_s_mixed))
    up_t = torch.where(is_convex, chord_b,
            torch.where(is_concave, tang_mid_b, up_t_mixed))
    # ABC-style direct-chord fallback in mixed case (matches their
    # bound_relax_impl in tanh.py:270): when k_direct < dfunc(lower),
    # the chord from (lower, f(lower)) to (upper, f(upper)) is itself
    # below the curve on the convex half (and thus a valid LB on the
    # whole interval — chord at extremes equals f, between extremes
    # chord stays below curve in convex half, and below the linear UB
    # in the concave half). Similarly chord is valid UB when
    # k_direct < dfunc(upper). For mixed leaves where this triggers,
    # the chord is much tighter than the tangent-from-endpoint
    # (verified on mscn_2048d_dual_240 leaf 5: sigmoid input
    # [-2.99, 0.32], chord_slope=0.160 < dfunc(0.32)=0.244 → use chord
    # for UB; our tangent path gave looser UB).
    mixed = ~is_convex & ~is_concave
    chord_lb_ok = mixed & (chord_slope < dact(lo))
    chord_ub_ok = mixed & (chord_slope < dact(hi))
    lo_s = torch.where(chord_lb_ok, chord_slope, lo_s)
    lo_t = torch.where(chord_lb_ok, chord_b, lo_t)
    up_s = torch.where(chord_ub_ok, chord_slope, up_s)
    up_t = torch.where(chord_ub_ok, chord_b, up_t)
    return lo_s, lo_t, up_s, up_t


def _make_slopes(lo, hi):
    """Compute CROWN adaptive slopes for ReLU relaxation.

    Returns (lo_s, up_s, up_t, active_mask, dead_mask, unstable_mask).
    Works for both 1-D (n,) and (B, n) inputs — operations are
    elementwise so the batched form needs no separate implementation.
    """
    DT = lo.dtype
    lb_r = torch.clamp(lo, max=0)
    ub_r = torch.clamp(hi, min=0)
    ub_r = torch.maximum(ub_r, lb_r + 1e-8)
    up_s = ub_r / (ub_r - lb_r)
    up_t = -lb_r * up_s
    active = lo >= 0
    dead = hi <= 0
    unstable = ~active & ~dead
    lm = active.to(DT)
    um = dead.to(DT)
    lo_s = (up_s > 0.5).to(DT) * (1 - lm) * (1 - um) + lm
    return lo_s, up_s, up_t, active, dead, unstable


def _point_bcast_bias(a, bias, op, sign):
    """Batched point  y = a + sign*bias  with full ONNX broadcast, matching the
    zono forward handlers EXACTLY (the PGD forward must reproduce the network or
    sat-finding is unsound). `a` is (B, n_in). Three cases:
      - elementwise (bias scalar or n_in)            -> a + sign*bias
      - aligned/last-dim (prod(out_shape) == n_in)   -> reshape a to out_shape,
        right-align bias and broadcast (the Add (3,C)+(C,) pattern)
      - genuine OUTER broadcast (a expands into out)  -> mirror
        `_zono_broadcast_const`'s live-shape reconstruction (Sub (N,)-(M,)->(N,M)).
    """
    bt = torch.as_tensor(np.asarray(bias), dtype=a.dtype, device=a.device)
    n_in = a.shape[1]
    B = a.shape[0]
    if bt.numel() == 1 or bt.numel() == n_in:
        return a + sign * bt.flatten()
    out_sh = tuple(int(d) for d in op.get('out_shape_nd'))
    if int(np.prod(out_sh)) == n_in:
        # aligned broadcast: a already has the output numel; bias right-aligns.
        a_nd = a.reshape(B, *out_sh)
        return (a_nd + sign * bt).reshape(B, -1)
    const_sh = tuple(int(d) for d in bt.shape) if bt.dim() > 0 else ()
    in_sh = tuple(int(d) for d in op.get('in_shapes_nd', [None])[0])
    live_sh, const_aligned = reconstruct_live_shape(
        out_sh, const_sh, in_sh, n_in)
    a_nd = a.reshape(B, *live_sh)
    const_nd = bt.reshape(const_aligned)
    return (a_nd + sign * const_nd).expand(B, *out_sh).reshape(B, -1)


def _forward_batch_graph(x, gg):
    """Batched forward pass for PGD on graph networks (supports skip connections)."""
    batch = x.shape[0]
    act = {gg['input_name']: x}
    forks = gg['fork_points']

    for op in gg['ops']:
        name = op['name']
        t = op['type']

        if t == 'conv':
            a = act[op['inputs'][0]]
            ins = op['in_shape']
            a = F.conv2d(a.reshape(batch, *ins), op['kernel'],
                         bias=op['bias'], stride=op['stride'],
                         padding=op['padding']).reshape(batch, -1)
            act[name] = a

        elif t == 'fc':
            a = act[op['inputs'][0]]
            W = op['W']
            in_nd = op.get('in_shapes_nd', [None])[0]
            if (W.ndim == 2 and in_nd is not None and len(in_nd) >= 2
                    and a.shape[1] != W.shape[1] and in_nd[-1] == W.shape[1]):
                # ND matmul: input is (batch, *lead, K) and W=(M,K) maps the
                # LAST dim (e.g. the piecewise-linear sigmoid decomposition's
                # per-row MatMul in ml4acopf-linear: (438,54) @ (54,1)). The
                # flat handler below assumes W maps the FULL flat dim and would
                # raise. Mirrors GemmNode.zonotope_propagate's ND branch
                # (network.py: center_nd @ W.T + b).
                a_nd = a.reshape(batch, *in_nd)
                act[name] = (a_nd @ W.T + op['bias']).reshape(batch, -1)
            else:
                act[name] = a @ W.T + op['bias']

        elif t == 'relu':
            act[name] = F.relu(act[op['inputs'][0]])

        elif t == 'add':
            if op.get('is_merge'):
                act[name] = act[op['inputs'][0]] + act[op['inputs'][1]]
            else:
                a = act[op['inputs'][0]]
                bias = op.get('bias')
                if bias is not None:
                    a = _point_bcast_bias(a, bias, op, 1.0)
                act[name] = a

        elif t == 'sub':
            a = act[op['inputs'][0]]
            bias = op.get('bias')
            if bias is not None:
                a = _point_bcast_bias(a, bias, op, -1.0)
            act[name] = a

        elif t == 'sub_bilinear':
            a = act[op['inputs'][0]]
            b = act[op['inputs'][1]]
            act[name] = a - b

        elif t == 'reshape':
            act[name] = act[op['inputs'][0]]

        elif t in ('slice', 'gather'):
            a = act[op['inputs'][0]]
            flat_idx = op.get('flat_idx')
            idx_t = torch.as_tensor(flat_idx, dtype=torch.long, device=a.device)
            act[name] = a.index_select(1, idx_t)

        elif t == 'concat':
            act[name] = torch.cat([act[inp] for inp in op['inputs']], dim=1)

        elif t == 'conv_transpose':
            a = act[op['inputs'][0]]
            ins = op['in_shape']
            a = F.conv_transpose2d(
                a.reshape(batch, *ins), op['kernel'], bias=op['bias'],
                stride=op['stride'], padding=op['padding'],
                output_padding=op['output_padding']).reshape(batch, -1)
            act[name] = a

        elif t == 'upsample':
            a = act[op['inputs'][0]]
            in_shape = op['in_shape']
            sH, sW = op['scale']
            a4 = a.reshape(batch, *in_shape)
            a4 = F.interpolate(a4, scale_factor=(sH, sW), mode='nearest')
            act[name] = a4.reshape(batch, -1)

        elif t == 'reduce_sum':
            a = act[op['inputs'][0]]
            in_shape_nd = op.get('in_shapes_nd', [None])[0]
            axes = op.get('axes', ())
            keep = op.get('keepdims', False)
            a_nd = a.reshape(batch, *in_shape_nd)
            # gg axes are relative to stripped-batch shape — add 1
            # to skip the batch dim in PGD's batched tensor.
            for ax in sorted(axes, reverse=True):
                a_nd = a_nd.sum(dim=ax + 1, keepdim=bool(keep))
            act[name] = a_nd.reshape(batch, -1)

        elif t == 'mul_bilinear':
            a = act[op['inputs'][0]]
            b = act[op['inputs'][1]]
            sh = op.get('in_shapes_nd', [None, None])
            if sh[0] is not None and sh[1] is not None and sh[0] != sh[1]:
                a_nd = a.reshape(batch, *sh[0])
                b_nd = b.reshape(batch, *sh[1])
                act[name] = (a_nd * b_nd).reshape(batch, -1)
            else:
                act[name] = a * b

        elif t == 'div_bilinear':
            a = act[op['inputs'][0]]
            b = act[op['inputs'][1]]
            sh = op.get('in_shapes_nd', [None, None])
            if sh[0] is not None and sh[1] is not None and sh[0] != sh[1]:
                a_nd = a.reshape(batch, *sh[0])
                b_nd = b.reshape(batch, *sh[1])
                act[name] = (a_nd / b_nd).reshape(batch, -1)
            else:
                act[name] = a / b

        elif t == 'pow':
            a = act[op['inputs'][0]]
            exp = op.get('exponent', 2.0)
            act[name] = a ** exp

        elif t == 'sigmoid':
            act[name] = torch.sigmoid(act[op['inputs'][0]])

        elif t == 'tanh':
            act[name] = torch.tanh(act[op['inputs'][0]])

        elif t == 'sin':
            act[name] = torch.sin(act[op['inputs'][0]])

        elif t == 'cos':
            act[name] = torch.cos(act[op['inputs'][0]])

        elif t == 'floor':
            act[name] = torch.floor(act[op['inputs'][0]])

        elif t in ('avg_pool', 'max_pool'):
            a = act[op['inputs'][0]]
            in_shape = op['in_shape']
            a4 = a.reshape(batch, *in_shape)
            fn = F.avg_pool2d if t == 'avg_pool' else F.max_pool2d
            a4 = fn(a4, kernel_size=op['kernel'], stride=op['stride'],
                      padding=op['padding'])
            act[name] = a4.reshape(batch, -1)

        elif t == 'squeeze':
            act[name] = act[op['inputs'][0]]

        elif t == 'mul':
            a = act[op['inputs'][0]]
            scale = op.get('scale')
            if scale is None:
                raise NotImplementedError(
                    f"_forward_batch_graph: mul op {name!r} has no 'scale' "
                    f"— treating it as identity would silently drop the "
                    f"multiply")
            s = torch.as_tensor(np.asarray(scale).flatten(),
                                  dtype=a.dtype, device=a.device)
            act[name] = a * s

        elif t == 'mul_bilinear':
            a = act[op['inputs'][0]]
            b = act[op['inputs'][1]]
            act[name] = a * b

        elif t == 'matmul_bilinear':
            a = act[op['inputs'][0]]
            b = act[op['inputs'][1]]
            shapes = op.get('in_shapes_nd', [None, None])
            sh_a, sh_b = shapes[0], shapes[1]
            assert sh_a and sh_b and len(sh_a) >= 2 and len(sh_b) >= 2, \
                f'matmul_bilinear needs ≥2-D shapes; got {sh_a}, {sh_b}'
            a_nd = a.reshape(batch, *sh_a)
            b_nd = b.reshape(batch, *sh_b)
            out_nd = a_nd @ b_nd
            act[name] = out_nd.reshape(batch, -1)

        elif t == 'exp':
            act[name] = torch.exp(act[op['inputs'][0]])

        elif t == 'reciprocal':
            act[name] = 1.0 / act[op['inputs'][0]]

        else:
            raise NotImplementedError(
                f'_forward_batch_graph: unknown op type {t!r} at {name!r}')

    return act[gg['ops'][-1]['name']]


def _forward_batch(x, fwd_data, nh):
    """Batched forward pass for PGD attack."""
    batch = x.shape[0]
    gpu_k = fwd_data['gpu_k']
    gpu_W_fwd = fwd_data['gpu_W_fwd']
    gpu_b_fwd = fwd_data['gpu_b_fwd']
    layer_types = fwd_data['layer_types']
    for l in range(nh + 1):
        lt, params = layer_types[l]
        if lt == 'conv':
            ins = params['input_shape']
            s = params['stride']
            p = params['padding']
            x = F.conv2d(x.reshape(batch, *ins), gpu_k[l],
                         bias=gpu_b_fwd[l], stride=s, padding=p
                         ).reshape(batch, -1)
        else:
            x = x @ gpu_W_fwd[l].T + gpu_b_fwd[l]
        if l < nh:
            x = F.relu(x)
    return x


def _pgd_attack(xl, xh, remaining_specs, pred, fwd_data, nh, settings):
    """Batched PGD with per-restart targets.

    Returns (is_sat, witness_np, best_adv_np).
    """
    DEV = xl.device
    DT = xl.dtype
    n_restarts = settings.pgd_restarts
    n_iter = settings.pgd_iter
    eps = (xh - xl) / 2
    step_size = eps * 0.2
    comps_list = sorted(remaining_specs)
    n_specs = len(comps_list)
    comps_t = torch.tensor(comps_list, device=DEV)
    target_idx = torch.arange(n_restarts, device=DEV) % n_specs
    target_comps = comps_t[target_idx]

    x_adv = xl + (xh - xl) * torch.rand(n_restarts, len(xl), dtype=DT,
                                         device=DEV)
    x_adv.requires_grad_(True)

    for _ in range(n_iter):
        out = _forward_batch(x_adv, fwd_data, nh)
        target_margins = (out[:, pred]
                          - out[torch.arange(n_restarts, device=DEV),
                                target_comps])
        all_margins = out[:, pred].unsqueeze(1) - out[:, comps_t]
        worst_per_sample = all_margins.min(dim=1).values
        if (worst_per_sample < 0).any():
            idx = worst_per_sample.argmin()
            return True, x_adv[idx].detach().cpu().numpy(), None
        loss = target_margins.sum()
        loss.backward()
        with torch.no_grad():
            x_new = x_adv - step_size * x_adv.grad.sign()
            x_adv = torch.clamp(x_new, xl, xh).clone().requires_grad_(True)

    with torch.no_grad():
        out = _forward_batch(x_adv, fwd_data, nh)
        all_margins = out[:, pred].unsqueeze(1) - out[:, comps_t]
        worst_per_sample = all_margins.min(dim=1).values
        if (worst_per_sample < 0).any():
            idx = worst_per_sample.argmin()
            return True, x_adv[idx].detach().cpu().numpy(), None
        best_idx = worst_per_sample.argmin()
        best_adv = x_adv[best_idx].detach().cpu().numpy()
    return False, None, best_adv


def _pgd_attack_graph(xl, xh, remaining_specs, pred, gg, settings):
    """Batched PGD on graph networks. Same interface as _pgd_attack."""
    DEV = xl.device
    DT = xl.dtype
    n_restarts = settings.pgd_restarts
    n_iter = settings.pgd_iter
    eps = (xh - xl) / 2
    step_size = eps * 0.2
    comps_list = sorted(remaining_specs)
    n_specs = len(comps_list)
    comps_t = torch.tensor(comps_list, device=DEV)
    target_idx = torch.arange(n_restarts, device=DEV) % n_specs
    target_comps = comps_t[target_idx]

    x_adv = xl + (xh - xl) * torch.rand(n_restarts, len(xl), dtype=DT,
                                         device=DEV)
    x_adv.requires_grad_(True)

    for _ in range(n_iter):
        out = _forward_batch_graph(x_adv, gg)
        target_margins = (out[:, pred]
                          - out[torch.arange(n_restarts, device=DEV),
                                target_comps])
        all_margins = out[:, pred].unsqueeze(1) - out[:, comps_t]
        worst_per_sample = all_margins.min(dim=1).values
        if (worst_per_sample < 0).any():
            idx = worst_per_sample.argmin()
            return True, x_adv[idx].detach().cpu().numpy(), None
        loss = target_margins.sum()
        loss.backward()
        with torch.no_grad():
            x_new = x_adv - step_size * x_adv.grad.sign()
            x_adv = torch.clamp(x_new, xl, xh).clone().requires_grad_(True)

    with torch.no_grad():
        out = _forward_batch_graph(x_adv, gg)
        all_margins = out[:, pred].unsqueeze(1) - out[:, comps_t]
        worst_per_sample = all_margins.min(dim=1).values
        if (worst_per_sample < 0).any():
            idx = worst_per_sample.argmin()
            return True, x_adv[idx].detach().cpu().numpy(), None
        best_idx = worst_per_sample.argmin()
        best_adv = x_adv[best_idx].detach().cpu().numpy()
    return False, None, best_adv


def _build_spec_ew(gpu_layers_list, pred, comps, device, dtype):
    """Precompute effective weight for spec backward pass.

    For the final layer, computes w_pred - w_comp and bias_pred - bias_comp.
    """
    spec_ew = {}
    final = gpu_layers_list[-1]
    if final['type'] == 'conv':
        in_shape = final['in_shape']
        n_prev = in_shape[0] * in_shape[1] * in_shape[2]
        kernel = final['kernel']
        bias = final['bias']
        out_shape = final['out_shape']
        n_out = final['n_out']
        for comp in comps:
            # Build one-hot for pred and comp, push through conv_transpose
            I_pred = torch.zeros(1, n_out, dtype=dtype, device=device)
            I_pred[0, pred] = 1.0
            wp = F.conv_transpose2d(
                I_pred.reshape(1, *out_shape), kernel,
                stride=final['stride'], padding=final['padding'],
                output_padding=final['output_padding']).flatten()
            I_comp = torch.zeros(1, n_out, dtype=dtype, device=device)
            I_comp[0, comp] = 1.0
            wc = F.conv_transpose2d(
                I_comp.reshape(1, *out_shape), kernel,
                stride=final['stride'], padding=final['padding'],
                output_padding=final['output_padding']).flatten()
            spatial = out_shape[1] * out_shape[2]
            b_diff = float(bias[pred // spatial]) - float(bias[comp // spatial])
            spec_ew[comp] = (wp - wc, b_diff)
    else:
        W = final['W']
        bias = final['bias']
        for comp in comps:
            spec_ew[comp] = (W[pred] - W[comp],
                             float(bias[pred]) - float(bias[comp]))
    return spec_ew


def _build_spec_ew_graph(gg, pred, comps, device, dtype):
    """Compute spec effective weights from gpu_graph's final linear layer."""
    # Find the last linear op (Conv or FC)
    last_linear = None
    for op in reversed(gg['ops']):
        if op['type'] in ('conv', 'fc'):
            last_linear = op
            break
    assert last_linear is not None, "No final linear layer found"

    spec_ew = {}
    if last_linear['type'] == 'fc':
        W = last_linear['W']
        bias = last_linear['bias']
        for comp in comps:
            spec_ew[comp] = (W[pred] - W[comp],
                             float(bias[pred]) - float(bias[comp]))
    else:
        kernel = last_linear['kernel']
        bias = last_linear['bias']
        out_shape = last_linear['out_shape']
        n_out = last_linear['n_out']
        for comp in comps:
            I_pred = torch.zeros(1, n_out, dtype=dtype, device=device)
            I_pred[0, pred] = 1.0
            wp = F.conv_transpose2d(
                I_pred.reshape(1, *out_shape), kernel,
                stride=last_linear['stride'],
                padding=last_linear['padding'],
                output_padding=last_linear['output_padding']).flatten()
            I_comp = torch.zeros(1, n_out, dtype=dtype, device=device)
            I_comp[0, comp] = 1.0
            wc = F.conv_transpose2d(
                I_comp.reshape(1, *out_shape), kernel,
                stride=last_linear['stride'],
                padding=last_linear['padding'],
                output_padding=last_linear['output_padding']).flatten()
            spatial = out_shape[1] * out_shape[2]
            b_diff = float(bias[pred // spatial]) - float(bias[comp // spatial])
            spec_ew[comp] = (wp - wc, b_diff)
    return spec_ew


def _ibp_forward_graph(xl, xh, gg, device, dtype):
    """Interval (IBP) forward over gpu_graph ops.

    Memory-light alternative to `_forward_zonotope_graph` for nets whose
    input-identity generator tensors don't fit GPU memory (e.g.
    challenging_certified_training tinyimagenet CNN7: 12 288 input gens
    x 262 144 first-layer activations = 12 GiB per layer at fp32, two
    consecutive such layers exceed a 24 GB card). IBP carries only
    (lo, hi) per activation — O(activations) total.

    Returns `sb`: {relu layer_idx: (lo, hi)} flattened PRE-activation
    bounds, the same contract as `_forward_zonotope_graph`'s first
    return value, so `_spec_backward_graph` (CROWN backward) and the
    per-layer bound consumers work unchanged (= CROWN-IBP overall).

    Sound by interval arithmetic: every op maps input enclosures to
    output enclosures (affine ops via mid/rad with |W|, monotone ops
    endpoint-wise).
    """
    lo = {gg['input_name']: xl.flatten()}
    hi = {gg['input_name']: xh.flatten()}
    sb = {}
    for op in gg['ops']:
        t = op['type']
        name = op['name']
        l, h = lo[op['inputs'][0]], hi[op['inputs'][0]]
        if t == 'conv':
            mid, rad = (l + h) / 2, (h - l) / 2
            mid4 = mid.reshape(1, *op['in_shape'])
            rad4 = rad.reshape(1, *op['in_shape'])
            mc = F.conv2d(mid4, op['kernel'], op['bias'],
                          stride=op['stride'], padding=op['padding'])
            rc = F.conv2d(rad4, op['kernel'].abs(), None,
                          stride=op['stride'], padding=op['padding'])
            lo[name], hi[name] = (mc - rc).flatten(), (mc + rc).flatten()
        elif t == 'fc':
            W, b = op['W'], op['bias']
            mid, rad = (l + h) / 2, (h - l) / 2
            mc = W @ mid + (b if b is not None else 0)
            rc = W.abs() @ rad
            lo[name], hi[name] = mc - rc, mc + rc
        elif t == 'relu':
            if 'layer_idx' in op:
                sb[op['layer_idx']] = (l.clone(), h.clone())
            lo[name], hi[name] = l.clamp(min=0), h.clamp(min=0)
        elif t == 'reshape':
            lo[name], hi[name] = l, h
        elif t == 'add':
            if op.get('is_merge'):
                l2, h2 = lo[op['inputs'][1]], hi[op['inputs'][1]]
                lo[name], hi[name] = l + l2, h + h2
            else:
                b = torch.as_tensor(op['bias'], dtype=dtype,
                                    device=device).reshape(-1)
                lo[name], hi[name] = l + b, h + b
        elif t == 'sub':
            b = torch.as_tensor(op['bias'], dtype=dtype,
                                device=device).reshape(-1)
            lo[name], hi[name] = l - b, h - b
        elif t == 'mul':
            s = torch.as_tensor(op['scale'], dtype=dtype,
                                device=device).reshape(-1)
            a, b2 = s * l, s * h
            lo[name], hi[name] = torch.minimum(a, b2), torch.maximum(a, b2)
        elif t == 'max_pool':
            kH, kW = op['kernel']
            lo4 = lo[op['inputs'][0]].reshape(1, *op['in_shape'])
            hi4 = hi[op['inputs'][0]].reshape(1, *op['in_shape'])
            lo[name] = F.max_pool2d(lo4, (kH, kW), stride=op['stride'],
                                    padding=op['padding']).flatten()
            hi[name] = F.max_pool2d(hi4, (kH, kW), stride=op['stride'],
                                    padding=op['padding']).flatten()
        elif t in ('slice', 'gather'):
            # bounds follow the gather/slice index map (maxpool_to_relu phase
            # extraction): out[k] = in[flat_idx[k]] for both lo and hi.
            idx = torch.as_tensor(op['flat_idx'], dtype=torch.long,
                                  device=device)
            lo[name], hi[name] = l[idx], h[idx]
        elif t == 'sub_bilinear':
            # y = a - b: [lo_a - hi_b, hi_a - lo_b]  (maxpool_to_relu b-a)
            l2, h2 = lo[op['inputs'][1]], hi[op['inputs'][1]]
            lo[name], hi[name] = l - h2, h - l2
        elif t == 'concat':
            lo[name] = torch.cat([lo[i] for i in op['inputs']])
            hi[name] = torch.cat([hi[i] for i in op['inputs']])
        else:
            raise NotImplementedError(
                f'_ibp_forward_graph: unsupported op {t!r} at {name!r} — '
                f'silently skipping would certify unsound bounds')
    return sb


def _ibp_forward_graph_batched(xl, xh, gg, device, dtype,
                               relu_clamps=None, root_sb=None):
    """Batched interval forward over gpu_graph ops (B domains at once).

    Same op semantics as `_ibp_forward_graph` but every tensor carries a
    leading batch dim. Used by the IBP-route ReLU-split BaB: each domain
    refreshes ALL intermediate bounds under its split clamps (an IBP
    forward costs ~ms, unlike the zonotope forward this trades against).

    Args:
        xl, xh: (B, n_in) per-domain input bounds.
        relu_clamps: optional {layer_idx: (cl, ch)} of (B, n_layer)
            pre-activation clamp tensors (±inf where unclamped). A ReLU
            split z_j <= 0 / z_j >= 0 is expressed as ch[b, j] = 0 /
            cl[b, j] = 0. Sound: on the subdomain the true pre-activation
            satisfies the clamp, so intersecting enclosures stays valid.
        root_sb: optional {layer_idx: (lo, hi)} flat (n_layer,) ROOT
            pre-activation bounds to intersect (broadcast over B) —
            domain bounds can never be looser than the root's.

    Returns `sb`: {relu layer_idx: (lo, hi)} of (B, n_layer) PRE-activation
    bounds (after clamp/root intersection — consumers see the bounds the
    ReLU relaxation will actually use).
    """
    B = xl.shape[0]
    lo = {gg['input_name']: xl}
    hi = {gg['input_name']: xh}
    sb = {}
    for op in gg['ops']:
        t = op['type']
        name = op['name']
        l, h = lo[op['inputs'][0]], hi[op['inputs'][0]]
        if t == 'conv':
            mid, rad = (l + h) / 2, (h - l) / 2
            mid4 = mid.reshape(B, *op['in_shape'])
            rad4 = rad.reshape(B, *op['in_shape'])
            mc = F.conv2d(mid4, op['kernel'], op['bias'],
                          stride=op['stride'], padding=op['padding'])
            rc = F.conv2d(rad4, op['kernel'].abs(), None,
                          stride=op['stride'], padding=op['padding'])
            lo[name] = (mc - rc).reshape(B, -1)
            hi[name] = (mc + rc).reshape(B, -1)
        elif t == 'fc':
            W, b = op['W'], op['bias']
            mid, rad = (l + h) / 2, (h - l) / 2
            mc = mid @ W.T + (b if b is not None else 0)
            rc = rad @ W.T.abs()
            lo[name], hi[name] = mc - rc, mc + rc
        elif t == 'relu':
            L = op.get('layer_idx')
            if L is not None:
                if root_sb is not None and L in root_sb:
                    l = torch.maximum(l, root_sb[L][0].unsqueeze(0))
                    h = torch.minimum(h, root_sb[L][1].unsqueeze(0))
                if relu_clamps is not None and L in relu_clamps:
                    cl, ch = relu_clamps[L]
                    l = torch.maximum(l, cl)
                    h = torch.minimum(h, ch)
                # Clamp crossings from intersecting independent
                # enclosures (empty boxes round to a point — sound:
                # an empty subdomain has no counterexample).
                h = torch.maximum(h, l)
                sb[L] = (l.clone(), h.clone())
            lo[name], hi[name] = l.clamp(min=0), h.clamp(min=0)
        elif t == 'reshape':
            lo[name], hi[name] = l, h
        elif t == 'add':
            if op.get('is_merge'):
                l2, h2 = lo[op['inputs'][1]], hi[op['inputs'][1]]
                lo[name], hi[name] = l + l2, h + h2
            else:
                b = torch.as_tensor(op['bias'], dtype=dtype,
                                    device=device).reshape(-1)
                lo[name], hi[name] = l + b, h + b
        elif t == 'sub':
            b = torch.as_tensor(op['bias'], dtype=dtype,
                                device=device).reshape(-1)
            lo[name], hi[name] = l - b, h - b
        elif t == 'mul':
            s = torch.as_tensor(op['scale'], dtype=dtype,
                                device=device).reshape(-1)
            a, b2 = s * l, s * h
            lo[name], hi[name] = torch.minimum(a, b2), torch.maximum(a, b2)
        else:
            raise NotImplementedError(
                f'_ibp_forward_graph_batched: unsupported op {t!r} at '
                f'{name!r} — silently skipping would certify unsound bounds')
    return sb


def _zono_broadcast_const(z, const_t, op_sign, in_shape_nd, out_shape_nd,
                          device, dtype):
    """Exact elementwise affine op  y = center op_sign*const  with full
    numpy/ONNX bidirectional broadcast of BOTH the live tensor and the
    constant to ``out_shape_nd``.

    op_sign = +1 for Add, -1 for Sub. Returns a TorchZonotope whose
    center is ``broadcast(c) + op_sign*broadcast(const)`` and whose
    generators are ``broadcast(g)`` along every dim the live tensor is
    stretched on (a constant has zero radius, so it adds no generators).
    EXACT — no relaxation. Used by ml4acopf's piecewise-linear
    Sigmoid/Sin/Cos decomposition (Sub of a (N,) tensor against an
    (M,) breakpoint vector → (N, M)).

    The live tensor's recorded N-D shape may have lost trailing-1 dims
    (an Unsqueeze that was aliased away), so its broadcast position in
    ``out_sh`` is reconstructed from the constant's shape: the constant
    right-aligns into ``out_sh``; every out dim the constant does NOT
    fully drive (const dim == 1) is a live dim, the rest are live-1.
    The reconstruction is validated to preserve the live numel and to be
    a sound numpy broadcast.
    """
    n_in = z.center.numel()
    out_sh = (tuple(int(d) for d in out_shape_nd)
              if out_shape_nd is not None else (n_in,))
    const_sh = tuple(int(d) for d in const_t.shape) if const_t.dim() > 0 \
        else ()
    in_sh = (tuple(int(d) for d in in_shape_nd)
             if in_shape_nd is not None else (n_in,))
    # Shared broadcast-shape reconstruction (single source of truth across the
    # zono + LiRPA reps; see broadcast_util).
    live_sh, const_aligned = reconstruct_live_shape(
        out_sh, const_sh, in_sh, n_in)
    c_nd = z.center.reshape(live_sh)
    const_nd = const_t.reshape(const_aligned) if const_t.dim() >= 0 \
        else const_t
    c_out = (c_nd + op_sign * const_nd).expand(out_sh).reshape(-1).clone()
    K = z.generators.shape[1]
    if K > 0:
        g_nd = z.generators.reshape(live_sh + (K,))
        g_out = g_nd.expand(out_sh + (K,)).reshape(-1, K).clone()
    else:
        g_out = z.generators.new_zeros(c_out.numel(), 0)
    return TorchZonotope(c_out, g_out)


def _forward_zonotope_graph(xl, xh, gg, device, dtype, settings=None,
                             rec_zono=None, tight_bounds=None,
                             relu_lambdas=None, op_bounds=None,
                             op_clamps=None, col_ids_out=None,
                             op_fresh_ids=None, op_geom_out=None):
    if relu_lambdas is None:
        # plain (non-differentiable) use: keep the historical no-grad
        # fast path; the alpha-zono caller passes relu_lambdas and needs
        # the autograd graph through the forward.
        with torch.no_grad():
            return _forward_zonotope_graph_impl(
                xl, xh, gg, device, dtype, settings=settings,
                rec_zono=rec_zono, tight_bounds=tight_bounds,
                op_bounds=op_bounds, op_clamps=op_clamps,
                col_ids_out=col_ids_out, op_fresh_ids=op_fresh_ids,
                op_geom_out=op_geom_out)
    return _forward_zonotope_graph_impl(
        xl, xh, gg, device, dtype, settings=settings, rec_zono=rec_zono,
        tight_bounds=tight_bounds, relu_lambdas=relu_lambdas,
        op_bounds=op_bounds, op_clamps=op_clamps,
        col_ids_out=col_ids_out, op_fresh_ids=op_fresh_ids,
        op_geom_out=op_geom_out)


def _forward_zonotope_graph_impl(xl, xh, gg, device, dtype, settings=None,
                             rec_zono=None, tight_bounds=None,
                             relu_lambdas=None, op_bounds=None,
                             op_clamps=None, col_ids_out=None,
                             op_fresh_ids=None, op_geom_out=None):
    """Graph-aware zonotope forward pass (supports skip connections).

    ``op_geom_out`` (optional dict): for each elementwise splittable nonlinear
    op (sigmoid/tanh/sin/cos/pow), record at op-time the PRE-ACTIVATION
    zonotope geometry the dual-ascent nonlinear-split state needs —
    ``op_geom_out[name] = (z.center, z.generators, input_global_col_ids)`` —
    captured before downstream pops drop the input's col-id entry.

    Args:
        xl, xh: input bounds (flat torch tensors)
        gg: gpu_graph dict from ComputeGraph.gpu_graph()
        settings: optional settings DotMap. When provided AND
            `settings.zono_impl == 'patches'` AND the input shape is
            image-like (C, H, W), the initial zonotope is built as a
            `PatchesZonotope` instead of a `TorchZonotope`. On
            TinyImageNet ResNet (3×56×56 = 9 408 input pixels) the
            dense path needs ~700 MB just for the input zonotope's
            generator-identity matrix and OOMs the RTX 3080 inside the
            first conv; the patches path uses ~0.6 MB for the same
            input. Defaults to dense when `settings is None` for
            backward compatibility with callers that don't carry the
            settings (e.g. unit tests, BaB-leaf shortcuts).
        rec_zono: optional dict to populate with ``{gen_rows_by_layer,
            col_origin, n_input}`` harvested at each layer's pre-ReLU.
            Same protocol as ``_forward_zonotope_interleaved`` —
            downstream Phase 7 (`state_from_phase1`) consumes this to
            avoid the multi-GB ``precompute_gen_state`` allocation.
            When None, behaves as before.
        tight_bounds: optional ``{layer_idx: (lo_np, hi_np)}`` dict of
            externally-computed (e.g. cascade-tightened) pre-activation
            bounds. When provided, ``apply_relu`` uses the intersection
            of ``z.bounds()`` with these tight bounds for the relaxation
            (sound). ``rec_zono`` entries also record the intersected
            (lo, hi), keeping the parametrization consistent so
            ``state_from_phase1``'s LP triangle constraints use the
            same (lo, hi) as the recorded μ, λ.
        op_clamps: optional ``{op_name: (lo_t, hi_t)}`` of input-bound
            clamps for exp/reciprocal ops (flat torch tensors, ±inf for
            unclamped coordinates). The BnB's bilinear-input value-split
            (z_j <= m / z_j >= m on the TRUE op input) intersects the
            zonotope input range with these before building the convex
            parallelogram — same soundness argument as ReLU
            ``tight_bounds``: on the subdomain, every true input value
            lies in the clamped range, so the clamped-range relaxation
            contains every true (x, f(x)) point, and the two children
            cover the parent. Recorded ``op_bounds`` carry the
            intersected range so backward planes tighten identically.

    Returns:
        sb: dict mapping layer_idx -> (lo, hi) bounds at each ReLU
        z_final: final zonotope (after last op, before output)
    """
    if settings is not None and str(getattr(
            settings, 'zono_impl', 'dense')) == 'patches':
        from .zonotope import make_input_zonotope
        in_shape = getattr(gg, 'input_shape', None) or gg.get('input_shape')
        z_init = make_input_zonotope(
            settings, xl, xh, device, dtype, in_shape=in_shape)
    else:
        z_init = TorchZonotope.from_input_bounds(xl, xh, device, dtype)
    zono_state = {gg['input_name']: z_init}
    gen_count = {gg['input_name']: z_init.n_gens}
    forks = gg['fork_points']
    sb = {}

    # Explicit per-column global noise IDs. Merges compute their sound shared
    # count as the common-prefix length of the two branches' ID arrays, which
    # stays correct across prefix-breaking ops (sigmoid/tanh, both-vary
    # mul/sub/div) — unlike the positional gen_count assumption. Merges/bilinear
    # set col_ids_state[name] in their handler; all other ops derive it
    # generically after the dispatch (preserve input[0]'s IDs, append fresh).
    col_ids_state = {gg['input_name']: list(range(z_init.n_gens))}
    _id_counter = [z_init.n_gens]

    def _alloc_ids(k):
        start = _id_counter[0]
        _id_counter[0] = start + max(0, int(k))
        return list(range(start, start + max(0, int(k))))

    if rec_zono is not None:
        rec_zono.setdefault('gen_rows_by_layer', {})
        rec_zono.setdefault('col_origin', {})
        rec_zono['n_input'] = z_init.n_gens

    # Precompute last consumer index for each op name → free memory eagerly
    last_use = {}
    for i, op2 in enumerate(gg['ops']):
        for inp in op2['inputs']:
            last_use[inp] = i

    def _get(name):
        if name in forks:
            return zono_state[name].copy()
        return zono_state[name]

    # True-value interval side-channel: iv_state[name] = (lo, hi) is a
    # sound enclosure of the op's TRUE output values on the (possibly
    # op_clamps-split) subdomain, sometimes strictly tighter than the
    # output zonotope's own range. Why needed: a clamped exp's linear
    # term k*z still ranges over the UNCLAMPED zonotope coordinate, so
    # z.bounds() of the output can dip below the true range (even below
    # 0, breaking the downstream reciprocal's domain). Seeded at
    # exp/reciprocal (monotone closed form on the intersected input
    # range), propagated through reduce_sum and mul_bilinear, and
    # intersected wherever a handler consumes input bounds.
    iv_state = {}

    def _iv_bounds(inp_name, z):
        lo, hi = z.bounds()
        if inp_name in iv_state:
            lo = torch.maximum(lo, iv_state[inp_name][0])
            hi = torch.minimum(hi, iv_state[inp_name][1])
            hi = torch.maximum(hi, lo)
        return lo, hi

    for op_idx, op in enumerate(gg['ops']):
        name = op['name']
        t = op['type']
        _pre_fresh_id = _id_counter[0]  # for op_fresh_ids attribution

        if t == 'conv':
            z = _get(op['inputs'][0])
            _did_reduce = False
            _conv_args = (op['kernel'], op['bias'], op['in_shape'],
                          op['stride'], op['padding'])
            try:
                z.propagate_conv(*_conv_args)
            except torch.cuda.OutOfMemoryError:
                # Box-reduce-on-OOM safety valve (gated): reduce the conv INPUT
                # to a per-neuron box (caps patch growth) and retry — only when
                # forced, so correlation is kept everywhere it fits.
                if not (isinstance(z, PatchesZonotope) and z._mode == 'patches'
                        and settings is not None
                        and getattr(settings, 'box_reduce_on_oom', False)):
                    raise
                torch.cuda.empty_cache()
                z.box_reduce()
                z.propagate_conv(*_conv_args)
                _did_reduce = True
            zono_state[name] = z
            # Gated box order-reduction by patch-byte budget (proactive): when
            # a conv's patch tensor exceeds `box_reduce_patch_budget` bytes,
            # collapse to a per-neuron box. Default off → no effect elsewhere.
            _brb = (getattr(settings, 'box_reduce_patch_budget', None)
                    if settings is not None else None)
            if (isinstance(_brb, (int, float)) and _brb > 0
                    and isinstance(z, PatchesZonotope)
                    and z._mode == 'patches' and z._patches is not None
                    and z._patches.numel() * z._patches.element_size() > _brb):
                z.box_reduce()
                _did_reduce = True
            # Box gens are fresh noise → reset this op's col-ids to fresh
            # (skips the generic prefix-preserving default below).
            if _did_reduce:
                col_ids_state[name] = _alloc_ids(z.n_gens)

        elif t == 'fc':
            z = _get(op['inputs'][0])
            in_shape_nd = op.get('in_shapes_nd', [None])[0]
            W = op['W']; bias = op['bias']
            # Standard 1D case: input is flat (n_in,), W is (n_out, n_in).
            # Batched MatMul case (nn4sys mscn, ≥2D in_shape): apply
            # F.linear over the last dim by reshaping center/gens to
            # (..., n_in_last). E.g., (3, 7) input with W=(128, 7) →
            # output (3, 128).
            if (in_shape_nd is not None and len(in_shape_nd) >= 2
                    and W.shape[1] == in_shape_nd[-1]
                    and z.center.numel() == int(np.prod(in_shape_nd))):
                prefix = in_shape_nd[:-1]
                n_last_in = in_shape_nd[-1]
                n_last_out = W.shape[0]
                K = z.generators.shape[1]
                # Center: (prefix..., n_in) → linear → (prefix..., n_out)
                c_nd = z.center.reshape(*prefix, n_last_in)
                c_out_nd = F.linear(c_nd, W, bias)
                new_c = c_out_nd.flatten()
                if K > 0:
                    # Generators: (prefix..., n_in, K) → linear (W on
                    # the second-to-last axis) → (prefix..., n_out, K).
                    # Use einsum to keep gen-axis intact.
                    g_nd = z.generators.reshape(*prefix, n_last_in, K)
                    # 'oi,...ik->...ok' but we need (...i k) → (...o k)
                    # apply W (o, i) along the second-to-last axis
                    # einsum with '...ik,oi->...ok'
                    g_out_nd = torch.einsum('...ik,oi->...ok', g_nd, W)
                    new_g = g_out_nd.reshape(-1, K)
                else:
                    new_g = z.generators.new_zeros(new_c.numel(), 0)
                zono_state[name] = TorchZonotope(new_c, new_g)
            else:
                z.propagate_fc(op['W'], op['bias'])
                zono_state[name] = z

        elif t == 'relu':
            z = _get(op['inputs'][0])
            layer_idx = op.get('layer_idx')
            if relu_lambdas is not None and layer_idx in relu_lambdas:
                # Parametrized ReLU relaxation with caller-supplied slopes
                # lam in [0,1] (alpha-zono forward): for unstable neurons,
                #   relu(z) - lam*z ∈ [0, max(-lam*lo, (1-lam)*hi)]
                # so y = lam*z + r/2 + (r/2)*e_new with
                # r = max(-lam*lo, (1-lam)*hi). Differentiable in lam;
                # ANY lam in [0,1] is a sound relaxation. Stable neurons
                # use the exact identity/zero.
                lam_t = relu_lambdas[layer_idx]
                lo_p, hi_p = z.bounds()
                if tight_bounds is not None and layer_idx in tight_bounds:
                    _tl, _th = tight_bounds[layer_idx]
                    lo_p = torch.maximum(lo_p, torch.as_tensor(
                        _tl, dtype=lo_p.dtype, device=device))
                    hi_p = torch.minimum(hi_p, torch.as_tensor(
                        _th, dtype=hi_p.dtype, device=device))
                dead = hi_p <= 0
                act = lo_p >= 0
                uns = (~dead) & (~act)
                eff = torch.where(uns, lam_t.clamp(0, 1),
                                  act.to(lam_t.dtype))
                r = torch.where(
                    uns,
                    torch.maximum(-eff * lo_p, (1 - eff) * hi_p),
                    torch.zeros_like(lo_p))
                c_new = eff * z.center + r / 2
                G_old = z.generators * eff.unsqueeze(1)
                nzr = torch.nonzero(r).flatten()
                G_new = torch.zeros(lo_p.numel(), nzr.numel(),
                                    dtype=z.center.dtype, device=device)
                G_new[nzr, torch.arange(nzr.numel(), device=device)] = \
                    r[nzr] / 2
                z2 = TorchZonotope(c_new, torch.cat([G_old, G_new], 1))
                if layer_idx is not None:
                    sb[layer_idx] = (lo_p.detach().clone(),
                                     hi_p.detach().clone())
                zono_state[name] = z2
                gen_count[name] = z2.n_gens
                # prefix-preserving (old gens scaled + fresh appended)
                _base = col_ids_state[op['inputs'][0]]
                col_ids_state[name] = (
                    list(_base) + _alloc_ids(z2.n_gens - len(_base)))
                for inp in op['inputs']:
                    if last_use.get(inp) == op_idx and inp in zono_state:
                        del zono_state[inp]
                        col_ids_state.pop(inp, None)
                continue
            # Build the (lo, hi) the relaxation will use: intersect z's
            # own bounds with any externally-supplied tight bounds. We
            # record this same (lo, hi) into rec_zono so the LP triangle
            # constraints in state_from_phase1 match the parametrization.
            # Gated CROWN-retighten: use the (tight) zono's own bounds to pick
            # the FEW unstable neurons, backward-CROWN only those (cheap), and
            # feed the tightened bounds into the relu relaxation — keeps the
            # zono tight without bounding IBP's grossly-inflated set. Default
            # off (no effect on any other path).
            _crt = (settings is not None and layer_idx is not None
                    and getattr(settings, 'crown_retighten_forward', False))
            need_pre_bounds = _crt or (
                rec_zono is not None and layer_idx is not None
            ) or (tight_bounds is not None and layer_idx in (tight_bounds or {}))
            if need_pre_bounds:
                pre_lo_z, pre_hi_z = z.bounds()
                if tight_bounds is not None and layer_idx in tight_bounds:
                    tlo_np, thi_np = tight_bounds[layer_idx]
                    tlo = torch.as_tensor(tlo_np, dtype=dtype, device=device)
                    thi = torch.as_tensor(thi_np, dtype=dtype, device=device)
                    pre_lo = torch.maximum(pre_lo_z, tlo)
                    pre_hi = torch.minimum(pre_hi_z, thi)
                else:
                    pre_lo, pre_hi = pre_lo_z.clone(), pre_hi_z.clone()
                if _crt:
                    uns = ((pre_lo < 0) & (pre_hi > 0)).nonzero().flatten()
                    # Cap: backward-CROWN only the WIDEST-N unstable neurons
                    # (largest hi-lo). The rest keep their sound (looser) zono
                    # bounds. Sound (we only tighten a subset); the widest
                    # neurons drive downstream looseness, so a modest cap keeps
                    # most of the tightening at a fraction of the cost (a deep
                    # full backward over thousands of unstable is the VGG16
                    # bottleneck). Default None = tighten all (unchanged).
                    _cap = getattr(settings,
                                   'crown_retighten_max_neurons', None)
                    if _cap is not None and uns.numel() > int(_cap):
                        w = (pre_hi[uns] - pre_lo[uns])
                        top = torch.topk(w, int(_cap)).indices
                        uns = uns[top]
                    _pre_3d = len(op.get('in_shapes_nd', [()])[0]) == 3
                    if uns.numel() > 0 and _pre_3d and getattr(
                            settings, 'crown_retighten_patches', False):
                        # Patches-mode backward CROWN: the relation stays a
                        # localized per-neuron patch (receptive-field sized)
                        # instead of a dense full-feature-map matrix — the
                        # backward dual of PatchesZonotope. ~5x faster at deep
                        # conv layers; the neuron batch self-splits on OOM.
                        # Retightens ALL unstable (the cap is a dense-only
                        # crutch). Validated bit-equivalent to the dense path.
                        from .patches_crown import patches_bounds
                        pre_op = op['inputs'][0]
                        out_shape = tuple(op['in_shapes_nd'][0])
                        if getattr(settings, 'crown_retighten_debug', False):
                            print(f'  [retighten L={layer_idx} '
                                  f'pre_unstable={uns.numel()} '
                                  f'shape={out_shape}]', flush=True)
                        lb, ub = patches_bounds(
                            gg, xl, xh, sb, pre_op, out_shape, uns,
                            device, dtype)
                        pre_lo[uns] = torch.maximum(pre_lo[uns], lb)
                        pre_hi[uns] = torch.minimum(pre_hi[uns], ub)
                    elif uns.numel() > 0:
                        from .alpha_crown import _crown_backward_matrix
                        pre_op = op['inputs'][0]
                        n_at = pre_lo.numel()
                        _ch = int(getattr(settings,
                                          'crown_retighten_chunk', 512))
                        for _s in range(0, uns.numel(), _ch):
                            idx = uns[_s:_s + _ch]
                            ew = torch.zeros(idx.numel(), n_at,
                                             dtype=dtype, device=device)
                            ew[torch.arange(idx.numel(), device=device),
                               idx] = 1.0
                            lb, _ = _crown_backward_matrix(
                                gg, xl, xh, {}, sb, pre_op, ew, device, dtype)
                            nlb, _ = _crown_backward_matrix(
                                gg, xl, xh, {}, sb, pre_op, -ew, device, dtype)
                            pre_lo[idx] = torch.maximum(pre_lo[idx], lb)
                            pre_hi[idx] = torch.minimum(pre_hi[idx], -nlb)
                if rec_zono is not None and layer_idx is not None:
                    from .verify_graph import _record_zono_pre_relu_rows
                    _record_zono_pre_relu_rows(
                        z, layer_idx,
                        (pre_lo.cpu().numpy(), pre_hi.cpu().numpy()),
                        rec_zono)
                lo, hi = z.apply_relu(tight_lo=pre_lo, tight_hi=pre_hi)
            else:
                lo, hi = z.apply_relu()
            if layer_idx is not None:
                sb[layer_idx] = (lo.clone(), hi.clone())
            zono_state[name] = z

        elif t == 'add':
            if op.get('is_merge'):
                z_a = _get(op['inputs'][0])
                z_b = _get(op['inputs'][1])
                # Shared = common-prefix length of the two branches' column IDs.
                ida = col_ids_state[op['inputs'][0]]
                idb = col_ids_state[op['inputs'][1]]
                shared = _common_prefix_len(ida, idb)
                zono_state[name] = z_a.add(z_b, shared)
                # add layout: [a[:s]+b[:s] | a[s:] | b[s:]]  ->  ida + idb[s:]
                col_ids_state[name] = list(ida) + list(idb[shared:])
            else:
                z = _get(op['inputs'][0])
                bias = op.get('bias')
                if bias is not None:
                    bias_t = torch.as_tensor(
                        bias, dtype=dtype, device=device)
                    if bias_t.numel() == 1:
                        # Scalar bias: broadcast to center shape.
                        bias_flat = bias_t.flatten().expand(
                            z.center.numel())
                    elif bias_t.numel() == z.center.numel():
                        bias_flat = bias_t.flatten()
                    else:
                        # Broadcast: bias shape (..., n_out) over center
                        # reshaped to (prefix..., n_out). nn4sys mscn:
                        # MatMul out (3, 128) + bias (128,) broadcasts.
                        out_shape_nd = op.get('out_shape_nd')
                        if (out_shape_nd is not None
                                and len(out_shape_nd) >= 1
                                and out_shape_nd[-1] == bias_t.numel()):
                            c_nd = z.center.reshape(*out_shape_nd)
                            bias_flat = (c_nd + bias_t).flatten() - z.center
                        else:
                            raise ValueError(
                                f'add bias shape {bias_t.shape} '
                                f'incompatible with center {z.center.shape} '
                                f'(out_shape_nd={out_shape_nd})')
                    z = TorchZonotope(z.center + bias_flat,
                                       z.generators.clone())
                zono_state[name] = z

        elif t == 'sub':
            z = _get(op['inputs'][0])
            bias = op.get('bias')
            if bias is not None:
                bias_t = torch.as_tensor(bias, dtype=dtype, device=device)
                n_in = z.center.numel()
                if bias_t.numel() == 1 or bias_t.numel() == n_in:
                    # Scalar or elementwise: no shape change.
                    z = TorchZonotope(
                        z.center - bias_t.flatten(), z.generators.clone())
                else:
                    # Genuine outer broadcast (ml4acopf sigmoid/sin/cos
                    # piecewise decomposition: (N,) − (M,) → (N, M)).
                    z = _zono_broadcast_const(
                        z, bias_t, -1.0,
                        op.get('in_shapes_nd', [None])[0],
                        op.get('out_shape_nd'), device, dtype)
            zono_state[name] = z

        elif t == 'sub_bilinear':
            # Sub(a, b) with both computed (skip-merge style). Same
            # shared-generator math as `add`'s skip-merge path, just
            # with z_b negated. Loadbearing for nn4sys
            # pensieve_*_parallel where output = MatMul1 - MatMul2.
            z_a = _get(op['inputs'][0])
            z_b = _get(op['inputs'][1])
            ida = col_ids_state[op['inputs'][0]]
            idb = col_ids_state[op['inputs'][1]]
            shared = _common_prefix_len(ida, idb)
            if isinstance(z_a, PatchesZonotope):
                # Patches-native subtraction (maxpool->relu's b - a, two
                # channel-slices of the same conv): stays in patches form,
                # column order [shared | a_extra | b_extra] matches the dense
                # path below so the col-id layout is unchanged.
                zono_state[name] = z_a.sub(z_b, shared)
            elif isinstance(z_b, PatchesZonotope):
                raise NotImplementedError(
                    'sub_bilinear with a dense lhs and a patches rhs is not '
                    'supported (would need to materialise the patches rhs '
                    'dense); the maxpool decomposition never produces it')
            else:
                # Dense layout: G_out = [G_a_shared - G_b_shared |
                #                        G_a_extra | -G_b_extra]
                g_a_shared = z_a.generators[:, :shared]
                g_b_shared = z_b.generators[:, :shared]
                g_a_extra = z_a.generators[:, shared:]
                g_b_extra = z_b.generators[:, shared:]
                g_out = torch.cat([
                    g_a_shared - g_b_shared,
                    g_a_extra,
                    -g_b_extra,
                ], dim=1)
                zono_state[name] = TorchZonotope(
                    z_a.center - z_b.center, g_out)
            # Same layout as add-merge: ida + idb[shared:].
            col_ids_state[name] = list(ida) + list(idb[shared:])

        elif t == 'reshape':
            zono_state[name] = _get(op['inputs'][0])

        elif t in ('slice', 'gather'):
            z = _get(op['inputs'][0])
            flat_idx = op.get('flat_idx')
            if flat_idx is None:
                raise ValueError("slice op missing 'flat_idx'")
            idx_t = torch.as_tensor(flat_idx, dtype=torch.long, device=device)
            if isinstance(z, PatchesZonotope):
                # Patches-native: a contiguous channel-block slice (the only
                # pattern maxpool->relu emits) stays in patches form with no
                # dense G materialisation — critical for full-image inputs
                # where dense G would be terabytes.
                zono_state[name] = z.slice(idx_t)
            else:
                c_flat = z.center.reshape(-1)
                g_flat = z.generators.reshape(c_flat.numel(), -1)
                zono_state[name] = TorchZonotope(
                    c_flat.index_select(0, idx_t),
                    g_flat.index_select(0, idx_t))

        elif t == 'permute':
            # Exact reordering of the element (row) dimension: a pure
            # index permutation of center + generators. Used by the
            # non-contiguous axis-1 Concat (flat-concat -> permute to the
            # true row-major (R, C) layout). EXACT — no relaxation.
            z = _get(op['inputs'][0])
            perm = op.get('perm')
            if perm is None:
                raise ValueError("permute op missing 'perm'")
            idx_t = torch.as_tensor(perm, dtype=torch.long, device=device)
            c_flat = z.center.reshape(-1)
            g_flat = z.generators.reshape(c_flat.numel(), -1)
            zono_state[name] = TorchZonotope(
                c_flat.index_select(0, idx_t),
                g_flat.index_select(0, idx_t))

        elif t == 'concat':
            # Stack rows. Inputs share only their common column-ID prefix;
            # each input's columns BEYOND that prefix are DISTINCT noise and
            # must get their OWN output columns (block-diagonal), with zeros
            # for other inputs' rows. The old pad-and-stack aliased different
            # inputs' noise into the same column index — sound for per-row
            # bounds but an UNDER-approximation of the joint range, which
            # corrupts downstream merges/bilinears (ml4acopf wide-box).
            zs = [_get(inp) for inp in op['inputs']]
            inp_ids = [col_ids_state[inp] for inp in op['inputs']]
            cp = list(inp_ids[0])
            for ids in inp_ids[1:]:
                cp = cp[:_common_prefix_len(cp, ids)]
            s = len(cp)
            cflats = [z.center.reshape(-1) for z in zs]
            gflats = [z.generators.reshape(c.numel(), -1)
                      for z, c in zip(zs, cflats)]
            extra = [g.shape[1] - s for g in gflats]
            total_rows = sum(c.numel() for c in cflats)
            total_cols = s + sum(extra)
            G = torch.zeros(total_rows, total_cols, dtype=dtype, device=device)
            out_ids = list(cp)
            r0, c0 = 0, s
            for ids, c, gf, ec in zip(inp_ids, cflats, gflats, extra):
                nr = c.numel()
                if s > 0:
                    G[r0:r0 + nr, :s] = gf[:, :s]
                if ec > 0:
                    G[r0:r0 + nr, c0:c0 + ec] = gf[:, s:]
                    out_ids.extend(ids[s:])
                    c0 += ec
                r0 += nr
            zono_state[name] = TorchZonotope(torch.cat(cflats, dim=0), G)
            col_ids_state[name] = out_ids

        elif t in ('sigmoid', 'tanh'):
            # Two relaxations, chosen by settings.sigmoid_relaxation:
            #  - 'box' (default): center = midpoint of the activation range,
            #    one new gen per cell of magnitude (hi-lo)/2. Discards ALL
            #    input correlation but keeps the generator count bounded (reset
            #    each activation) — safe for sigmoid-heavy nets (dist_shift).
            #  - 'affine_band': the sound DeepZ affine-band transform (like
            #    sin/cos) — y = lam*x + mu, lam-scaled gens preserved + one
            #    fresh delta gen per element. Far tighter (keeps correlation),
            #    but GROWS the generator count, so opt in where the net has few
            #    sigmoid cells (ml4acopf: one 24-wide layer).
            z = _get(op['inputs'][0])
            lo_pre, hi_pre = z.bounds()
            if op_geom_out is not None:
                op_geom_out[name] = {
                    'c_in': z.center.detach().clone(),
                    'gens': z.generators.detach().clone(),
                    'in_ids': list(col_ids_state[op['inputs'][0]])}
            mode = (settings.get('sigmoid_relaxation', 'box')
                    if settings is not None else 'box')
            # nonlinear-split clamp (sound; children cover the parent).
            _tl = _th = None
            if op_clamps is not None and name in op_clamps:
                _tl, _th = op_clamps[name]
            if mode == 'affine_band':
                from .nonlinear_relax import REGISTRY, zono_affine_transform
                relax = REGISTRY['Sigmoid' if t == 'sigmoid' else 'Tanh']()
                _alpha = (relu_lambdas[name] if relu_lambdas is not None
                          and name in relu_lambdas else None)
                if op_geom_out is not None and name in op_geom_out:
                    new_c, new_g, _band = zono_affine_transform(
                        relax, z.center, z.generators, tight_lo=_tl,
                        tight_hi=_th, alpha=_alpha, return_band=True)
                    op_geom_out[name]['band'] = tuple(
                        b.detach().clone() for b in _band)
                else:
                    new_c, new_g = zono_affine_transform(
                        relax, z.center, z.generators, tight_lo=_tl,
                        tight_hi=_th, alpha=_alpha)
                zono_state[name] = TorchZonotope(new_c, new_g)
                # preserve-append: keep input[0]'s col IDs + fresh per element
                _base = col_ids_state[op['inputs'][0]]
                col_ids_state[name] = (
                    list(_base) + _alloc_ids(new_g.shape[1] - len(_base)))
            else:
                act = torch.sigmoid if t == 'sigmoid' else torch.tanh
                _lo_c = (torch.maximum(lo_pre, _tl.to(lo_pre.dtype))
                         if _tl is not None else lo_pre)
                _hi_c = (torch.minimum(hi_pre, _th.to(hi_pre.dtype))
                         if _th is not None else hi_pre)
                _hi_c = torch.maximum(_hi_c, _lo_c)
                s_lo = act(_lo_c); s_hi = act(_hi_c)
                c_out = (s_lo + s_hi) / 2
                mu = (s_hi - s_lo) / 2
                # zero old gens (no preserved correlation), diag(mu) fresh.
                zono_state[name] = TorchZonotope(c_out, torch.diag(mu))
                col_ids_state[name] = _alloc_ids(c_out.numel())
            if op_bounds is not None:
                _ol = (torch.maximum(lo_pre, _tl.to(lo_pre.dtype))
                       if _tl is not None else lo_pre)
                _oh = (torch.minimum(hi_pre, _th.to(hi_pre.dtype))
                       if _th is not None else hi_pre)
                op_bounds[name] = (_ol.detach().clone(), _oh.detach().clone())
            layer_idx = op.get('layer_idx')
            if layer_idx is not None:
                sb[layer_idx] = (lo_pre.clone(), hi_pre.clone())

        elif t in ('sin', 'cos', 'floor'):
            # Sound elementwise nonlinearity via the affine-band DeepZ
            # transformer (nonlinear_relax): y = lam*x + mu (+ lam-scaled gens)
            # + one fresh delta error generator per element. Preserves input
            # correlation (tighter than the sigmoid box-collapse above) and is
            # sound by the relaxation's band guarantee.
            from .nonlinear_relax import REGISTRY, zono_affine_transform
            z = _get(op['inputs'][0])
            lo_pre, hi_pre = z.bounds()
            _cap = op_geom_out is not None and t in ('sin', 'cos')
            if _cap:
                op_geom_out[name] = {
                    'c_in': z.center.detach().clone(),
                    'gens': z.generators.detach().clone(),
                    'in_ids': list(col_ids_state[op['inputs'][0]])}
            # nonlinear-split clamp: tighten the band's input range to the
            # split sub-interval (sound — children cover the parent).
            _tl = _th = None
            if op_clamps is not None and name in op_clamps:
                _tl, _th = op_clamps[name]
            relax = REGISTRY[{'sin': 'Sin', 'cos': 'Cos', 'floor': 'Floor'}[t]]()
            # α-CROWN: caller-optimized per-element slope (sound for any α,
            # differentiable). floor has no tunable slope, so only sin/cos.
            _alpha = (relu_lambdas[name] if t in ('sin', 'cos')
                      and relu_lambdas is not None and name in relu_lambdas
                      else None)
            if _cap:
                new_c, new_g, _band = zono_affine_transform(
                    relax, z.center, z.generators, tight_lo=_tl, tight_hi=_th,
                    alpha=_alpha, return_band=True)
                op_geom_out[name]['band'] = tuple(
                    b.detach().clone() for b in _band)
            else:
                new_c, new_g = zono_affine_transform(
                    relax, z.center, z.generators, tight_lo=_tl, tight_hi=_th,
                    alpha=_alpha)
            zono_state[name] = TorchZonotope(new_c, new_g)
            if op_bounds is not None:
                _ol = (torch.maximum(lo_pre, _tl.to(lo_pre.dtype))
                       if _tl is not None else lo_pre)
                _oh = (torch.minimum(hi_pre, _th.to(hi_pre.dtype))
                       if _th is not None else hi_pre)
                op_bounds[name] = (_ol.detach().clone(), _oh.detach().clone())
            layer_idx = op.get('layer_idx')
            if layer_idx is not None:
                sb[layer_idx] = (lo_pre.clone(), hi_pre.clone())

        elif t == 'mul':
            # Constant scalar / per-channel multiply: y = scale * x.
            z = _get(op['inputs'][0])
            scale_t = op.get('scale')
            if scale_t is None:
                raise ValueError("mul op missing 'scale' for forward zono")
            if isinstance(scale_t, np.ndarray):
                scale_t = torch.from_numpy(scale_t).to(device=device, dtype=dtype)
            elif not isinstance(scale_t, torch.Tensor):
                scale_t = torch.tensor(scale_t, dtype=dtype, device=device)
            else:
                scale_t = scale_t.to(device=device, dtype=dtype)
            sflat = scale_t.flatten()
            n = z.center.numel()
            if sflat.numel() == 1:
                new_c = z.center * sflat
                new_g = z.generators * sflat
            elif sflat.numel() == n:
                new_c = z.center * sflat
                new_g = z.generators * sflat.unsqueeze(-1)
            else:
                in_shape = op.get('in_shapes_nd', [None])[0]
                if in_shape is None or len(in_shape) != 3:
                    raise ValueError(
                        f'mul: scale shape {sflat.shape} incompatible with '
                        f'input ({n}); no spatial shape')
                C, H, W = in_shape
                assert sflat.numel() == C
                scale_4d = sflat.view(1, C, 1, 1).expand(
                    1, C, H, W).reshape(-1)
                new_c = z.center * scale_4d
                new_g = z.generators * scale_4d.unsqueeze(-1)
            zono_state[name] = TorchZonotope(new_c, new_g)

        elif t == 'mul_bilinear':
            # Element-wise Mul. nn4sys mscn uses Mul(features, mask)
            # where the mask side is constant per-disjunct (zero
            # radius). When both vary, the helper returns the sound
            # zonotope product (linear exact + boxed remainder).
            from .zonotope import _torch_zono_mul_bilinear
            z_a = _get(op['inputs'][0])
            z_b = _get(op['inputs'][1])
            _al, _ah = _iv_bounds(op['inputs'][0], z_a)
            _bl, _bh = _iv_bounds(op['inputs'][1], z_b)
            if op_bounds is not None:
                op_bounds[name] = (
                    (_al.detach().clone(), _ah.detach().clone()),
                    (_bl.detach().clone(), _bh.detach().clone()))
            in_shapes = op.get('in_shapes_nd', [None, None])
            out_shape = op.get('out_shape_nd')
            if (in_shapes[0] is not None and in_shapes[1] is not None
                    and (op['inputs'][0] in iv_state
                         or op['inputs'][1] in iv_state)):
                # interval product (4 broadcast corner products) brackets
                # the true elementwise products
                _av = (_al.reshape(in_shapes[0]), _ah.reshape(in_shapes[0]))
                _bv = (_bl.reshape(in_shapes[1]), _bh.reshape(in_shapes[1]))
                _corners = torch.stack([
                    (_av[i] * _bv[j]).reshape(-1)
                    for i in (0, 1) for j in (0, 1)])
                iv_state[name] = (_corners.min(dim=0).values.detach(),
                                  _corners.max(dim=0).values.detach())
            # Only the shared generator prefix (the branches' common column-ID
            # prefix) is the SAME noise in both operands; columns beyond it are
            # branch-specific and must stay independent or the bilinear product
            # under-approximates (unsound — ml4acopf wide-box).
            ida = col_ids_state[op['inputs'][0]]
            idb = col_ids_state[op['inputs'][1]]
            shared = _common_prefix_len(ida, idb)
            new_c, new_g = _torch_zono_mul_bilinear(
                z_a.center, z_a.generators, z_b.center, z_b.generators,
                shape_a=in_shapes[0], shape_b=in_shapes[1],
                shape_out=out_shape, shared_gens=shared)
            zono_state[name] = TorchZonotope(new_c, new_g)
            nout = zono_state[name].n_gens
            a_pt = (z_a.generators.numel() == 0
                    or bool(z_a.generators.abs().max() < 1e-12))
            b_pt = (z_b.generators.numel() == 0
                    or bool(z_b.generators.abs().max() < 1e-12))
            if not (a_pt or b_pt):
                # both-vary layout: [shared | a_extra | b_extra | box].
                # shared+a_extra = ida (a's full IDs); b_extra = b's own IDs
                # beyond the shared prefix; box = fresh second-order columns.
                col_ids_state[name] = (
                    list(ida) + list(idb[shared:])
                    + _alloc_ids(nout - len(ida) - (len(idb) - shared)))
            else:
                # point-mask: preserves the VARYING side's columns positionally
                base = idb if a_pt else ida
                col_ids_state[name] = (
                    list(base[:nout]) if nout <= len(base)
                    else list(base) + _alloc_ids(nout - len(base)))

        elif t == 'div_bilinear':
            # Element-wise Div. Point denominator → exact. Non-point
            # denominator → box fallback if settings.nonlin_div_fallback
            # is 'box' AND denominator is sign-stable; otherwise raises.
            # When fallback fires, stash on op so backward switches to
            # the sound decorrelated bound (slope-to-input = 0, accum
            # += ep·box_lo + en·box_hi).
            from .zonotope import _torch_zono_div_bilinear
            z_a = _get(op['inputs'][0])
            z_b = _get(op['inputs'][1])
            fb = (settings.nonlin_div_fallback
                  if settings is not None and 'nonlin_div_fallback' in settings
                  else 'raise')
            b_is_point = (z_b.generators.numel() == 0
                          or bool(z_b.generators.abs().max() < 1e-12))
            op['_div_decoupled'] = not b_is_point
            if not b_is_point:
                # Forward zono will use box fallback; cache the input
                # boxes so backward can return the sound decorrelated
                # bound without recomputing.
                rad_a = (z_a.generators.abs().sum(dim=1)
                          if z_a.generators.numel() > 0
                          else torch.zeros_like(z_a.center))
                rad_b = z_b.generators.abs().sum(dim=1)
                op['_div_a_lo'] = (z_a.center - rad_a).detach()
                op['_div_a_hi'] = (z_a.center + rad_a).detach()
                op['_div_b_lo'] = (z_b.center - rad_b).detach()
                op['_div_b_hi'] = (z_b.center + rad_b).detach()
            new_c, new_g = _torch_zono_div_bilinear(
                z_a.center, z_a.generators, z_b.center, z_b.generators,
                fallback=fb)
            zono_state[name] = TorchZonotope(new_c, new_g)

        elif t == 'reduce_sum':
            # Linear reduction along given axes. Centers + gens both
            # sum along the same axes.
            from .zonotope import _torch_zono_reduce_sum
            z = _get(op['inputs'][0])
            in_shape_nd = op.get('in_shapes_nd', [None])[0]
            if in_shape_nd is None:
                raise ValueError(
                    f'reduce_sum: missing in_shapes_nd for {name!r}')
            new_c, new_g = _torch_zono_reduce_sum(
                z.center, z.generators, in_shape_nd,
                op.get('axes', ()), op.get('keepdims', False))
            if op['inputs'][0] in iv_state:
                # sum is monotone per coordinate: summed iv brackets the
                # true reduced values
                _l0, _h0 = iv_state[op['inputs'][0]]
                _g_empty = z.generators[:, :0]
                _lr, _ = _torch_zono_reduce_sum(
                    _l0, _g_empty, in_shape_nd, op.get('axes', ()),
                    op.get('keepdims', False))
                _hr, _ = _torch_zono_reduce_sum(
                    _h0, _g_empty, in_shape_nd, op.get('axes', ()),
                    op.get('keepdims', False))
                iv_state[name] = (_lr.detach(), _hr.detach())
            zono_state[name] = TorchZonotope(new_c, new_g)

        elif t == 'pow':
            # x^p as sound zonotope. Chord-tangent parallelogram per
            # element preserves x-y correlation via slope λ. Box fallback
            # used per-element when the [lo, hi] crosses curvature change.
            from .zonotope import _torch_zono_pow_int
            z = _get(op['inputs'][0])
            if op_geom_out is not None:
                op_geom_out[name] = {
                    'c_in': z.center.detach().clone(),
                    'gens': z.generators.detach().clone(),
                    'in_ids': list(col_ids_state[op['inputs'][0]])}
            exp = op.get('exponent', 2.0)
            assert float(int(exp)) == float(exp), (
                f'pow: only integer exponents supported, got {exp}')
            # Cache pre-pow input bounds + chord coeffs on the op dict
            # so backward can reuse them (no layer_idx → no `sb` slot).
            in_rad = (z.generators.abs().sum(dim=1)
                       if z.generators.numel() > 0
                       else torch.zeros_like(z.center))
            _ptl = _pth = None
            if op_clamps is not None and name in op_clamps:
                _ptl, _pth = op_clamps[name]
            _pl = (torch.maximum(z.center - in_rad, _ptl.to(z.center.dtype))
                   if _ptl is not None else z.center - in_rad)
            _ph = (torch.minimum(z.center + in_rad, _pth.to(z.center.dtype))
                   if _pth is not None else z.center + in_rad)
            op['_pow_in_lo'] = _pl.detach()
            op['_pow_in_hi'] = torch.maximum(_ph, _pl).detach()
            if op_bounds is not None:
                op_bounds[name] = (_pl.detach().clone(),
                                   torch.maximum(_ph, _pl).detach().clone())
            _relax = (settings.get('pow_relaxation', 'chord')
                       if settings is not None else 'chord')
            _alpha = (relu_lambdas[name] if relu_lambdas is not None
                      and name in relu_lambdas else None)
            if _alpha is not None and int(exp) == 2:
                # α-CROWN Sqr: convex band y=λx+μ±δ over the (clamped) range
                # with a DIFFERENTIABLE slope λ(α)=2lo+α(2hi-2lo). Sound for any
                # α (μ,δ are the exact midpoint/half-range of g=x²-λx, whose
                # only interior extremum is the tangent x*=λ/2); gradient flows
                # λ<-α. Preserves x-correlation (λ·gens) — no box.
                _a = _alpha.clamp(0.0, 1.0)
                lam = 2.0 * _pl + _a * (2.0 * _ph - 2.0 * _pl)
                xstar = (0.5 * lam).clamp(min=_pl, max=torch.maximum(_ph, _pl))
                g_lo = _pl * _pl - lam * _pl
                g_hi = _ph * _ph - lam * _ph
                g_st = xstar * xstar - lam * xstar
                gmax = torch.maximum(g_lo, g_hi)
                gmin = torch.minimum(torch.minimum(g_lo, g_hi), g_st)
                mu = 0.5 * (gmax + gmin)
                delta = (0.5 * (gmax - gmin)).clamp(min=0)
                new_c = lam * z.center + mu
                new_g = lam.unsqueeze(-1) * z.generators
                _pad = 1e-6 * (new_c.abs() + new_g.abs().sum(1) + 1.0)
                new_g = torch.cat([new_g, torch.diag(delta + _pad)], dim=1)
                if op_geom_out is not None and name in op_geom_out:
                    op_geom_out[name]['band'] = (
                        lam.detach().clone(), mu.detach().clone(),
                        (delta + _pad).detach().clone())
            else:
                new_c, new_g = _torch_zono_pow_int(
                    z.center, z.generators, int(exp), relaxation=_relax,
                    tight_lo=_ptl, tight_hi=_pth)
            op['_pow_relaxation'] = _relax
            zono_state[name] = TorchZonotope(new_c, new_g)

        elif t == 'matmul_bilinear':
            # A @ B with BOTH inputs perturbed (vit attention Q@K^T and
            # attn@V). True zonotope product: with shared noise symbols
            # e (columns aligned by the zonotope.py prefix invariant),
            #   (Ac + Σ A_i e_i) @ (Bc + Σ B_j e_j)
            #     = Ac@Bc + Σ_i (A_i@Bc + Ac@B_i) e_i + R,
            # the LINEAR terms are exact and column-aligned; the
            # quadratic remainder R = Σ_{ij} (A_i@B_j) e_i e_j is boxed
            # soundly by radius-matmul:  |R| ≤ (Σ_i|A_i|) @ (Σ_j|B_j|)
            # elementwise (each |e_i e_j| ≤ 1), appended as fresh
            # diagonal columns.
            za = _get(op['inputs'][0]); zb = _get(op['inputs'][1])
            sa = op.get('in_shapes_nd', [None, None])[0]
            sb_nd = op.get('in_shapes_nd', [None, None])[1]
            if sa is None or sb_nd is None:
                raise NotImplementedError(
                    f'matmul_bilinear {name!r}: need static N-D shapes')
            K_a = za.n_gens; K_b = zb.n_gens
            K = max(K_a, K_b)
            if op_bounds is not None:
                _al, _ah = _iv_bounds(op['inputs'][0], za)
                _bl, _bh = _iv_bounds(op['inputs'][1], zb)
                op_bounds[name] = (
                    (_al.detach().clone(), _ah.detach().clone()),
                    (_bl.detach().clone(), _bh.detach().clone()))
            Ac = za.center.reshape(sa)
            Bc = zb.center.reshape(sb_nd)
            # gens to (K, *shape), zero-padded to the common width
            GA = za.generators
            GB = zb.generators
            if K_a < K:
                GA = torch.cat([GA, torch.zeros(GA.shape[0], K - K_a,
                                                dtype=dtype, device=device)], 1)
            if K_b < K:
                GB = torch.cat([GB, torch.zeros(GB.shape[0], K - K_b,
                                                dtype=dtype, device=device)], 1)
            GA = GA.t().reshape(K, *sa)
            GB = GB.t().reshape(K, *sb_nd)
            center = Ac @ Bc
            lin = GA @ Bc + Ac.unsqueeze(0) @ GB        # (K, ..., n, p)
            radA = GA.abs().sum(dim=0)
            radB = GB.abs().sum(dim=0)
            R = radA @ radB                              # (..., n, p) ≥ 0
            n_out = center.numel()
            G_lin = lin.reshape(K, n_out).t().contiguous()
            Rf = R.reshape(-1)
            nz = torch.nonzero(Rf).flatten()
            G_quad = torch.zeros(n_out, nz.numel(), dtype=dtype,
                                 device=device)
            G_quad[nz, torch.arange(nz.numel(), device=device)] = Rf[nz]
            zono_state[name] = TorchZonotope(
                center.reshape(-1), torch.cat([G_lin, G_quad], dim=1))

        elif t in ('exp', 'reciprocal'):
            # 1-D convex relaxation as a sound parallelogram:
            #   y = k*x + (g_end + g_min)/2 + ((g_end - g_min)/2) * e_new
            # with k the chord slope on [l, u]; the gap g(x) = f(x) - k*x
            # is maximal (equal) at the endpoints and minimal at the
            # tangency point x* (f'(x*) = k). Mirrors alpha,beta-CROWN's
            # BoundExp / BoundReciprocal chord-vs-tangent planes; sound
            # because the parallelogram contains f pointwise on [l, u].
            z = _get(op['inputs'][0])
            lo_in, hi_in = _iv_bounds(op['inputs'][0], z)
            if op_clamps is not None and name in op_clamps:
                cl, ch = op_clamps[name]
                lo_in = torch.maximum(lo_in, cl.to(lo_in.dtype))
                hi_in = torch.minimum(hi_in, ch.to(hi_in.dtype))
                # an empty intersection means the subdomain is infeasible
                # for this abstract state; keep numerics valid (any bound
                # over the degenerate point interval is sound)
                hi_in = torch.maximum(hi_in, lo_in)
            if op_bounds is not None:
                op_bounds[name] = (lo_in.detach().clone(),
                                   hi_in.detach().clone())
            if t == 'reciprocal':
                if float(lo_in.min()) <= 0:
                    raise NotImplementedError(
                        f'reciprocal {name!r}: input lower bound '
                        f'{float(lo_in.min()):.3g} <= 0 — relaxation only '
                        f'valid on positive inputs')
                f_l = 1.0 / lo_in
                f_u = 1.0 / hi_in
            else:
                f_l = torch.exp(lo_in)
                f_u = torch.exp(hi_in)
            # monotone closed form exactly brackets the true outputs on
            # the intersected input range
            iv_state[name] = (torch.minimum(f_l, f_u).detach(),
                              torch.maximum(f_l, f_u).detach())
            w_in = (hi_in - lo_in).clamp(min=1e-12)
            k = (f_u - f_l) / w_in
            if relu_lambdas is not None and name in relu_lambdas:
                # alpha-zono: caller-optimized slope, ANY value within
                # [f'(l), f'(u)] is sound for a convex f (offsets below
                # are recomputed from the slope, so the parallelogram
                # always contains f on [l, u]). The raw param in [0, 1]
                # interpolates the derivative range.
                _s01 = relu_lambdas[name].clamp(0, 1)
                if t == 'exp':
                    k = f_l + _s01 * (f_u - f_l)        # f' = e^x
                else:
                    d_lo = -1.0 / lo_in.pow(2)          # most negative
                    d_hi = -1.0 / hi_in.pow(2)
                    k = d_lo + _s01 * (d_hi - d_lo)
            if t == 'exp':
                xs = torch.log(k.clamp(min=1e-300))
                fs = k.clamp(min=1e-300)    # f(x*) = e^{x*} = k
            else:
                xs = torch.sqrt(1.0 / (-k).clamp(min=1e-300))
                fs = 1.0 / xs
            xs = torch.minimum(torch.maximum(xs, lo_in), hi_in)
            fxs = torch.exp(xs) if t == 'exp' else 1.0 / xs
            fs = fxs
            g_l_end = f_l - k * lo_in
            g_u_end = f_u - k * hi_in
            g_min = fs - k * xs
            g_hi = torch.maximum(torch.maximum(g_l_end, g_u_end), g_min)
            g_lo = torch.minimum(torch.minimum(g_l_end, g_u_end), g_min)
            # positivity guard: exp/recip outputs are strictly positive,
            # but the parallelogram's lower edge can dip below 0 on wide
            # inputs — downstream reciprocal then loses its domain. Where
            # that happens, fall back per-element to the exact positive
            # interval box (slope 0). Sound either way.
            _edge_lo = (torch.minimum(k * lo_in, k * hi_in) + g_lo)
            _boxm = _edge_lo <= 0
            _fmin = torch.minimum(f_l, f_u)
            _fmax = torch.maximum(f_l, f_u)
            k = torch.where(_boxm, torch.zeros_like(k), k)
            c_off = torch.where(_boxm, (_fmin + _fmax) / 2,
                                (g_hi + g_lo) / 2)
            r_off = torch.where(_boxm, (_fmax - _fmin) / 2,
                                (g_hi - g_lo) / 2)
            new_c = k * z.center + c_off
            G_old = z.generators * k.unsqueeze(1)
            nzr = torch.nonzero(r_off).flatten()
            G_new = torch.zeros(new_c.numel(), nzr.numel(), dtype=dtype,
                                device=device)
            G_new[nzr, torch.arange(nzr.numel(), device=device)] = \
                r_off[nzr]
            zono_state[name] = TorchZonotope(
                new_c, torch.cat([G_old, G_new], 1))

        else:
            raise NotImplementedError(
                f'_forward_zonotope_graph: unsupported op {t!r} '
                f'(name={name!r}). Silent skip would propagate stale zono — '
                'add a forward handler before using this op.')

        gen_count[name] = zono_state[name].n_gens

        # Generic per-column IDs for ops that didn't set them in-handler.
        # Default: prefix-preserving — keep input[0]'s IDs (linear / relu /
        # sin / cos / pow / exp / slice / reshape / fc / reduce_sum / div /
        # matmul preserve their first input's columns as a leading block) and
        # append fresh IDs for any extra columns. (Merges, concat, and
        # sigmoid/tanh set their own IDs in-handler.) No input -> all fresh.
        if name not in col_ids_state:
            nout = zono_state[name].n_gens
            in0 = op['inputs'][0] if op['inputs'] else None
            base = col_ids_state.get(in0) if in0 is not None else None
            if base is None:
                col_ids_state[name] = _alloc_ids(nout)
            elif nout <= len(base):
                col_ids_state[name] = list(base[:nout])
            else:
                col_ids_state[name] = list(base) + _alloc_ids(nout - len(base))

        # Record the fresh global-noise IDs this op allocated (the nonlinear
        # ops' per-element δ columns), so the nonlinear-split BaB can attribute
        # output margin-slack back to each op (bbps-style scoring).
        if op_fresh_ids is not None and _id_counter[0] > _pre_fresh_id:
            op_fresh_ids[name] = (_pre_fresh_id, _id_counter[0])

        # Free zonotopes (and their col IDs) that are no longer needed
        for inp in op['inputs']:
            if last_use.get(inp) == op_idx and inp in zono_state:
                del zono_state[inp]
                col_ids_state.pop(inp, None)

    # Debug hook: expose the final per-column ID map (used by soundness probes
    # to inspect merge alignment). Off unless a dict is supplied.
    if col_ids_out is not None:
        col_ids_out.update(col_ids_state)

    # Find last op's zonotope
    last_name = gg['ops'][-1]['name']
    return sb, zono_state[last_name]


def _find_shared_gens_count(name_a, name_b, gg, gen_count):
    """Find shared generator count at fork point for two merging branches.

    Walks backward through gpu_graph ops to find the deepest common ancestor
    that is a fork point. NOTE: this assumes a positional noise-identity
    invariant (column k of every descendant of fork F is F's k-th noise) that
    holds only for prefix-preserving ops (linear / relu / sin / cos / pow /
    exp / merges). The unbatched forward instead tracks explicit per-column
    global IDs (``col_ids_state``) and uses ``_common_prefix_len`` at merges,
    which stays correct across prefix-breaking ops (sigmoid/tanh, both-vary
    mul/sub/div). This function remains for the batched/alpha paths whose
    graphs are prefix-preserving.
    """
    forks = gg['fork_points']
    input_name = gg['input_name']

    # Build predecessor map from ops
    pred_map = {}
    for op in gg['ops']:
        pred_map[op['name']] = op['inputs']

    def _ancestors(name):
        visited = []
        stack = [name]
        seen = set()
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            visited.append(n)
            if n in pred_map:
                for inp in pred_map[n]:
                    stack.append(inp)
        return visited

    anc_a = _ancestors(name_a)
    anc_b_set = set(_ancestors(name_b))
    for anc in anc_a:
        if anc in anc_b_set and anc in forks:
            return gen_count.get(anc, 0)
    return gen_count.get(input_name, 0)


def _common_prefix_len(ids_a, ids_b):
    """Length of the longest common leading run of two column-ID sequences.

    Two branches forking from a common ancestor carry that ancestor's noise as
    a shared prefix of their column-ID arrays; they diverge at the first column
    whose global ID differs. That divergence index is exactly the number of
    generator columns the two branches genuinely share (same noise, same
    position) — the sound ``shared`` count for a zonotope merge. Robust to
    prefix-breaking ops, which alter the ID arrays so the common prefix
    shrinks accordingly.
    """
    n = min(len(ids_a), len(ids_b))
    i = 0
    while i < n and ids_a[i] == ids_b[i]:
        i += 1
    return i


@torch.no_grad()
def _spec_backward_graph(tight, xl, xh, gg, spec_ew,
                          remaining_specs, nh, device, dtype,
                          return_ew=False, return_input_linear=False,
                          op_bounds=None, seed_node=None):
    """Graph-aware spec backward pass for networks with skip connections.

    spec_ew maps query_id -> (w, bias) where w is in OUTPUT space.
    Propagates backward through ALL ops including the final linear layer.

    seed_node: if given, the backward starts at this op's output (w is in
        that node's space, not OUTPUT space) instead of the final op — used
        by the topo-order intermediate-bound refinement to bound an
        arbitrary interior node by CROWN-to-input. Ops topologically after
        seed_node carry no ew and are skipped.

    Returns (spec_lbs, still_open). Optional flags add tuple tails:
      - return_ew=True: appends ew_at_relu (qid -> {layer_idx -> ew_numpy})
      - return_input_linear=True: appends input_linear
        (qid -> (ew_inp_numpy, acc_float)), the linear lower-bound
        coefficients in input space such that for all x in [xl, xh]:
            spec(x) >= ew_inp · x + acc
        Used by `_input_split_fast_leaf`'s joint-AND infeasibility LP.
    """
    ops = gg['ops']
    # Shared lazy cache for bilinear-op point-side centers (mscn).
    point_centers_cache = [None]

    spec_lbs = {}
    all_ew_at_relu = {} if return_ew else None
    input_linear = {} if return_input_linear else None
    for qid in remaining_specs:
        ew_init, b_spec = spec_ew[qid]
        ew_at = {}
        acc = b_spec
        qid_ew_at_relu = {} if return_ew else None

        # Seed ew at the output of the last op (or an interior node when
        # refining that node's bound by CROWN-to-input).
        last_name = seed_node if seed_node is not None else ops[-1]['name']
        ew_at[last_name] = ew_init.clone()

        # Walk backward through ALL ops
        for op in reversed(ops):
            name = op['name']
            if name not in ew_at:
                continue
            ew = ew_at[name]
            t = op['type']
            _tr = __import__('os').environ.get('VC_BWD_TRACE')
            if _tr and (_tr == 'all' or str(qid) == _tr):
                _ewmag = float(sum(v.abs().sum() for v in ew_at.values()))
                print(f'[bwd q{qid}] {name} ({t}) ew={tuple(ew.shape)} '
                      f'acc={float(acc):+.5g} |ew|1={_ewmag:.5g}', flush=True)

            if t == 'conv':
                acc += float(
                    ew.reshape(1, *op['out_shape']).reshape(
                        op['out_shape'][0], -1).sum(dim=1) @ op['bias'])
                ew_back = F.conv_transpose2d(
                    ew.reshape(1, *op['out_shape']), op['kernel'],
                    stride=op['stride'], padding=op['padding'],
                    output_padding=op['output_padding']).flatten()
                inp = op['inputs'][0]
                ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_back)) + ew_back

            elif t == 'fc':
                W = op['W']; bias = op['bias']
                in_shape_nd = op.get('in_shapes_nd', [None])[0]
                # Batched/ND matmul: in (prefix..., k), W (m, k) -> y (prefix...,
                # m), where the trailing m may be SQUEEZED in out_shape (m==1 ->
                # out (prefix,)). Detect by the contraction dim + numel rather
                # than out_shape_nd[-1] (which can have lost the trailing 1).
                if (in_shape_nd is not None and len(in_shape_nd) >= 2
                        and W.shape[1] == in_shape_nd[-1]
                        and ew.numel() == int(np.prod(in_shape_nd[:-1]))
                        * W.shape[0]):
                    prefix = tuple(int(d) for d in in_shape_nd[:-1])
                    ew_nd = ew.reshape(*prefix, W.shape[0])
                    acc += float((ew_nd * bias).sum())
                    ew_back = (ew_nd @ W).reshape(-1)
                else:
                    acc += float((ew * bias).sum())   # bias may be scalar/bcast
                    ew_back = ew @ W
                inp = op['inputs'][0]
                ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_back)) + ew_back

            elif t == 'relu':
                if 'layer_idx' in op:
                    if return_ew:
                        qid_ew_at_relu[op['layer_idx']] = ew.cpu().numpy()
                    lo_k, hi_k = tight[op['layer_idx']]
                    lo_s, up_s, up_t, _, _, _ = _make_slopes(lo_k, hi_k)
                    ep = ew.clamp(min=0)
                    en = ew.clamp(max=0)
                    acc += float((en * up_t).sum())
                    ew_back = ep * lo_s + en * up_s
                else:
                    ew_back = ew
                inp = op['inputs'][0]
                ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_back)) + ew_back

            elif t == 'add':
                # Add backward: ew goes to both inputs unchanged.
                # Non-merge bias-add contributes the bias-dot to acc.
                if op.get('is_merge'):
                    for inp in op['inputs']:
                        ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew)) + ew
                else:
                    bias = op.get('bias')
                    if bias is not None:
                        from .alpha_crown import _bias_dot_ew
                        acc += float(_bias_dot_ew(
                            ew, bias, ew.dtype, ew.device,
                            out_shape=op.get('out_shape_nd')))
                    inp = op['inputs'][0]
                    ew_b = _bcast_ew_back(ew, op)
                    ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_b)) + ew_b

            elif t == 'sub':
                # Sub backward: ew passes through (reduce-summed if the forward
                # outer-broadcast the input), bias contributes to acc.
                bias = op.get('bias')
                if bias is not None:
                    from .alpha_crown import _bias_dot_ew
                    acc -= float(_bias_dot_ew(
                        ew, bias, ew.dtype, ew.device,
                        out_shape=op.get('out_shape_nd')))
                inp = op['inputs'][0]
                ew_b = _bcast_ew_back(ew, op)
                ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_b)) + ew_b

            elif t == 'sub_bilinear':
                # Sub(a, b) backward: y = a - b → ew_a = ew, ew_b = -ew.
                ia, ib = op['inputs'][0], op['inputs'][1]
                ew_at[ia] = ew_at.get(ia, torch.zeros_like(ew)) + ew
                ew_at[ib] = ew_at.get(ib, torch.zeros_like(ew)) + (-ew)

            elif t == 'reshape':
                inp = op['inputs'][0]
                ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew)) + ew

            elif t in ('slice', 'gather'):
                flat_idx = op.get('flat_idx')
                in_shape_nd = op.get('in_shapes_nd', [None])[0]
                if flat_idx is None or in_shape_nd is None:
                    raise ValueError(
                        f"slice backward missing flat_idx/in_shape")
                n_in = int(np.prod(in_shape_nd))
                idx_t = torch.as_tensor(flat_idx, dtype=torch.long,
                                          device=ew.device)
                ew_back = torch.zeros(n_in, dtype=ew.dtype, device=ew.device)
                # Adjoint of a gather (out[k]=in[idx[k]]) is scatter-ADD:
                # in_grad[idx[k]] += out_grad[k]. When flat_idx has DUPLICATE
                # indices (a fan-out gather — one input feeds many outputs,
                # as power-flow nets do gathering a bus voltage into many line
                # equations), index_copy_ keeps only the LAST write and DROPS
                # the rest, producing an unsound backward coefficient. Use
                # index_add_ (== index_copy_ when indices are unique).
                ew_back.index_add_(-1, idx_t, ew)
                inp = op['inputs'][0]
                ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_back)) + ew_back

            elif t == 'concat':
                in_shapes = op.get('in_shapes_nd', [])
                offset = 0
                for inp, in_shape_nd in zip(op['inputs'], in_shapes):
                    n_in = int(np.prod(in_shape_nd))
                    ew_i = ew[offset:offset + n_in]
                    ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_i)) + ew_i
                    offset += n_in

            elif t == 'mul':
                # y = scale * x → ew_back = ew * scale.
                scale_t = op.get('scale')
                if isinstance(scale_t, np.ndarray):
                    scale_t = torch.from_numpy(scale_t).to(
                        device=ew.device, dtype=ew.dtype)
                elif not isinstance(scale_t, torch.Tensor):
                    scale_t = torch.tensor(scale_t, dtype=ew.dtype,
                                            device=ew.device)
                else:
                    scale_t = scale_t.to(device=ew.device, dtype=ew.dtype)
                sflat = scale_t.flatten()
                n = ew.numel()
                if sflat.numel() == 1 or sflat.numel() == n:
                    ew_back = ew * sflat
                else:
                    in_shape = op.get('in_shapes_nd', [None])[0]
                    C, H, W = in_shape
                    assert sflat.numel() == C
                    scale_4d = sflat.view(1, C, 1, 1).expand(
                        1, C, H, W).reshape(-1)
                    ew_back = ew * scale_4d
                inp = op['inputs'][0]
                ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_back)) + ew_back

            elif t == 'reduce_sum':
                from .alpha_crown import _reduce_sum_backward
                in_shape_nd = op.get('in_shapes_nd', [None])[0]
                out_shape_nd = op.get('out_shape_nd')
                # ew is 1D (n_out,); _reduce_sum_backward expects lead
                # dims. Add a dummy lead, then squeeze.
                ew_back = _reduce_sum_backward(
                    ew.unsqueeze(0), in_shape_nd, op.get('axes', ()),
                    op.get('keepdims', False), out_shape_nd).squeeze(0)
                inp = op['inputs'][0]
                ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_back)) + ew_back

            elif (t in ('exp', 'reciprocal') and op_bounds is not None
                  and name in op_bounds):
                # 1-D convex backward (alpha,beta-CROWN BoundExp /
                # BoundReciprocal planes): positive ew takes the LOWER
                # plane (tangent), negative ew the UPPER plane (chord).
                l_in, u_in = op_bounds[name]
                l_in = l_in.to(device=ew.device, dtype=ew.dtype)
                u_in = u_in.to(device=ew.device, dtype=ew.dtype)
                w_ = (u_in - l_in).clamp(min=1e-12)
                if t == 'exp':
                    f_l = torch.exp(l_in); f_u = torch.exp(u_in)
                    k_up = (f_u - f_l) / w_
                    b_up = f_l - k_up * l_in
                    m_t = torch.minimum((l_in + u_in) / 2, l_in + 0.99)
                    k_lo = torch.exp(m_t)
                    b_lo = k_lo * (1 - m_t)
                else:
                    assert bool((l_in > 0).all()), (
                        f'reciprocal backward {name!r}: nonpositive input '
                        f'lower bound')
                    k_up = -1.0 / (l_in * u_in)
                    b_up = 1.0 / l_in + 1.0 / u_in
                    m_t = (l_in + u_in) / 2
                    k_lo = -1.0 / (m_t * m_t)
                    b_lo = 2.0 / m_t
                ep = ew.clamp(min=0); en = ew.clamp(max=0)
                acc += float((ep * b_lo + en * b_up).sum())
                ew_back = ep * k_lo + en * k_up
                inp = op['inputs'][0]
                ew_at[inp] = ew_at.get(
                    inp, torch.zeros_like(ew_back)) + ew_back

            elif (t == 'matmul_bilinear' and op_bounds is not None
                  and name in op_bounds):
                # McCormick corner planes per product term (the
                # alpha,beta-CROWN MulHelper default, r at the corner):
                #   x*y >= yl*x + xl*y - xl*yl   (lower)
                #   x*y <= yh*x + xl*y - xl*yh   (upper)
                # ew on Z distributes to BOTH inputs through the planes.
                (xl_b, xh_b), (yl_b, yh_b) = op_bounds[name]
                sa = op['in_shapes_nd'][0]
                sb_ = op['in_shapes_nd'][1]
                so = op['out_shape_nd']
                Xl = xl_b.to(ew.dtype).reshape(sa).unsqueeze(-1)
                Yl = yl_b.to(ew.dtype).reshape(sb_).unsqueeze(-3)
                Yh = yh_b.to(ew.dtype).reshape(sb_).unsqueeze(-3)
                ew_nd = ew.reshape(so)
                ep = ew_nd.clamp(min=0).unsqueeze(-2)   # (.., n, 1, p)
                en = ew_nd.clamp(max=0).unsqueeze(-2)
                acc += float((ep * (-(Xl * Yl))
                              + en * (-(Xl * Yh))).sum())
                ew_x = (ep * Yl + en * Yh).sum(dim=-1)
                ew_y = ((ep + en) * Xl).sum(dim=-3)
                ia, ib = op['inputs'][0], op['inputs'][1]
                ew_x = ew_x.reshape(-1); ew_y = ew_y.reshape(-1)
                ew_at[ia] = ew_at.get(ia, torch.zeros_like(ew_x)) + ew_x
                ew_at[ib] = ew_at.get(ib, torch.zeros_like(ew_y)) + ew_y

            elif (t == 'mul_bilinear' and op['inputs'][0] == op['inputs'][1]
                  and op_bounds is not None and name in op_bounds):
                # Self-product x*x = x^2 (ml4acopf /Pow_* squares). The
                # general McCormick branch below treats the two operands as
                # INDEPENDENT (corner plane y_l*x + x_l*y - x_l*y_l), which
                # over a wide sign-mixed [-a, a] gives a lower bound ~ -a^2
                # at x=0 — catastrophic (this was the -444 spec bound). x^2
                # is convex everywhere, so use the exact square two-line
                # CROWN (tangent LB + chord UB) — the same helper the `pow`
                # backward uses, and a single coefficient to the shared input.
                (xl_b, xh_b), _ = op_bounds[name]
                lo_sq = xl_b.to(device=ew.device, dtype=ew.dtype).flatten()
                hi_sq = xh_b.to(device=ew.device, dtype=ew.dtype).flatten()
                assert lo_sq.numel() == ew.numel(), (
                    f'self-product square backward: operand/output numel '
                    f'mismatch at {name!r} ({lo_sq.numel()} vs {ew.numel()})')
                (lb_slope, lb_const, ub_slope, ub_const,
                 use_two_line, box_lo_v, box_hi_v) = _pow_two_line_coeffs(
                    lo_sq, hi_sq, 2)
                ep = ew.clamp(min=0); en = ew.clamp(max=0)
                slope_back = ep * lb_slope + en * ub_slope
                const_back = ep * lb_const + en * ub_const
                # Box fallback is unreachable for p=2 now (convex everywhere)
                # but kept for parity/safety with the pow backward.
                slope_back = torch.where(use_two_line, slope_back,
                                          torch.zeros_like(slope_back))
                const_back = torch.where(use_two_line, const_back,
                                          torch.where(ep > 0, box_lo_v,
                                              torch.zeros_like(box_lo_v))
                                          + torch.where(en < 0, box_hi_v,
                                              torch.zeros_like(box_hi_v)))
                acc += float(const_back.sum())
                if __import__('os').environ.get('VC_SQR_DUMP') == name:
                    _nz = (ew.abs() > 1e-12).nonzero().flatten().tolist()
                    print(f'[sqr {name}] ew nonzero elems: '
                          f'{[(i, round(float(ew[i]),4), round(float(lo_sq[i]),3), round(float(hi_sq[i]),3), round(float(const_back[i]),3)) for i in _nz]}',
                          flush=True)
                    print(f'  (i, ew, lo, hi, const_contrib); sum const={float(const_back.sum()):+.4g}',
                          flush=True)
                inp = op['inputs'][0]
                ew_at[inp] = ew_at.get(
                    inp, torch.zeros_like(slope_back)) + slope_back

            elif (t == 'mul_bilinear' and op_bounds is not None
                  and name in op_bounds):
                # Element-wise McCormick (broadcast-aware): same corner
                # planes as matmul_bilinear, per element.
                from .alpha_crown import _sum_to_shape
                (xl_b, xh_b), (yl_b, yh_b) = op_bounds[name]
                sa = op['in_shapes_nd'][0]
                sb_ = op['in_shapes_nd'][1]
                so = op['out_shape_nd']
                ones_o = torch.ones(so, dtype=ew.dtype, device=ew.device)
                Xl = xl_b.to(ew.dtype).reshape(sa) * ones_o
                Yl = yl_b.to(ew.dtype).reshape(sb_) * ones_o
                Yh = yh_b.to(ew.dtype).reshape(sb_) * ones_o
                ew_nd = ew.reshape(so)
                ep = ew_nd.clamp(min=0); en = ew_nd.clamp(max=0)
                acc += float((ep * (-(Xl * Yl))
                              + en * (-(Xl * Yh))).sum())
                cx = ep * Yl + en * Yh
                cy = (ep + en) * Xl
                ia, ib = op['inputs'][0], op['inputs'][1]
                ew_x = _sum_to_shape(cx, (), sa).reshape(-1)
                ew_y = _sum_to_shape(cy, (), sb_).reshape(-1)
                ew_at[ia] = ew_at.get(ia, torch.zeros_like(ew_x)) + ew_x
                ew_at[ib] = ew_at.get(ib, torch.zeros_like(ew_y)) + ew_y

            elif t in ('mul_bilinear', 'div_bilinear'):
                # Sound linearised backward for Div with non-point
                # denominator. Uses analytic R-bounds at 4 corners +
                # 2 edge criticals (where ∂R/∂b = 0 for a=a_lo, a=a_hi).
                if t == 'div_bilinear' and op.get('_div_decoupled'):
                    sh_in = op.get('in_shapes_nd', [None, None])
                    sh_out = op.get('out_shape_nd')
                    ia, ib = op['inputs'][0], op['inputs'][1]
                    from .alpha_crown import _sum_to_shape
                    a_lo = op['_div_a_lo'].to(device=ew.device, dtype=ew.dtype)
                    a_hi = op['_div_a_hi'].to(device=ew.device, dtype=ew.dtype)
                    b_lo = op['_div_b_lo'].to(device=ew.device, dtype=ew.dtype)
                    b_hi = op['_div_b_hi'].to(device=ew.device, dtype=ew.dtype)
                    assert bool((b_lo > 0).all()), (
                        f'div_bilinear backward only supports b > 0; '
                        f'got b_lo={b_lo}')
                    c_a_loc = (a_lo + a_hi) / 2
                    c_b_loc = (b_lo + b_hi) / 2
                    inv_cb = 1.0 / c_b_loc
                    neg_ca_over_cb2 = -c_a_loc / (c_b_loc * c_b_loc)
                    L_const = c_a_loc / c_b_loc
                    def _R_at(a_eval, b_eval):
                        L_val = (a_eval * inv_cb + b_eval * neg_ca_over_cb2
                                  + L_const)
                        return a_eval / b_eval - L_val
                    ones_out_pre = torch.ones(*sh_out, dtype=ew.dtype,
                                                device=ew.device)
                    a_lo_out = ones_out_pre * a_lo.reshape(*sh_in[0])
                    a_hi_out = ones_out_pre * a_hi.reshape(*sh_in[0])
                    b_lo_out = ones_out_pre * b_lo.reshape(*sh_in[1])
                    b_hi_out = ones_out_pre * b_hi.reshape(*sh_in[1])
                    c_a_out = (a_lo_out + a_hi_out) / 2
                    pos_a_lo = (a_lo_out > 0) & (c_a_out > 1e-30)
                    pos_a_hi = (a_hi_out > 0) & (c_a_out > 1e-30)
                    b_crit_lo = torch.where(pos_a_lo,
                        c_b_loc.reshape(*sh_in[1]) * ones_out_pre *
                            torch.sqrt(torch.clamp(
                                a_lo_out / c_a_out.clamp(min=1e-30), min=0)),
                        b_lo_out)
                    b_crit_hi = torch.where(pos_a_hi,
                        c_b_loc.reshape(*sh_in[1]) * ones_out_pre *
                            torch.sqrt(torch.clamp(
                                a_hi_out / c_a_out.clamp(min=1e-30), min=0)),
                        b_lo_out)
                    b_crit_lo = torch.maximum(torch.minimum(b_crit_lo,
                                                              b_hi_out), b_lo_out)
                    b_crit_hi = torch.maximum(torch.minimum(b_crit_hi,
                                                              b_hi_out), b_lo_out)
                    R_pts = torch.stack([
                        _R_at(a_lo_out, b_lo_out),
                        _R_at(a_lo_out, b_hi_out),
                        _R_at(a_hi_out, b_lo_out),
                        _R_at(a_hi_out, b_hi_out),
                        _R_at(a_lo_out, b_crit_lo),
                        _R_at(a_hi_out, b_crit_hi),
                    ])
                    R_min = R_pts.min(dim=0).values
                    R_max = R_pts.max(dim=0).values
                    R_mid = (R_min + R_max) / 2
                    R_half = (R_max - R_min) / 2
                    ones_out = torch.ones(*sh_out, dtype=ew.dtype,
                                            device=ew.device)
                    inv_cb_out = ones_out * inv_cb.reshape(*sh_in[1])
                    neg_grad_b_out = ones_out * neg_ca_over_cb2.reshape(*sh_out)
                    L_const_out = L_const.reshape(*sh_out)
                    R_mid_out = R_mid.reshape(*sh_out)
                    R_half_out = R_half.reshape(*sh_out)
                    ew_nd = ew.reshape(*sh_out)
                    ew_a_nd = _sum_to_shape(ew_nd * inv_cb_out, (), sh_in[0])
                    ew_b_nd = _sum_to_shape(ew_nd * neg_grad_b_out, (), sh_in[1])
                    ew_a = ew_a_nd.reshape(-1)
                    ew_b = ew_b_nd.reshape(-1)
                    acc += float(
                        (ew_nd * (L_const_out + R_mid_out)).reshape(-1).sum()
                        - (ew_nd.abs() * R_half_out).reshape(-1).sum())
                    ew_at[ia] = ew_at.get(ia, torch.zeros_like(ew_a)) + ew_a
                    ew_at[ib] = ew_at.get(ib, torch.zeros_like(ew_b)) + ew_b
                else:
                    # Sound linearization path: requires one side to be
                    # point (zero radius) — assertion enforced here so
                    # an accidental call on non-point inputs raises
                    # rather than silently un-sound.
                    from .alpha_crown import _compute_point_centers, _sum_to_shape
                    if point_centers_cache[0] is None:
                        x_center = ((xl + xh) / 2).to(
                            device=ew.device, dtype=ew.dtype)
                        point_centers_cache[0] = _compute_point_centers(
                            gg, x_center, ew.device, ew.dtype)
                    point_centers = point_centers_cache[0]
                    sh_in = op.get('in_shapes_nd', [None, None])
                    sh_out = op.get('out_shape_nd')
                    c_a = point_centers[op['inputs'][0]]
                    c_b = point_centers[op['inputs'][1]]
                    ew_nd = ew.reshape(*sh_out)
                    a_nd = c_a.reshape(*sh_in[0])
                    b_nd = c_b.reshape(*sh_in[1])
                    if t == 'mul_bilinear':
                        ew_a_nd = _sum_to_shape(ew_nd * b_nd, (), sh_in[0])
                        ew_b_nd = _sum_to_shape(ew_nd * a_nd, (), sh_in[1])
                    else:
                        if bool((b_nd == 0).any()):
                            raise ZeroDivisionError(
                                f'div_bilinear backward: denom zero at {name!r}')
                        inv_b = b_nd.reciprocal()
                        ew_a_nd = _sum_to_shape(ew_nd * inv_b, (), sh_in[0])
                        ew_b_nd = _sum_to_shape(
                            -ew_nd * a_nd * inv_b * inv_b, (), sh_in[1])
                    ew_a = ew_a_nd.reshape(-1)
                    ew_b = ew_b_nd.reshape(-1)
                    ia, ib = op['inputs'][0], op['inputs'][1]
                    ew_at[ia] = ew_at.get(ia, torch.zeros_like(ew_a)) + ew_a
                    ew_at[ib] = ew_at.get(ib, torch.zeros_like(ew_b)) + ew_b

            elif t in ('sigmoid', 'tanh'):
                # CROWN backward: closed-form linear slopes via the
                # same `_sigmoid_tanh_linear_bounds` helper used by the
                # batched pipeline. Pre-activation bounds from `tight`.
                L = op.get('layer_idx')
                lo_pre, hi_pre = tight[L]
                lo_s, lo_t, up_s, up_t = _sigmoid_tanh_linear_bounds(
                    lo_pre, hi_pre, t)
                ep = ew.clamp(min=0); en = ew.clamp(max=0)
                acc += float((ep * lo_t).sum() + (en * up_t).sum())
                ew_back = ep * lo_s + en * up_s
                inp = op['inputs'][0]
                ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_back)) + ew_back

            elif t == 'pow':
                # Pow backward — two-line CROWN (separate LB & UB slopes,
                # tighter than chord+slack). Matches α,β-CROWN BoundPow
                # `bound_relax_branch`. The tangent point on the convex
                # side defaults to the midpoint of [lo, hi]; an
                # α-optimizable point can be passed in later for further
                # tightening.
                lo_pre = op.get('_pow_in_lo')
                hi_pre = op.get('_pow_in_hi')
                assert lo_pre is not None and hi_pre is not None, (
                    f"pow backward: missing _pow_in_lo/_pow_in_hi for "
                    f"{name!r} — forward must run before backward.")
                lo_pre_t = lo_pre.to(device=ew.device, dtype=ew.dtype)
                hi_pre_t = hi_pre.to(device=ew.device, dtype=ew.dtype)
                p = int(op.get('exponent', 2))
                inp = op['inputs'][0]
                # Allow α-CROWN to inject a per-element tangent point.
                tan_pos_alpha = op.get('_pow_tangent_alpha')
                if tan_pos_alpha is not None:
                    tan_pos_alpha = tan_pos_alpha.to(
                        device=ew.device, dtype=ew.dtype)
                (lb_slope, lb_const, ub_slope, ub_const,
                 use_two_line, box_lo_v, box_hi_v) = _pow_two_line_coeffs(
                    lo_pre_t, hi_pre_t, p, tangent_pos=tan_pos_alpha)
                ep = ew.clamp(min=0); en = ew.clamp(max=0)
                # For ew_pos: lower-bound the linear contribution → use LB.
                # For ew_neg: lower-bound the linear contribution → use UB.
                slope_back = ep * lb_slope + en * ub_slope
                const_back = ep * lb_const + en * ub_const
                # Where two_line isn't valid (sign-mixed): fall back to
                # box (zero slope, ep·box_lo + en·box_hi).
                not_tl = ~use_two_line
                slope_back = torch.where(use_two_line, slope_back,
                                          torch.zeros_like(slope_back))
                const_back = torch.where(use_two_line, const_back,
                                          torch.where(ep > 0, box_lo_v,
                                              torch.zeros_like(box_lo_v))
                                          + torch.where(en < 0, box_hi_v,
                                              torch.zeros_like(box_hi_v)))
                acc += float(const_back.sum())
                ew_back = slope_back
                ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_back)) + ew_back

            elif t == 'floor':
                # Floor backward via the SAME sound affine band the forward
                # DeepZ uses (nl_floor.FloorRelax): within one integer cell
                # floor is the exact constant (lam=0, mu=floor(lo), delta=0 —
                # matches ABC's exact floor node); spanning a boundary uses
                # floor(x) in (x-1, x] => lam=1, mu=-0.5, delta=0.5. The lower
                # / upper linear bounds are lam*x + (mu -/+ delta); CROWN picks
                # per sign of ew (both slopes == lam, so ew_back = ew*lam).
                from .nl_floor import FloorRelax
                L = op.get('layer_idx')
                if L is not None and L in tight:
                    lo_pre, hi_pre = tight[L]
                elif op_bounds is not None and name in op_bounds:
                    # Forward stores floor's (clamp-tightened) input pre-bounds
                    # under op_bounds[name]; floor often carries no layer_idx.
                    lo_pre, hi_pre = op_bounds[name]
                else:
                    raise KeyError(
                        f'floor backward: no input bounds for {name!r} '
                        f'(layer_idx={L}); forward must populate tight[L] or '
                        f'op_bounds[name] before the backward pass.')
                lam, mu, delta = FloorRelax().affine_band(
                    lo_pre.to(device=ew.device, dtype=ew.dtype),
                    hi_pre.to(device=ew.device, dtype=ew.dtype))
                ep = ew.clamp(min=0); en = ew.clamp(max=0)
                acc += float((ep * (mu - delta)).sum()
                             + (en * (mu + delta)).sum())
                ew_back = ew * lam
                inp = op['inputs'][0]
                ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_back)) + ew_back

            else:
                raise NotImplementedError(
                    f'_spec_backward_graph: unsupported op {t!r} (name={name!r}) — '
                    'unhandled ops silently drop ew, producing unsound bounds. '
                    'Add a backward handler before using this op type.')

        # At input: interval bound
        input_name = gg['input_name']
        ew_inp = ew_at.get(input_name, torch.zeros_like(xl))
        spec_lbs[qid] = acc + float(
            ew_inp.clamp(min=0) @ xl + ew_inp.clamp(max=0) @ xh)
        if return_ew:
            all_ew_at_relu[qid] = qid_ew_at_relu
        if return_input_linear:
            input_linear[qid] = (
                ew_inp.detach().cpu().numpy().astype(np.float64),
                float(acc))

    still_open = {c for c in remaining_specs if spec_lbs[c] <= 0}
    if return_input_linear and return_ew:
        return spec_lbs, still_open, all_ew_at_relu, input_linear
    if return_input_linear:
        return spec_lbs, still_open, input_linear
    if return_ew:
        return spec_lbs, still_open, all_ew_at_relu
    return spec_lbs, still_open


# ---------------------------------------------------------------------------
# Batched forward zono + spec-backward CROWN for input-split BaB.
# Each batch element is an INDEPENDENT input box [xl[b], xh[b]] processed
# through the SAME network graph. Centers / generators carry a leading
# batch dim B; intermediate tensors are (B, n, K) for generators and
# (B, n) for centers/bounds. After each ReLU we append `n_layer` new
# generator columns (one per neuron, with mu padded to zero for stable
# neurons) — keeps gen-count uniform across the batch so ops stay
# vectorized. For cersyve (200 ReLU total, input dim 4) this caps gens
# at ~260 → at batch=4096 the largest intermediate is ~850 MB (fits 10
# GB GPU).
#
# Not supported: conv, add-merge with non-trivially-shared gens, patches
# zonotope. Add-merge with shared_gens equal to fork K is supported by
# concat. The driver `_input_split_batched` falls back to the scalar
# path when the graph requires unsupported ops.
# ---------------------------------------------------------------------------


@torch.no_grad()
def _forward_zonotope_graph_batched(xl, xh, gg, device, dtype):
    """Batched forward zonotope on the graph.

    Args:
        xl, xh: (B, n_in) input bounds — one box per batch element.
        gg: gpu_graph dict.

    Returns:
        sb: dict layer_idx → (lo, hi) shape (B, n_layer) per ReLU.
        z_final: (c, G) tuple where c is (B, n_out) and G is (B, n_out, K).

    Raises:
        ValueError on unsupported op types (conv, add-merge with extras).
    """
    B, n_in = xl.shape
    c = (xl + xh) / 2
    radii = (xh - xl) / 2
    # G is (B, n_in, K) where K = number of input dims with any
    # non-zero radius across the batch. Dropping zero-radius columns
    # is exact (a zero column contributes nothing to any bound) and
    # for mscn dual where K=1 out of 308 it cuts gen-tensor memory by
    # 308×, enabling B≥32 instead of B=8 on 2048d.
    varying_mask = (radii.abs().max(dim=0).values > 0)  # (n_in,)
    K = int(varying_mask.sum())
    if K == n_in:
        G = torch.diag_embed(radii)
    else:
        # Build (B, n_in, K) with one column per varying dim.
        var_idx = varying_mask.nonzero(as_tuple=True)[0]  # (K,)
        G = torch.zeros(B, n_in, K, dtype=dtype, device=device)
        G[:, var_idx, torch.arange(K, device=device)] = radii[:, var_idx]
    state = {gg['input_name']: (c, G)}
    gen_count = {gg['input_name']: G.shape[2]}
    forks = gg['fork_points']
    sb = {}
    # Stash (lo, hi) for ops that feed a mul/div_bilinear input, so
    # backward CROWN can use proper McCormick envelopes instead of
    # point linearization.
    bilinear_input_names = set()
    for op_ in gg['ops']:
        if op_['type'] in ('mul_bilinear', 'div_bilinear'):
            for inp_ in op_['inputs']:
                bilinear_input_names.add(inp_)
    op_bounds = {}

    last_use = {}
    for i, op2 in enumerate(gg['ops']):
        for inp in op2['inputs']:
            last_use[inp] = i

    def _get(name):
        # `forks` means the value is consumed twice — clone (cheap on GPU)
        c_, G_ = state[name]
        if name in forks:
            return c_.clone(), G_.clone()
        return c_, G_

    for op_idx, op in enumerate(gg['ops']):
        name = op['name']
        t = op['type']

        if t == 'fc':
            c_in, G_in = _get(op['inputs'][0])
            W = op['W']  # (n_out, n_in_layer)
            bias = op['bias']  # (n_out,)
            in_shape_nd = op.get('in_shapes_nd', [None])[0]
            # ND batched MatMul (nn4sys mscn: input (3, 7), W=(128, 7)
            # → (3, 128)). Reshape (B, prod(prefix*n_in_last)) to
            # (B, *prefix, n_in_last), apply F.linear (broadcasts bias
            # over prefix), reshape back.
            if (in_shape_nd is not None and len(in_shape_nd) >= 2
                    and W.shape[1] == in_shape_nd[-1]
                    and c_in.shape[1] == int(np.prod(in_shape_nd))):
                prefix = in_shape_nd[:-1]
                n_in_last = in_shape_nd[-1]
                n_out_last = W.shape[0]
                K = G_in.shape[2]
                c_nd = c_in.reshape(B, *prefix, n_in_last)
                c_out_nd = F.linear(c_nd, W, bias)
                c_out = c_out_nd.reshape(B, -1)
                # Gens: (B, *prefix, n_in_last, K) → contract n_in_last
                # axis with W's input axis to get (B, *prefix, n_out, K).
                if K > 0:
                    G_nd = G_in.reshape(B, *prefix, n_in_last, K)
                    G_out_nd = torch.einsum(
                        '...ik,oi->...ok', G_nd, W)
                    G_out = G_out_nd.reshape(B, -1, K)
                else:
                    G_out = G_in.new_zeros(B, c_out.shape[1], 0)
                state[name] = (c_out, G_out)
            else:
                c_out = c_in @ W.T + bias  # (B, n_out)
                # G_out[b, o, k] = sum_i W[o, i] * G_in[b, i, k]
                G_out = torch.einsum('oi,bik->bok', W, G_in)
                state[name] = (c_out, G_out)

        elif t == 'relu':
            c_in, G_in = _get(op['inputs'][0])
            abs_sum = G_in.abs().sum(dim=2)  # (B, n)
            lo = c_in - abs_sum
            hi = c_in + abs_sum
            ust = (lo < 0) & (hi > 0)
            dead = hi <= 0
            lam = torch.where(ust, hi / (hi - lo),
                               torch.where(dead, torch.zeros_like(hi),
                                            torch.ones_like(hi)))  # (B, n)
            mu = torch.where(ust, -hi * lo / (2 * (hi - lo)),
                              torch.zeros_like(hi))  # (B, n)
            c_out = lam * c_in + mu
            G_scaled = G_in * lam.unsqueeze(-1)  # (B, n, K)
            # Compact gen append: only one new column per UNSTABLE neuron
            # (stable neurons have mu=0; full diag was 800MB+ on cGAN at
            # n=28800). Per-batch unstable counts may differ; pad with
            # zeros to max across batch.
            ust_cnt = ust.sum(dim=1)  # (B,)
            max_K = int(ust_cnt.max().item())
            if max_K == 0:
                G_out = G_scaled
            else:
                new_gens = torch.zeros(B, c_in.shape[1], max_K,
                                          dtype=dtype, device=device)
                # k-index within each batch's unstable list
                ust_rank = ust.long().cumsum(dim=1) - 1  # (B, n)
                b_idx = torch.arange(B, device=device).unsqueeze(-1).expand(
                    -1, c_in.shape[1])  # (B, n)
                r_idx = torch.arange(c_in.shape[1],
                                       device=device).unsqueeze(0).expand(
                    B, -1)  # (B, n)
                new_gens[b_idx[ust], r_idx[ust], ust_rank[ust]] = mu[ust]
                G_out = torch.cat([G_scaled, new_gens], dim=2)
            state[name] = (c_out, G_out)
            if 'layer_idx' in op:
                sb[op['layer_idx']] = (lo.clone(), hi.clone())

        elif t == 'add':
            if op.get('is_merge'):
                c_a, G_a = _get(op['inputs'][0])
                c_b, G_b = _get(op['inputs'][1])
                shared = _find_shared_gens_count(
                    op['inputs'][0], op['inputs'][1], gg, gen_count)
                K_a, K_b = G_a.shape[2], G_b.shape[2]
                assert 0 <= shared <= K_a and 0 <= shared <= K_b
                if K_b == shared:
                    # Fast path: mutate a's first `shared` cols.
                    G_a[:, :, :shared] = G_a[:, :, :shared] + G_b[:, :, :shared]
                    state[name] = (c_a + c_b, G_a)
                else:
                    K_out = K_a + K_b - shared
                    n = c_a.shape[1]
                    G_out = torch.empty(B, n, K_out, dtype=dtype, device=device)
                    G_out[:, :, :shared] = G_a[:, :, :shared] + G_b[:, :, :shared]
                    if K_a > shared:
                        G_out[:, :, shared:K_a] = G_a[:, :, shared:]
                    if K_b > shared:
                        G_out[:, :, K_a:] = G_b[:, :, shared:]
                    state[name] = (c_a + c_b, G_out)
            else:
                c_in, G_in = _get(op['inputs'][0])
                bias = op.get('bias')
                if bias is not None:
                    bt = torch.as_tensor(bias,
                                          dtype=dtype, device=device)
                    n = c_in.shape[1]
                    if bt.numel() == n:
                        c_in = c_in + bt.flatten()
                    else:
                        # ND broadcast (mscn: out (6, 128) + bias (128,))
                        out_shape_nd = op.get('out_shape_nd')
                        if (out_shape_nd is not None
                                and out_shape_nd[-1] == bt.numel()):
                            c_nd = c_in.reshape(B, *out_shape_nd)
                            c_in = (c_nd + bt).reshape(B, -1)
                        else:
                            raise ValueError(
                                f'batched add bias shape {bt.shape} '
                                f'incompatible (out_shape={out_shape_nd})')
                state[name] = (c_in, G_in)

        elif t == 'sub':
            c_in, G_in = _get(op['inputs'][0])
            bias = op.get('bias')
            if bias is not None:
                bt = torch.as_tensor(bias.flatten(),
                                      dtype=dtype, device=device)
                c_in = c_in - bt
            state[name] = (c_in, G_in)

        elif t == 'sub_bilinear':
            c_a, G_a = _get(op['inputs'][0])
            c_b, G_b = _get(op['inputs'][1])
            # Batched skip-style sub. Generators are batched (B, n, K).
            # Pad gen dims to max so they can be concat'd; subtract on
            # the prefix that's shared, append non-shared (with -G_b for
            # the b-side extras).
            ka = G_a.shape[2] if G_a.numel() > 0 else 0
            kb = G_b.shape[2] if G_b.numel() > 0 else 0
            shared = min(ka, kb)
            G_a_shared = G_a[:, :, :shared]
            G_b_shared = G_b[:, :, :shared]
            parts = [G_a_shared - G_b_shared]
            if ka > shared:
                parts.append(G_a[:, :, shared:])
            if kb > shared:
                parts.append(-G_b[:, :, shared:])
            G_out = torch.cat(parts, dim=2) if parts else \
                c_a.new_zeros(c_a.shape[0], c_a.shape[1], 0)
            state[name] = (c_a - c_b, G_out)

        elif t == 'reshape':
            state[name] = _get(op['inputs'][0])

        elif t in ('slice', 'gather'):
            c_in, G_in = _get(op['inputs'][0])
            flat_idx = op.get('flat_idx')
            if flat_idx is None:
                raise ValueError("slice op missing 'flat_idx' for batched fwd")
            idx_t = torch.as_tensor(flat_idx, dtype=torch.long, device=device)
            c_out = c_in.index_select(1, idx_t)
            G_out = G_in.index_select(1, idx_t)
            state[name] = (c_out, G_out)

        elif t == 'concat':
            cs_l, gs_l = [], []
            inputs = [_get(inp) for inp in op['inputs']]
            n_gens = max(G.shape[2] for _, G in inputs)
            for c_in, G_in in inputs:
                if G_in.shape[2] < n_gens:
                    pad = torch.zeros(B, G_in.shape[1],
                                       n_gens - G_in.shape[2],
                                       dtype=dtype, device=device)
                    G_in = torch.cat([G_in, pad], dim=2)
                cs_l.append(c_in); gs_l.append(G_in)
            state[name] = (torch.cat(cs_l, dim=1), torch.cat(gs_l, dim=1))

        elif t == 'conv_transpose':
            c_in, G_in = _get(op['inputs'][0])
            kernel = op['kernel']
            bias = op['bias']
            in_shape = op['in_shape']  # (C_in, H_in, W_in)
            stride = op['stride']
            padding = op['padding']
            output_padding = op['output_padding']
            n_in = c_in.shape[1]
            assert n_in == in_shape[0] * in_shape[1] * in_shape[2]
            # Center: (B, n_in) → (B, C_in, H, W) → conv_transpose → flatten
            c_4d = c_in.reshape(B, *in_shape)
            c_out_4d = F.conv_transpose2d(
                c_4d, kernel, bias=bias, stride=stride, padding=padding,
                output_padding=output_padding)
            c_out = c_out_4d.reshape(B, -1)
            # Generators: (B, n_in, K) → (B*K, C_in, H, W) per K → reshape
            K = G_in.shape[2]
            if K == 0:
                G_out = torch.zeros(B, c_out.shape[1], 0,
                                       dtype=dtype, device=device)
            else:
                # permute to (B, K, n_in) → (B*K, C, H, W)
                g_perm = G_in.permute(0, 2, 1).reshape(B * K, *in_shape)
                g_out = F.conv_transpose2d(
                    g_perm, kernel, bias=None, stride=stride, padding=padding,
                    output_padding=output_padding)
                n_out = c_out.shape[1]
                G_out = g_out.reshape(B, K, n_out).permute(0, 2, 1).contiguous()
            state[name] = (c_out, G_out)

        elif t in ('sigmoid', 'tanh'):
            c_in, G_in = _get(op['inputs'][0])
            abs_sum = G_in.abs().sum(dim=2)
            lo_pre = c_in - abs_sum
            hi_pre = c_in + abs_sum
            act = torch.sigmoid if t == 'sigmoid' else torch.tanh
            s_lo = act(lo_pre); s_hi = act(hi_pre)
            c_out = (s_lo + s_hi) / 2
            mu = (s_hi - s_lo) / 2
            if 'layer_idx' in op:
                sb[op['layer_idx']] = (lo_pre.clone(), hi_pre.clone())
            # Collapse old gens (no preserved correlation through nonlinearity).
            # Compact: only add gen columns for neurons with non-zero slack.
            G_scaled = torch.zeros(B, c_in.shape[1], G_in.shape[2],
                                      dtype=dtype, device=device)
            nonzero_mask = mu.abs() > 1e-9
            ust_cnt = nonzero_mask.sum(dim=1)
            max_K = int(ust_cnt.max().item())
            if max_K == 0:
                G_out = G_scaled  # (B, n, K_old) zeros
            else:
                new_gens = torch.zeros(B, c_in.shape[1], max_K,
                                          dtype=dtype, device=device)
                rank = nonzero_mask.long().cumsum(dim=1) - 1
                b_idx = torch.arange(B, device=device).unsqueeze(-1).expand(
                    -1, c_in.shape[1])
                r_idx = torch.arange(c_in.shape[1],
                                       device=device).unsqueeze(0).expand(
                    B, -1)
                new_gens[b_idx[nonzero_mask], r_idx[nonzero_mask],
                          rank[nonzero_mask]] = mu[nonzero_mask]
                G_out = torch.cat([G_scaled, new_gens], dim=2)
            state[name] = (c_out, G_out)

        elif t == 'upsample':
            c_in, G_in = _get(op['inputs'][0])
            in_shape = op['in_shape']
            sH, sW = op['scale']
            n_in_layer = in_shape[0] * in_shape[1] * in_shape[2]
            assert c_in.shape[1] == n_in_layer
            c_4d = c_in.reshape(B, *in_shape)
            c_out_4d = F.interpolate(c_4d, scale_factor=(sH, sW),
                                       mode='nearest')
            c_out = c_out_4d.reshape(B, -1)
            K = G_in.shape[2]
            if K == 0:
                G_out = torch.zeros(B, c_out.shape[1], 0,
                                       dtype=dtype, device=device)
            else:
                g_perm = G_in.permute(0, 2, 1).reshape(B * K, *in_shape)
                g_out = F.interpolate(g_perm, scale_factor=(sH, sW),
                                        mode='nearest')
                n_out_layer = c_out.shape[1]
                G_out = g_out.reshape(B, K, n_out_layer).permute(0, 2, 1).contiguous()
            state[name] = (c_out, G_out)

        elif t == 'conv':
            c_in, G_in = _get(op['inputs'][0])
            kernel = op['kernel']
            bias = op['bias']
            in_shape = op['in_shape']
            stride = op['stride']
            padding = op['padding']
            n_in_layer = in_shape[0] * in_shape[1] * in_shape[2]
            assert c_in.shape[1] == n_in_layer
            c_4d = c_in.reshape(B, *in_shape)
            c_out_4d = F.conv2d(c_4d, kernel, bias=bias,
                                  stride=stride, padding=padding)
            c_out = c_out_4d.reshape(B, -1)
            K = G_in.shape[2]
            if K == 0:
                G_out = torch.zeros(B, c_out.shape[1], 0,
                                       dtype=dtype, device=device)
            else:
                g_perm = G_in.permute(0, 2, 1).reshape(B * K, *in_shape)
                g_out = F.conv2d(g_perm, kernel, bias=None,
                                   stride=stride, padding=padding)
                n_out_layer = c_out.shape[1]
                G_out = g_out.reshape(B, K, n_out_layer).permute(0, 2, 1).contiguous()
            state[name] = (c_out, G_out)

        elif t == 'avg_pool':
            # avg_pool is linear: y = (1/k^2) * sum over window. Apply
            # F.avg_pool2d to center and each generator column. No bound
            # loss.
            c_in, G_in = _get(op['inputs'][0])
            in_shape = op['in_shape']
            kH, kW = op['kernel']
            sH, sW = op['stride']
            pH, pW = op['padding']
            n_in_layer = in_shape[0] * in_shape[1] * in_shape[2]
            assert c_in.shape[1] == n_in_layer
            c_4d = c_in.reshape(B, *in_shape)
            c_out_4d = F.avg_pool2d(c_4d, kernel_size=(kH, kW),
                                       stride=(sH, sW), padding=(pH, pW))
            c_out = c_out_4d.reshape(B, -1)
            K = G_in.shape[2]
            if K == 0:
                G_out = torch.zeros(B, c_out.shape[1], 0,
                                       dtype=dtype, device=device)
            else:
                g_perm = G_in.permute(0, 2, 1).reshape(B * K, *in_shape)
                g_out = F.avg_pool2d(g_perm, kernel_size=(kH, kW),
                                       stride=(sH, sW), padding=(pH, pW))
                n_out_layer = c_out.shape[1]
                G_out = g_out.reshape(B, K, n_out_layer).permute(
                    0, 2, 1).contiguous()
            state[name] = (c_out, G_out)

        elif t == 'max_pool':
            # max_pool is nonlinear. Box approximation: per-cell bounds
            # are lo_out=max(lo_in over window), hi_out=max(hi_in over
            # window). Collapse correlations into a new gen column per
            # cell with non-zero slack. Sound but loose; suffices for
            # cgan small_transformer's attention (4 MaxPool ops).
            c_in, G_in = _get(op['inputs'][0])
            in_shape = op['in_shape']
            kH, kW = op['kernel']
            sH, sW = op['stride']
            pH, pW = op['padding']
            n_in_layer = in_shape[0] * in_shape[1] * in_shape[2]
            abs_sum = G_in.abs().sum(dim=2)
            lo_pre = (c_in - abs_sum).reshape(B, *in_shape)
            hi_pre = (c_in + abs_sum).reshape(B, *in_shape)
            lo_out = F.max_pool2d(lo_pre, (kH, kW), stride=(sH, sW),
                                     padding=(pH, pW))
            hi_out = F.max_pool2d(hi_pre, (kH, kW), stride=(sH, sW),
                                     padding=(pH, pW))
            n_out_layer = lo_out.shape[1] * lo_out.shape[2] * lo_out.shape[3]
            lo_flat = lo_out.reshape(B, n_out_layer)
            hi_flat = hi_out.reshape(B, n_out_layer)
            c_out = (lo_flat + hi_flat) / 2
            mu = (hi_flat - lo_flat) / 2
            # Compact gen append (mirrors sigmoid/tanh).
            nonzero_mask = mu.abs() > 1e-9
            ust_cnt = nonzero_mask.sum(dim=1)
            max_K = int(ust_cnt.max().item())
            G_zeros = torch.zeros(B, n_out_layer, G_in.shape[2],
                                     dtype=dtype, device=device)
            if max_K == 0:
                G_out = G_zeros
            else:
                new_gens = torch.zeros(B, n_out_layer, max_K,
                                          dtype=dtype, device=device)
                rank = nonzero_mask.long().cumsum(dim=1) - 1
                b_idx = torch.arange(B, device=device).unsqueeze(-1).expand(
                    -1, n_out_layer)
                r_idx = torch.arange(n_out_layer,
                                       device=device).unsqueeze(0).expand(B, -1)
                new_gens[b_idx[nonzero_mask], r_idx[nonzero_mask],
                          rank[nonzero_mask]] = mu[nonzero_mask]
                G_out = torch.cat([G_zeros, new_gens], dim=2)
            state[name] = (c_out, G_out)
            # Record pre-act bounds for backward (used by box CROWN).
            sb[op['name'] + '__maxpool_box'] = (lo_flat.clone(),
                                                   hi_flat.clone())

        elif t == 'mul':
            # Constant scalar/per-channel multiply: y = scale * x.
            c_in, G_in = _get(op['inputs'][0])
            scale_t = op.get('scale')
            if scale_t is None:
                raise ValueError("mul op missing 'scale' for forward zono")
            if isinstance(scale_t, np.ndarray):
                scale_t = torch.from_numpy(scale_t).to(device=device, dtype=dtype)
            elif not isinstance(scale_t, torch.Tensor):
                scale_t = torch.tensor(scale_t, dtype=dtype, device=device)
            else:
                scale_t = scale_t.to(device=device, dtype=dtype)
            sflat = scale_t.flatten()
            # Broadcast: per-channel or scalar. Per-channel must match
            # spatial layout; assume scalar or matches c_in.shape[1].
            if sflat.numel() == 1:
                c_out = c_in * sflat
                G_out = G_in * sflat
            elif sflat.numel() == c_in.shape[1]:
                c_out = c_in * sflat.unsqueeze(0)
                G_out = G_in * sflat.unsqueeze(0).unsqueeze(-1)
            else:
                # Per-channel broadcast over spatial: use op's input
                # shape to expand.
                in_shape = op.get('in_shapes_nd', [None])[0]
                if in_shape is None or len(in_shape) != 3:
                    raise ValueError(
                        f'mul: scale shape {sflat.shape} incompatible with '
                        f'input ({c_in.shape[1]}); no spatial shape known')
                C, H, W = in_shape
                assert sflat.numel() == C, (
                    f'mul per-channel scale {sflat.numel()} != C={C}')
                scale_4d = sflat.view(1, C, 1, 1).expand(1, C, H, W).reshape(1, -1)
                c_out = c_in * scale_4d
                G_out = G_in * scale_4d.unsqueeze(-1)
            state[name] = (c_out, G_out)

        elif t in ('exp', 'reciprocal'):
            # Batched 1-D convex parallelogram (same construction as the
            # unbatched forward): y = k*x + c_off + r_off*e_new with the
            # chord slope k on [l, u]; offsets bracket the gap
            # g(x) = f(x) - k*x between its endpoint max and tangency min.
            c_in, G_in = _get(op['inputs'][0])
            abs_in = G_in.abs().sum(dim=2)
            lo_in = c_in - abs_in; hi_in = c_in + abs_in
            if t == 'reciprocal':
                if float(lo_in.min()) <= 0:
                    raise ValueError(
                        f'reciprocal {name!r}: input lower bound <= 0')
                f_l = 1.0 / lo_in; f_u = 1.0 / hi_in
            else:
                f_l = torch.exp(lo_in); f_u = torch.exp(hi_in)
            w_in = (hi_in - lo_in).clamp(min=1e-12)
            k = (f_u - f_l) / w_in
            if t == 'exp':
                xs = torch.log(k.clamp(min=1e-300))
                fxs_fn = torch.exp
            else:
                xs = torch.sqrt(1.0 / (-k).clamp(min=1e-300))
                fxs_fn = torch.reciprocal
            xs = torch.minimum(torch.maximum(xs, lo_in), hi_in)
            fxs = fxs_fn(xs)
            g_l_end = f_l - k * lo_in
            g_u_end = f_u - k * hi_in
            g_min = fxs - k * xs
            g_hi = torch.maximum(torch.maximum(g_l_end, g_u_end), g_min)
            g_lo = torch.minimum(torch.minimum(g_l_end, g_u_end), g_min)
            # positivity guard (see unbatched handler)
            _edge_lo = (torch.minimum(k * lo_in, k * hi_in) + g_lo)
            _boxm = _edge_lo <= 0
            _fmin = torch.minimum(f_l, f_u)
            _fmax = torch.maximum(f_l, f_u)
            k = torch.where(_boxm, torch.zeros_like(k), k)
            c_off = torch.where(_boxm, (_fmin + _fmax) / 2,
                                (g_hi + g_lo) / 2)
            r_off = torch.where(_boxm, (_fmax - _fmin) / 2,
                                (g_hi - g_lo) / 2)
            c_out = k * c_in + c_off
            G_out = torch.cat(
                [G_in * k.unsqueeze(-1), torch.diag_embed(r_off)], dim=2)
            state[name] = (c_out, G_out)

        elif t in ('mul_bilinear', 'matmul_bilinear', 'softmax'):
            # Nonlinear / variable-x-variable ops: collapse to box.
            # Forward computes interval bounds and emits a single new
            # gen column per non-zero-slack cell. Center = midpoint.
            c_a, G_a = _get(op['inputs'][0])
            abs_a = G_a.abs().sum(dim=2)
            lo_a = c_a - abs_a; hi_a = c_a + abs_a
            if t == 'mul_bilinear':
                c_b, G_b = _get(op['inputs'][1])
                # Fast path: if one side has zero generator radius
                # everywhere (point per-disjunct, e.g. nn4sys mscn mask),
                # the result is linear in the other side. Skip the box
                # collapse, preserve generator correlation.
                a_pt = (G_a.numel() == 0
                         or bool(G_a.abs().max() < 1e-12))
                b_pt = (G_b.numel() == 0
                         or bool(G_b.abs().max() < 1e-12))
                if a_pt or b_pt:
                    sh_in = op.get('in_shapes_nd', [None, None])
                    sh_out = op.get('out_shape_nd')
                    K = max(G_a.shape[2] if G_a.numel() > 0 else 0,
                             G_b.shape[2] if G_b.numel() > 0 else 0)
                    if sh_in[0] is not None and sh_in[1] is not None:
                        a_nd = c_a.reshape(B, *sh_in[0])
                        b_nd = c_b.reshape(B, *sh_in[1])
                    else:
                        a_nd = c_a; b_nd = c_b
                    c_out_nd = a_nd * b_nd
                    c_out_local = c_out_nd.reshape(B, -1)
                    if K == 0:
                        state[name] = (
                            c_out_local,
                            c_out_local.new_zeros(B, c_out_local.shape[1], 0))
                    else:
                        # Scale gens of the varying side by the point
                        # side's center, broadcast to out shape.
                        if b_pt and not a_pt:
                            G_a_nd = G_a.reshape(B, *sh_in[0], K)
                            G_out_nd = (G_a_nd * b_nd.unsqueeze(-1))
                        else:
                            G_b_nd = G_b.reshape(B, *sh_in[1], K)
                            G_out_nd = (G_b_nd * a_nd.unsqueeze(-1))
                        if sh_out is not None:
                            G_out_nd = G_out_nd.expand(B, *sh_out, K)
                        state[name] = (
                            c_out_local,
                            G_out_nd.contiguous().reshape(B, -1, K))
                    gen_count[name] = state[name][1].shape[2]
                    for inp in op['inputs']:
                        if last_use.get(inp) == op_idx and inp in state:
                            del state[inp]
                    continue
                # Slow path: both sides varying — box collapse with
                # broadcast support via ND shapes.
                abs_b = G_b.abs().sum(dim=2)
                lo_b = c_b - abs_b; hi_b = c_b + abs_b
                sh_in = op.get('in_shapes_nd', [None, None])
                sh_out = op.get('out_shape_nd')
                if (sh_in[0] is not None and sh_in[1] is not None
                        and sh_in[0] != sh_in[1]):
                    lo_a_nd = lo_a.reshape(B, *sh_in[0])
                    hi_a_nd = hi_a.reshape(B, *sh_in[0])
                    lo_b_nd = lo_b.reshape(B, *sh_in[1])
                    hi_b_nd = hi_b.reshape(B, *sh_in[1])
                    corners = torch.stack(
                        [lo_a_nd * lo_b_nd, lo_a_nd * hi_b_nd,
                          hi_a_nd * lo_b_nd, hi_a_nd * hi_b_nd], dim=-1)
                    lo_out = corners.min(dim=-1).values.reshape(B, -1)
                    hi_out = corners.max(dim=-1).values.reshape(B, -1)
                else:
                    corners = torch.stack(
                        [lo_a * lo_b, lo_a * hi_b,
                          hi_a * lo_b, hi_a * hi_b], dim=-1)
                    lo_out = corners.min(dim=-1).values
                    hi_out = corners.max(dim=-1).values
            elif t == 'matmul_bilinear':
                # (B, .., M, K) @ (B, .., K, N) -> (B, .., M, N).
                # Reshape to N-D via op['in_shapes_nd'] and ['out_shape_nd'].
                c_b, G_b = _get(op['inputs'][1])
                abs_b = G_b.abs().sum(dim=2)
                lo_b = c_b - abs_b; hi_b = c_b + abs_b
                sh_a = op['in_shapes_nd'][0]
                sh_b = op['in_shapes_nd'][1]
                sh_o = op['out_shape_nd']
                lo_a_nd = lo_a.reshape(B, *sh_a)
                hi_a_nd = hi_a.reshape(B, *sh_a)
                lo_b_nd = lo_b.reshape(B, *sh_b)
                hi_b_nd = hi_b.reshape(B, *sh_b)
                # For y = a @ b with each a_ij in [lo_a_ij, hi_a_ij] and
                # b_jk in [lo_b_jk, hi_b_jk], element y_ik = sum_j a_ij * b_jk.
                # Bound sum_j of min/max over corners.
                # Per-pair corners:
                #   p_jk^lo = min(lo_a_ij*lo_b_jk, lo_a_ij*hi_b_jk,
                #                  hi_a_ij*lo_b_jk, hi_a_ij*hi_b_jk)
                # Sum over j gives sound lower bound. (Conservative but
                # straightforward; auto_LiRPA does tighter via McCormick.)
                # Implementation via four matmuls + min/max.
                pp = lo_a_nd.unsqueeze(-1) * lo_b_nd.unsqueeze(-3)  # (B, ..., M, K, N)
                pn = lo_a_nd.unsqueeze(-1) * hi_b_nd.unsqueeze(-3)
                np_ = hi_a_nd.unsqueeze(-1) * lo_b_nd.unsqueeze(-3)
                nn = hi_a_nd.unsqueeze(-1) * hi_b_nd.unsqueeze(-3)
                cmin = torch.minimum(torch.minimum(pp, pn),
                                       torch.minimum(np_, nn))  # (B,...,M,K,N)
                cmax = torch.maximum(torch.maximum(pp, pn),
                                       torch.maximum(np_, nn))
                lo_out_nd = cmin.sum(dim=-2)  # (B, ..., M, N)
                hi_out_nd = cmax.sum(dim=-2)
                n_out_layer = 1
                for d in sh_o:
                    n_out_layer *= d
                lo_out = lo_out_nd.reshape(B, n_out_layer)
                hi_out = hi_out_nd.reshape(B, n_out_layer)
            else:  # softmax
                # auto_LiRPA's interval bound:
                #   lower = exp(lo - shift) / (sum exp(hi - shift)
                #                              - exp(hi - shift) + exp(lo - shift) + eps)
                #   upper = exp(hi - shift) / (sum exp(lo - shift)
                #                              - exp(lo - shift) + exp(hi - shift) + eps)
                # where shift = max(hi) per row.
                axis = int(op.get('axis', -1))
                # Reshape to (B, ..., n_axis) per op's in_shape.
                sh_a = op['in_shapes_nd'][0]
                if sh_a is None:
                    raise ValueError('softmax requires in_shape_nd')
                # Normalize axis to the reshaped (B, *sh_a) tensor.
                ax = axis if axis >= 0 else axis + 1 + len(sh_a)
                lo_nd = lo_a.reshape(B, *sh_a)
                hi_nd = hi_a.reshape(B, *sh_a)
                shift = hi_nd.max(dim=ax, keepdim=True).values
                exp_lo = torch.exp(lo_nd - shift)
                exp_hi = torch.exp(hi_nd - shift)
                sum_hi = exp_hi.sum(dim=ax, keepdim=True)
                sum_lo = exp_lo.sum(dim=ax, keepdim=True)
                eps = 1e-12
                lo_out_nd = exp_lo / (sum_hi - exp_hi + exp_lo + eps)
                hi_out_nd = exp_hi / (sum_lo - exp_lo + exp_hi + eps)
                lo_out = lo_out_nd.reshape(B, -1)
                hi_out = hi_out_nd.reshape(B, -1)
            n_out_layer = lo_out.shape[1]
            c_out = (lo_out + hi_out) / 2
            mu = (hi_out - lo_out) / 2
            nonzero_mask = mu.abs() > 1e-9
            ust_cnt = nonzero_mask.sum(dim=1)
            max_K = int(ust_cnt.max().item())
            G_zeros = torch.zeros(B, n_out_layer, G_a.shape[2],
                                     dtype=dtype, device=device)
            if max_K == 0:
                G_out = G_zeros
            else:
                new_gens = torch.zeros(B, n_out_layer, max_K,
                                          dtype=dtype, device=device)
                rank = nonzero_mask.long().cumsum(dim=1) - 1
                b_idx = torch.arange(B, device=device).unsqueeze(-1).expand(
                    -1, n_out_layer)
                r_idx = torch.arange(n_out_layer,
                                       device=device).unsqueeze(0).expand(B, -1)
                new_gens[b_idx[nonzero_mask], r_idx[nonzero_mask],
                          rank[nonzero_mask]] = mu[nonzero_mask]
                G_out = torch.cat([G_zeros, new_gens], dim=2)
            state[name] = (c_out, G_out)
            sb[op['name'] + f'__{t}_box'] = (lo_out.clone(), hi_out.clone())


        elif t == 'squeeze':
            # Reshape-only: data unchanged.
            c_in, G_in = _get(op['inputs'][0])
            state[name] = (c_in, G_in)

        elif t == 'reduce_sum':
            # Linear sum along given axes. mscn uses axis=1 with
            # keepdims=0 to sum (3, 7) → (3,) for masked-mean numerator.
            c_in, G_in = _get(op['inputs'][0])
            in_shape_nd = op.get('in_shapes_nd', [None])[0]
            axes = op.get('axes', ())
            keep = op.get('keepdims', False)
            K = G_in.shape[2]
            c_nd = c_in.reshape(B, *in_shape_nd)
            # gg axes are relative to stripped-batch shape — add 1 to
            # skip the batch dim in the batched tensor.
            for ax in sorted(axes, reverse=True):
                c_nd = c_nd.sum(dim=ax + 1, keepdim=bool(keep))
            c_out = c_nd.reshape(B, -1)
            if K > 0:
                G_nd = G_in.reshape(B, *in_shape_nd, K)
                for ax in sorted(axes, reverse=True):
                    G_nd = G_nd.sum(dim=ax + 1, keepdim=bool(keep))
                G_out = G_nd.reshape(B, -1, K)
            else:
                G_out = G_in.new_zeros(B, c_out.shape[1], 0)
            state[name] = (c_out, G_out)

        elif t == 'mul_bilinear':
            # Element-wise Mul. mscn: Mul(features, mask) where mask is
            # point per-disjunct (zero radius). Helper raises when both
            # sides vary; ND broadcast supported.
            from .zonotope import _torch_zono_mul_bilinear
            c_a, G_a = _get(op['inputs'][0])
            c_b, G_b = _get(op['inputs'][1])
            sh = op.get('in_shapes_nd', [None, None])
            sh_out = op.get('out_shape_nd')
            # Process per batch element (broadcasting differs per side).
            # Most mscn cases: G_b is zero everywhere (mask is point per
            # subbox). Use the helper's check.
            K = max(G_a.shape[2] if G_a.numel() > 0 else 0,
                     G_b.shape[2] if G_b.numel() > 0 else 0)
            a_pt = (G_a.numel() == 0 or bool(G_a.abs().max() < 1e-12))
            b_pt = (G_b.numel() == 0 or bool(G_b.abs().max() < 1e-12))
            if not (a_pt or b_pt):
                raise NotImplementedError(
                    'batched mul_bilinear: both sides varying')
            # Reshape to ND and broadcast.
            if sh[0] is not None and sh[1] is not None:
                a_nd = c_a.reshape(B, *sh[0])
                b_nd = c_b.reshape(B, *sh[1])
                c_out_nd = a_nd * b_nd
                c_out = c_out_nd.reshape(B, -1)
                if K > 0 and b_pt:
                    G_a_nd = G_a.reshape(B, *sh[0], K)
                    G_out_nd = (G_a_nd * b_nd.unsqueeze(-1)).expand(
                        B, *sh_out, K)
                    G_out = G_out_nd.contiguous().reshape(B, -1, K)
                elif K > 0:
                    G_b_nd = G_b.reshape(B, *sh[1], K)
                    G_out_nd = (G_b_nd * a_nd.unsqueeze(-1)).expand(
                        B, *sh_out, K)
                    G_out = G_out_nd.contiguous().reshape(B, -1, K)
                else:
                    G_out = G_a.new_zeros(B, c_out.shape[1], 0)
            else:
                c_out = c_a * c_b
                if b_pt:
                    G_out = G_a * c_b.unsqueeze(-1) if K > 0 else G_a
                else:
                    G_out = G_b * c_a.unsqueeze(-1) if K > 0 else G_b
            state[name] = (c_out, G_out)

        elif t == 'div_bilinear':
            # Element-wise Div. Point denom → exact path. Non-point
            # denom → box-decorrelated fallback (one new gen per output
            # element) keyed off `_div_decoupled` so the backward
            # uses the matching sound linearisation.
            c_a, G_a = _get(op['inputs'][0])
            c_b, G_b = _get(op['inputs'][1])
            sh = op.get('in_shapes_nd', [None, None])
            sh_out = op.get('out_shape_nd')
            b_pt = (G_b.numel() == 0 or bool(G_b.abs().max() < 1e-12))
            K = G_a.shape[2] if G_a.numel() > 0 else 0
            op['_div_decoupled'] = not b_pt
            if not b_pt:
                # Compute per-batch-element box bounds; stash them so
                # the backward can recompute its linearisation. Note
                # these are PER-BATCH-ELEMENT tensors (B, n_a) / (B, n_b).
                K_a = G_a.shape[2] if G_a.numel() > 0 else 0
                K_b = G_b.shape[2] if G_b.numel() > 0 else 0
                rad_a = G_a.abs().sum(dim=2) if K_a > 0 else torch.zeros_like(c_a)
                rad_b = G_b.abs().sum(dim=2)
                a_lo_b = c_a - rad_a
                a_hi_b = c_a + rad_a
                b_lo_b = c_b - rad_b
                b_hi_b = c_b + rad_b
                op['_div_a_lo'] = a_lo_b.detach()
                op['_div_a_hi'] = a_hi_b.detach()
                op['_div_b_lo'] = b_lo_b.detach()
                op['_div_b_hi'] = b_hi_b.detach()
                if not bool((b_lo_b > 0).all()):
                    raise ZeroDivisionError(
                        'batched div_bilinear: b not sign-stable > 0')
                # Scalar-b path: shared-gen 1/b chord-tangent +
                # bilinear product. Preserves a-correlations + a↔b
                # correlation via shared eps cols. Tighter downstream
                # than the decorrelated 4-corner box.
                is_scalar_b = (c_b.shape[1] == 1)
                if is_scalar_b:
                    # (B, 1) tensors.
                    c_b_s = c_b[:, :1]
                    rad_b_s = rad_b[:, :1]
                    b_lo_s = b_lo_b[:, :1]
                    b_hi_s = b_hi_b[:, :1]
                    f_lo = 1.0 / b_lo_s
                    f_hi = 1.0 / b_hi_s
                    diff_b = (b_hi_s - b_lo_s).clamp(min=1e-30)
                    lam_s = (f_hi - f_lo) / diff_b  # (B, 1)
                    bstar_mag = 1.0 / lam_s.abs().clamp(min=1e-30).sqrt()
                    bstar = torch.where(b_hi_s < 0, -bstar_mag, bstar_mag)
                    bstar = torch.maximum(torch.minimum(bstar, b_hi_s), b_lo_s)
                    f_star = 1.0 / bstar
                    chord_int = f_lo - lam_s * b_lo_s
                    tan_int = f_star - lam_s * bstar
                    mu_s = (chord_int + tan_int) / 2
                    gamma_s = (chord_int - tan_int).abs() / 2  # (B, 1)
                    c_v = lam_s * c_b_s + mu_s             # (B, 1)
                    # Pad K_a and K_b to common K.
                    K_a = G_a.shape[2] if G_a.numel() > 0 else 0
                    K_b = G_b.shape[2] if G_b.numel() > 0 else 0
                    K_max = max(K_a, K_b)
                    if K_a < K_max:
                        G_a = torch.cat([G_a, G_a.new_zeros(
                            B, G_a.shape[1] if G_a.numel() > 0 else c_a.shape[1],
                            K_max - K_a)], dim=2)
                    if K_b < K_max:
                        G_b = torch.cat([G_b, G_b.new_zeros(
                            B, 1, K_max - K_b)], dim=2)
                    # c_out = c_a · c_v (broadcast scalar over n).
                    c_out = c_a * c_v  # (B, n)
                    n_y = c_a.shape[1]
                    if K_max > 0:
                        # Shared linear gens: c_a · lam · g_b  +  c_v · g_a.
                        # c_a:(B,n,1) · lam:(B,1,1) · g_b:(B,1,K)  + c_v:(B,1,1) · g_a:(B,n,K)
                        g_y_lin = (c_a.unsqueeze(-1)
                                   * (lam_s.unsqueeze(-1) * G_b)
                                   + c_v.unsqueeze(-1) * G_a)  # (B, n, K_max)
                    else:
                        g_y_lin = G_a.new_zeros(B, n_y, 0)
                    # 1 new shared eps col for gamma.
                    g_y_slack = (c_a * gamma_s).unsqueeze(-1)  # (B, n, 1)
                    # n diag cols for bilinear remainder.
                    quad_mag = rad_a * (
                        lam_s.abs() * rad_b_s + gamma_s)  # (B, n)
                    g_y_quad = torch.zeros(
                        B, n_y, n_y, dtype=dtype, device=device)
                    idx = torch.arange(n_y, device=device)
                    g_y_quad[:, idx, idx] = quad_mag
                    G_new = torch.cat([g_y_lin, g_y_slack, g_y_quad], dim=2)
                    # Softmax pattern detection + sound [0, 1] clamp.
                    # See `_torch_zono_div_scalar_b` for rationale.
                    a_lo_b_v = c_a - rad_a   # (B, n_y)
                    is_softmax = (
                        bool((a_lo_b_v >= -1e-9).all())
                        and bool(((c_b_s.reshape(B) -
                                    c_a.sum(dim=1)).abs() <
                                   1e-6 * c_b_s.reshape(B).abs().clamp(min=1.0)
                                  ).all()))
                    if is_softmax and K_max > 0:
                        # g_b shape (B, 1, K), g_a (B, n, K). sum gens.
                        g_a_sum_b = G_a.sum(dim=1, keepdim=True)  # (B, 1, K)
                        if bool(((G_b - g_a_sum_b).abs().max(dim=2).values.max(dim=1).values
                                 < 1e-6 * G_b.abs().amax(dim=(1,2)).clamp(min=1.0)).all()):
                            # Apply per-element [0, 1] clamp.
                            y_lo_enc = c_out - G_new.abs().sum(dim=2)
                            y_hi_enc = c_out + G_new.abs().sum(dim=2)
                            y_lo_c = torch.maximum(y_lo_enc,
                                                    torch.zeros_like(c_out))
                            y_hi_c = torch.minimum(y_hi_enc,
                                                    torch.ones_like(c_out))
                            needs = ((y_lo_enc < y_lo_c - 1e-12)
                                     | (y_hi_enc > y_hi_c + 1e-12))
                            if bool(needs.any()):
                                tight_mask = needs  # (B, n_y)
                                new_rad = (y_hi_c - y_lo_c) / 2
                                new_center = (y_lo_c + y_hi_c) / 2
                                # Zero out existing gen rows for tight elements.
                                G_new_keep = G_new.clone()
                                # Mask shape: (B, n_y, 1)
                                G_new_keep = torch.where(
                                    tight_mask.unsqueeze(-1),
                                    torch.zeros_like(G_new_keep),
                                    G_new_keep)
                                # Replace center for tight elements.
                                c_out = torch.where(tight_mask, new_center, c_out)
                                # New diag gens of new_rad for tight elements.
                                new_diag = torch.zeros(B, n_y, n_y,
                                                        dtype=dtype, device=device)
                                idx_t = torch.arange(n_y, device=device)
                                diag_mag = torch.where(tight_mask, new_rad,
                                                        torch.zeros_like(new_rad))
                                new_diag[:, idx_t, idx_t] = diag_mag
                                G_new = torch.cat([G_new_keep, new_diag], dim=2)
                    state[name] = (c_out, G_new)
                    gen_count[name] = G_new.shape[2]
                    # Free inputs early.
                    for inp in op['inputs']:
                        if last_use.get(inp) == op_idx and inp in state:
                            del state[inp]
                    continue
                # Vector-b fallback: 4-corner decorrelated.
                if sh[0] is not None and sh[1] is not None:
                    a_lo_nd = a_lo_b.reshape(B, *sh[0])
                    a_hi_nd = a_hi_b.reshape(B, *sh[0])
                    b_lo_nd = b_lo_b.reshape(B, *sh[1])
                    b_hi_nd = b_hi_b.reshape(B, *sh[1])
                    inv_lo = 1.0 / b_hi_nd
                    inv_hi = 1.0 / b_lo_nd
                    corners = torch.stack([
                        a_lo_nd * inv_lo, a_lo_nd * inv_hi,
                        a_hi_nd * inv_lo, a_hi_nd * inv_hi])
                    out_lo_nd = corners.min(dim=0).values
                    out_hi_nd = corners.max(dim=0).values
                    c_out = ((out_lo_nd + out_hi_nd) / 2).reshape(B, -1)
                    rad_out = ((out_hi_nd - out_lo_nd) / 2).reshape(B, -1)
                else:
                    inv_lo = 1.0 / b_hi_b
                    inv_hi = 1.0 / b_lo_b
                    corners = torch.stack([
                        c_a * inv_lo, c_a * inv_hi,  # placeholder
                        c_a * inv_lo, c_a * inv_hi])
                    out_lo_flat = (a_lo_b / b_hi_b).minimum(a_lo_b / b_lo_b
                        ).minimum(a_hi_b / b_hi_b).minimum(a_hi_b / b_lo_b)
                    out_hi_flat = (a_lo_b / b_hi_b).maximum(a_lo_b / b_lo_b
                        ).maximum(a_hi_b / b_hi_b).maximum(a_hi_b / b_lo_b)
                    c_out = (out_lo_flat + out_hi_flat) / 2
                    rad_out = (out_hi_flat - out_lo_flat) / 2
                # New decorrelated gen column per output element per
                # batch (diagonal). Total K_new = c_out.shape[1].
                n_out_per_b = c_out.shape[1]
                G_new = torch.zeros(B, n_out_per_b, n_out_per_b,
                                      dtype=dtype, device=device)
                idx = torch.arange(n_out_per_b, device=device)
                G_new[:, idx, idx] = rad_out
                state[name] = (c_out, G_new)
                gen_count[name] = G_new.shape[2]
                # Free inputs early.
                for inp in op['inputs']:
                    if last_use.get(inp) == op_idx and inp in state:
                        del state[inp]
                continue
            if bool((c_b == 0).any()):
                raise ZeroDivisionError(
                    'batched div_bilinear: denominator has zero element')
            if sh[0] is not None and sh[1] is not None:
                a_nd = c_a.reshape(B, *sh[0])
                b_nd = c_b.reshape(B, *sh[1])
                inv_b = b_nd.reciprocal()
                c_out = (a_nd * inv_b).reshape(B, -1)
                if K > 0:
                    G_a_nd = G_a.reshape(B, *sh[0], K)
                    G_out_nd = (G_a_nd * inv_b.unsqueeze(-1)).expand(
                        B, *sh_out, K)
                    G_out = G_out_nd.contiguous().reshape(B, -1, K)
                else:
                    G_out = G_a.new_zeros(B, c_out.shape[1], 0)
            else:
                inv_b = c_b.reciprocal()
                c_out = c_a * inv_b
                G_out = (G_a * inv_b.unsqueeze(-1)) if K > 0 else G_a
            state[name] = (c_out, G_out)

        elif t == 'pow':
            # x^p batched. Stash batched pre-pow box bounds for the
            # backward to use. Per-batch independent processing for the
            # gen encoding (new gen count may differ per box).
            from .zonotope import _torch_zono_pow_int
            c_in, G_in = _get(op['inputs'][0])
            exp = op.get('exponent', 2.0)
            assert float(int(exp)) == float(exp), (
                f'batched pow: integer exponent required, got {exp}')
            p = int(exp)
            # Batched pre-pow bounds: (B, n_in).
            K_in = G_in.shape[2] if G_in.numel() > 0 else 0
            rad_in = G_in.abs().sum(dim=2) if K_in > 0 else torch.zeros_like(c_in)
            op['_pow_in_lo'] = (c_in - rad_in).detach()
            op['_pow_in_hi'] = (c_in + rad_in).detach()
            op['_pow_relaxation'] = 'chord'
            outs = []
            max_new_K = 0
            for bi in range(B):
                c_b = c_in[bi]
                g_b = G_in[bi] if G_in.numel() > 0 else c_b.new_zeros(c_b.numel(), 0)
                c_out_b, g_out_b = _torch_zono_pow_int(c_b, g_b, p)
                outs.append((c_out_b, g_out_b))
                max_new_K = max(max_new_K, g_out_b.shape[1])
            n_out = outs[0][0].numel()
            c_out = torch.stack([o[0] for o in outs], dim=0).reshape(B, -1)
            G_out = c_in.new_zeros(B, n_out, max_new_K)
            for bi, (_, g) in enumerate(outs):
                G_out[bi, :, :g.shape[1]] = g
            state[name] = (c_out, G_out)

        else:
            raise ValueError(
                f'batched zono forward: unknown op type {t!r}')

        gen_count[name] = state[name][1].shape[2]
        if name in bilinear_input_names:
            c_o, G_o = state[name]
            rad_o = G_o.abs().sum(dim=-1)
            op_bounds[name] = (c_o - rad_o, c_o + rad_o)
        import os as _os_dwl
        if _os_dwl.environ.get('DUMP_WIDTHS_LEAF', '') != '':
            _li = int(_os_dwl.environ['DUMP_WIDTHS_LEAF'])
            _c, _G = state[name]
            if _li < _c.shape[0]:
                _lo = (_c[_li] - _G[_li].abs().sum(dim=-1)).flatten()
                _hi = (_c[_li] + _G[_li].abs().sum(dim=-1)).flatten()
                _w = (_hi - _lo)
                print(f"[ZONO] leaf{_li} op {name} ({t}) shape={list(_c[_li].shape)} K={_G.shape[2]} w_max={_w.max().item():.4f} w_mean={_w.mean().item():.4f}")
        for inp in op['inputs']:
            if last_use.get(inp) == op_idx and inp in state:
                del state[inp]

    last_name = gg['ops'][-1]['name']
    _forward_zonotope_graph_batched.last_bilinear_op_bounds = op_bounds
    return sb, state[last_name]


def _crown_intermediate_batched(gg, xl, xh, device, dtype):
    """Backward-CROWN intermediate pre-ReLU bounds (min-area slopes, no α).

    This is what AB-CROWN uses for ACAS Xu (`bound_prop_method: crown`) and it
    is ~2x TIGHTER and CHEAPER than the forward zonotope (acasxu 3_3 prop_2 root:
    forward-zono -1597 / 226 ms vs CROWN -722 / 8 ms). For each ReLU layer L,
    seed the spec backward at L's pre-activation op with [+I, -I] (lower + upper
    of every neuron), propagate to the input through the EARLIER layers'
    min-area ReLU relaxations (which use the bounds tightened so far), and
    intersect with the running forward-zono bound (max of los / min of his — both
    are sound over-approximations, so the intersection stays sound). One layer
    sweep converges for these nets. Drop-in replacement for
    `_forward_zonotope_graph_batched`'s intermediate bounds in the input-split
    BaB; the spec backward is then much tighter -> far fewer leaves.

    (A mutual zono<->CROWN tightening was tried — feeding each layer's
    intersection back into the next forward-zono pass — and measured to add only
    ~3% margin, sub-threshold to close a leaf one bisection earlier; a net
    end-to-end loss. Removed. See scratch/acasxu_p2_33/plan.md explore11-14.)

    Returns `tight = {layer_idx: (lo, hi)}` with (B, n_layer) tensors.
    """
    sb, _ = _forward_zonotope_graph_batched(xl, xh, gg, device, dtype)
    tight = {L: (lo.clone(), hi.clone()) for L, (lo, hi) in sb.items()}
    relu_op_by_L = {op['layer_idx']: op for op in gg['ops']
                     if op['type'] == 'relu' and 'layer_idx' in op}
    B = xl.shape[0]
    # `tight` also holds string-keyed bilinear/pool boxes (kept for the
    # backward McCormick pass); iterate ONLY the int ReLU-layer keys so
    # sorted() doesn't choke on mixed str/int (lsnc_relu quadratic forms).
    for L in sorted(L for L in tight if isinstance(L, int)):
        if L not in relu_op_by_L:
            continue
        feed = relu_op_by_L[L]['inputs'][0]
        n = tight[L][0].shape[1]
        eye = torch.eye(n, dtype=dtype, device=device)
        seed = {feed: torch.cat([eye, -eye], dim=0)
                .unsqueeze(0).expand(B, -1, -1)}
        sl = _spec_backward_graph_batched(
            tight, xl, xh, gg, None, device, dtype,
            seed_ew_at=seed,
            seed_acc=torch.zeros(B, 2 * n, dtype=dtype, device=device))
        # Intersect the backward-CROWN bound with the running (forward-zono)
        # bound; both sound, so the tighter one stays sound.
        lo_t = torch.maximum(tight[L][0], sl[:, :n])
        hi_t = torch.maximum(torch.minimum(tight[L][1], -sl[:, n:]), lo_t)
        tight[L] = (lo_t, hi_t)
    return tight


@torch.no_grad()
def _crown_refine_intermediate_graph(gg, xl, xh, sb, op_bounds, device, dtype,
                                     deadline=None, print_progress=False,
                                     max_node=1024):
    """Topo-order backward-CROWN intermediate-bound refinement for graphs with
    bilinear / scalar-nonlinear ops (ml4acopf). This is the per-node analog of
    `_crown_intermediate_batched` (which only refines ReLU pre-activations and
    needs the batched forward that ml4acopf's outer-broadcast ops break): for
    EACH nonlinear op's input node it bounds that node by backward-CROWN to the
    network input (seeding `_spec_backward_graph` at the node via `seed_node`,
    +e_i for the lower and -e_i for the upper of every element) and INTERSECTS
    with the forward-zono bound already in `sb` / `op_bounds`.

    Soundness: both the CROWN-to-input bound and the forward-zono bound are
    sound over-approximations, so their intersection (max of los, min of his)
    stays sound; refining FEWER nodes (deadline hit) is also sound — the
    un-refined nodes keep their forward-zono bounds. ABC reaches its tight
    init-CROWN bound the same way (tight intermediates feeding the relaxations);
    on the ml4acopf linear nets this flips the spec bound from hugely negative
    to ABC-or-better (e.g. 14_ieee-linear-residual prop2: -452 -> +3.98).

    Mutates `sb` (int layer_idx keys: relu/sigmoid/tanh/sin/cos/floor
    pre-activations) and `op_bounds` (bilinear input tuples; scalar-nonlin
    input boxes) IN PLACE, and returns them. Pass COPIES if the caller needs
    the un-refined bounds afterwards.

    deadline: optional perf_counter() time after which no further nodes are
    refined (the spec backward the caller runs next still uses every
    refinement done so far).
    """
    ops = gg['ops']
    NL = ('relu', 'sigmoid', 'tanh', 'sin', 'cos', 'floor',
          'mul_bilinear', 'sub_bilinear', 'div_bilinear')
    refined = {}
    n_done = 0
    # Per-node refinement cost = 2n sequential backward passes; on the big
    # nets (300_ieee: n up to ~3.8k) ONE node can run for minutes and blow the
    # `deadline` (which is only checked between nodes). Track the worst node
    # time and stop proactively before a node that would overrun; skip nodes
    # with n > max_node outright (too costly, and the refinement is grossly
    # insufficient on the big nets anyway — keep their forward-zono bound).
    max_node_dt = 0.0

    def _crown_node(node, flo, fhi):
        n = flo.numel()
        eye = torch.eye(n, dtype=dtype, device=device)
        se = {}
        for i in range(n):
            se[2 * i] = (eye[i], 0.0)        # lower bound of node[i]
            se[2 * i + 1] = (-eye[i], 0.0)   # lower bound of -node[i] -> -upper
        lbs, _ = _spec_backward_graph(
            sb, xl, xh, gg, se, list(range(2 * n)), len(sb), device, dtype,
            op_bounds=op_bounds, seed_node=node)
        clo = torch.tensor([lbs[2 * i] for i in range(n)],
                           dtype=dtype, device=device)
        chi = torch.tensor([-lbs[2 * i + 1] for i in range(n)],
                           dtype=dtype, device=device)
        rlo = torch.maximum(flo, clo)
        # keep hi >= lo even if the intersect would invert (numerical guard).
        rhi = torch.minimum(fhi, torch.maximum(chi, rlo))
        return rlo, rhi

    def _is_bilin(op):
        nm = op['name']
        return (op['type'] in ('mul_bilinear', 'sub_bilinear', 'div_bilinear')
                and nm in op_bounds
                and isinstance(op_bounds[nm], tuple)
                and isinstance(op_bounds[nm][0], tuple))

    for op in ops:
        if deadline is not None and time.perf_counter() > deadline:
            if print_progress:
                print(f'[nl-bwd-crown] refinement deadline hit after '
                      f'{n_done} nodes', flush=True)
            break
        t = op['type']
        if t not in NL:
            continue
        nm = op['name']
        ins = op['inputs']
        if _is_bilin(op):
            (al, ah), (bl, bh) = op_bounds[nm]
            fb = {ins[0]: (al, ah), ins[1]: (bl, bh)}
        elif nm in op_bounds:
            fb = {ins[0]: op_bounds[nm]}
        elif 'layer_idx' in op and op['layer_idx'] in sb:
            fb = {ins[0]: sb[op['layer_idx']]}
        else:
            continue
        nfb = {}
        for nd, (flo, fhi) in fb.items():
            if nd in refined:
                nfb[nd] = refined[nd]
                continue
            flo = flo.to(device=device, dtype=dtype)
            fhi = fhi.to(device=device, dtype=dtype)
            now = time.perf_counter()
            # Skip oversized nodes, and stop before a node likely to overrun
            # the deadline (estimated from the worst node seen so far). Either
            # way keep the forward-zono bound for this node (sound).
            if (flo.numel() > max_node
                    or (deadline is not None
                        and now + 1.3 * max_node_dt > deadline)):
                refined[nd] = (flo, fhi)
                nfb[nd] = (flo, fhi)
                continue
            r = _crown_node(nd, flo, fhi)
            max_node_dt = max(max_node_dt, time.perf_counter() - now)
            refined[nd] = r
            nfb[nd] = r
            n_done += 1
        if _is_bilin(op):
            op_bounds[nm] = (nfb[ins[0]], nfb[ins[1]])
        elif nm in op_bounds:
            op_bounds[nm] = nfb[ins[0]]
        if 'layer_idx' in op and op['layer_idx'] in sb:
            sb[op['layer_idx']] = nfb[ins[0]]
    return sb, op_bounds


def _spec_backward_graph_batched(tight, xl, xh, gg, spec_ew, device, dtype,
                                   return_input_linear=False,
                                   alpha_at_layer=None,
                                   relu_split_beta=None,
                                   alpha_mccormick=None,
                                   alpha_recip=None,
                                   bilinear_op_bounds=None,
                                   seed_ew_at=None, seed_acc=None):
    """Batched CROWN spec backward.

    Args:
        tight: {layer_idx: (lo, hi)} where lo, hi are (B, n_layer).
        xl, xh: (B, n_in) per-batch input bounds.
        spec_ew: dict {qid: (w, bias)} with w (n_out,), bias scalar.
            Same w/bias applied across the batch.
        return_input_linear: if True, also returns the linear lower
            bound coefficients in input space:
              A: (B, Q, n_in) such that spec_q(x) >= A[b,q] · x + acc[b,q]
              acc: (B, Q) per-batch per-query bias
            Used by domain clipping in `_input_split_batched`.
        alpha_at_layer: optional dict {layer_idx: alpha} where alpha is
            (B, n_layer) tensor with values in [0, 1] giving the
            per-(leaf, neuron) lower slope for unstable ReLUs.
            Stable+ neurons use slope 1, stable- use 0, unstable use
            `alpha`. When provided, gradients flow through alpha →
            enables α-CROWN optimization. When None, falls back to
            min-area: lower slope = (up_s > 0.5). Caller controls
            torch.no_grad / torch.enable_grad as appropriate.

    Returns:
        spec_lbs: (B, Q) per-batch per-query lower bound on
        `w_q · y(x) + bias_q` over x in the batch's box.
        If return_input_linear: (spec_lbs, A, acc).
    """
    ops = gg['ops']
    B, n_in = xl.shape
    _pcs_batched = None  # lazy-built shared point-centers for bilinears

    if seed_ew_at is not None:
        # Caller-provided seed: used for intermediate-layer tightening
        # (start the backward at an arbitrary op rather than the spec).
        ew_at = {name: seed.clone() for name, seed in seed_ew_at.items()}
        any_seed = next(iter(seed_ew_at.values()))
        Q = any_seed.shape[1]
        if seed_acc is None:
            acc = torch.zeros(B, Q, dtype=dtype, device=device)
        else:
            acc = seed_acc.clone()
    else:
        qids = sorted(spec_ew.keys())
        Q = len(qids)
        W_q = torch.stack([spec_ew[qid][0].flatten() for qid in qids])  # (Q, n_out)
        b_q = torch.tensor([float(spec_ew[qid][1]) for qid in qids],
                            dtype=dtype, device=device)  # (Q,)
        # Seed ew at output: (B, Q, n_out) — broadcast queries across batch.
        last_name = ops[-1]['name']
        ew_at = {last_name: W_q.unsqueeze(0).expand(B, -1, -1).clone()}
        acc = b_q.unsqueeze(0).expand(B, -1).clone()  # (B, Q)

    # Bilinear (mul_bilinear) backward needs each product's per-leaf input
    # boxes to build the SOUND McCormick envelope. Every caller computes
    # `tight` via `_forward_zonotope_graph_batched(xl, xh, ...)` immediately
    # before this backward, and that forward stashes the boxes on
    # `.last_bilinear_op_bounds` for exactly the same (xl, xh). Pull them in
    # when the caller didn't pass them explicitly — otherwise the
    # mul_bilinear branch falls to a point-linearization that is a TANGENT,
    # not a bound: it omits the product's `-a*·b*` constant and so DOUBLE-
    # counts, putting the "lower bound" ABOVE the true value. That silently
    # false-verified lsnc_relu SAT cases (the quadratic Lyapunov value Y_1's
    # `<= b` query came back +0.83 when the true value was -0.01, refuting
    # the level-set band that the real counterexample lives in).
    if bilinear_op_bounds is None:
        bilinear_op_bounds = getattr(
            _forward_zonotope_graph_batched, 'last_bilinear_op_bounds', None)

    import os as _os_gdbg
    _grad_dbg = _os_gdbg.environ.get('DEBUG_GRAD_FLOW', '') == '1'
    _trace_dbg = _os_gdbg.environ.get('VIB_TRACE_BWD', '') == '1'
    def _spec_lb_now(_ew_at, _acc):
        """Compute current spec_lb assuming chain stops here: lb = acc +
        sum over remaining ew_at of ew@x_bound. Concretize via worst-case
        per-element box bound."""
        tot = _acc.clone()
        for _nm, _ev in _ew_at.items():
            if _ev is None:
                continue
            # The bound on ew_at[input] @ x is computed at input dim
            # for unprocessed ops. For arbitrary intermediate names we
            # can't directly concretize (would need tight[L] for that op).
            # Skip — only report acc here for trace.
        return tot
    for op in reversed(ops):
        name = op['name']
        if name not in ew_at:
            continue
        ew = ew_at[name]  # (B, Q, n)
        t = op['type']
        if _grad_dbg:
            print(f'[grad] op={name} type={t} ew.requires_grad={ew.requires_grad} '
                  f'acc.requires_grad={acc.requires_grad if isinstance(acc, torch.Tensor) else "N/A"}',
                  flush=True)
        if _trace_dbg:
            # Sum |ew| over ew_at - rough measure of "remaining work"
            _ew_total = sum(float(v.abs().sum()) for v in ew_at.values() if v is not None)
            _ew_max = max(float(v.abs().max()) for v in ew_at.values() if v is not None)
            print(f'[vib-trace] BEFORE op={name} type={t} '
                  f'acc[0,0]={float(acc[0,0]):.6f} '
                  f'ew_total={_ew_total:.4f} ew_max={_ew_max:.4f} '
                  f'ew_at_name_shape={tuple(ew.shape)}', flush=True)
        if t == 'fc':
            W = op['W']
            bias = op['bias']
            # ND-batched MatMul case (mscn dual: input (3, 7), W (128, 7),
            # output (3, 128) → ew shape (B, Q, 3*128 = 384)).
            in_shape_nd = op.get('in_shapes_nd', [None])[0]
            out_shape_nd = op.get('out_shape_nd')
            if (in_shape_nd is not None and len(in_shape_nd) >= 2
                    and W.shape[1] == in_shape_nd[-1]
                    and out_shape_nd is not None and len(out_shape_nd) >= 2):
                prefix = out_shape_nd[:-1]
                n_out_inner = out_shape_nd[-1]
                n_in_inner = in_shape_nd[-1]
                prefix_size = int(np.prod(prefix))
                # ew: (B, Q, prefix_size * n_out_inner)
                # Reshape to (B, Q, prefix_size, n_out_inner).
                ew_nd = ew.reshape(*ew.shape[:-1], prefix_size, n_out_inner)
                # bias (n_out_inner,) broadcasts over prefix.
                acc = acc + (ew_nd * bias).sum(dim=(-2, -1))
                # ew_back at input: (B, Q, prefix_size, n_in_inner) = ew_nd @ W
                ew_back_nd = ew_nd @ W  # (B, Q, prefix_size, n_in_inner)
                ew_back = ew_back_nd.reshape(
                    *ew_back_nd.shape[:-2], prefix_size * n_in_inner)
            else:
                acc = acc + ew @ bias  # (B, Q)
                ew_back = ew @ W  # (B, Q, n_in_layer)
            inp = op['inputs'][0]
            existing = ew_at.get(inp)
            ew_at[inp] = ew_back if existing is None else existing + ew_back

        elif t == 'relu':
            if 'layer_idx' in op:
                L = op['layer_idx']
                lo, hi = tight[L]  # (B, n)
                lo_s_def, up_s, up_t, active, dead, unstable = _make_slopes(
                    lo, hi)
                ep = ew.clamp(min=0)  # (B, Q, n)
                en = ew.clamp(max=0)
                # acc += (en * up_t).sum over n, per (b, q)
                acc = acc + (en * up_t.unsqueeze(1)).sum(dim=-1)
                # α-CROWN: replace default lower slope with α[L] for
                # unstable neurons; stable+ → 1, stable- → 0 (already
                # the case via masks).
                if alpha_at_layer is not None and L in alpha_at_layer:
                    alpha_L = alpha_at_layer[L]  # (B, n) or (B, Q, n)
                    DT = lo.dtype
                    if alpha_L.dim() == 2:
                        # shared α across queries
                        lo_s = (active.to(DT)
                                + unstable.to(DT) * alpha_L).unsqueeze(1)
                    else:
                        # per-query α: (B, Q, n)
                        lo_s = (active.to(DT).unsqueeze(1)
                                + unstable.to(DT).unsqueeze(1) * alpha_L)
                    ew_back = ep * lo_s + en * up_s.unsqueeze(1)
                else:
                    lo_s = lo_s_def
                    ew_back = ep * lo_s.unsqueeze(1) + en * up_s.unsqueeze(1)
                # β-CROWN split enforcement: for a ReLU-split constraint
                # in the form s(x) <= 0 on the subdomain (OFF: z_j <= 0
                # → s = +z; ON: z_j >= 0 → s = -z), adding β·s (β >= 0)
                # to the objective keeps a valid lower bound by weak
                # duality: min_relaxed(f + β·s) <= min_feas(f + β·s)
                # <= min_feas(f). The caller passes the SIGNED β
                # (sign·β, zero at non-split neurons); it adds to the
                # PRE-activation coefficient so the constraint couples
                # linearly through all earlier layers (the part the box
                # clamp alone cannot capture).
                if relu_split_beta is not None and L in relu_split_beta:
                    ew_back = ew_back + relu_split_beta[L].unsqueeze(1)
            else:
                ew_back = ew
            inp = op['inputs'][0]
            existing = ew_at.get(inp)
            ew_at[inp] = ew_back if existing is None else existing + ew_back

        elif t == 'add':
            # This batched pass propagates ew through unchanged — only sound if
            # no operand is outer-broadcast (the reduce-sum adjoint lives in the
            # scalar _spec_backward_graph). Fail loud rather than silently
            # unsound if a future outer-broadcast Add reaches here.
            assert_no_outer_broadcast(op.get('in_shapes_nd'),
                                      op.get('out_shape_nd'), t, 'batched')
            if op.get('is_merge'):
                # Skip connection: y = z_a + z_b. ew flows to both
                # inputs unchanged; no constant contribution to acc.
                for inp in op['inputs']:
                    existing = ew_at.get(inp)
                    ew_at[inp] = ew.clone() if existing is None else existing + ew
            else:
                # Constant bias-add: y = z + bias. CROWN backward
                # passes ew through to z but must accumulate the bias
                # contribution `(ew · bias).sum` into acc. Dropping it
                # is silently UNSOUND when the bias contribution
                # changes sign across α-CROWN iters (acasxu prop_8:
                # plain CROWN sound by accident, α-CROWN explodes to
                # +6 when true min is +0.03). Same pattern as the
                # bias-drop bugs fixed in `verify_milp.py`.
                bias = op.get('bias')
                if bias is not None:
                    # Lazy-cache torch conversion on op dict. Without this
                    # we re-convert numpy → torch every BAB iter (76 calls
                    # × 0.17ms = 13ms per spec_backward, biggest hotspot).
                    bt = op.get('_bias_t_cached')
                    if bt is None or bt.dtype != dtype or bt.device != device:
                        bt = torch.as_tensor(bias.flatten(),
                                              dtype=dtype, device=device)
                        op['_bias_t_cached'] = bt
                    # Broadcast bias if smaller than ew's last dim
                    # (mscn pattern: bias (128,) over ew (B, Q, 256=
                    # 2 segments × 128) needs tile).
                    if bt.numel() < ew.shape[-1] and ew.shape[-1] % bt.numel() == 0:
                        out_shape_nd = op.get('out_shape_nd')
                        if (out_shape_nd is not None
                                and len(out_shape_nd) >= 1
                                and out_shape_nd[-1] == bt.numel()):
                            ew_nd = ew.reshape(*ew.shape[:-1], *out_shape_nd)
                            acc = acc + (ew_nd * bt).reshape(
                                *ew.shape[:-1], -1).sum(dim=-1)
                        else:
                            # Fallback: tile bt to match ew's last dim.
                            tile = ew.shape[-1] // bt.numel()
                            bt_tiled = bt.repeat(tile)
                            acc = acc + (ew * bt_tiled).sum(dim=-1)
                    else:
                        acc = acc + (ew * bt).sum(dim=-1)
                inp = op['inputs'][0]
                existing = ew_at.get(inp)
                ew_at[inp] = ew.clone() if existing is None else existing + ew

        elif t == 'sub':
            assert_no_outer_broadcast(op.get('in_shapes_nd'),
                                      op.get('out_shape_nd'), t, 'batched')
            bias = op.get('bias')
            if bias is not None:
                bt = op.get('_bias_t_cached')
                if bt is None or bt.dtype != dtype or bt.device != device:
                    bt = torch.as_tensor(bias.flatten(),
                                          dtype=dtype, device=device)
                    op['_bias_t_cached'] = bt
                # acc -= (ew * bt).sum over n
                acc = acc - (ew * bt).sum(dim=-1)
            inp = op['inputs'][0]
            existing = ew_at.get(inp)
            ew_at[inp] = ew.clone() if existing is None else existing + ew

        elif t == 'sub_bilinear':
            # ew flows + to inp[0], - to inp[1]; no const.
            ia, ib = op['inputs'][0], op['inputs'][1]
            ea = ew_at.get(ia)
            eb = ew_at.get(ib)
            ew_at[ia] = ew.clone() if ea is None else ea + ew
            ew_at[ib] = (-ew).clone() if eb is None else eb + (-ew)

        elif t == 'reshape':
            inp = op['inputs'][0]
            existing = ew_at.get(inp)
            ew_at[inp] = ew.clone() if existing is None else existing + ew

        elif t in ('slice', 'gather'):
            flat_idx = op.get('flat_idx')
            in_shape_nd = op.get('in_shapes_nd', [None])[0]
            if flat_idx is None or in_shape_nd is None:
                raise ValueError("slice backward (batched) missing flat_idx/in_shape")
            n_in_layer = int(np.prod(in_shape_nd))
            idx_t = op.get('_flat_idx_t_cached')
            if idx_t is None or idx_t.device != device:
                idx_t = torch.as_tensor(flat_idx, dtype=torch.long, device=device)
                op['_flat_idx_t_cached'] = idx_t
            ew_back = torch.zeros(B, ew.shape[1], n_in_layer, dtype=dtype, device=device)
            # scatter-ADD adjoint: a fan-out gather (duplicate flat_idx) must
            # SUM the per-output contributions, not overwrite (index_copy_
            # drops all but the last → unsound). See _spec_backward_graph.
            ew_back.index_add_(-1, idx_t, ew)
            inp = op['inputs'][0]
            existing = ew_at.get(inp)
            ew_at[inp] = ew_back if existing is None else existing + ew_back

        elif t == 'concat':
            in_shapes = op.get('in_shapes_nd', [])
            offset = 0
            for inp, in_shape_nd in zip(op['inputs'], in_shapes):
                n_in_layer = int(np.prod(in_shape_nd))
                ew_i = ew[..., offset:offset + n_in_layer]
                existing = ew_at.get(inp)
                ew_at[inp] = ew_i.clone() if existing is None else existing + ew_i
                offset += n_in_layer

        elif t == 'conv':
            # Backward of conv2d is conv_transpose2d. ew shape (B, Q, n_out).
            kernel = op['kernel']
            bias = op['bias']
            out_shape = op['out_shape']
            in_shape = op['in_shape']
            stride = op['stride']
            padding = op['padding']
            output_padding = op['output_padding']
            # acc += sum over neurons of ew * bias_per_neuron.
            # bias is per-channel, broadcast over spatial.
            C_out, H_out, W_out = out_shape
            spatial = H_out * W_out
            bias_flat = bias.repeat_interleave(spatial)  # (C_out * spatial,)
            acc = acc + (ew * bias_flat).sum(dim=-1)
            # ew_back: reshape ew to (B*Q, C_out, H_out, W_out), apply
            # conv_transpose2d with kernel, flatten back.
            ew_4d = ew.reshape(B * Q, *out_shape)
            ew_back_4d = F.conv_transpose2d(
                ew_4d, kernel, bias=None, stride=stride, padding=padding,
                output_padding=output_padding)
            n_in_layer = in_shape[0] * in_shape[1] * in_shape[2]
            ew_back = ew_back_4d.reshape(B, Q, n_in_layer)
            inp = op['inputs'][0]
            existing = ew_at.get(inp)
            ew_at[inp] = ew_back if existing is None else existing + ew_back

        elif t == 'conv_transpose':
            # Backward of conv_transpose2d is conv2d. ew shape (B, Q, n_out).
            kernel = op['kernel']
            bias = op['bias']
            out_shape = op['out_shape']
            in_shape = op['in_shape']
            stride = op['stride']
            padding = op['padding']
            C_out, H_out, W_out = out_shape
            spatial = H_out * W_out
            bias_flat = bias.repeat_interleave(spatial)
            acc = acc + (ew * bias_flat).sum(dim=-1)
            ew_4d = ew.reshape(B * Q, *out_shape)
            ew_back_4d = F.conv2d(
                ew_4d, kernel, bias=None, stride=stride, padding=padding)
            n_in_layer = in_shape[0] * in_shape[1] * in_shape[2]
            assert ew_back_4d.shape[2] * ew_back_4d.shape[3] * ew_back_4d.shape[1] == n_in_layer, \
                (f"conv_transpose backward shape mismatch: got "
                 f"{ew_back_4d.shape} expected total {n_in_layer}")
            ew_back = ew_back_4d.reshape(B, Q, n_in_layer)
            inp = op['inputs'][0]
            existing = ew_at.get(inp)
            ew_at[inp] = ew_back if existing is None else existing + ew_back

        elif t in ('sigmoid', 'tanh'):
            # CROWN backward through sigmoid/tanh — closed-form linear
            # bounds from `_sigmoid_tanh_linear_bounds`.
            # Pre-activation bounds come from `tight[L]`, recorded by
            # the forward zono.
            L = op['layer_idx']
            lo_pre, hi_pre = tight[L]
            lo_s, lo_t, up_s, up_t = _sigmoid_tanh_linear_bounds(
                lo_pre, hi_pre, t)
            ep = ew.clamp(min=0)
            en = ew.clamp(max=0)
            acc = acc + (ep * lo_t.unsqueeze(1)).sum(dim=-1) + \
                    (en * up_t.unsqueeze(1)).sum(dim=-1)
            ew_back = ep * lo_s.unsqueeze(1) + en * up_s.unsqueeze(1)
            inp = op['inputs'][0]
            existing = ew_at.get(inp)
            ew_at[inp] = ew_back if existing is None else existing + ew_back

        elif t == 'upsample':
            # Nearest-mode upsample y[c, h*sH+a, w*sW+b] = x[c, h, w] for
            # all (a, b) in [0, sH)×[0, sW). Adjoint sums over the
            # repeated output cells per input cell: avg_pool2d with
            # divisor_override=1.
            in_shape = op['in_shape']
            out_shape = op['out_shape']
            sH, sW = op['scale']
            ew_4d = ew.reshape(B * Q, *out_shape)
            ew_back_4d = F.avg_pool2d(
                ew_4d, kernel_size=(sH, sW), stride=(sH, sW),
                divisor_override=1)
            n_in_layer = in_shape[0] * in_shape[1] * in_shape[2]
            ew_back = ew_back_4d.reshape(B, Q, n_in_layer)
            inp = op['inputs'][0]
            existing = ew_at.get(inp)
            ew_at[inp] = ew_back if existing is None else existing + ew_back

        elif t == 'avg_pool':
            # avg_pool y = (1/(kH*kW)) * sum window. Adjoint = depthwise
            # conv_transpose with (1/(kH*kW))-uniform kernel. For
            # non-overlapping (stride==kernel) just F.conv_transpose2d
            # works. For overlapping windows the same call gives the
            # correct sum-of-broadcasted-gradients.
            in_shape = op['in_shapes_nd'][0]
            out_shape = op['out_shape_nd']
            C, H_in, W_in = in_shape
            kH, kW = op['kernel']
            sH, sW = op['stride']
            pH, pW = op['padding']
            ew_4d = ew.reshape(B * Q, *out_shape)
            w_avg = torch.full((C, 1, kH, kW), 1.0 / (kH * kW),
                                  dtype=dtype, device=device)
            ew_back_4d = F.conv_transpose2d(
                ew_4d, w_avg, bias=None, stride=(sH, sW),
                padding=(pH, pW), groups=C)
            # Crop or pad to match input shape exactly.
            if ew_back_4d.shape[2] != H_in or ew_back_4d.shape[3] != W_in:
                ew_back_4d = ew_back_4d[:, :, :H_in, :W_in]
                if ew_back_4d.shape[2] < H_in or ew_back_4d.shape[3] < W_in:
                    pad_h = H_in - ew_back_4d.shape[2]
                    pad_w = W_in - ew_back_4d.shape[3]
                    ew_back_4d = F.pad(ew_back_4d, (0, pad_w, 0, pad_h))
            n_in_layer = C * H_in * W_in
            ew_back = ew_back_4d.reshape(B, Q, n_in_layer)
            inp = op['inputs'][0]
            existing = ew_at.get(inp)
            ew_at[inp] = ew_back if existing is None else existing + ew_back

        elif t == 'mul':
            # y = scale * x (constant scale). Backward: ew_back = ew * scale.
            scale_t = op.get('scale')
            if isinstance(scale_t, np.ndarray):
                scale_t = torch.from_numpy(scale_t).to(
                    device=device, dtype=dtype)
            elif not isinstance(scale_t, torch.Tensor):
                scale_t = torch.tensor(scale_t, dtype=dtype, device=device)
            else:
                scale_t = scale_t.to(device=device, dtype=dtype)
            sflat = scale_t.flatten()
            n_in_layer = ew.shape[-1]
            if sflat.numel() == 1:
                ew_back = ew * sflat
            elif sflat.numel() == n_in_layer:
                ew_back = ew * sflat.unsqueeze(0).unsqueeze(0)
            else:
                in_shape = op['in_shapes_nd'][0]
                C, H, W = in_shape
                assert sflat.numel() == C
                scale_4d = sflat.view(1, C, 1, 1).expand(
                    1, C, H, W).reshape(1, 1, -1)
                ew_back = ew * scale_4d
            inp = op['inputs'][0]
            existing = ew_at.get(inp)
            ew_at[inp] = ew_back if existing is None else existing + ew_back

        elif t == 'reduce_sum':
            # Linear reduction backward: ew at output broadcasts to input.
            # ew has shape (B, Q, n_out). Reshape to (B, Q, *sh_out_nd),
            # then expand into (B, Q, *sh_in_nd) via broadcast.
            in_shape_nd = op.get('in_shapes_nd', [None])[0]
            out_shape_nd = op.get('out_shape_nd')
            axes = op.get('axes', ())
            keepdims = op.get('keepdims', False)
            n_in_layer = int(np.prod(in_shape_nd))
            ew_nd = ew.reshape(B, Q, *out_shape_nd)
            # Expand by un-summing the reduced axes.
            for ax in sorted(axes):
                if not keepdims:
                    ew_nd = ew_nd.unsqueeze(2 + ax)
                ew_nd = ew_nd.expand(*ew_nd.shape[:2 + ax],
                                       in_shape_nd[ax],
                                       *ew_nd.shape[3 + ax:])
            ew_back = ew_nd.reshape(B, Q, n_in_layer).contiguous()
            inp = op['inputs'][0]
            existing = ew_at.get(inp)
            ew_at[inp] = ew_back if existing is None else existing + ew_back

        elif t == 'div_bilinear' and op.get('_div_decoupled'):
            # ABC-style Div backward: Mul(a, Reciprocal(b)) with α-tunable
            # McCormick + α-tunable Recip tangent. Strictly tighter than
            # single-point Taylor + R-bound (validated on pensieve test
            # case: ~2.3x tighter LB on a representative (a, b) box).
            # Mirrors α,β-CROWN's BoundDiv = BoundMul · BoundReciprocal.
            from .alpha_crown import _sum_to_shape
            sh_in = op.get('in_shapes_nd', [None, None])
            sh_out = op.get('out_shape_nd')
            ia, ib = op['inputs'][0], op['inputs'][1]
            a_lo = op['_div_a_lo'].to(device=device, dtype=dtype)
            a_hi = op['_div_a_hi'].to(device=device, dtype=dtype)
            b_lo = op['_div_b_lo'].to(device=device, dtype=dtype)
            b_hi = op['_div_b_hi'].to(device=device, dtype=dtype)
            # Broadcast (a, b) bounds to (B, *sh_out) per batch element.
            ones_out = torch.ones(B, *sh_out, dtype=dtype, device=device)
            a_lo_o = ones_out * a_lo.reshape(B, *sh_in[0])
            a_hi_o = ones_out * a_hi.reshape(B, *sh_in[0])
            b_lo_o = ones_out * b_lo.reshape(B, *sh_in[1])
            b_hi_o = ones_out * b_hi.reshape(B, *sh_in[1])
            # α tunables (only used when shape matches the per-leaf box;
            # otherwise fall back to default midpoint).
            def _alpha_for(name, default_shape):
                a_t = op.get(name)
                if a_t is None:
                    return None
                a_t = a_t.to(device=device, dtype=dtype)
                if a_t.shape == default_shape:
                    return a_t
                return None
            alpha_r = _alpha_for('_div_recip_alpha', b_lo.shape)
            r_l = _alpha_for('_div_mc_rl', a_lo.shape)
            r_u = _alpha_for('_div_mc_ru', a_lo.shape)
            # Broadcast α's to (B, *sh_out).
            if alpha_r is not None:
                alpha_r_o = ones_out * alpha_r.reshape(B, *sh_in[1])
            else:
                alpha_r_o = None
            if r_l is not None:
                r_l_o = ones_out * r_l.reshape(B, *sh_in[0])
            else:
                r_l_o = None
            if r_u is not None:
                r_u_o = ones_out * r_u.reshape(B, *sh_in[0])
            else:
                r_u_o = None
            # ew shape (B, Q, n_out) → (B, Q, *sh_out)
            ew_nd = ew.reshape(B, Q, *sh_out)
            # Helper expects ew and (a_lo,...) to broadcast. Add Q dim
            # to bounds by unsqueezing; helper acts per element.
            acc_contrib, ew_a_nd, ew_b_nd = _div_backward_rm_mccormick(
                a_lo_o.unsqueeze(1), a_hi_o.unsqueeze(1),
                b_lo_o.unsqueeze(1), b_hi_o.unsqueeze(1),
                ew_nd,
                alpha_r=(alpha_r_o.unsqueeze(1) if alpha_r_o is not None else None),
                r_l=(r_l_o.unsqueeze(1) if r_l_o is not None else None),
                r_u=(r_u_o.unsqueeze(1) if r_u_o is not None else None))
            # `acc_contrib` shape: (B, Q). Add directly.
            acc = acc + acc_contrib
            # `ew_a_nd`, `ew_b_nd` shape: (B, Q, *sh_out). Sum-to-shape
            # back to (B, Q, *sh_in[*]) via broadcast adjoint.
            ew_a_in_nd = _sum_to_shape(ew_a_nd, (B, Q), sh_in[0])
            ew_b_in_nd = _sum_to_shape(ew_b_nd, (B, Q), sh_in[1])
            ew_a = ew_a_in_nd.reshape(B, Q, -1)
            ew_b = ew_b_in_nd.reshape(B, Q, -1)
            ea = ew_at.get(ia)
            eb = ew_at.get(ib)
            ew_at[ia] = ew_a if ea is None else ea + ew_a
            ew_at[ib] = ew_b if eb is None else eb + ew_b

        elif t == 'div_bilinear' and not op.get('_div_decoupled'):
            # Point-side linearization: b is point per-sub. Use exact
            # 1/c_b slopes (b's variation is 0 → no slack needed).
            # Forward: y = a · (1/c_b). Backward: ew_a = ew/c_b, ew_b = -ew·c_a/c_b².
            # Need batched point centers — compute via single forward.
            from .alpha_crown import (
                _compute_point_centers_batched, _sum_to_shape)
            # Use per-leaf input midpoint as the linearisation point.
            x_centers = ((xl + xh) / 2)  # (B, n_in)
            sh_in = op.get('in_shapes_nd', [None, None])
            sh_out = op.get('out_shape_nd')
            ia, ib = op['inputs'][0], op['inputs'][1]
            # Cache: same x_centers across all bilinear ops in this
            # backward. Compute once and reuse.
            if '_pcs_batched' not in locals() or _pcs_batched is None:
                _pcs_batched = _compute_point_centers_batched(
                    gg, x_centers, device, dtype)
            c_a_batched = _pcs_batched[ia]  # (B, n_a)
            c_b_batched = _pcs_batched[ib]  # (B, n_b)
            ew_nd = ew.reshape(B, Q, *sh_out)
            a_nd = c_a_batched.reshape(B, *sh_in[0])
            b_nd = c_b_batched.reshape(B, *sh_in[1])
            if bool((b_nd == 0).any()):
                raise ZeroDivisionError(
                    f'batched div_bilinear backward: denom zero at {name!r}')
            if (bilinear_op_bounds is not None
                    and ia in bilinear_op_bounds
                    and ib in bilinear_op_bounds):
                # ABC-style: decompose y = a/b as a * (1/b).
                # McCormick on (a, t=1/b) with bounds on (a, t),
                # then substitute t with linear LB/UB in b (1/b is convex
                # on positive reals — tangent for LB, secant for UB).
                # Sign-conditional dispatch on McCormick β (coefficient
                # on t) to pick LB vs UB linearization of 1/b.
                a_lo_o, a_hi_o = bilinear_op_bounds[ia]
                b_lo_o, b_hi_o = bilinear_op_bounds[ib]
                a_lo_nd = a_lo_o.reshape(B, *sh_in[0])
                a_hi_nd = a_hi_o.reshape(B, *sh_in[0])
                b_lo_nd = b_lo_o.reshape(B, *sh_in[1])
                b_hi_nd = b_hi_o.reshape(B, *sh_in[1])
                # Require b > 0 (mscn softmax-style — denom is sum of
                # positives + epsilon). If b_lo <= 0, fall through to
                # point linearization below.
                if not bool((b_lo_nd > 0).all()):
                    raise NotImplementedError(
                        'div_bilinear McCormick path requires b_lo > 0; '
                        f'got min b_lo={b_lo_nd.min().item()}')
                # 1/b bounds at INPUT (decreasing function):
                t_lo_nd = 1.0 / b_hi_nd  # min of 1/b
                t_hi_nd = 1.0 / b_lo_nd  # max of 1/b
                # McCormick envelopes for product a*t with (a, t) bounds.
                # r_l, r_u ∈ [0, 1] interpolate corners (ABC-style).
                if (alpha_mccormick is not None
                        and name in alpha_mccormick):
                    r_l, r_u = alpha_mccormick[name]
                else:
                    r_l = torch.full(sh_out, 0.5, dtype=dtype, device=device)
                    r_u = torch.full(sh_out, 0.5, dtype=dtype, device=device)
                # Need shape (B, *sh_out) for broadcast with a/t bounds.
                r_l_b = r_l.unsqueeze(0) if r_l.dim() == len(sh_out) else r_l
                r_u_b = r_u.unsqueeze(0) if r_u.dim() == len(sh_out) else r_u
                alpha_l = (t_lo_nd - t_hi_nd) * r_l_b + t_hi_nd
                beta_l = (a_lo_nd - a_hi_nd) * r_l_b + a_hi_nd
                gamma_l = ((t_hi_nd * a_hi_nd - t_lo_nd * a_lo_nd) * r_l_b
                           - t_hi_nd * a_hi_nd)
                alpha_u = (t_hi_nd - t_lo_nd) * r_u_b + t_lo_nd
                beta_u = (a_lo_nd - a_hi_nd) * r_u_b + a_hi_nd
                gamma_u = ((t_lo_nd * a_hi_nd - t_hi_nd * a_lo_nd) * r_u_b
                           - t_lo_nd * a_hi_nd)
                # Reciprocal LB tangent at b0 ∈ [b_lo, b_hi].
                # ABC tunes b0 per spec output (alpha[start_node]); we
                # parameterize b0 = b_lo + r_recip * (b_hi - b_lo)
                # with r_recip ∈ [0, 1]. Default r_recip = 0.5 = midpoint.
                if alpha_recip is not None and name in alpha_recip:
                    r_recip = alpha_recip[name]
                    r_recip_b = (r_recip.unsqueeze(0)
                                 if r_recip.dim() == len(sh_in[1]) else r_recip)
                    b0 = b_lo_nd + r_recip_b * (b_hi_nd - b_lo_nd)
                else:
                    b0 = (b_lo_nd + b_hi_nd) * 0.5
                m_lb_inv = -1.0 / (b0 * b0)
                c_lb_inv = 2.0 / b0
                # UB_inv: 1/b <= m_u_inv * b + c_u_inv (secant b_lo → b_hi)
                m_ub_inv = -1.0 / (b_lo_nd * b_hi_nd)
                c_ub_inv = 1.0 / b_lo_nd + 1.0 / b_hi_nd
                # For LB on y = a*t: substitute t.
                #   if β_l >= 0: use LB_inv (smaller t → smaller LB,
                #     still valid LB);
                #   if β_l < 0: use UB_inv (smaller |β_l|*t → larger,
                #     still valid LB).
                beta_l_pos = (beta_l >= 0).to(dtype)
                beta_l_neg = 1.0 - beta_l_pos
                m_l_sub = beta_l_pos * m_lb_inv + beta_l_neg * m_ub_inv
                c_l_sub = beta_l_pos * c_lb_inv + beta_l_neg * c_ub_inv
                beta_y_l = beta_l * m_l_sub
                gamma_y_l = beta_l * c_l_sub + gamma_l
                # For UB on y: dual.
                beta_u_pos = (beta_u >= 0).to(dtype)
                beta_u_neg = 1.0 - beta_u_pos
                m_u_sub = beta_u_pos * m_ub_inv + beta_u_neg * m_lb_inv
                c_u_sub = beta_u_pos * c_ub_inv + beta_u_neg * c_lb_inv
                beta_y_u = beta_u * m_u_sub
                gamma_y_u = beta_u * c_u_sub + gamma_u
                # Sign-conditional backward (per output element).
                ew_nd_d = ew_nd
                ew_pos = ew_nd_d.clamp(min=0)
                ew_neg = ew_nd_d.clamp(max=0)
                ew_a_full = (ew_pos * alpha_l.unsqueeze(1)
                             + ew_neg * alpha_u.unsqueeze(1))
                ew_b_full = (ew_pos * beta_y_l.unsqueeze(1)
                             + ew_neg * beta_y_u.unsqueeze(1))
                acc_contrib = (ew_pos * gamma_y_l.unsqueeze(1)
                               + ew_neg * gamma_y_u.unsqueeze(1))
                acc = acc + acc_contrib.reshape(B, Q, -1).sum(dim=-1)
                ew_a_nd = _sum_to_shape(ew_a_full, (B, Q), sh_in[0])
                ew_b_nd = _sum_to_shape(ew_b_full, (B, Q), sh_in[1])
                ew_a = ew_a_nd.reshape(B, Q, -1)
                ew_b = ew_b_nd.reshape(B, Q, -1)
            else:
                inv_b = b_nd.reciprocal()
                # ew_a = ew * inv_b, broadcast (B, Q, *sh_out) by (B, *sh_in_b)
                # Need to add Q dim to b_nd / a_nd for broadcasting.
                ew_a_nd_full = ew_nd * inv_b.unsqueeze(1)
                ew_b_nd_full = -ew_nd * a_nd.unsqueeze(1) * inv_b.unsqueeze(1) * inv_b.unsqueeze(1)
                # Sum-to-shape per batch element (broadcast adjoint).
                ew_a_nd = _sum_to_shape(ew_a_nd_full, (B, Q), sh_in[0])
                ew_b_nd = _sum_to_shape(ew_b_nd_full, (B, Q), sh_in[1])
                ew_a = ew_a_nd.reshape(B, Q, -1)
                ew_b = ew_b_nd.reshape(B, Q, -1)
            ea = ew_at.get(ia)
            eb = ew_at.get(ib)
            ew_at[ia] = ew_a.clone() if ea is None else ea + ew_a
            ew_at[ib] = ew_b.clone() if eb is None else eb + ew_b

        elif t == 'mul_bilinear' and t != 'mul_bilinear_box_relax':
            # Will fall through to box-relax below if needed.
            # For point-side, similar to div but multiplication.
            from .alpha_crown import _sum_to_shape
            sh_in = op.get('in_shapes_nd', [None, None])
            sh_out = op.get('out_shape_nd')
            ia, ib = op['inputs'][0], op['inputs'][1]
            # NOTE: the SOUND McCormick path below uses only the per-leaf
            # input BOXES (bilinear_op_bounds), never the input point
            # centers. The point-center forward (`_compute_point_centers_
            # batched`) is only needed by the legacy point-linearization,
            # which is now an unsound dead branch (raises). Computing it
            # eagerly was a full redundant forward pass per spec backward —
            # ~20% of the bound cost on lsnc. Compute it lazily ONLY in the
            # else branch (which immediately raises, so in practice never).
            if (bilinear_op_bounds is not None
                    and ia in bilinear_op_bounds
                    and ib in bilinear_op_bounds):
                # ABC-style McCormick + α (sign-conditional envelope).
                # LB: y >= α_l·a + β_l·b + γ_l  (used where ew > 0)
                # UB: y <= α_u·a + β_u·b + γ_u  (used where ew < 0)
                # r_l, r_u ∈ [0,1] interpolate between 2 McCormick
                # corners (ABC's `interpolated_relaxation`).
                a_lo, a_hi = bilinear_op_bounds[ia]
                b_lo, b_hi = bilinear_op_bounds[ib]
                a_lo_nd = a_lo.reshape(B, *sh_in[0])
                a_hi_nd = a_hi.reshape(B, *sh_in[0])
                b_lo_nd = b_lo.reshape(B, *sh_in[1])
                b_hi_nd = b_hi.reshape(B, *sh_in[1])
                if alpha_mccormick is not None and name in alpha_mccormick:
                    r_l, r_u = alpha_mccormick[name]
                else:
                    r_l = torch.full(sh_out, 0.5, dtype=dtype, device=device)
                    r_u = torch.full(sh_out, 0.5, dtype=dtype, device=device)
                r_l_b = r_l.unsqueeze(0) if r_l.dim() == len(sh_out) else r_l
                r_u_b = r_u.unsqueeze(0) if r_u.dim() == len(sh_out) else r_u
                alpha_l = (b_lo_nd - b_hi_nd) * r_l_b + b_hi_nd
                beta_l = (a_lo_nd - a_hi_nd) * r_l_b + a_hi_nd
                gamma_l = ((b_hi_nd * a_hi_nd - b_lo_nd * a_lo_nd) * r_l_b
                           - b_hi_nd * a_hi_nd)
                alpha_u = (b_hi_nd - b_lo_nd) * r_u_b + b_lo_nd
                beta_u = (a_lo_nd - a_hi_nd) * r_u_b + a_hi_nd
                gamma_u = ((b_lo_nd * a_hi_nd - b_hi_nd * a_lo_nd) * r_u_b
                           - b_lo_nd * a_hi_nd)
                ew_nd = ew.reshape(B, Q, *sh_out)
                ew_pos = ew_nd.clamp(min=0)
                ew_neg = ew_nd.clamp(max=0)
                ew_a_full = (ew_pos * alpha_l.unsqueeze(1)
                             + ew_neg * alpha_u.unsqueeze(1))
                ew_b_full = (ew_pos * beta_l.unsqueeze(1)
                             + ew_neg * beta_u.unsqueeze(1))
                acc_contrib = (ew_pos * gamma_l.unsqueeze(1)
                               + ew_neg * gamma_u.unsqueeze(1))
                acc = acc + acc_contrib.reshape(B, Q, -1).sum(dim=-1)
                ew_a_nd = _sum_to_shape(ew_a_full, (B, Q), sh_in[0])
                ew_b_nd = _sum_to_shape(ew_b_full, (B, Q), sh_in[1])
                ew_a = ew_a_nd.reshape(B, Q, -1)
                ew_b = ew_b_nd.reshape(B, Q, -1)
            else:
                # No per-leaf input boxes -> the only alternative is a point
                # linearization at the centers (ew_a = ew·b_c, ew_b = ew·a_c).
                # That is a TANGENT to the bilinear surface, NOT a sound
                # bound: it drops the `-a_c·b_c` constant, so at the center it
                # evaluates to 2·a_c·b_c instead of a_c·b_c and the resulting
                # "lower bound" can sit ABOVE the true value. Using it as a
                # CROWN lower bound silently false-verifies (lsnc_relu). The
                # forward always stashes these boxes, and this function pulls
                # them above when the caller passes None — so reaching here
                # means the boxes are genuinely missing. Fail loudly rather
                # than certify an unsound bound.
                raise NotImplementedError(
                    f'mul_bilinear backward at {name!r}: missing per-leaf '
                    f'input boxes (bilinear_op_bounds) for inputs '
                    f'{ia!r}, {ib!r}. A point linearization here is unsound '
                    f'(tangent, not a bound). Ensure the forward zono ran on '
                    f'this box so .last_bilinear_op_bounds is populated, or '
                    f'pass bilinear_op_bounds explicitly.')
            ea = ew_at.get(ia)
            eb = ew_at.get(ib)
            ew_at[ia] = ew_a.clone() if ea is None else ea + ew_a
            ew_at[ib] = ew_b.clone() if eb is None else eb + ew_b

        elif t == 'pow':
            # Pow batched backward — two-line (LB tangent, UB chord)
            # per element per batch. ew shape (B, Q, n). Uses
            # α-optimized tangent position when present (set by the
            # α-CROWN Adam loop in `_run_alpha_crown_inputsplit_batched`).
            lo_pre = op.get('_pow_in_lo')
            hi_pre = op.get('_pow_in_hi')
            assert lo_pre is not None and hi_pre is not None, (
                f"batched pow backward: missing pre-pow bounds")
            lo_pre_t = lo_pre.to(device=device, dtype=dtype)
            hi_pre_t = hi_pre.to(device=device, dtype=dtype)
            p = int(op.get('exponent', 2))
            tan_alpha = op.get('_pow_tangent_alpha')
            # Phase 0.5's `_pow_tangent_alpha` may be shape (n,) — only
            # valid at root box. Skip unless its shape matches lo_pre_t.
            if tan_alpha is not None:
                tan_alpha = tan_alpha.to(device=device, dtype=dtype)
                if tan_alpha.shape != lo_pre_t.shape:
                    tan_alpha = None
            (lb_slope, lb_const, ub_slope, ub_const,
             use_tl, box_lo_v, box_hi_v) = _pow_two_line_coeffs(
                lo_pre_t, hi_pre_t, p, tangent_pos=tan_alpha)
            # All these are (B, n_layer) shape.
            ep = ew.clamp(min=0); en = ew.clamp(max=0)
            slope_back = ep * lb_slope.unsqueeze(1) \
                + en * ub_slope.unsqueeze(1)
            const_back = ep * lb_const.unsqueeze(1) \
                + en * ub_const.unsqueeze(1)
            use_tl_u = use_tl.unsqueeze(1)
            box_lo_u = box_lo_v.unsqueeze(1)
            box_hi_u = box_hi_v.unsqueeze(1)
            slope_back = torch.where(use_tl_u, slope_back,
                                       torch.zeros_like(slope_back))
            const_back = torch.where(use_tl_u, const_back,
                                       torch.where(ep > 0, box_lo_u,
                                           torch.zeros_like(box_lo_u))
                                       + torch.where(en < 0, box_hi_u,
                                           torch.zeros_like(box_hi_u)))
            acc = acc + const_back.sum(dim=-1)
            ew_back = slope_back
            inp = op['inputs'][0]
            existing = ew_at.get(inp)
            ew_at[inp] = ew_back if existing is None else existing + ew_back

        elif t in ('max_pool', 'matmul_bilinear', 'softmax'):
            # Box-relaxation CROWN: y has constant bounds (lo_out, hi_out)
            # stamped at forward time. Linear lower bound is the constant
            # lo_out (slope 0); the contribution to acc is
            # sum_n max(0, ew[n]) * lo_out[n] + sum_n min(0, ew[n]) * hi_out[n].
            # No backward signal to inputs.
            key = op['name'] + f'__{t}_box'
            if key not in tight:
                raise ValueError(
                    f'batched backward: missing box bounds for {key}')
            lo_box, hi_box = tight[key]  # (B, n_out)
            ep = ew.clamp(min=0)
            en = ew.clamp(max=0)
            acc = acc + (ep * lo_box.unsqueeze(1)).sum(dim=-1) + \
                    (en * hi_box.unsqueeze(1)).sum(dim=-1)
            # No ew_back: inputs aren't propagated (box loses correlation).

        elif t == 'transpose':
            # Inverse-permutation of the gen layout.
            sh_in = op['in_shapes_nd'][0]
            sh_out = op['out_shape_nd']
            perm = op['perm']
            perm_b = [0] + [p for p in perm if p != 0]
            # Inverse permutation.
            inv_perm = [0] * len(perm_b)
            for i, p in enumerate(perm_b):
                inv_perm[p] = i
            ew_nd = ew.reshape(B, Q, *sh_out)
            # ew is (B, Q, *sh_out). We want to permute the sh_out axes
            # by inv_perm. ew has 2 leading dims (B, Q) — shift inv_perm
            # by +1 to account for the Q dim while keeping B at 0.
            perm_eq = [0, 1] + [p + 1 for p in inv_perm if p != 0]
            ew_back_nd = ew_nd.permute(*perm_eq).contiguous()
            n_in_layer = ew_back_nd.numel() // (B * Q)
            ew_back = ew_back_nd.reshape(B, Q, n_in_layer)
            inp = op['inputs'][0]
            existing = ew_at.get(inp)
            ew_at[inp] = ew_back if existing is None else existing + ew_back

        elif t == 'squeeze':
            # No-op: shape change only, data unchanged.
            inp = op['inputs'][0]
            existing = ew_at.get(inp)
            ew_at[inp] = ew if existing is None else existing + ew

        else:
            raise ValueError(f'batched spec backward: unknown op {t!r}')

    input_name = gg['input_name']
    ew_inp = ew_at.get(input_name)
    if ew_inp is None:
        if return_input_linear:
            zeros = torch.zeros(B, Q, n_in, dtype=dtype, device=device)
            return acc, zeros, acc
        return acc
    # Per-batch interval bound: spec_lb[b, q] = acc[b, q]
    # + sum_i [pos(ew[b, q, i]) * xl[b, i] + neg(ew[b, q, i]) * xh[b, i]]
    pos = ew_inp.clamp(min=0)
    neg = ew_inp.clamp(max=0)
    spec_lbs = (acc
                 + (pos * xl.unsqueeze(1)).sum(dim=-1)
                 + (neg * xh.unsqueeze(1)).sum(dim=-1))
    if return_input_linear:
        # spec_q(x) >= ew_inp[b,q] · x + acc[b,q]
        return spec_lbs, ew_inp, acc
    return spec_lbs


def _run_alpha_crown_inputsplit_batched(xl, xh, gg, spec_ew, device, dtype,
                                           n_iters=10, lr=0.25, lr_decay=0.98,
                                           early_stop_eps=1e-6):
    """Batched α-CROWN for input-split BaB boundary leaves.

    Optimizes per-(leaf, layer, neuron) lower-slope α to maximize
    per-query spec lb across a BATCH of leaves on GPU. Uses Adam.

    Args:
        xl, xh: (B, n_in) input bounds per leaf.
        gg: gpu_graph dict.
        spec_ew: dict {qid: (w (n_out,), bias float)} — same query
            family for all leaves; spec_lbs returned as (B, Q).
        n_iters: max Adam iters.
        lr, lr_decay: optimizer schedule.
        early_stop_eps: if no leaf's spec_lb improves by more than
            `eps` for one full iter, stop early.

    Returns:
        best_spec_lbs: (B, Q) — best spec lb seen across iterations.

    Notes:
      - α is initialized to min-area choice (1.0 where up_s > 0.5, else
        0.0). First iter equals plain CROWN.
      - α is clamped to [0, 1] after each Adam step.
      - Loss = -sum over (b, q) of spec_lbs (maximize lb).
      - For ACASXU 6-layer × 50-neuron net at B=100: ~10 ms per iter.
    """
    B, n_in = xl.shape
    with torch.no_grad():
        sb_init, _ = _forward_zonotope_graph_batched(
            xl, xh, gg, device, dtype)
    alpha_at_layer = {}
    for L, (lo, hi) in sb_init.items():
        _, up_s, _, active, dead, unstable = _make_slopes(lo, hi)
        init_alpha = ((up_s > 0.5).to(dtype) * unstable.to(dtype))
        alpha_at_layer[L] = init_alpha.detach().clone().requires_grad_(True)
    # α-Pow: NORMALIZED α ∈ [0, 1]; tangent_pos = lo + α·(hi - lo).
    # Reparam makes Adam's per-param step lr=0.25 cover ~25% of [lo,hi]
    # regardless of absolute magnitude (Pow ranges from O(1) to O(1e8)
    # across the chain — uniform [0,1] avoids per-op lr tuning).
    alpha_pow_norm = {}  # op_name -> (norm_alpha_tensor in [0,1], lo, hi)
    for op in gg['ops']:
        if op['type'] == 'pow':
            lo_pre = op.get('_pow_in_lo')
            hi_pre = op.get('_pow_in_hi')
            if lo_pre is not None and hi_pre is not None:
                lo_t = lo_pre.to(device=device, dtype=dtype)
                hi_t = hi_pre.to(device=device, dtype=dtype)
                a = torch.full_like(lo_t, 0.5).requires_grad_(True)
                alpha_pow_norm[op['name']] = (a, lo_t, hi_t)
    # α-Div: NORMALIZED α ∈ [0, 1]; cb = b_lo + α·(b_hi - b_lo).
    alpha_div_norm = {}
    for op in gg['ops']:
        if op['type'] == 'div_bilinear' and op.get('_div_decoupled'):
            b_lo = op.get('_div_b_lo')
            b_hi = op.get('_div_b_hi')
            if b_lo is not None and b_hi is not None:
                b_lo_t = b_lo.to(device=device, dtype=dtype)
                b_hi_t = b_hi.to(device=device, dtype=dtype)
                a = torch.full_like(b_lo_t, 0.5).requires_grad_(True)
                alpha_div_norm[op['name']] = (a, b_lo_t, b_hi_t)
    # α-Mul (McCormick r_l, r_u) and α-Recip (b0 tangent point) for
    # div_bilinear ops. Matches ABC's BoundReciprocal + BoundMul
    # structural split (their MulHelper has 4-channel alpha; we use
    # r_l, r_u — 2-channel — for our LB-only spec). Adam tunes per leaf
    # per neuron. Default r=0.5 = "middle" interpolation == current
    # static behavior.
    # α-Mul (McCormick r_l, r_u) and α-Recip (b0 tangent point) were
    # tried in this session — NO EFFECT for mscn. Measured grid sweep
    # over r_l, r_u ∈ {0, 0.3, 0.5, 0.7, 1} on card_1_1 16-leaf split:
    # all give same spec_lb to 1e-6 ulp. Reason: 5 of 6 div_bilinear ops
    # in mscn have CONSTANT denominator (sum of constant-branch ReLUs);
    # when b is constant McCormick degenerates to a single linear bound
    # regardless of r. Only div 163 has perturbed b — its α gives ~1e-6
    # spec movement. Adam can't escape the local optimum because
    # gradient is effectively zero. The actual ABC mechanism that
    # tightens to -0.053 is `compute_bounds(method='forward+crown')`:
    # joint forward+backward α-opt that REFRESHES INTERMEDIATE BOUNDS
    # per Adam iter. Vibe's α-CROWN uses static intermediates → caps at
    # -0.082. Real fix needs joint-opt rewrite of _run_alpha_crown.
    alpha_mccormick = {}
    alpha_recip = {}
    # Push initial values into ops.
    for op_name, (a, lo_t, hi_t) in alpha_pow_norm.items():
        for op in gg['ops']:
            if op['name'] == op_name:
                op['_pow_tangent_alpha'] = lo_t + a * (hi_t - lo_t)
    for op_name, (a, blo, bhi) in alpha_div_norm.items():
        for op in gg['ops']:
            if op['name'] == op_name:
                op['_div_cb_alpha'] = blo + a * (bhi - blo)
    optimizer_params = [alpha_at_layer[L] for L in alpha_at_layer]
    optimizer_params += [a for (a, _, _) in alpha_pow_norm.values()]
    optimizer_params += [a for (a, _, _) in alpha_div_norm.values()]
    if not optimizer_params:
        # No tunable α — every ReLU is stable (all active or all dead) over
        # these leaves and there are no Pow/Div ops. There is nothing to
        # optimise; the basic backward bound (init α) IS the answer and is
        # sound. (torch.optim.Adam raises on an empty parameter list; a fully-
        # stable leaf is reachable both here and via small BaB sub-boxes.)
        with torch.no_grad():
            return _spec_backward_graph_batched(
                sb_init, xl, xh, gg, spec_ew, device, dtype,
                alpha_at_layer=alpha_at_layer)
    optimizer = torch.optim.Adam(optimizer_params, lr=lr)
    # Backward compat for legacy variable names used in clamp loop below
    alpha_pow = alpha_pow_norm
    alpha_div = alpha_div_norm
    best_spec_lbs = None
    prev_max = -float('inf')
    for it in range(n_iters):
        optimizer.zero_grad()
        spec_lbs = _spec_backward_graph_batched(
            sb_init, xl, xh, gg, spec_ew, device, dtype,
            alpha_at_layer=alpha_at_layer)
        with torch.no_grad():
            if best_spec_lbs is None:
                best_spec_lbs = spec_lbs.detach().clone()
            else:
                best_spec_lbs = torch.maximum(best_spec_lbs, spec_lbs.detach())
            curr_max = float(best_spec_lbs.max().item())
        if (best_spec_lbs > 0).all().item():
            break
        loss = -spec_lbs.sum()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for L in alpha_at_layer:
                alpha_at_layer[L].clamp_(0.0, 1.0)
            # Clamp normalized α ∈ [0, 1]; recompute consumed values.
            for op_name, (a, lo_t, hi_t) in alpha_pow.items():
                a.data.clamp_(0.0, 1.0)
            for op_name, (a, blo, bhi) in alpha_div.items():
                a.data.clamp_(0.0, 1.0)
        # Refresh op refs to use NEW α values (after step). Build a
        # fresh tensor with current α (still requires_grad via mul chain).
        for op in gg['ops']:
            if op['type'] == 'pow' and op['name'] in alpha_pow:
                a, lo_t, hi_t = alpha_pow[op['name']]
                op['_pow_tangent_alpha'] = lo_t + a * (hi_t - lo_t)
            if op['type'] == 'div_bilinear' and op['name'] in alpha_div:
                a, blo, bhi = alpha_div[op['name']]
                op['_div_cb_alpha'] = blo + a * (bhi - blo)
        with torch.no_grad():
            pass
        if it > 0 and curr_max - prev_max < early_stop_eps:
            break
        prev_max = curr_max
        for g in optimizer.param_groups:
            g['lr'] *= lr_decay
    return best_spec_lbs



@torch.no_grad()
def _evaluate_region(xl, xh, remaining_specs, gpu_layers_list, spec_ew,
                     pred, nh, device, dtype):
    """Three-phase evaluation: forward zonotope, backward tighten, spec backward.

    Returns (spec_lbs, still_open, split_dim).
    """
    # Phase 1: Forward zonotope
    z = TorchZonotope.from_input_bounds(xl, xh, device, dtype)
    sb = {}
    for l in range(nh):
        gl = gpu_layers_list[l]
        if gl['type'] == 'conv':
            z.propagate_conv(gl['kernel'], gl['bias'], gl['in_shape'],
                             gl['stride'], gl['padding'])
        else:
            z.propagate_fc(gl['W'], gl['bias'])
        lo, hi = z.apply_relu()
        sb[l] = (lo.clone(), hi.clone())

    # Phase 2: Backward tighten unstable neurons
    if nh > 0:
        tight = {0: (sb[0][0].clone(), sb[0][1].clone())}
    else:
        tight = {}
    for l in range(1, nh):
        lo_std, hi_std = sb[l]
        ust_idx = torch.where((lo_std < 0) & (hi_std > 0))[0]
        n_ust = len(ust_idx)
        if n_ust == 0:
            tight[l] = (lo_std.clone(), hi_std.clone())
            continue

        # Precompute layer info for backward pass
        layer_info = {}
        for k in range(l):
            lo_k, hi_k = tight[k]
            lo_s, up_s, up_t, active, dead, ust_k = _make_slopes(lo_k, hi_k)
            act_idx = torch.where(active)[0]
            ust_k_idx = torch.where(ust_k)[0]
            dead_idx = torch.where(dead)[0]
            pct = len(ust_k_idx) / len(lo_k) if len(lo_k) > 0 else 1.0
            glk = gpu_layers_list[k]
            info = {
                'act_idx': act_idx, 'ust_idx': ust_k_idx,
                'dead_idx': dead_idx,
                'up_s_ust': up_s[ust_k_idx], 'up_t_ust': up_t[ust_k_idx],
                'lo_s_ust': lo_s[ust_k_idx],
                'is_conv': glk['type'] == 'conv', 'glk': glk,
                'lo_s_full': lo_s, 'up_s_full': up_s, 'up_t_full': up_t,
                'pct': pct,
            }
            if not info['is_conv']:
                info['W_act'] = glk['W'][act_idx]
                info['b_act'] = glk['bias'][act_idx]
                info['W_ust'] = glk['W'][ust_k_idx]
                info['b_ust'] = glk['bias'][ust_k_idx]
            layer_info[k] = info

        gl = gpu_layers_list[l]
        lbs = torch.empty(n_ust, dtype=dtype, device=device)
        ubs = torch.empty(n_ust, dtype=dtype, device=device)

        for cs in range(0, n_ust, 512):
            ce = min(cs + 512, n_ust)
            cidx = ust_idx[cs:ce]
            batch = len(cidx)

            if gl['type'] == 'conv':
                I_p = torch.zeros(batch, gl['n_out'], dtype=dtype,
                                  device=device)
                I_p[torch.arange(batch, device=device), cidx] = 1.0
                EW = F.conv_transpose2d(
                    I_p.reshape(batch, *gl['out_shape']), gl['kernel'],
                    stride=gl['stride'], padding=gl['padding'],
                    output_padding=gl['output_padding']).reshape(batch, -1)
                spatial = gl['out_shape'][1] * gl['out_shape'][2]
                bi = gl['bias'][cidx // spatial]
            else:
                EW = gl['W'][cidx].clone()
                bi = gl['bias'][cidx].clone()

            bias_lb = bi.clone()
            bias_ub = bi.clone()
            EW_lb = EW.clone()
            EW_ub = EW.clone()

            for k in range(l - 1, -1, -1):
                info = layer_info[k]
                if info['is_conv']:
                    if info['pct'] < 0.5:
                        ust_k_idx = info['ust_idx']
                        dead_idx = info['dead_idx']
                        EW_lb[:, dead_idx] = 0
                        EW_ub[:, dead_idx] = 0
                        ew_u = EW_lb[:, ust_k_idx]
                        ep = ew_u.clamp(min=0)
                        en = ew_u.clamp(max=0)
                        bias_lb += (en * info['up_t_ust']).sum(dim=1)
                        EW_lb[:, ust_k_idx] = (ep * info['lo_s_ust']
                                               + en * info['up_s_ust'])
                        ew_u = EW_ub[:, ust_k_idx]
                        ep = ew_u.clamp(min=0)
                        en = ew_u.clamp(max=0)
                        bias_ub += (ep * info['up_t_ust']).sum(dim=1)
                        EW_ub[:, ust_k_idx] = (ep * info['up_s_ust']
                                               + en * info['lo_s_ust'])
                    else:
                        ep = EW_lb.clamp(min=0)
                        en = EW_lb.clamp(max=0)
                        bias_lb += (en * info['up_t_full']).sum(dim=1)
                        EW_lb = (ep * info['lo_s_full']
                                 + en * info['up_s_full'])
                        ep = EW_ub.clamp(min=0)
                        en = EW_ub.clamp(max=0)
                        bias_ub += (ep * info['up_t_full']).sum(dim=1)
                        EW_ub = (ep * info['up_s_full']
                                 + en * info['lo_s_full'])
                    glk = info['glk']
                    os_k = glk['out_shape']
                    bias_lb += (EW_lb.reshape(batch, *os_k).sum(dim=(2, 3))
                                @ glk['bias'])
                    bias_ub += (EW_ub.reshape(batch, *os_k).sum(dim=(2, 3))
                                @ glk['bias'])
                    EW_lb = F.conv_transpose2d(
                        EW_lb.reshape(batch, *os_k), glk['kernel'],
                        stride=glk['stride'], padding=glk['padding'],
                        output_padding=glk['output_padding']
                    ).reshape(batch, -1)
                    EW_ub = F.conv_transpose2d(
                        EW_ub.reshape(batch, *os_k), glk['kernel'],
                        stride=glk['stride'], padding=glk['padding'],
                        output_padding=glk['output_padding']
                    ).reshape(batch, -1)
                else:
                    act_idx = info['act_idx']
                    ust_k_idx = info['ust_idx']
                    n_act = len(act_idx)
                    n_ust_k = len(ust_k_idx)
                    out_dim = (info['W_act'].shape[1] if n_act > 0
                               else info['W_ust'].shape[1])
                    EW_lb_new = torch.zeros(batch, out_dim, dtype=dtype,
                                            device=device)
                    EW_ub_new = torch.zeros_like(EW_lb_new)
                    if n_act > 0:
                        EW_lb_new += EW_lb[:, act_idx] @ info['W_act']
                        bias_lb += EW_lb[:, act_idx] @ info['b_act']
                        EW_ub_new += EW_ub[:, act_idx] @ info['W_act']
                        bias_ub += EW_ub[:, act_idx] @ info['b_act']
                    if n_ust_k > 0:
                        ep = EW_lb[:, ust_k_idx].clamp(min=0)
                        en = EW_lb[:, ust_k_idx].clamp(max=0)
                        bias_lb += (en * info['up_t_ust']).sum(dim=1)
                        ew_a = ep * info['lo_s_ust'] + en * info['up_s_ust']
                        EW_lb_new += ew_a @ info['W_ust']
                        bias_lb += ew_a @ info['b_ust']
                        ep = EW_ub[:, ust_k_idx].clamp(min=0)
                        en = EW_ub[:, ust_k_idx].clamp(max=0)
                        bias_ub += (ep * info['up_t_ust']).sum(dim=1)
                        ew_a = ep * info['up_s_ust'] + en * info['lo_s_ust']
                        EW_ub_new += ew_a @ info['W_ust']
                        bias_ub += ew_a @ info['b_ust']
                    EW_lb = EW_lb_new
                    EW_ub = EW_ub_new

            lbs[cs:ce] = (bias_lb + EW_lb.clamp(min=0) @ xl
                          + EW_lb.clamp(max=0) @ xh)
            ubs[cs:ce] = (bias_ub + EW_ub.clamp(min=0) @ xh
                          + EW_ub.clamp(max=0) @ xl)

        new_lo = lo_std.clone()
        new_hi = hi_std.clone()
        new_lo[ust_idx] = torch.maximum(lo_std[ust_idx], lbs)
        new_hi[ust_idx] = torch.minimum(hi_std[ust_idx], ubs)
        tight[l] = (new_lo, new_hi)

    # Phase 3: Spec backward
    spec_lbs = {}
    input_weights = {}
    for comp in remaining_specs:
        ew, b_spec = spec_ew[comp]
        ew = ew.clone()
        acc = b_spec
        for k in range(nh - 1, -1, -1):
            lo_k, hi_k = tight[k]
            lo_s, up_s, up_t, _, _, _ = _make_slopes(lo_k, hi_k)
            ep = ew.clamp(min=0)
            en = ew.clamp(max=0)
            acc += float((en * up_t).sum())
            ew = ep * lo_s + en * up_s
            glk = gpu_layers_list[k]
            if glk['type'] == 'conv':
                os_k = glk['out_shape']
                ew_4d = ew.reshape(1, *os_k)
                acc += float(
                    ew_4d.reshape(os_k[0], -1).sum(dim=1) @ glk['bias'])
                ew = F.conv_transpose2d(
                    ew_4d, glk['kernel'], stride=glk['stride'],
                    padding=glk['padding'],
                    output_padding=glk['output_padding']).flatten()
            else:
                acc += float(ew @ glk['bias'])
                ew = ew @ glk['W']
        spec_lbs[comp] = acc + float(
            ew.clamp(min=0) @ xl + ew.clamp(max=0) @ xh)
        input_weights[comp] = ew.detach()

    still_open = {c for c in remaining_specs if spec_lbs[c] <= 0}
    if still_open:
        w = (xh - xl).cpu().numpy()
        score = np.zeros(len(w))
        for comp in still_open:
            score += np.abs(input_weights[comp].cpu().numpy())
        split_dim = int(np.argmax(score * w))
    else:
        split_dim = -1
    return spec_lbs, still_open, split_dim


@torch.no_grad()
def _spec_backward(tight, xl, xh, gpu_layers_list, spec_ew,
                   remaining_specs, nh, device, dtype):
    """Spec backward pass using provided tight bounds.

    Returns (spec_lbs, still_open) without split_dim computation.
    """
    spec_lbs = {}
    for comp in remaining_specs:
        ew, b_spec = spec_ew[comp]
        ew = ew.clone()
        acc = b_spec
        for k in range(nh - 1, -1, -1):
            lo_k, hi_k = tight[k]
            lo_s, up_s, up_t, _, _, _ = _make_slopes(lo_k, hi_k)
            ep = ew.clamp(min=0)
            en = ew.clamp(max=0)
            acc += float((en * up_t).sum())
            ew = ep * lo_s + en * up_s
            glk = gpu_layers_list[k]
            if glk['type'] == 'conv':
                os_k = glk['out_shape']
                ew_4d = ew.reshape(1, *os_k)
                acc += float(
                    ew_4d.reshape(os_k[0], -1).sum(dim=1) @ glk['bias'])
                ew = F.conv_transpose2d(
                    ew_4d, glk['kernel'], stride=glk['stride'],
                    padding=glk['padding'],
                    output_padding=glk['output_padding']).flatten()
            else:
                acc += float(ew @ glk['bias'])
                ew = ew @ glk['W']
        spec_lbs[comp] = acc + float(
            ew.clamp(min=0) @ xl + ew.clamp(max=0) @ xh)
    still_open = {c for c in remaining_specs if spec_lbs[c] <= 0}
    return spec_lbs, still_open


def _fmt_eta(seconds):
    """Format ETA for display."""
    if seconds < 60:
        return '%.1fs' % seconds
    if seconds < 3600:
        return '%dm%02ds' % (int(seconds) // 60, int(seconds) % 60)
    if seconds < 86400:
        return '%dh%02dm' % (int(seconds) // 3600,
                             (int(seconds) % 3600) // 60)
    days = seconds / 86400
    if days > 99:
        return '>99days'
    return '%dd%02dh' % (int(days), int((seconds % 86400) / 3600))


def _run_bnb(evaluate_fn, pgd_fn, x_lo, x_hi, comps, settings):
    """Queue-based Branch-and-Bound loop.

    Returns ('verified', 'unknown', or 'sat', details_dict).
    """
    mode = settings.bnb_order
    print_progress = settings.print_progress
    timeout = settings.bnb_timeout

    if print_progress:
        print('=== BnB: %s + PGD-guided + progress tracking ===' % mode.upper())

    t_wall = time.perf_counter()

    # Initial PGD
    t0 = time.perf_counter()
    is_sat, witness, best_adv = pgd_fn(x_lo, x_hi, set(comps))
    t_pgd_init = time.perf_counter() - t0

    if is_sat:
        if print_progress:
            print('SAT found by initial PGD in %.1fms!' % (t_pgd_init * 1000))
        return 'sat', {'witness': witness, 'n_evals': 0,
                        'time': time.perf_counter() - t_wall}

    if print_progress:
        print('Initial PGD: UNSAT (%.1fms)' % (t_pgd_init * 1000))

    queue = [(x_lo.copy(), x_hi.copy(), set(comps), 0)]
    n_evals = 0
    n_verified = 0
    max_depth = 0
    volume_proven = 0.0
    depth_sum = 0
    depth_count = 0

    t_bab_start = time.perf_counter()

    while queue:
        if mode == 'dfs':
            x_l, x_h, remaining, depth = queue.pop(-1)
        else:
            x_l, x_h, remaining, depth = queue.pop(0)

        max_depth = max(max_depth, depth)

        if depth >= settings.bnb_max_depth:
            if print_progress:
                print('MAX DEPTH %d reached, giving up on this branch'
                      % settings.bnb_max_depth)
            continue

        spec_lbs, still_open, split_dim = evaluate_fn(x_l, x_h, remaining)
        n_evals += 1

        if not still_open:
            n_verified += 1
            volume_proven += 2.0 ** (-depth) if depth < 1024 else 0.0
            depth_sum += depth
            depth_count += 1
            if print_progress:
                avg_depth = depth_sum / depth_count
                elapsed = time.perf_counter() - t_bab_start
                eta = (elapsed * (1.0 - volume_proven) / volume_proven
                       if 0 < volume_proven < 1.0 else 0)
                print('UNSAT leaf d=%d | proven=%.1f%% | q=%d | evals=%d'
                      ' | elapsed=%.1fs | avg_d=%.1f | ETA=%s' % (
                          depth, volume_proven * 100, len(queue), n_evals,
                          elapsed, avg_depth, _fmt_eta(eta)))
            continue

        # PGD attack on subregion
        is_sat, witness, best_adv = pgd_fn(x_l, x_h, still_open)

        if is_sat:
            if print_progress:
                print('\nSAT! Counterexample at eval %d, depth %d'
                      % (n_evals, depth))
            return 'sat', {'witness': witness, 'n_evals': n_evals,
                            'time': time.perf_counter() - t_wall}

        # Split
        mid = (x_l[split_dim] + x_h[split_dim]) / 2
        xh1 = x_h.copy()
        xh1[split_dim] = mid
        xl2 = x_l.copy()
        xl2[split_dim] = mid

        adv_in_left = best_adv is not None and best_adv[split_dim] < mid

        if mode == 'dfs':
            if adv_in_left:
                queue.append((xl2, x_h, still_open, depth + 1))
                queue.append((x_l, xh1, still_open, depth + 1))
            else:
                queue.append((x_l, xh1, still_open, depth + 1))
                queue.append((xl2, x_h, still_open, depth + 1))
        else:
            if adv_in_left:
                queue.append((x_l, xh1, still_open, depth + 1))
                queue.append((xl2, x_h, still_open, depth + 1))
            else:
                queue.append((xl2, x_h, still_open, depth + 1))
                queue.append((x_l, xh1, still_open, depth + 1))

        if print_progress and (n_evals <= 5 or n_evals % 10 == 0):
            elapsed = time.perf_counter() - t_bab_start
            worst = min(spec_lbs[c] for c in still_open)
            print('split d=%d dim=%d | open=%d worst=%.4f | q=%d evals=%d'
                  ' | elapsed=%.1fs' % (depth, split_dim, len(still_open),
                                        worst, len(queue), n_evals, elapsed))

        if time.perf_counter() - t_bab_start > timeout:
            if print_progress:
                print('\nTIMEOUT %.0fs' % timeout)
            break

    t_total = time.perf_counter() - t_wall

    if print_progress:
        print('\nEvals: %d, Verified: %d, MaxDepth: %d, Queue: %d'
              % (n_evals, n_verified, max_depth, len(queue)))
        print('Volume proven: %.2f%%' % (volume_proven * 100))
        print('Total: %.1fms' % (t_total * 1000))

    if not queue and volume_proven >= 1.0 - 1e-9:
        return 'verified', {'n_evals': n_evals, 'time': t_total,
                             'volume_proven': volume_proven}
    return 'unknown', {'n_evals': n_evals, 'time': t_total,
                        'volume_proven': volume_proven, 'queue_remaining': len(queue)}


def zonotope_bnb_verify(graph, spec, settings=None):
    """BnB verification: forward zonotope + CROWN backward + input splitting.

    Args:
        graph: ComputeGraph loaded from ONNX
        spec: VNNSpec with input bounds and pairwise constraints
        settings: DotMap settings (or None for defaults)

    Returns:
        (result, details) where result is 'verified', 'unknown', or 'sat'
    """
    if settings is None:
        settings = default_settings()
    device, dtype = resolve_torch(settings)

    torch.set_num_threads(1)

    pw = spec.as_pairwise()
    assert pw is not None, (
        "BnB verification requires pairwise constraints (Y_comp >= Y_pred)")
    pred, comps = pw

    gpu_layers_list, fwd_data = graph.gpu_layers(device, dtype)
    nh = len(gpu_layers_list) - 1

    spec_ew = _build_spec_ew(gpu_layers_list, pred, comps, device, dtype)

    x_lo_np = spec.x_lo.astype(np.float32 if settings.bits == 32
                                else np.float64)
    x_hi_np = spec.x_hi.astype(np.float32 if settings.bits == 32
                                else np.float64)

    xl_g = torch.tensor(x_lo_np, dtype=dtype, device=device)
    xh_g = torch.tensor(x_hi_np, dtype=dtype, device=device)

    # Warmup
    _evaluate_region(xl_g, xh_g, set(comps), gpu_layers_list, spec_ew,
                     pred, nh, device, dtype)
    _pgd_attack(xl_g, xh_g, set(comps), pred, fwd_data, nh, settings)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    def evaluate_fn(x_l, x_h, remaining):
        xl_t = torch.tensor(x_l, dtype=dtype, device=device)
        xh_t = torch.tensor(x_h, dtype=dtype, device=device)
        return _evaluate_region(xl_t, xh_t, remaining, gpu_layers_list,
                                spec_ew, pred, nh, device, dtype)

    def pgd_fn(x_l, x_h, remaining):
        xl_t = torch.tensor(x_l, dtype=dtype, device=device)
        xh_t = torch.tensor(x_h, dtype=dtype, device=device)
        return _pgd_attack(xl_t, xh_t, remaining, pred, fwd_data, nh,
                           settings)

    return _run_bnb(evaluate_fn, pgd_fn, x_lo_np, x_hi_np, comps, settings)


def retarget_softmax_shifts(gg, x_center):
    """Point the softmax max-shift gathers at the row argmax of the
    CENTER input (alpha,beta-CROWN's fixed-index trick).

    The emission's k=0 placeholder is already exact (softmax is shift
    invariant for any fixed per-row index); retargeting only improves
    relaxation quality: with k = argmax, the shifted scores' upper
    bounds sit near 0, so exp operates in its tame range. A point
    forward records the (shifted) scores at the center; the row argmax
    of the shifted values equals the row argmax of the raw scores
    (constant per-row shift). Mutates flat_idx in place; call BEFORE
    any bound computation. No-op for graphs without softmax shifts."""
    shift_ops = [op for op in gg['ops']
                 if op['type'] == 'slice' and 'softmax_axis' in op]
    if not shift_ops:
        return 0
    ob = {}
    _forward_zonotope_graph(x_center, x_center, gg, x_center.device,
                            x_center.dtype, op_bounds=ob)
    n_re = 0
    for op in shift_ops:
        base = op['name'][:-len('__shift')]
        exp_name = f'{base}__exp'
        if exp_name not in ob:
            raise NotImplementedError(
                f'retarget_softmax_shifts: no recorded bounds for '
                f'{exp_name!r} — decomposition shape changed?')
        v = ob[exp_name][0].detach().cpu().numpy()
        sh = tuple(op['in_shapes_nd'][0])
        ax = int(op['softmax_axis'])
        pos = np.arange(int(np.prod(sh))).reshape(sh)
        k = v.reshape(sh).argmax(axis=ax, keepdims=True)
        flat_idx = np.take_along_axis(
            pos, np.broadcast_to(k, sh), axis=ax).reshape(-1)
        op['flat_idx'] = flat_idx
        n_re += 1
    return n_re
