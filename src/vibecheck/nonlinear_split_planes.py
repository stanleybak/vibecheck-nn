"""Per-op relaxation planes and how they CHANGE under a nonlinear split — the
foundation for the dual-ascent nonlinear-split BaB.

Every supported 1-D activation is relaxed over a pre-activation interval [lo,hi]
by a PARALLELOGRAM = two linear planes:
    lower:  f(x) >= s_L*x + t_L
    upper:  f(x) <= s_U*x + t_U          for all x in [lo, hi].

A split at point p replaces [lo,hi] with children [lo,p] and [p,hi]; the planes
are RE-EVALUATED on each sub-interval (tighter by construction) — that is the
entire "relaxation change under split" (plus a x<=p / x>=p halfspace the dual
adds as a beta term; that lives in the BaB, not here).

Two plane FORMS, both sound (ablated by the BaB):
  - 'band'      : parallel planes from the affine band (lam*x + mu +- delta);
                  one slope, alpha-tunable; consistent with the forward
                  alpha-CROWN. s_L=s_U=lam, t_L=mu-delta, t_U=mu+delta.
  - 'two_plane' : non-parallel tangent + secant (ABC's bound_relax). For a
                  CONVEX element: lower = tangent at an alpha-chosen point,
                  upper = secant chord. CONCAVE: swapped. MIXED-curvature
                  elements fall back to the (always-sound) band.

Soundness holds for ANY alpha (the band recomputes a sound mu/delta; the
tangent/secant bracket a convex/concave f). Validated in
tests/test_split_planes.py by dense sampling.
"""
import torch


def op_planes(relax, lo, hi, alpha=None, form='band'):
    """Return (s_L, t_L, s_U, t_U): planes bounding f over [lo, hi].

    relax: a ScalarNonlinearRelax (func, slope_at, affine_band[_alpha],
    curvature). alpha: per-element tangent/slope fraction in [0,1] (or None ->
    chord). form: 'band' or 'two_plane'.
    """
    lo = torch.as_tensor(lo, dtype=torch.float64)
    hi = torch.as_tensor(hi, dtype=torch.float64)
    lo, hi = torch.broadcast_tensors(lo, hi)
    lo = lo.contiguous(); hi = hi.contiguous()

    # band planes (always available, sound for any curvature)
    if alpha is not None:
        lam, mu, delta = relax.affine_band_alpha(lo, hi, alpha)
    else:
        lam, mu, delta = relax.affine_band(lo, hi)
    lam = torch.as_tensor(lam, dtype=torch.float64)
    mu = torch.as_tensor(mu, dtype=torch.float64)
    delta = torch.as_tensor(delta, dtype=torch.float64).clamp(min=0)
    b_sL, b_tL, b_sU, b_tU = lam, mu - delta, lam, mu + delta
    if form == 'band':
        return b_sL, b_tL, b_sU, b_tU

    # two_plane: tangent + secant, chosen by per-element curvature.
    width = (hi - lo).clamp(min=1e-300)
    f_lo = relax.func(lo); f_hi = relax.func(hi)
    s_sec = (f_hi - f_lo) / width            # secant slope
    t_sec = f_lo - s_sec * lo
    a = (alpha.clamp(0.0, 1.0) if torch.is_tensor(alpha)
         else (0.5 if alpha is None else float(alpha)))
    t_pt = lo + a * (hi - lo)                # tangent point (alpha-chosen)
    s_tan = relax.slope_at(t_pt)
    t_tan = relax.func(t_pt) - s_tan * t_pt
    code = relax.curvature(lo, hi)           # 0 convex, 1 concave, 2 mixed
    convex = code < 0.5
    concave = (code >= 0.5) & (code < 1.5)
    # convex -> lower=tangent, upper=secant ; concave -> swapped ;
    # mixed -> fall back to the sound parallel band.
    s_L = torch.where(convex, s_tan, torch.where(concave, s_sec, b_sL))
    t_L = torch.where(convex, t_tan, torch.where(concave, t_sec, b_tL))
    s_U = torch.where(convex, s_sec, torch.where(concave, s_tan, b_sU))
    t_U = torch.where(convex, t_sec, torch.where(concave, t_tan, b_tU))
    return s_L, t_L, s_U, t_U


# Option A — split-point rule. 1-D ops whose relaxation has a kink / inflection
# / min AT 0 get split THERE when the pre-activation straddles 0 (it separates
# the two monotone/convex pieces — a big tightening). Periodic Sin/Cos and the
# operands of a bilinear Mul get the MIDPOINT: zero is not special for them and
# is gap-suboptimal when the interval is asymmetric (see the option-A analysis —
# McCormick/zono-box gap ∝ Δa·Δb, minimized by the midpoint cut).
_ZERO_SPLIT_OP_TYPES = frozenset({'sigmoid', 'tanh', 'pow', 'relu'})


def split_point(op_type, lo, hi):
    """Option-A split point for a 1-D op element over [lo, hi]: 0 if the op has
    a feature at 0 and the range straddles it, else the midpoint. Scalars or
    tensors (element-wise)."""
    if isinstance(lo, float):
        if op_type in _ZERO_SPLIT_OP_TYPES and lo < 0.0 < hi:
            return 0.0
        return 0.5 * (lo + hi)
    import torch as _t
    mid = 0.5 * (lo + hi)
    if op_type in _ZERO_SPLIT_OP_TYPES:
        straddle = (lo < 0.0) & (hi > 0.0)
        return _t.where(straddle, _t.zeros_like(mid), mid)
    return mid


def bilinear_axis_score(rad_a, rad_b, sens_a=1.0, sens_b=1.0):
    """Option-A operand choice for a bilinear Mul w=a·b (a TIEBREAK, not a
    gap-size decision).

    The McCormick / zono-box gap is ∝ radA·radB, and after K midpoint splits it
    is radA·radB·2^-K REGARDLESS of how the K splits are allocated between a and
    b — so bisecting either axis reduces the gap equally; there is no
    "more-gap" axis. The real risk is STARVATION: always cutting `a` leaves radB
    untouched (the gap still →0 via radA→0, but slowly and lopsidedly, and any
    spec component sensitive to b alone never tightens). So balance: split the
    axis with the larger (sensitivity-weighted) radius, keeping radA≈radB and
    driving the product →0 evenly. Halving the winner flips the choice to the
    other axis next round. Returns ('a' or 'b', score)."""
    sa = float(sens_a) * float(rad_a)
    sb = float(sens_b) * float(rad_b)
    return ('a', sa) if sa >= sb else ('b', sb)


def split_planes(relax, lo, hi, p, alpha_l=None, alpha_r=None, form='band'):
    """How the parallelogram relaxation CHANGES under a split at p: returns
    (left_planes, right_planes), each a (s_L,t_L,s_U,t_U) tuple — the op's
    planes re-evaluated over [lo,p] and [p,hi]. Sound for any alpha; the two
    children's input domains cover [lo,hi] (x<=p or x>=p), so combining their
    certified bounds covers the parent (the BaB adds the halfspace beta term)."""
    left = op_planes(relax, lo, p, alpha_l, form)
    right = op_planes(relax, p, hi, alpha_r, form)
    return left, right
