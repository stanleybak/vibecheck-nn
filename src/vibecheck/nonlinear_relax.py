"""Generic sound relaxations for elementwise scalar nonlinearities.

Used to give vibecheck sound zonotope/interval bounds for the nonlinear ops in
benchmarks like ml4acopf (Sin, Cos, Pow, Floor, ...). Each op is a
`ScalarNonlinearRelax` subclass providing, for an elementwise function f and a
per-element input interval [lo, hi]:

  - `func(x)`              : the exact f (torch), for validation/sampling.
  - `interval(lo, hi)`     : a SOUND output interval (out_lo, out_hi).
  - `affine_band(lo, hi)`  : a SOUND affine over-approximation (lam, mu, delta)
        with   |f(x) - (lam*x + mu)| <= delta   for ALL x in [lo, hi],
        delta >= 0, all three tensors element-wise the same shape as lo/hi.

SOUNDNESS IS BY CONSTRUCTION — closed form / monotonicity / convexity /
critical-point (root of f'(x) = lam) analysis. NEVER sample to derive a bound
(the worst case can lie between samples). `assert_band_sound` /
`assert_interval_sound` below sample only to TEST a provided closed-form band,
which is the correct use of sampling per the project soundness rule.

The zonotope transformer built on `affine_band` (wired into the forward pass)
is the DeepZ-style elementwise transformer:
    y_center = lam*center + mu ;  y_gens = lam .* existing_gens ;
    + one FRESH error generator of magnitude delta per element
which is sound: for every noise assignment, f(z) stays within the band.
"""
import torch

# op_type (ONNX string) -> ScalarNonlinearRelax subclass
REGISTRY = {}


def register(onnx_op):
    """Class decorator: register a relaxation under its ONNX op type."""
    def deco(cls):
        cls.onnx_op = onnx_op
        REGISTRY[onnx_op] = cls
        return cls
    return deco


class ScalarNonlinearRelax:
    """Base class for an elementwise scalar-nonlinearity relaxation.

    Subclasses MUST implement `func`, `interval`, and `affine_band`. All three
    operate element-wise and broadcast; `lo`/`hi` are torch tensors and the
    returned tensors share their shape.
    """
    onnx_op = None

    def func(self, x):
        """Exact elementwise function (torch tensor -> torch tensor)."""
        raise NotImplementedError

    def interval(self, lo, hi):
        """Sound output interval: returns (out_lo, out_hi), each elementwise
        bounding f over [lo, hi]. Must be sound by construction."""
        raise NotImplementedError

    def affine_band(self, lo, hi, lam=None):
        """Sound affine over-approximation. Returns (lam, mu, delta) with
        |f(x) - (lam*x + mu)| <= delta  for all x in [lo, hi], delta >= 0.
        Must be sound by construction (no sampling).

        ``lam`` (optional): use THIS slope instead of the default chord slope.
        ANY real lam is sound — mu/delta are recomputed as the exact midpoint /
        half-range of g(x)=f(x)-lam*x over [lo,hi] (endpoints + critical points
        where f'(x)=lam). This is the α-CROWN hook: a differentiable lam lets
        gradient pick the slope that maximises the downstream margin."""
        raise NotImplementedError

    def slope_at(self, x):
        """f'(x) — used by ``affine_band_alpha`` to span a sound slope range."""
        raise NotImplementedError

    def curvature(self, lo, hi):
        """Per-element curvature code over [lo,hi]: 0=convex, 1=concave,
        2=mixed. Used by the split-relaxation two-plane form to pick
        tangent-vs-secant. Default = mixed (always falls back to the parallel
        band, which is sound for any curvature)."""
        lo = torch.as_tensor(lo, dtype=torch.float64)
        hi = torch.as_tensor(hi, dtype=torch.float64)
        lo, hi = torch.broadcast_tensors(lo, hi)
        return 2.0 * torch.ones_like(lo)

    def affine_band_alpha(self, lo, hi, alpha):
        """α-parametrised band: lam = (1-α)·f'(lo) + α·f'(hi), α in [0,1].
        Differentiable in α; sound for every α (``affine_band`` recomputes a
        sound mu/delta for whatever lam it is given). Mirrors the convex-op
        α-zono path (exp/reciprocal) used by ``_zono_alpha_close``."""
        a = alpha.clamp(0.0, 1.0) if torch.is_tensor(alpha) else alpha
        d_lo = self.slope_at(lo)
        d_hi = self.slope_at(hi)
        lam = d_lo + a * (d_hi - d_lo)
        return self.affine_band(lo, hi, lam=lam)


def zono_affine_transform(relax, center, generators, rel_pad=None,
                          tight_lo=None, tight_hi=None, alpha=None,
                          return_band=False):
    """Sound DeepZ elementwise transformer for f, using relax.affine_band.

    Zonotope layout: center (n,), generators (n, k) [row i = element i's gens].
    Returns (new_center (n,), new_gens (n, k+n)); with ``return_band=True`` also
    returns the per-element band ``(lam, mu, delta+pad)`` actually used — the
    EXACT parent geometry the dual-ascent nonlinear-split state needs (delta is
    returned WITH the float-rounding pad, so it equals the fresh e_new column
    magnitude in new_gens; the dual sensitivity g_k = d[e_new]/that value is then
    exact). All three are cast to ``center.dtype``.

    For z_i = center_i + generators_i . e (e in [-1,1]^k), with affine band
    |f(x) - (lam_i x + mu_i)| <= delta_i over [lo_i, hi_i] = z_i's range:
        f(z_i) = (lam_i center_i + mu_i) + lam_i (generators_i . e) + delta_i e'_i
    i.e. scale the existing gens by lam (preserves input correlation, unlike a
    box collapse) and add ONE fresh error generator of magnitude delta per
    element. Sound for any noise assignment.

    ``tight_lo``/``tight_hi`` (per-element, optional): intersect the band's input
    range with [tight_lo, tight_hi] — the nonlinear-split clamp. The band is then
    computed over the TIGHTER range (smaller delta, often different lam) but still
    applied to the FULL zonotope. This is sound for the sub-domain where the op's
    input lies in the clamp: on that sub-domain every (x, f(x)) is inside the
    clamped band, so the resulting zonotope's bound over-approximates f there; the
    two children of a split (clamp [lo,m] and [m,hi]) cover the parent's range, so
    "both children verified ⟹ parent verified" (same argument as ReLU
    ``tight_bounds`` / the exp/reciprocal op_clamps path).

    The band is computed in float64 then cast (avoids float32 rounding making
    delta too small); `rel_pad` adds a small value-relative slack covering the
    float32 rounding of lam*center+mu (sound inflation; ~1e-6 at ACOPF scales).
    """
    dt = center.dtype
    if rel_pad is None:
        # The pad is a sound inflation covering the FLOAT ROUNDING of
        # lam*center+mu. It was sized ~1e-6 for float32 (eps~1.2e-7), but that
        # is ~10 orders too large for the float64 path: a CONSTANT ~1e-6
        # inflation (independent of lam/alpha) then dwarfs sub-1e-6 spec margins
        # and caps the achievable bound — e.g. ml4acopf full prop3 plateaued at
        # -1.56e-7 (within-tol sat) no matter how alpha tuned the slope, because
        # no slope removes the pad. Scale the pad to the working dtype's epsilon:
        # 1e-6 for float32, 1e-12 for float64 (still ~4 orders above float64
        # eps~2.2e-16, so soundly covers float64 rounding while not swamping the
        # margin). Callers may still pass an explicit rel_pad.
        rel_pad = 1e-6 if dt == torch.float32 else 1e-12
    rad = generators.abs().sum(dim=1)
    lo = (center - rad).double()
    hi = (center + rad).double()
    if tight_lo is not None:
        lo = torch.maximum(lo, torch.as_tensor(
            tight_lo, dtype=torch.float64, device=center.device))
    if tight_hi is not None:
        hi = torch.minimum(hi, torch.as_tensor(
            tight_hi, dtype=torch.float64, device=center.device))
        # empty intersection (infeasible sub-domain) -> collapse to a point;
        # any bound is sound there since no real input reaches it.
        hi = torch.maximum(hi, lo)
    if alpha is not None:
        # α-CROWN: differentiable slope; gradient flows lam<-alpha<-spec margin.
        a = alpha.double() if torch.is_tensor(alpha) else alpha
        lam, mu, delta = relax.affine_band_alpha(lo, hi, a)
    else:
        lam, mu, delta = relax.affine_band(lo, hi)
    lam = torch.as_tensor(lam, dtype=dt, device=center.device)
    mu = torch.as_tensor(mu, dtype=dt, device=center.device)
    delta = torch.as_tensor(delta, dtype=dt, device=center.device).clamp(min=0)
    new_center = lam * center + mu
    new_gens = lam.unsqueeze(-1) * generators
    pad = rel_pad * (new_center.abs() + new_gens.abs().sum(dim=1) + 1.0)
    err = torch.diag(delta + pad)
    out = (new_center, torch.cat([new_gens, err], dim=1))
    if return_band:
        return (*out, (lam, mu, delta + pad))
    return out


def _as64(*ts):
    return tuple(torch.as_tensor(t, dtype=torch.float64) for t in ts)


def assert_band_sound(relax, lo, hi, n_samples=50000, atol=1e-6):
    """TEST helper: densely sample [lo, hi] (+ endpoints) and assert the
    affine band holds everywhere. Returns the worst observed gap. This TESTS a
    closed-form band; it does not produce one."""
    lo, hi = _as64(lo, hi)
    lam, mu, delta = _as64(*relax.affine_band(lo, hi))
    assert bool((delta >= -atol).all()), 'delta must be >= 0'
    u = torch.rand(n_samples, *lo.shape, dtype=torch.float64)
    u = torch.cat([u, torch.zeros((1,) + tuple(lo.shape), dtype=torch.float64),
                   torch.ones((1,) + tuple(lo.shape), dtype=torch.float64)], dim=0)
    xs = lo + (hi - lo) * u
    gap = (relax.func(xs) - (lam * xs + mu)).abs()
    worst = float(gap.max())
    assert bool((gap <= delta + atol).all()), (
        f'AFFINE BAND VIOLATED for {relax.onnx_op}: worst gap {worst:.3e} '
        f'> delta {float(delta.max()):.3e}')
    return worst


def assert_interval_sound(relax, lo, hi, n_samples=50000, atol=1e-6):
    """TEST helper: assert interval(lo, hi) contains every sampled f(x)."""
    lo, hi = _as64(lo, hi)
    olo, ohi = _as64(*relax.interval(lo, hi))
    u = torch.rand(n_samples, *lo.shape, dtype=torch.float64)
    xs = lo + (hi - lo) * u
    f = relax.func(xs)
    assert bool((f >= olo - atol).all()) and bool((f <= ohi + atol).all()), (
        f'INTERVAL NOT SOUND for {relax.onnx_op}')


# Import the per-op relaxation modules so their @register decorators run and
# populate REGISTRY on first import of this module. Done at the bottom to
# avoid a circular import (each nl_* module imports `register` from here).
# Without this, consumers (forward zono sin/cos/floor, pow) saw an empty
# REGISTRY → KeyError on the first ml4acopf Floor/Sin/Cos/Pow op.
from . import nl_sin, nl_cos, nl_floor, nl_pow  # noqa: E402,F401
from . import nl_sigmoid_tanh  # noqa: E402,F401
