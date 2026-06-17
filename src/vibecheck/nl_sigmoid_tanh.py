"""Sound zonotope/interval relaxations for elementwise Sigmoid and Tanh.

Both are smooth, strictly increasing, S-shaped (derivative is unimodal, peaking
at x=0), so for a chosen slope ``lam`` the deviation ``g(x) = f(x) - lam*x`` has
at most TWO interior stationary points (where ``f'(x) = lam``) plus the two
endpoints. We enumerate them in closed form (no sampling) and set
``mu = (gmax+gmin)/2``, ``delta = (gmax-gmin)/2`` over the in-range candidates —
sound by construction (the extrema of a smooth g on a closed interval are at the
endpoints or stationary points).

This is the affine-band DeepZ transformer (`zono_affine_transform`) and is far
tighter than the historical box-collapse (`new_g = diag(mu)`, which discards ALL
input correlation): it scales the existing generators by ``lam`` (preserving
input correlation) and adds one fresh ``delta`` error generator per element.

Critical points:
  - sigmoid: f'(x) = s(1-s) where s = sigmoid(x); f'(x)=lam => s = (1±√(1-4λ))/2
    (real for 0 ≤ λ ≤ 1/4, which always holds since max f' = 1/4), x = logit(s).
  - tanh:    f'(x) = 1 - t² where t = tanh(x); f'(x)=lam => t = ±√(1-λ)
    (real for 0 ≤ λ ≤ 1), x = atanh(t).
"""
import torch

from .nonlinear_relax import ScalarNonlinearRelax, register

_EPS = 1e-12


def _band_from_candidates(f, lo, hi, lam, crit_xs):
    """g(x)=f(x)-lam*x; extrema over {lo, hi} ∪ (in-range crit_xs). Returns
    (mu, delta). crit_xs is a list of element-wise candidate tensors."""
    def g(x):
        return f(x) - lam * x
    g_lo = g(lo)
    g_hi = g(hi)
    gmax = torch.maximum(g_lo, g_hi)
    gmin = torch.minimum(g_lo, g_hi)
    for xc in crit_xs:
        in_range = (xc >= lo) & (xc <= hi) & torch.isfinite(xc)
        if not bool(in_range.any()):
            continue
        gc = g(xc)
        gmax = torch.where(in_range, torch.maximum(gmax, gc), gmax)
        gmin = torch.where(in_range, torch.minimum(gmin, gc), gmin)
    mu = 0.5 * (gmax + gmin)
    delta = (0.5 * (gmax - gmin)).clamp_min(0.0)
    return mu, delta


@register('Sigmoid')
class SigmoidRelax(ScalarNonlinearRelax):
    def func(self, x):
        return torch.sigmoid(x)

    def interval(self, lo, hi):
        # monotone increasing => endpoints bracket the range
        lo = torch.as_tensor(lo, dtype=torch.float64)
        hi = torch.as_tensor(hi, dtype=torch.float64)
        return torch.sigmoid(lo), torch.sigmoid(hi)

    def slope_at(self, x):
        s = torch.sigmoid(torch.as_tensor(x, dtype=torch.float64))
        return s * (1.0 - s)

    def curvature(self, lo, hi):
        # sigmoid: convex on x<=0, concave on x>=0 (inflection at 0).
        lo = torch.as_tensor(lo, dtype=torch.float64)
        hi = torch.as_tensor(hi, dtype=torch.float64)
        lo, hi = torch.broadcast_tensors(lo, hi)
        return torch.where(hi <= 0.0, torch.zeros_like(lo),
                           torch.where(lo >= 0.0, torch.ones_like(lo),
                                       2.0 * torch.ones_like(lo)))

    def affine_band(self, lo, hi, lam=None):
        lo = torch.as_tensor(lo, dtype=torch.float64)
        hi = torch.as_tensor(hi, dtype=torch.float64)
        lo, hi = torch.broadcast_tensors(lo, hi)
        lo = lo.contiguous(); hi = hi.contiguous()
        if lam is None:
            width = hi - lo
            degenerate = width <= 0.0
            denom = torch.where(degenerate, torch.ones_like(width), width)
            s_lo = torch.sigmoid(lo)
            lam = torch.where(degenerate, s_lo * (1.0 - s_lo),
                              (torch.sigmoid(hi) - s_lo) / denom)
        # s(1-s)=lam => s = (1±√(1-4λ))/2 ; x = logit(s). Real for lam<=1/4
        # (always so for an interpolated σ' slope); clamp guards the boundary.
        disc = (1.0 - 4.0 * lam).clamp_min(0.0)
        sq = torch.sqrt(disc)
        s1 = ((1.0 + sq) * 0.5).clamp(_EPS, 1.0 - _EPS)
        s2 = ((1.0 - sq) * 0.5).clamp(_EPS, 1.0 - _EPS)
        x1 = torch.log(s1) - torch.log1p(-s1)
        x2 = torch.log(s2) - torch.log1p(-s2)
        mu, delta = _band_from_candidates(torch.sigmoid, lo, hi, lam, [x1, x2])
        return lam, mu, delta


@register('Tanh')
class TanhRelax(ScalarNonlinearRelax):
    def func(self, x):
        return torch.tanh(x)

    def interval(self, lo, hi):
        lo = torch.as_tensor(lo, dtype=torch.float64)
        hi = torch.as_tensor(hi, dtype=torch.float64)
        return torch.tanh(lo), torch.tanh(hi)

    def slope_at(self, x):
        t = torch.tanh(torch.as_tensor(x, dtype=torch.float64))
        return 1.0 - t * t

    def curvature(self, lo, hi):
        # tanh: convex on x<=0, concave on x>=0 (inflection at 0).
        lo = torch.as_tensor(lo, dtype=torch.float64)
        hi = torch.as_tensor(hi, dtype=torch.float64)
        lo, hi = torch.broadcast_tensors(lo, hi)
        return torch.where(hi <= 0.0, torch.zeros_like(lo),
                           torch.where(lo >= 0.0, torch.ones_like(lo),
                                       2.0 * torch.ones_like(lo)))

    def affine_band(self, lo, hi, lam=None):
        lo = torch.as_tensor(lo, dtype=torch.float64)
        hi = torch.as_tensor(hi, dtype=torch.float64)
        lo, hi = torch.broadcast_tensors(lo, hi)
        lo = lo.contiguous(); hi = hi.contiguous()
        if lam is None:
            width = hi - lo
            degenerate = width <= 0.0
            denom = torch.where(degenerate, torch.ones_like(width), width)
            t_lo = torch.tanh(lo)
            lam = torch.where(degenerate, 1.0 - t_lo * t_lo,
                              (torch.tanh(hi) - t_lo) / denom)
        # 1 - t² = lam => t = ±√(1-λ) ; x = atanh(t)
        val = torch.sqrt((1.0 - lam).clamp_min(0.0)).clamp(max=1.0 - _EPS)
        x1 = torch.atanh(val)
        x2 = -x1
        mu, delta = _band_from_candidates(torch.tanh, lo, hi, lam, [x1, x2])
        return lam, mu, delta
