"""RelaxLib: one relaxation object per elementwise nonlinearity (design 3.4).

Each entry owns everything the rest of the core needs to know about its op:

  point(x)                exact evaluation (torch)
  planes(lo, hi)          sound elementwise linear bounds on [lo, hi]:
                          (al, bl, au, bu) with al*x+bl <= f(x) <= au*x+bu.
                          Closed-form / provably bracketing ONLY, never
                          sampled (CLAUDE.md). Where a plane is optimizable,
                          the default is the sound midpoint choice; alpha
                          parameterization arrives with the backward pass.

Adversarial sampling VALIDATES planes in tests; it never defines them.
Ops are registered in REL by name; an unknown fn raises KeyError loudly.
"""
from __future__ import annotations

import torch


class Relu:
    def point(self, x, params=None):
        return torch.relu(x)

    def planes(self, lo, hi, params=None):
        """DeepZ/CROWN triangle: exact on stable neurons, slope=hi/(hi-lo)
        chord above, adaptive (0 or 1) tangent below on unstable ones."""
        unstable = (lo < 0) & (hi > 0)
        pos = lo >= 0
        # upper: chord through (lo, relu(lo)), (hi, relu(hi))
        denom = (hi - lo).clamp_min(1e-30)
        au = torch.where(unstable, hi / denom, (pos).to(lo.dtype))
        bu = torch.where(unstable, -hi * lo / denom, torch.zeros_like(lo))
        # lower: adaptive tangent y=0 or y=x, whichever is tighter (|lo| vs hi)
        al = torch.where(unstable, (hi >= -lo).to(lo.dtype), pos.to(lo.dtype))
        bl = torch.zeros_like(lo)
        return al, bl, au, bu

    def band(self, lo, hi, params=None):
        """DeepZ affine band (lam, mu, delta): f(x) in lam*x + mu +/- delta."""
        unstable = (lo < 0) & (hi > 0)
        lam = torch.where(unstable, hi / (hi - lo).clamp_min(1e-30),
                          (lo >= 0).to(lo.dtype))
        mu = torch.where(unstable,
                         -hi * lo / (hi - lo).clamp_min(1e-30) / 2,
                         torch.zeros_like(lo))
        return lam, mu, mu.clone()


class LeakyRelu:
    def point(self, x, params=None):
        alpha = (params or {}).get('alpha', 0.01)
        return torch.nn.functional.leaky_relu(x, alpha)


def _band(f, lo, hi, lam, crit_xs):
    """Closed-form affine band for smooth f: with slope lam, the deviation
    g(x) = f(x) - lam*x on [lo, hi] attains its extrema at the endpoints or
    at the finitely many stationary points f'(x) = lam (supplied in closed
    form via crit_xs). Returns (bl, bu) with lam*x+bl <= f(x) <= lam*x+bu.
    Ported from v1 nl_sigmoid_tanh._band_from_candidates (sound by
    construction; no sampling)."""
    g_lo, g_hi = f(lo) - lam * lo, f(hi) - lam * hi
    gmax = torch.maximum(g_lo, g_hi)
    gmin = torch.minimum(g_lo, g_hi)
    for xc in crit_xs:
        ok = (xc >= lo) & (xc <= hi) & torch.isfinite(xc)
        gx = torch.where(ok, f(xc) - lam * xc, gmin)
        gmin = torch.minimum(gmin, gx)
        gmax = torch.maximum(gmax, torch.where(ok, gx, gmax))
    return gmin, gmax


class _SShaped:
    """Shared plane construction for strictly increasing S-shaped ops
    (sigmoid, tanh): chord slope + closed-form critical points."""

    def planes(self, lo, hi, params=None):
        f = self.point
        lam = (f(hi) - f(lo)) / (hi - lo).clamp_min(1e-12)
        lam = torch.where(hi > lo, lam, self._slope_at(lo)).clamp_min(0.0)
        bl, bu = _band(f, lo, hi, lam, self._crit(lam))
        return lam, bl, lam, bu

    def band(self, lo, hi, params=None):
        al, bl, _au, bu = self.planes(lo, hi)
        return al, (bl + bu) / 2, (bu - bl) / 2


class Sigmoid(_SShaped):
    def point(self, x, params=None):
        return torch.sigmoid(x)

    def _slope_at(self, x):
        s = torch.sigmoid(x)
        return s * (1 - s)

    def _crit(self, lam):
        # f'(x) = s(1-s) = lam  =>  s = (1 +/- sqrt(1-4*lam))/2, x = logit(s)
        root = torch.sqrt((1 - 4 * lam).clamp_min(0.0))
        xs = []
        for s in ((1 + root) / 2, (1 - root) / 2):
            s = s.clamp(1e-12, 1 - 1e-12)
            xs.append(torch.log(s / (1 - s)))
        return xs


class Tanh(_SShaped):
    def point(self, x, params=None):
        return torch.tanh(x)

    def _slope_at(self, x):
        t = torch.tanh(x)
        return 1 - t * t

    def _crit(self, lam):
        # f'(x) = 1 - t^2 = lam  =>  t = +/- sqrt(1-lam), x = atanh(t)
        t = torch.sqrt((1 - lam).clamp_min(0.0)).clamp(max=1 - 1e-12)
        return [torch.atanh(t), torch.atanh(-t)]


class _V1Band:
    """Adapter over a v1 ScalarNonlinearRelax (nl_sin/nl_cos/nl_pow): those
    modules already provide the closed-form affine band (lam, mu, delta)
    with exhaustive critical-point enumeration; this is that ONE
    implementation behind the RelaxLib interface."""

    def _rel(self, params=None):
        raise NotImplementedError

    def band(self, lo, hi, params=None):
        lam, mu, delta = self._rel(params).affine_band(lo, hi)
        return (lam.to(lo.dtype), mu.to(lo.dtype), delta.to(lo.dtype))

    def planes(self, lo, hi, params=None):
        lam, mu, delta = self.band(lo, hi, params)
        return lam, mu - delta, lam, mu + delta


class Sin(_V1Band):
    def point(self, x, params=None):
        return torch.sin(x)

    def _rel(self, params=None):
        from vibecheck.nl_sin import SinRelax
        return SinRelax()


class Cos(_V1Band):
    def point(self, x, params=None):
        return torch.cos(x)

    def _rel(self, params=None):
        from vibecheck.nl_cos import CosRelax
        return CosRelax()


class Exp:
    def point(self, x, params=None):
        return torch.exp(x)


class Pow(_V1Band):
    def point(self, x, params=None):
        return x ** (params or {})['exponent']

    def _rel(self, params=None):
        from vibecheck.nl_pow import PowRelax
        return PowRelax((params or {})['exponent'])


class SignFn:
    def point(self, x, params=None):
        return torch.sign(x)


class Floor:
    def point(self, x, params=None):
        return torch.floor(x)


REL = {'relu': Relu(), 'leaky_relu': LeakyRelu(), 'sigmoid': Sigmoid(),
       'tanh': Tanh(), 'sin': Sin(), 'cos': Cos(), 'exp': Exp(),
       'pow': Pow(), 'sign': SignFn(), 'floor': Floor()}
