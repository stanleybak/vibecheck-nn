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
    """Piecewise-linear with slope a<1 on x<0: convex, so tangents (slope a
    or 1, both through 0) bound below and the chord bounds above."""

    def point(self, x, params=None):
        alpha = (params or {}).get('alpha', 0.01)
        return torch.nn.functional.leaky_relu(x, alpha)

    def planes(self, lo, hi, params=None):
        a = float((params or {}).get('alpha', 0.01))
        if a > 1.0:
            raise NotImplementedError('leaky_relu with slope > 1 (concave)')
        unstable = (lo < 0) & (hi > 0)
        pos = lo >= 0
        denom = (hi - lo).clamp_min(1e-30)
        chord = (hi - a * lo) / denom
        au = torch.where(unstable, chord,
                         torch.where(pos, torch.ones_like(lo),
                                     torch.full_like(lo, a)))
        bu = torch.where(unstable, hi * (1 - chord), torch.zeros_like(lo))
        al = torch.where(unstable,
                         torch.where(hi >= -lo, torch.ones_like(lo),
                                     torch.full_like(lo, a)),
                         au)
        bl = torch.zeros_like(lo)
        return al, bl, au, bu

    def band(self, lo, hi, params=None):
        al, bl, au, bu = self.planes(lo, hi, params)
        # single-slope band at the chord: deviation spans [0, bu]
        return au, bu / 2, bu / 2


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
    """Plane construction for strictly increasing S-shaped ops (sigmoid,
    tanh; inflection at 0, convex left, concave right).

    Asymmetric CROWN planes per region:
      convex side (hi <= 0):  lower = tangent at midpoint, upper = chord
      concave side (lo >= 0): lower = chord, upper = tangent at midpoint
      crossing:  lower = tangent from (lo, f(lo)) touching the concave arm,
                 upper = tangent from (hi, f(hi)) touching the convex arm,
                 tangent point found by bisection CONVERGING FROM THE SAFE
                 SIDE (a tangent point further out only rotates the line
                 away from f), so the result is sound at any precision.
    """

    def _cross_lower_point(self, lo, hi, iters=30):
        """Safe tangent point d <= 0 for the crossing-region LOWER plane
        (line through (hi, f(hi)) tangent to the convex arm). Any d' in
        [d, 0] has slope >= the critical tangent slope, hence stays sound;
        the bisection returns the sound endpoint of its bracket."""
        f, fp = self.point, self._slope_at
        y1 = f(hi)
        a = -torch.full_like(lo, 24.0)
        b = torch.zeros_like(lo)
        for _ in range(iters):
            d = (a + b) / 2
            sound = y1 + fp(d) * (d - hi) <= f(d)     # slope already >= s*
            b = torch.where(sound, d, b)
            a = torch.where(sound, a, d)
        return b

    def _cross_upper_point(self, lo, hi, iters=30):
        """Safe tangent point d >= 0 for the crossing-region UPPER plane
        (line through (lo, f(lo)) tangent to the concave arm); any d' in
        [0, d] stays sound."""
        f, fp = self.point, self._slope_at
        y0 = f(lo)
        a = torch.zeros_like(lo)
        b = torch.full_like(lo, 24.0)
        for _ in range(iters):
            d = (a + b) / 2
            sound = y0 + fp(d) * (d - lo) >= f(d)     # slope already >= s*
            a = torch.where(sound, d, a)
            b = torch.where(sound, b, d)
        return a

    def planes(self, lo, hi, params=None, t_low=None, t_up=None):
        """Sound planes; t_low/t_up in [0,1] are OPTIONAL tangent-position
        parameters (alpha): every value in [0,1] yields a sound plane, so
        the optimizer may move them freely.

        one-sided regions: the tangent point slides across [lo, hi];
        crossing: the tangent point slides from the safe bisection bracket
        toward the inflection (slope >= the critical tangent slope stays
        sound; see _cross_lower/_cross_upper)."""
        f = self.point
        # defaults: midpoint tangent on one-sided regions, the exact
        # anchored tangent (t=0) on crossing ones (tightest of the family)
        t_side_l = 0.5 if t_low is None else t_low
        t_side_u = 0.5 if t_up is None else t_up
        t_cross_l = 0.0 if t_low is None else t_low
        t_cross_u = 0.0 if t_up is None else t_up
        chord = (f(hi) - f(lo)) / (hi - lo).clamp_min(1e-12)
        chord = torch.where(hi > lo, chord, self._slope_at(lo)).clamp_min(0.0)
        convex = hi <= 0
        concave = lo >= 0
        tl = lo + t_side_l * (hi - lo)       # tangent positions (broadcast)
        tu = lo + t_side_u * (hi - lo)
        sl_t, su_t = self._slope_at(tl), self._slope_at(tu)
        al = torch.where(convex, sl_t, chord * torch.ones_like(sl_t))
        bl = torch.where(convex, f(tl) - sl_t * tl, f(lo) - chord * lo)
        au = torch.where(concave, su_t, chord * torch.ones_like(su_t))
        bu = torch.where(concave, f(tu) - su_t * tu, f(lo) - chord * lo)
        crossing = ~(convex | concave)
        if bool(crossing.any()):
            dl = self._cross_lower_point(lo, hi)
            d = dl * (1 - t_cross_l)         # slide toward 0: slope >= s*
            s = self._slope_at(d)
            al = torch.where(crossing, s, al)
            bl = torch.where(crossing, f(hi) - s * hi, bl)
            du = self._cross_upper_point(lo, hi)
            d2 = du * (1 - t_cross_u)
            s2 = self._slope_at(d2)
            au = torch.where(crossing, s2, au)
            bu = torch.where(crossing, f(lo) - s2 * lo, bu)
        return al, bl, au, bu

    def alpha_planes(self, lo, hi, alpha, params=None):
        """planes() with optimizer-controlled tangent positions; alpha is
        (..., 2, n): channel 0 moves the lower plane, channel 1 the upper."""
        return self.planes(lo, hi, params, t_low=alpha[..., 0, :],
                           t_up=alpha[..., 1, :])

    def band(self, lo, hi, params=None):
        # single-slope zono band: chord + closed-form critical points
        f = self.point
        lam = (f(hi) - f(lo)) / (hi - lo).clamp_min(1e-12)
        lam = torch.where(hi > lo, lam, self._slope_at(lo)).clamp_min(0.0)
        bl, bu = _band(f, lo, hi, lam, self._crit(lam))
        return lam, (bl + bu) / 2, (bu - bl) / 2


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
    """Convex: chord above; below, the tangent whose slope equals the chord
    (at x* = ln(chord)), the tightest same-slope band."""

    def point(self, x, params=None):
        return torch.exp(x)

    def planes(self, lo, hi, params=None):
        w = (hi - lo).clamp_min(1e-12)
        chord = (torch.exp(hi) - torch.exp(lo)) / w
        chord = torch.where(hi > lo, chord, torch.exp(lo))
        bu = torch.exp(lo) - chord * lo
        # exp(x) >= chord*x + chord*(1 - ln(chord))  (tangent at ln(chord))
        xstar = torch.log(chord.clamp_min(1e-30))
        bl = chord * (1 - xstar)
        return chord, bl, chord, bu

    def band(self, lo, hi, params=None):
        al, bl, _au, bu = self.planes(lo, hi)
        return al, (bl + bu) / 2, (bu - bl) / 2


class Reciprocal:
    """1/y on sign-definite ranges: convex for y>0 (tangent below, chord
    above), concave for y<0 (mirrored). A range straddling 0 has no sound
    linear relaxation and raises loudly."""

    def point(self, x, params=None):
        return 1.0 / x

    def planes(self, lo, hi, params=None):
        if bool(((lo <= 0) & (hi >= 0)).any()):
            raise NotImplementedError(
                'reciprocal over a range containing 0 is unbounded')
        m = (lo + hi) / 2
        tan_a = -1.0 / (m * m)
        tan_b = 2.0 / m
        chord_a = -1.0 / (lo * hi)
        chord_b = 1.0 / lo + 1.0 / hi
        pos = lo > 0
        al = torch.where(pos, tan_a, chord_a)
        bl = torch.where(pos, tan_b, chord_b)
        au = torch.where(pos, chord_a, tan_a)
        bu = torch.where(pos, chord_b, tan_b)
        return al, bl, au, bu

    def band(self, lo, hi, params=None):
        # chord slope band: g(y) = 1/y - lam*y has its interior stationary
        # point at y = -1/sqrt(-lam... ) i.e. g'(y) = -1/y^2 - lam = 0 ->
        # y* = +/- sqrt(-1/lam) (lam < 0 always for 1/y on sign-definite y)
        lam = -1.0 / (lo * hi)
        y_star = torch.sqrt((-1.0 / lam).clamp_min(1e-30))
        y_star = torch.where(lo > 0, y_star, -y_star)
        bl, bu = _band(lambda y: 1.0 / y, lo, hi, lam, [y_star])
        return lam, (bl + bu) / 2, (bu - bl) / 2


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
       'pow': Pow(), 'sign': SignFn(), 'floor': Floor(),
       'reciprocal': Reciprocal()}
