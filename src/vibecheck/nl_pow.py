"""Sound zonotope/interval relaxation for elementwise integer Pow (x**p).

Registered under the ONNX op type ``Pow``. Constructed with a fixed integer
exponent ``p >= 2`` (default 2 — ACOPF benchmarks square variables). Provides:

  - ``func``        : exact ``x ** p`` (torch).
  - ``interval``    : sound output interval for x**p over [lo, hi].
  - ``affine_band`` : sound affine over-approximation (lam, mu, delta) with
        ``|x**p - (lam*x + mu)| <= delta`` for all x in [lo, hi].

Soundness is BY CONSTRUCTION:

  - INTERVAL. For EVEN p, x**p is U-shaped: f'' = p(p-1)x**(p-2) >= 0 so f is
    convex on all of R, the minimum is 0 at x=0 (if 0 in [lo,hi]) else at the
    endpoint nearest 0, and the maximum is at the endpoint with the larger
    |x|. For ODD p, x**p is strictly increasing on all of R, so the extrema are
    the endpoint values (lo**p, hi**p).

  - BAND. g(x) = x**p - lam*x is smooth, so its max and min over the closed
    interval [lo, hi] occur at the endpoints or at interior stationary points
    g'(x) = p*x**(p-1) - lam = 0, i.e. x**(p-1) = lam/p. We enumerate the real
    roots of that equation that fall inside [lo, hi] (at most two: +m and, when
    p is odd, also -m, with m = |lam/p|**(1/(p-1))), evaluate g at lo, hi and
    each in-range root, then set mu = (gmax+gmin)/2, delta = (gmax-gmin)/2.
    Including a few extra *real, in-range* candidate points can only widen the
    [gmin, gmax] bracket — it never shrinks it — so the band stays sound even if
    a candidate turns out not to be a true stationary point. We NEVER derive
    delta by sampling.

Fractional-power edge case: the stationary-point magnitude uses an odd-root
(``1/(p-1)``) that is ill-defined for negative bases in torch. We compute it on
the non-negative magnitude ``|lam/p|`` and reattach the sign explicitly, and we
gate every candidate by ``lo <= xc <= hi`` so out-of-range / non-real-branch
points are discarded.
"""
import torch

from .nonlinear_relax import ScalarNonlinearRelax, register


@register('Pow')
class PowRelax(ScalarNonlinearRelax):
    """Sound relaxation for elementwise x**p, integer p >= 2."""

    def __init__(self, p=2):
        p_int = int(p)
        if p_int != p:
            raise ValueError(f'PowRelax requires an integer exponent, got {p!r}')
        if p_int < 2:
            raise ValueError(f'PowRelax requires p >= 2, got {p_int}')
        self.p = p_int

    def func(self, x):
        return x ** self.p

    def interval(self, lo, hi):
        """Sound (out_lo, out_hi) for x**p over [lo, hi], element-wise.

        EVEN p (convex, U-shaped, min 0 at the origin):
            out_lo = 0          where 0 in [lo, hi]
                   = min(lo**p, hi**p)   otherwise (= value at endpoint nearest 0)
            out_hi = max(lo**p, hi**p)   (endpoint with larger |x|)
        ODD p (monotone increasing):
            out_lo = lo**p,  out_hi = hi**p.
        """
        lo = torch.as_tensor(lo, dtype=torch.float64)
        hi = torch.as_tensor(hi, dtype=torch.float64)
        lo, hi = torch.broadcast_tensors(lo, hi)

        p = self.p
        f_lo = lo ** p
        f_hi = hi ** p
        if p % 2 == 1:
            # Strictly increasing: endpoints are the extrema.
            return f_lo, f_hi
        # Even: convex U-shape.
        zero_in = (lo <= 0.0) & (hi >= 0.0)
        out_lo = torch.where(zero_in, torch.zeros_like(lo),
                             torch.minimum(f_lo, f_hi))
        out_hi = torch.maximum(f_lo, f_hi)
        return out_lo, out_hi

    def affine_band(self, lo, hi):
        """Sound affine band (lam, mu, delta): |x**p - (lam*x + mu)| <= delta
        for all x in [lo, hi].

        lam is the chord slope (hi**p - lo**p)/(hi - lo); on the degenerate
        hi == lo the band collapses and lam = p*lo**(p-1) (the local derivative).
        g(x) = x**p - lam*x is smooth; its extrema over [lo, hi] are at the
        endpoints or at stationary points x**(p-1) = lam/p. We enumerate those
        roots in-range and bracket gmin/gmax over {lo, hi, roots}.
        """
        lo = torch.as_tensor(lo, dtype=torch.float64)
        hi = torch.as_tensor(hi, dtype=torch.float64)
        lo, hi = torch.broadcast_tensors(lo, hi)
        lo = lo.contiguous()
        hi = hi.contiguous()

        p = self.p
        width = hi - lo
        degenerate = width <= 0.0
        denom = torch.where(degenerate, torch.ones_like(width), width)
        lam = torch.where(degenerate,
                          p * lo ** (p - 1),
                          (hi ** p - lo ** p) / denom)

        def g(x):
            return x ** p - lam * x

        # Endpoints seed the running bracket.
        g_lo = g(lo)
        g_hi = g(hi)
        gmax = torch.maximum(g_lo, g_hi)
        gmin = torch.minimum(g_lo, g_hi)

        # Stationary points: x**(p-1) = lam/p.  m = |lam/p|**(1/(p-1)) >= 0.
        # Compute the magnitude on the non-negative |lam/p| (avoids a fractional
        # power of a negative base in torch) and reattach the sign per branch.
        ratio = lam / p
        m = ratio.abs().pow(1.0 / (p - 1))

        if p % 2 == 1:
            # p-1 even: x**(p-1) >= 0, so a real root needs lam/p >= 0, and then
            # BOTH x = +m and x = -m solve it.
            real = ratio >= 0.0
            candidates = (m, -m)
        else:
            # p-1 odd: x**(p-1) is odd/monotone -> exactly one real root,
            # x = sign(lam/p) * m.
            real = torch.ones_like(lam, dtype=torch.bool)
            xc_even = torch.sign(ratio) * m
            candidates = (xc_even,)

        for xc in candidates:
            in_range = real & (xc >= lo) & (xc <= hi)
            if not bool(in_range.any()):
                continue
            gc = g(xc)
            gmax = torch.where(in_range, torch.maximum(gmax, gc), gmax)
            gmin = torch.where(in_range, torch.minimum(gmin, gc), gmin)

        mu = 0.5 * (gmax + gmin)
        delta = 0.5 * (gmax - gmin)
        delta = delta.clamp_min(0.0)
        return lam, mu, delta
