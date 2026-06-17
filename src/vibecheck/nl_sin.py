"""Sound zonotope/interval relaxation for elementwise Sin.

Registered under the ONNX op type ``Sin``. Provides:

  - ``func``        : exact ``torch.sin``.
  - ``interval``    : sound output interval for sin over [lo, hi].
  - ``affine_band`` : sound affine over-approximation (lam, mu, delta) with
        ``|sin(x) - (lam*x + mu)| <= delta`` for all x in [lo, hi].

Soundness is BY CONSTRUCTION — closed form for the interval (sin attains its
global extrema +-1 exactly at the points pi/2 + 2*pi*k / -pi/2 + 2*pi*k, and is
otherwise monotone between them so the segment extrema are the endpoints) and
exhaustive critical-point enumeration for the band (g(x) = sin(x) - lam*x is
smooth, so its extrema over a closed interval are at the endpoints or at
stationary points cos(x) = lam). NEVER derive delta by sampling.
"""
import math

import torch

from .nonlinear_relax import ScalarNonlinearRelax, register

_TWO_PI = 2.0 * math.pi
_HALF_PI = 0.5 * math.pi


def _interval_contains_congruent(lo, hi, theta):
    """Element-wise boolean: does [lo, hi] contain some x congruent to ``theta``
    modulo 2*pi, i.e. exists integer k with lo <= theta + 2*pi*k <= hi?

    Equivalent to floor((hi - theta) / 2pi) >= ceil((lo - theta) / 2pi).
    Vectorized over the broadcast shape of lo/hi.
    """
    k_hi = torch.floor((hi - theta) / _TWO_PI)
    k_lo = torch.ceil((lo - theta) / _TWO_PI)
    return k_hi >= k_lo


@register('Sin')
class SinRelax(ScalarNonlinearRelax):
    """Sound relaxation for elementwise sin."""

    def func(self, x):
        return torch.sin(x)

    def interval(self, lo, hi):
        """Sound (out_lo, out_hi) for sin over [lo, hi], element-wise.

        out_hi = 1.0  wherever some maximizer pi/2 + 2*pi*k is in [lo, hi],
                 else max(sin(lo), sin(hi)).
        out_lo = -1.0 wherever some minimizer -pi/2 + 2*pi*k is in [lo, hi],
                 else min(sin(lo), sin(hi)).

        Sound: between consecutive extrema sin is monotone, so if no extremum
        lies inside the interval the extrema of sin over [lo, hi] are the
        endpoint values; if one does, sin attains the global +-1 there.
        """
        lo = torch.as_tensor(lo, dtype=torch.float64)
        hi = torch.as_tensor(hi, dtype=torch.float64)
        lo, hi = torch.broadcast_tensors(lo, hi)

        s_lo = torch.sin(lo)
        s_hi = torch.sin(hi)
        endpoint_max = torch.maximum(s_lo, s_hi)
        endpoint_min = torch.minimum(s_lo, s_hi)

        has_peak = _interval_contains_congruent(lo, hi, _HALF_PI)
        has_trough = _interval_contains_congruent(lo, hi, -_HALF_PI)

        out_hi = torch.where(has_peak, torch.ones_like(endpoint_max), endpoint_max)
        out_lo = torch.where(has_trough, -torch.ones_like(endpoint_min), endpoint_min)
        return out_lo, out_hi

    def slope_at(self, x):
        return torch.cos(torch.as_tensor(x, dtype=torch.float64))

    def affine_band(self, lo, hi, lam=None):
        """Sound affine band (lam, mu, delta): |sin(x) - (lam*x + mu)| <= delta
        for all x in [lo, hi].

        lam = chord slope (sin(hi) - sin(lo)) / (hi - lo), or cos(lo) when
        hi == lo (or a caller-supplied α-CROWN slope). The deviation
        g(x) = sin(x) - lam*x is smooth, so its max and min over [lo, hi] occur
        at the endpoints or at interior stationary points where
        g'(x) = cos(x) - lam = 0, i.e. cos(x) = lam. Those are
        x = +-arccos(lam) + 2*pi*k. We enumerate every such x inside [lo, hi]
        (a bounded count: at most ~ (hi - lo)/(2*pi) + 2 per branch), evaluate g
        at lo, hi and each in-range critical point, then set
        mu = (gmax + gmin)/2, delta = (gmax - gmin)/2. Sound for ANY lam.
        """
        lo = torch.as_tensor(lo, dtype=torch.float64)
        hi = torch.as_tensor(hi, dtype=torch.float64)
        lo, hi = torch.broadcast_tensors(lo, hi)
        lo = lo.contiguous()
        hi = hi.contiguous()

        width = hi - lo
        degenerate = width <= 0.0
        if lam is None:
            # chord slope; on hi == lo use the local derivative cos(lo).
            denom = torch.where(degenerate, torch.ones_like(width), width)
            lam = torch.where(degenerate,
                              torch.cos(lo),
                              (torch.sin(hi) - torch.sin(lo)) / denom)

        def g(x):
            return torch.sin(x) - lam * x

        # Start the running max/min from the two endpoints.
        g_lo = g(lo)
        g_hi = g(hi)
        gmax = torch.maximum(g_lo, g_hi)
        gmin = torch.minimum(g_lo, g_hi)

        # Interior stationary points: cos(x) = lam. Real only when |lam| <= 1.
        has_root = lam.abs() <= 1.0
        lam_clamped = lam.clamp(-1.0, 1.0)
        base = torch.arccos(lam_clamped)  # principal value in [0, pi]

        # The two arccos branches: x = +base + 2*pi*k and x = -base + 2*pi*k.
        # Bound the integer-k range by the interval width.
        max_periods = int(math.floor(float(width.max().item()) / _TWO_PI)) + 2 \
            if width.numel() > 0 else 2

        for sign in (1.0, -1.0):
            theta = sign * base  # element-wise candidate phase in [-pi, pi]
            # Smallest k such that theta + 2*pi*k >= lo:
            k_start = torch.ceil((lo - theta) / _TWO_PI)
            for j in range(max_periods + 1):
                k = k_start + j
                xc = theta + _TWO_PI * k
                in_range = has_root & (xc >= lo) & (xc <= hi)
                if not bool(in_range.any()):
                    continue
                gc = g(xc)
                # Only let in-range critical points move the extrema.
                gmax = torch.where(in_range, torch.maximum(gmax, gc), gmax)
                gmin = torch.where(in_range, torch.minimum(gmin, gc), gmin)

        mu = 0.5 * (gmax + gmin)
        delta = 0.5 * (gmax - gmin)
        delta = delta.clamp_min(0.0)
        return lam, mu, delta
