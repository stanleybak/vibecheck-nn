"""Sound zonotope/interval relaxation for elementwise Cos.

`CosRelax` provides, for f(x) = cos(x) and a per-element input interval
[lo, hi]:

  - `func(x)`             : exact torch.cos.
  - `interval(lo, hi)`    : SOUND output interval (out_lo, out_hi).
  - `affine_band(lo, hi)` : SOUND affine over-approximation (lam, mu, delta)
        with  |cos(x) - (lam*x + mu)| <= delta  for ALL x in [lo, hi].

SOUNDNESS IS BY CONSTRUCTION (exhaustive critical-point enumeration of a
smooth function on a closed interval — extrema lie at endpoints or stationary
points). We NEVER sample to derive a bound. See `nonlinear_relax.py` for the
contract and the test helpers that *test* (not produce) these bounds.
"""
import math

import torch

from .nonlinear_relax import ScalarNonlinearRelax, register

_TWO_PI = 2.0 * math.pi


def _interval_contains_congruent(lo, hi, theta):
    """Elementwise boolean: does [lo, hi] contain some point congruent to
    `theta` modulo 2*pi?  i.e. exists integer k with  theta + 2*pi*k in [lo, hi].

    A residue r in [0, 2*pi) congruent to theta lies in [lo, hi] iff, after
    shifting lo into [theta, theta + 2*pi), it is <= hi. Concretely: the
    smallest x >= lo with x = theta (mod 2*pi) is
        x0 = lo + ((theta - lo) mod 2*pi)
    and the interval contains such a point iff x0 <= hi. `(theta - lo) mod 2*pi`
    is in [0, 2*pi), so x0 in [lo, lo + 2*pi); thus x0 <= hi handles every
    period (including hi - lo >= 2*pi, where x0 <= lo + 2*pi <= hi always).
    """
    # torch.remainder returns a result with the sign of the divisor (>0 here),
    # i.e. a value in [0, 2*pi).
    offset = torch.remainder(theta - lo, _TWO_PI)
    x0 = lo + offset
    return x0 <= hi


@register('Cos')
class CosRelax(ScalarNonlinearRelax):
    """Sound relaxation for elementwise cos."""

    def func(self, x):
        return torch.cos(x)

    def interval(self, lo, hi):
        """Sound output interval.

        cos attains its max (=1) at x = 2*pi*k and its min (=-1) at
        x = pi + 2*pi*k. If such a maximizer/minimizer lies inside [lo, hi] the
        bound is the global extremum; otherwise cos is monotone-piece-free of
        interior extrema on [lo, hi], so the extreme values are at the
        endpoints.
        """
        lo = torch.as_tensor(lo)
        hi = torch.as_tensor(hi)
        lo, hi = torch.broadcast_tensors(lo, hi)
        flo = torch.cos(lo)
        fhi = torch.cos(hi)
        ep_max = torch.maximum(flo, fhi)
        ep_min = torch.minimum(flo, fhi)
        one = torch.ones_like(ep_max)
        neg_one = -one
        has_peak = _interval_contains_congruent(lo, hi, 0.0)          # max at 2*pi*k
        has_trough = _interval_contains_congruent(lo, hi, math.pi)    # min at pi + 2*pi*k
        out_hi = torch.where(has_peak, one, ep_max)
        out_lo = torch.where(has_trough, neg_one, ep_min)
        return out_lo, out_hi

    def slope_at(self, x):
        return -torch.sin(torch.as_tensor(x, dtype=torch.float64))

    def affine_band(self, lo, hi, lam=None):
        """Sound affine band around the chord (or a caller-supplied α-CROWN
        slope — sound for ANY lam).

        Let lam be the chord slope and g(x) = cos(x) - lam*x. g is smooth, so on
        the closed interval [lo, hi] its extrema occur at the endpoints or at
        stationary points g'(x) = -sin(x) - lam = 0  <=>  sin(x) = -lam.

        Stationary points (when |lam| <= 1):
            x = -arcsin(lam) + 2*pi*k          (from sin(x) = -lam)
            x = pi + arcsin(lam) + 2*pi*k
        We enumerate every integer k that can place such a point in [lo, hi]
        (the count of full periods is (hi - lo)/(2*pi); +2 slack covers the
        partial periods at both ends and arcsin offsets), evaluate g at each
        in-range stationary point and at both endpoints, take gmax/gmin, then
            mu    = (gmax + gmin) / 2
            delta = (gmax - gmin) / 2
        so  gmin <= cos(x) - lam*x <= gmax  =>  |cos(x) - (lam*x + mu)| <= delta
        for all x in [lo, hi]. Exact (tightest for this lam) and sound.
        """
        lo = torch.as_tensor(lo)
        hi = torch.as_tensor(hi)
        lo, hi = torch.broadcast_tensors(lo, hi)
        work_dtype = lo.dtype if lo.dtype.is_floating_point else torch.float64
        lo = lo.to(work_dtype)
        hi = hi.to(work_dtype)

        width = hi - lo
        degenerate = width.abs() <= 0.0  # hi == lo

        if lam is None:
            # Chord slope; guard hi == lo with the exact derivative -sin(lo).
            safe_width = torch.where(degenerate, torch.ones_like(width), width)
            lam_chord = (torch.cos(hi) - torch.cos(lo)) / safe_width
            lam = torch.where(degenerate, -torch.sin(lo), lam_chord)
        else:
            lam = torch.as_tensor(lam, dtype=work_dtype, device=lo.device)

        def g(x):
            return torch.cos(x) - lam * x

        g_lo = g(lo)
        g_hi = g(hi)
        gmax = torch.maximum(g_lo, g_hi)
        gmin = torch.minimum(g_lo, g_hi)

        # Stationary points need a real arcsin(lam); clamp keeps arcsin finite,
        # the `valid` mask below discards contributions where |lam| > 1.
        valid = lam.abs() <= 1.0
        asin = torch.asin(torch.clamp(lam, -1.0, 1.0))
        base_a = -asin                # x = -arcsin(lam) + 2*pi*k
        base_b = math.pi + asin       # x = pi + arcsin(lam) + 2*pi*k

        # Anchor the k-sweep to the interval's LOCATION (not just its width):
        # for each stationary-point family `base + 2*pi*k`, the integer k that
        # lands the point nearest the interval centre is
        #     k0(elt) = round((centre - base) / (2*pi))
        # which is a per-element tensor (an x ~ 30 interval needs k ~ 5, an
        # x ~ -30 interval needs k ~ -5 — a width-only cap misses both). We then
        # sweep a small WINDOW of integer offsets around k0 wide enough to span
        # the periods inside [lo, hi]. The interval covers width/(2*pi) periods,
        # so +-(ceil(width/2*pi) + 1) offsets from k0 is guaranteed to enumerate
        # every in-range stationary point of each family. Per-element [lo, hi]
        # masking discards out-of-range candidates.
        centre = (lo + hi) * 0.5
        half_win = int(math.ceil(float((width.abs() / _TWO_PI).max().item()))) + 1 \
            if width.numel() else 1
        neg_inf = torch.full_like(lo, float('-inf'))
        pos_inf = torch.full_like(lo, float('inf'))
        for base in (base_a, base_b):
            k0 = torch.round((centre - base) / _TWO_PI)
            for d in range(-half_win, half_win + 1):
                x = base + _TWO_PI * (k0 + d)
                in_range = valid & (x >= lo) & (x <= hi)
                gx = g(x)
                cand_max = torch.where(in_range, gx, neg_inf)
                cand_min = torch.where(in_range, gx, pos_inf)
                gmax = torch.maximum(gmax, cand_max)
                gmin = torch.minimum(gmin, cand_min)

        mu = (gmax + gmin) * 0.5
        delta = (gmax - gmin) * 0.5
        delta = torch.clamp(delta, min=0.0)
        return lam, mu, delta
