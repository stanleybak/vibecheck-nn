"""Sound elementwise relaxation for ONNX ``Floor``.

``floor(x)`` is the greatest integer ``<= x``. It is a right-continuous
staircase: constant on each half-open interval ``[n, n+1)`` and monotonic
non-decreasing. Two facts drive the bounds below:

  1. Monotonicity  -> a sound interval is ``(floor(lo), floor(hi))``.
  2. ``floor(x) in (x-1, x]`` for all x  -> the line ``y = x - 0.5`` over-
     approximates floor with worst-case error exactly ``0.5``:
         floor(x) - (x - 0.5)  in  (-0.5, 0.5]
     so  ``|floor(x) - (x - 0.5)| <= 0.5``  everywhere, including at exact
     integers where floor(n) = n and the gap is |n - (n - 0.5)| = 0.5.

When the whole input interval lands inside a single integer cell
(``floor(lo) == floor(hi)``) floor is *constant* there, so we return the exact
zero-error band ``lam=0, mu=floor(lo), delta=0``.

Soundness is by construction (monotonicity + the (x-1, x] containment); no
sampling is used to derive any bound.
"""
import torch

from .nonlinear_relax import ScalarNonlinearRelax, register


@register('Floor')
class FloorRelax(ScalarNonlinearRelax):
    """Sound relaxation for elementwise floor(x) = greatest integer <= x."""

    def func(self, x):
        """Exact elementwise floor."""
        return torch.floor(x)

    def interval(self, lo, hi):
        """Sound output interval. floor is monotonic non-decreasing, so the
        endpoints map directly: (floor(lo), floor(hi))."""
        return torch.floor(lo), torch.floor(hi)

    def affine_band(self, lo, hi):
        """Sound affine band (lam, mu, delta) with
        |floor(x) - (lam*x + mu)| <= delta for all x in [lo, hi].

        Two cases, applied element-wise via torch.where so a batched lo/hi can
        mix within-cell and spanning elements:

        - Within one integer cell (floor(lo) == floor(hi)): floor is constant,
          lam=0, mu=floor(lo), delta=0 (exact).
        - Spanning >= one integer boundary: lam=1, mu=-0.5, delta=0.5, since
          floor(x) in (x-1, x] => |floor(x) - (x - 0.5)| <= 0.5.
        """
        lo = torch.as_tensor(lo)
        hi = torch.as_tensor(hi)

        flo = torch.floor(lo)
        # Constant where the interval stays inside a single integer cell,
        # i.e. floor(lo) == floor(hi). Right-closed endpoints are handled
        # correctly because, e.g. [1.9, 2.0] has floor(lo)=1 != floor(hi)=2
        # and is treated as spanning (floor really does take value 2 at 2.0).
        same_cell = torch.floor(hi) == flo

        zero = torch.zeros_like(flo)
        one = torch.ones_like(flo)

        lam = torch.where(same_cell, zero, one)
        mu = torch.where(same_cell, flo, torch.full_like(flo, -0.5))
        delta = torch.where(same_cell, zero, torch.full_like(flo, 0.5))
        return lam, mu, delta
