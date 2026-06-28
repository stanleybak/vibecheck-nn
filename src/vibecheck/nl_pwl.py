"""Sound zonotope/interval relaxation for a 1-D piecewise-linear lookup table.

This is the relaxation for the merged ReLU-lookup-table node (see
``onnx_optimizer.merge_relu_lookup_table``): the linear-surrogate ml4acopf models
encode the nonlinear physics (sigmoid/sin/cos) as

    f(x) = bias + sum_i weight_i * ReLU(x - offset_i)

applied element-wise. Propagating through the *expanded* ReLU sum loses all
input->output correlation (each ReLU bounded independently), which blows VC's
forward-zonotope bound up by orders of magnitude (118-linear-residual: layer
widths 10 -> 343, root margin -38.69). Bounding the merged 1-D PWL directly is
EXACT: f is piecewise-linear in x with breakpoints exactly at the offsets, so its
extrema over any [lo, hi] are at the endpoints or the in-range breakpoints, and so
is g(x) = f(x) - lam*x. Hence both ``interval`` and ``affine_band`` are exact (the
tightest sound parallel band) by construction — NEVER sampled.

Unlike the registered scalar ops (Sin/Cos/...), this relax is PARAMETRIZED per
node, so it is instantiated directly as ``PWLRelax(offsets, weights, bias)`` rather
than via the ONNX-op REGISTRY.
"""
import torch

from .nonlinear_relax import ScalarNonlinearRelax


class PWLRelax(ScalarNonlinearRelax):
    """Sound relaxation for f(x) = bias + sum_i weight_i * ReLU(x - offset_i)."""

    onnx_op = 'PWLLookup'

    def __init__(self, offsets, weights, bias=0.0):
        # 1-D parameter vectors (K,), shared across all elements.
        self.offsets = torch.as_tensor(offsets, dtype=torch.float64).reshape(-1)
        self.weights = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
        self.bias = float(bias)
        assert self.offsets.numel() == self.weights.numel(), \
            'PWLRelax: offsets/weights length mismatch'

    def func(self, x):
        x = torch.as_tensor(x, dtype=torch.float64)
        o = self.offsets.to(x.device)
        w = self.weights.to(x.device)
        # relu(x[...,None] - o) * w, summed over the lookup dim.
        return self.bias + (torch.clamp(x.unsqueeze(-1) - o, min=0.0) * w).sum(-1)

    def slope_at(self, x):
        x = torch.as_tensor(x, dtype=torch.float64)
        o = self.offsets.to(x.device)
        w = self.weights.to(x.device)
        # f'(x) = sum_i w_i * [x > o_i] (left-continuous; value at a breakpoint is
        # a subgradient — fine, slope_at only seeds candidate band slopes).
        return ((x.unsqueeze(-1) > o).to(torch.float64) * w).sum(-1)

    def _candidates(self, lo, hi):
        """x-points where f (and g = f - lam*x) can attain an extremum over
        [lo, hi]: the endpoints plus every offset clamped into [lo, hi]. Clamping
        an out-of-range offset to an endpoint just re-evaluates that endpoint
        (harmless); in-range offsets are the actual PWL breakpoints. Shape
        (..., K+2)."""
        lo = torch.as_tensor(lo, dtype=torch.float64)
        hi = torch.as_tensor(hi, dtype=torch.float64)
        lo, hi = torch.broadcast_tensors(lo, hi)
        o = self.offsets.to(lo.device)
        # offsets broadcast to (..., K), clamped element-wise into [lo, hi].
        o_clamped = torch.minimum(torch.maximum(
            o.expand(*lo.shape, o.numel()), lo.unsqueeze(-1)), hi.unsqueeze(-1))
        return torch.cat([lo.unsqueeze(-1), hi.unsqueeze(-1), o_clamped], dim=-1)

    def interval(self, lo, hi):
        pts = self._candidates(lo, hi)         # (..., K+2)
        fp = self.func(pts)                    # (..., K+2)
        return fp.min(-1).values, fp.max(-1).values

    def affine_band(self, lo, hi, lam=None):
        lo = torch.as_tensor(lo, dtype=torch.float64)
        hi = torch.as_tensor(hi, dtype=torch.float64)
        lo, hi = torch.broadcast_tensors(lo, hi)
        width = hi - lo
        degenerate = width <= 0.0
        if lam is None:
            denom = torch.where(degenerate, torch.ones_like(width), width)
            lam = torch.where(degenerate, self.slope_at(lo),
                              (self.func(hi) - self.func(lo)) / denom)
        lam = torch.as_tensor(lam, dtype=torch.float64)
        pts = self._candidates(lo, hi)                 # (..., K+2)
        g = self.func(pts) - lam.unsqueeze(-1) * pts   # (..., K+2)
        gmax = g.max(-1).values
        gmin = g.min(-1).values
        mu = 0.5 * (gmax + gmin)
        delta = (0.5 * (gmax - gmin)).clamp_min(0.0)
        return lam, mu, delta
