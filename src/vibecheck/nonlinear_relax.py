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

    def affine_band(self, lo, hi):
        """Sound affine over-approximation. Returns (lam, mu, delta) with
        |f(x) - (lam*x + mu)| <= delta  for all x in [lo, hi], delta >= 0.
        Must be sound by construction (no sampling)."""
        raise NotImplementedError


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
