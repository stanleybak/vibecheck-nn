"""Soundness tests for the elementwise integer-Pow relaxation (PowRelax).

Covers p in {2, 3, 4} over interval families: fully positive, fully negative,
crossing 0, tiny, wide, and a vector shape (N,). Asserts both the affine band
and the interval are sound (densely sampled — TEST-only sampling, per the
project rule), checks the even-p U-shape (min 0 when 0 in [lo, hi]) and the
odd-p monotone interval, and validates ``func`` against torch directly.
"""
import torch

from vibecheck.nl_pow import PowRelax
from vibecheck.nonlinear_relax import (
    REGISTRY, assert_band_sound, assert_interval_sound,
)


# (lo, hi) scalar interval families exercised for every exponent.
_SCALAR_INTERVALS = [
    (0.5, 3.0),       # fully positive
    (-3.0, -0.5),     # fully negative
    (-2.0, 2.0),      # symmetric crossing 0
    (-0.3, 1.7),      # asymmetric crossing 0
    (1.0, 1.0 + 1e-9),  # tiny (near-degenerate)
    (-50.0, 50.0),    # wide crossing 0
    (-7.0, -7.0),     # degenerate point, negative
    (0.0, 4.0),       # touches 0 at lo
    (-4.0, 0.0),      # touches 0 at hi
]

_PS = [2, 3, 4]


def test_registered():
    assert REGISTRY['Pow'] is PowRelax


def test_func_matches_torch():
    x = torch.linspace(-5.0, 5.0, 101, dtype=torch.float64)
    for p in _PS:
        relax = PowRelax(p=p)
        assert torch.allclose(relax.func(x), x ** p)


def test_init_validates_exponent():
    for bad in (1, 0, -2, 2.5):
        try:
            PowRelax(p=bad)
        except ValueError:
            continue
        raise AssertionError(f'PowRelax({bad!r}) should have raised ValueError')
    # Float that is integer-valued is accepted and coerced.
    assert PowRelax(p=3.0).p == 3


def test_band_sound_scalar():
    worst = 0.0
    for p in _PS:
        relax = PowRelax(p=p)
        for lo, hi in _SCALAR_INTERVALS:
            g = assert_band_sound(relax, lo, hi)
            worst = max(worst, g)
    # The band holds (assert above); report worst observed gap stays finite.
    assert worst < float('inf')


def test_interval_sound_scalar():
    for p in _PS:
        relax = PowRelax(p=p)
        for lo, hi in _SCALAR_INTERVALS:
            assert_interval_sound(relax, lo, hi)


def test_band_and_interval_sound_vector():
    # Vector shape (N,) mixing all interval kinds in one call.
    los = torch.tensor([lo for lo, _ in _SCALAR_INTERVALS], dtype=torch.float64)
    his = torch.tensor([hi for _, hi in _SCALAR_INTERVALS], dtype=torch.float64)
    for p in _PS:
        relax = PowRelax(p=p)
        assert_band_sound(relax, los, his)
        assert_interval_sound(relax, los, his)


def test_even_p_u_shape_min_zero_when_zero_in_interval():
    # Even p is U-shaped: when 0 in [lo, hi] the interval min is exactly 0,
    # and the max is at the endpoint with the larger magnitude.
    for p in (2, 4):
        relax = PowRelax(p=p)
        lo = torch.tensor([-2.0, -0.3, 0.0, -5.0], dtype=torch.float64)
        hi = torch.tensor([2.0, 1.7, 4.0, -1.0], dtype=torch.float64)
        olo, ohi = relax.interval(lo, hi)
        zero_in = (lo <= 0) & (hi >= 0)
        # min is 0 wherever 0 is inside.
        assert torch.allclose(olo[zero_in], torch.zeros_like(olo[zero_in]))
        # fully-negative element [-5,-1]: min at endpoint nearest 0 (-1)**p.
        assert torch.isclose(olo[3], torch.tensor((-1.0) ** p, dtype=torch.float64))
        # max is the larger-magnitude endpoint to the p.
        expect_hi = torch.maximum(lo ** p, hi ** p)
        assert torch.allclose(ohi, expect_hi)


def test_odd_p_monotone_interval():
    # Odd p is strictly increasing: interval is exactly (lo**p, hi**p).
    for p in (3,):
        relax = PowRelax(p=p)
        lo = torch.tensor([-2.0, 0.5, -5.0, 0.0], dtype=torch.float64)
        hi = torch.tensor([3.0, 4.0, -1.0, 2.0], dtype=torch.float64)
        olo, ohi = relax.interval(lo, hi)
        assert torch.allclose(olo, lo ** p)
        assert torch.allclose(ohi, hi ** p)


def test_band_degenerate_collapses():
    # On a degenerate (hi == lo) interval the band collapses: delta == 0 and
    # the affine line passes through the point exactly.
    for p in _PS:
        relax = PowRelax(p=p)
        lo = torch.tensor([-3.0, 2.0], dtype=torch.float64)
        hi = lo.clone()
        lam, mu, delta = relax.affine_band(lo, hi)
        assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-9)
        assert torch.allclose(lam * lo + mu, lo ** p, atol=1e-9)
