"""Soundness + tightness tests for the elementwise Cos relaxation.

Every band/interval is checked by dense sampling (the TEST use of sampling per
the project rule — we verify a closed-form bound, never derive one). Covers
random intervals plus adversarial cases: a peak at 0 / 2*pi, the trough at pi,
the inflections at +-pi/2, widths > 2*pi spanning multiple periods, fully
negative ranges, |chord slope| > 1, degenerate hi == lo, and vector shapes.
"""
import math

import torch

from vibecheck.nonlinear_relax import (
    REGISTRY,
    assert_band_sound,
    assert_interval_sound,
)
from vibecheck.nl_cos import CosRelax

PI = math.pi
TWO_PI = 2.0 * math.pi


def _relax():
    # Exercise the registry path too: import side-effect registers 'Cos'.
    assert REGISTRY['Cos'] is CosRelax
    return CosRelax()


def test_func_matches_torch_cos():
    relax = _relax()
    x = torch.linspace(-12.0, 12.0, 257, dtype=torch.float64)
    assert torch.allclose(relax.func(x), torch.cos(x))


# Adversarial / structured intervals -----------------------------------------

ADVERSARIAL = [
    (-0.3, 0.3),                 # crosses the peak at x = 0
    (-1e-9, 1e-9),               # tiny interval at the peak
    (TWO_PI - 0.4, TWO_PI + 0.4),  # crosses the peak at x = 2*pi
    (PI - 0.5, PI + 0.5),        # crosses the trough at x = pi
    (-PI - 0.5, -PI + 0.5),      # crosses the trough at x = -pi
    (PI / 2 - 0.3, PI / 2 + 0.3),  # inflection at +pi/2 (steepest descent)
    (-PI / 2 - 0.3, -PI / 2 + 0.3),  # inflection at -pi/2 (steepest ascent)
    (0.1, 0.1 + TWO_PI),         # exactly one full period, width = 2*pi
    (-0.7, -0.7 + 3.0 * TWO_PI),  # multi-period, width = 6*pi
    (-9.5, -8.9),                # fully negative range
    (-10.0, 10.0),               # very wide, both extrema interior, many periods
    (0.0, 0.05),                 # |chord slope| > 1 region near steepest part?
    (PI / 2 - 0.02, PI / 2 + 0.02),  # narrow band where slope ~ -sin ~ -1
    (PI / 2 + 0.4, PI / 2 + 0.6),  # short, monotone-decreasing, steep
    (5.0, 5.0),                  # degenerate hi == lo
    (-3.0, -3.0),                # degenerate negative
    (1.234, 1.234 + 1e-7),       # near-degenerate
    # Regression: stationary points must be anchored to the interval LOCATION,
    # not just its width. These |x| ~ 30 cases produced unsound bands when the
    # k-sweep was width-only (it never reached k ~ +-5). gap-delta was ~2 then.
    (29.03696, 32.75979),        # far-positive, ~width 3.7
    (-29.94671, -26.45948),      # far-negative
    (28.54383, 33.92880),        # far-positive, ~width 5.4
    (-150.3, -147.1),            # very far negative
    (197.6, 201.0),              # very far positive
]


def test_affine_band_sound_adversarial():
    relax = _relax()
    worst = 0.0
    for lo, hi in ADVERSARIAL:
        g = assert_band_sound(relax, torch.tensor(lo), torch.tensor(hi))
        worst = max(worst, g)
    # Sanity: some band actually has appreciable curvature gap.
    assert worst > 1e-3, f'expected a nontrivial worst gap, got {worst:.3e}'


def test_interval_sound_adversarial():
    relax = _relax()
    for lo, hi in ADVERSARIAL:
        assert_interval_sound(relax, torch.tensor(lo), torch.tensor(hi))


def test_interval_hits_global_extrema():
    """When [lo, hi] straddles a peak/trough the interval must reach +-1."""
    relax = _relax()
    f64 = lambda v: torch.tensor(v, dtype=torch.float64)
    olo, ohi = relax.interval(f64(-0.3), f64(0.3))
    assert abs(float(ohi) - 1.0) < 1e-12  # peak at 0
    olo, ohi = relax.interval(f64(PI - 0.2), f64(PI + 0.2))
    assert abs(float(olo) + 1.0) < 1e-12  # trough at pi
    # No extremum inside a short monotone piece -> endpoints only.
    a, b = 0.2, 0.8
    olo, ohi = relax.interval(f64(a), f64(b))
    assert abs(float(ohi) - math.cos(a)) < 1e-12
    assert abs(float(olo) - math.cos(b)) < 1e-12


# Slope guard: chord slope can exceed 1 in magnitude on a steep short interval.
def test_chord_slope_exceeds_one():
    relax = _relax()
    lo, hi = torch.tensor(PI / 2 - 0.4), torch.tensor(PI / 2 + 0.4)
    lam, mu, delta = relax.affine_band(lo, hi)
    # Around pi/2 the chord of cos is steep; with the |lam|>1 path forced we
    # still want soundness. Construct a definitely-steep one by hand too.
    assert_band_sound(relax, lo, hi)
    # A short interval right where |cos'| ~ 1: ensure even if |lam|>1 the
    # endpoint-only extrema path stays sound.
    lo2, hi2 = torch.tensor(PI / 2 - 0.05), torch.tensor(PI / 2 + 0.05)
    assert_band_sound(relax, lo2, hi2)


# Random coverage -------------------------------------------------------------

def test_affine_band_sound_random():
    relax = _relax()
    g = torch.Generator().manual_seed(20260615)
    worst = 0.0
    for _ in range(500):
        # Large magnitudes (|x| up to 60) exercise the location-anchored k-sweep;
        # widths up to ~12 span multiple periods.
        a = (torch.rand(1, generator=g, dtype=torch.float64) * 120.0 - 60.0)
        w = (torch.rand(1, generator=g, dtype=torch.float64) * 12.0)
        lo = a
        hi = a + w
        worst = max(worst, assert_band_sound(relax, lo, hi, n_samples=8000))
    assert worst >= 0.0


def test_interval_sound_random():
    relax = _relax()
    g = torch.Generator().manual_seed(11)
    for _ in range(500):
        a = (torch.rand(1, generator=g, dtype=torch.float64) * 120.0 - 60.0)
        w = (torch.rand(1, generator=g, dtype=torch.float64) * 12.0)
        assert_interval_sound(relax, a, a + w, n_samples=8000)


def test_far_from_origin_band_sound():
    """Regression for the width-only k-sweep bug: bands far from x=0 must stay
    sound. Sweeps |x| up to 200 with widths up to ~30 (vectorized)."""
    relax = _relax()
    g = torch.Generator().manual_seed(424242)
    a = torch.rand(2000, generator=g, dtype=torch.float64) * 400.0 - 200.0
    w = torch.rand(2000, generator=g, dtype=torch.float64) * 30.0
    assert_band_sound(relax, a, a + w, n_samples=4000)


# Vector / broadcast shapes ---------------------------------------------------

def test_vector_shape_band_and_interval():
    relax = _relax()
    lo = torch.tensor([-0.3, PI - 0.4, 0.1, -9.5, 5.0, -10.0])
    hi = torch.tensor([0.3, PI + 0.4, 0.1 + TWO_PI, -8.9, 5.0, 10.0])
    lam, mu, delta = relax.affine_band(lo, hi)
    assert lam.shape == lo.shape and mu.shape == lo.shape and delta.shape == lo.shape
    assert bool((delta >= 0).all())
    assert_band_sound(relax, lo, hi)
    olo, ohi = relax.interval(lo, hi)
    assert olo.shape == lo.shape and ohi.shape == lo.shape
    assert_interval_sound(relax, lo, hi)


def test_vector_2d_shape():
    relax = _relax()
    lo = torch.tensor([[-1.0, 0.0], [PI - 0.3, 2.0]])
    hi = torch.tensor([[1.0, 0.5], [PI + 0.3, 9.0]])
    assert_band_sound(relax, lo, hi)
    assert_interval_sound(relax, lo, hi)
    lam, mu, delta = relax.affine_band(lo, hi)
    assert lam.shape == lo.shape


# Tightness sanity: delta -> 0 as the interval shrinks ------------------------

def test_delta_shrinks_with_interval():
    relax = _relax()
    center = 0.9  # generic point, nonzero curvature
    prev = None
    for w in [1.0, 0.3, 0.1, 0.03, 0.01, 0.003]:
        lo = torch.tensor(center - w / 2)
        hi = torch.tensor(center + w / 2)
        _, _, delta = relax.affine_band(lo, hi)
        d = float(delta)
        assert d >= 0.0
        if prev is not None:
            assert d < prev + 1e-12, f'delta should not grow as width shrinks: {d} vs {prev}'
        prev = d
    assert prev < 1e-5, f'delta should approach 0 for tiny interval, got {prev:.3e}'


def test_degenerate_zero_width_band():
    relax = _relax()
    for c in [0.0, PI, PI / 2, -3.3, 5.0]:
        lo = hi = torch.tensor(c, dtype=torch.float64)
        lam, mu, delta = relax.affine_band(lo, hi)
        assert float(delta) < 1e-9
        # band value at the point equals cos(c)
        val = float(lam * c + mu)
        assert abs(val - math.cos(c)) < 1e-7
