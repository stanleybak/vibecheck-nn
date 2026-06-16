"""Soundness + tightness tests for the elementwise Sin relaxation.

Sampling is used here ONLY to TEST closed-form bounds (the correct use per the
project soundness rule), via assert_band_sound / assert_interval_sound.
"""
import math

import torch
import pytest

from vibecheck.nl_sin import SinRelax
from vibecheck.nonlinear_relax import (
    REGISTRY, assert_band_sound, assert_interval_sound,
)

PI = math.pi
HALF_PI = 0.5 * PI


@pytest.fixture
def relax():
    return SinRelax()


def test_registered():
    assert REGISTRY['Sin'] is SinRelax
    assert SinRelax.onnx_op == 'Sin'


def test_func_matches_torch_sin(relax):
    x = torch.linspace(-20.0, 20.0, 1001, dtype=torch.float64)
    assert torch.allclose(relax.func(x), torch.sin(x))


# ---------------------------------------------------------------------------
# Adversarial single-interval cases: tiny, peak/trough/inflection crossing,
# multi-period, negative, |chord slope| > 1.
# ---------------------------------------------------------------------------
ADVERSARIAL = {
    'tiny_positive': (0.30, 0.3000001),
    'tiny_near_peak': (HALF_PI - 1e-7, HALF_PI + 1e-7),
    'cross_peak': (1.0, 2.2),                     # straddles pi/2
    'cross_trough': (-2.2, -1.0),                 # straddles -pi/2
    'cross_trough_shifted': (3.0 * HALF_PI - 0.4, 3.0 * HALF_PI + 0.4),  # 3pi/2
    'inflection_zero': (-0.7, 0.9),               # straddles x = 0
    'inflection_pi': (PI - 0.6, PI + 0.6),        # straddles x = pi
    'monotone_quarter': (0.1, 0.1 + HALF_PI - 0.05),
    'full_period': (0.3, 0.3 + 2.0 * PI),         # contains one peak + trough
    'multi_period': (-1.0, -1.0 + 5.0 * PI),      # width > 2pi, several extrema
    'wide_negative': (-15.0, -3.0),
    'chord_slope_gt1': (-0.2, 0.2),               # slope ~ near 1; symmetric
    'steep_small': (-0.01, 0.01),                 # slope ~ 1, |lam| can be ~1
    'large_offset': (100.0, 103.5),               # big x, multiple extrema
    'span_peak_and_trough': (HALF_PI - 0.3, 3.0 * HALF_PI + 0.3),
}


@pytest.mark.parametrize('name', list(ADVERSARIAL))
def test_band_sound_adversarial(relax, name):
    lo, hi = ADVERSARIAL[name]
    lo = torch.tensor(lo, dtype=torch.float64)
    hi = torch.tensor(hi, dtype=torch.float64)
    assert_band_sound(relax, lo, hi)


@pytest.mark.parametrize('name', list(ADVERSARIAL))
def test_interval_sound_adversarial(relax, name):
    lo, hi = ADVERSARIAL[name]
    lo = torch.tensor(lo, dtype=torch.float64)
    hi = torch.tensor(hi, dtype=torch.float64)
    assert_interval_sound(relax, lo, hi)


# ---------------------------------------------------------------------------
# Interval-correctness: when the interval is known to contain a peak/trough the
# returned bound must be exactly +-1; when it doesn't, the endpoint values.
# ---------------------------------------------------------------------------
def test_interval_hits_plus_one_on_peak(relax):
    lo = torch.tensor([1.0, HALF_PI - 0.1], dtype=torch.float64)
    hi = torch.tensor([2.2, HALF_PI + 0.1], dtype=torch.float64)
    out_lo, out_hi = relax.interval(lo, hi)
    assert torch.allclose(out_hi, torch.ones_like(out_hi))


def test_interval_hits_minus_one_on_trough(relax):
    lo = torch.tensor([-2.2, -HALF_PI - 0.1], dtype=torch.float64)
    hi = torch.tensor([-1.0, -HALF_PI + 0.1], dtype=torch.float64)
    out_lo, out_hi = relax.interval(lo, hi)
    assert torch.allclose(out_lo, -torch.ones_like(out_lo))


def test_interval_monotone_uses_endpoints(relax):
    # On [0.1, 1.0] sin is increasing, no interior extremum.
    lo = torch.tensor(0.1, dtype=torch.float64)
    hi = torch.tensor(1.0, dtype=torch.float64)
    out_lo, out_hi = relax.interval(lo, hi)
    assert torch.allclose(out_lo, torch.sin(lo))
    assert torch.allclose(out_hi, torch.sin(hi))


# ---------------------------------------------------------------------------
# Batched / vector lo, hi of shape (N,) and (M, N).
# ---------------------------------------------------------------------------
def test_band_sound_batched_random(relax):
    torch.manual_seed(0)
    N = 400
    centers = (torch.rand(N, dtype=torch.float64) - 0.5) * 60.0  # +-30
    widths = torch.rand(N, dtype=torch.float64) * 8.0 + 1e-4     # up to 8 rad
    lo = centers - widths / 2
    hi = centers + widths / 2
    assert_band_sound(relax, lo, hi, n_samples=20000)
    assert_interval_sound(relax, lo, hi, n_samples=20000)


def test_band_sound_2d_batched(relax):
    torch.manual_seed(1)
    M, N = 16, 24
    centers = (torch.rand(M, N, dtype=torch.float64) - 0.5) * 40.0
    widths = torch.rand(M, N, dtype=torch.float64) * 6.0 + 1e-3
    lo = centers - widths / 2
    hi = centers + widths / 2
    assert_band_sound(relax, lo, hi, n_samples=8000)
    assert_interval_sound(relax, lo, hi, n_samples=8000)


def test_band_sound_tiny_widths_batched(relax):
    torch.manual_seed(2)
    N = 300
    centers = (torch.rand(N, dtype=torch.float64) - 0.5) * 40.0
    widths = torch.rand(N, dtype=torch.float64) * 1e-5 + 1e-9
    lo = centers - widths / 2
    hi = centers + widths / 2
    assert_band_sound(relax, lo, hi, n_samples=2000)
    assert_interval_sound(relax, lo, hi, n_samples=2000)


def test_degenerate_point_interval(relax):
    # hi == lo: band should collapse to ~0 delta, interval to [sin(lo), sin(lo)].
    lo = torch.tensor([0.3, 2.0, -1.7], dtype=torch.float64)
    hi = lo.clone()
    lam, mu, delta = relax.affine_band(lo, hi)
    assert torch.all(delta >= 0)
    assert float(delta.max()) < 1e-9
    out_lo, out_hi = relax.interval(lo, hi)
    assert torch.allclose(out_lo, torch.sin(lo))
    assert torch.allclose(out_hi, torch.sin(lo))


# ---------------------------------------------------------------------------
# Tightness sanity: delta -> 0 as the interval shrinks; small on a within-
# quarter-period monotone interval.
# ---------------------------------------------------------------------------
def test_delta_shrinks_to_zero(relax):
    center = torch.tensor(0.8, dtype=torch.float64)
    prev = float('inf')
    for w in [1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.001]:
        lo = center - w / 2
        hi = center + w / 2
        _, _, delta = relax.affine_band(lo, hi)
        d = float(delta)
        assert d <= prev + 1e-12, f'delta not monotone shrinking at w={w}'
        prev = d
    assert prev < 1e-5  # at w=0.001 the band is essentially exact


def test_delta_small_on_quarter_period_monotone(relax):
    # [0.1, 0.1 + pi/2 - 0.05]: monotone increasing, no interior extremum.
    lo = torch.tensor(0.1, dtype=torch.float64)
    hi = torch.tensor(0.1 + HALF_PI - 0.05, dtype=torch.float64)
    worst = assert_band_sound(relax, lo, hi)
    # chord vs sin over a smooth monotone arc: deviation is modest, < 0.2.
    assert worst < 0.2


def test_delta_grows_with_more_extrema(relax):
    # A multi-period interval has a large delta (chord can't track many waves).
    lo = torch.tensor(0.0, dtype=torch.float64)
    hi = torch.tensor(4.0 * PI, dtype=torch.float64)
    _, _, delta = relax.affine_band(lo, hi)
    # sin spans [-1, 1] and the chord slope is ~0, so delta ~ 1.
    assert float(delta) > 0.9
    assert_band_sound(relax, lo, hi)


# ---------------------------------------------------------------------------
# A broad random fuzz over many seeds to stress the critical-point enumeration.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('seed', list(range(12)))
def test_random_fuzz(relax, seed):
    torch.manual_seed(100 + seed)
    N = 150
    centers = (torch.rand(N, dtype=torch.float64) - 0.5) * 200.0  # +-100
    widths = torch.rand(N, dtype=torch.float64) * 12.0 + 1e-6     # up to 12 rad
    lo = centers - widths / 2
    hi = centers + widths / 2
    assert_band_sound(relax, lo, hi, n_samples=5000)
    assert_interval_sound(relax, lo, hi, n_samples=5000)
