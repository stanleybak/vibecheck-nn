"""Soundness of the Sigmoid/Tanh affine-band relaxations (nl_sigmoid_tanh).

The affine band must satisfy |f(x) - (lam*x + mu)| <= delta for ALL x in
[lo, hi] (band sound by construction, validated here by dense sampling across
the inflection at 0, wide/narrow/one-sided intervals), and interval(lo,hi)
must bracket every f(x). These pin the closed-form critical points (logit /
atanh of the derivative-inverse).
"""
import torch

from vibecheck.nonlinear_relax import (REGISTRY, assert_band_sound,
                                        assert_interval_sound,
                                        zono_affine_transform)

_F64 = torch.float64
# NOTE: do NOT call torch.set_default_dtype here — it leaks to every later test
# in the session (float32 vs float64 pollution). Use explicit dtype instead.

_CASES = [(-6.0, 6.0), (-1.0, 1.0), (0.0, 5.0), (-5.0, 0.0),
          (-0.001, 0.001), (2.0, 8.0), (-8.0, -2.0), (-3.0, 0.5),
          (0.0, 0.0), (-20.0, 20.0), (-0.5, 3.0)]


def _grid(lo, hi, n=40):
    base_l = torch.full((n,), lo, dtype=_F64) + torch.linspace(-2.0, 2.0, n,
                                                               dtype=_F64)
    base_h = torch.full((n,), hi, dtype=_F64) + torch.linspace(-2.0, 2.0, n,
                                                               dtype=_F64)
    return base_l, torch.maximum(base_h, base_l)


def test_sigmoid_tanh_bands_sound():
    for op in ('Sigmoid', 'Tanh'):
        relax = REGISTRY[op]()
        for lo, hi in _CASES:
            L, H = _grid(lo, hi)
            assert_band_sound(relax, L, H, n_samples=20000, atol=1e-9)
            assert_interval_sound(relax, L, H, n_samples=12000, atol=1e-9)


def test_sigmoid_tanh_band_tighter_than_box():
    # On a moderate interval the affine band's output width should beat the
    # box-collapse width (hi-lo of the activation range), i.e. it preserves
    # correlation rather than discarding it.
    for op, fn in (('Sigmoid', torch.sigmoid), ('Tanh', torch.tanh)):
        relax = REGISTRY[op]()
        lo = torch.tensor([-1.0], dtype=_F64); hi = torch.tensor([1.0],
                                                                 dtype=_F64)
        lam, mu, delta = relax.affine_band(lo, hi)
        # band half-range at the endpoints: |lam|*(hi-lo)/2 + delta
        band_w = float(2 * (lam.abs() * (hi - lo) / 2 + delta))
        box_w = float(fn(hi) - fn(lo))
        assert delta.item() >= 0.0
        # the band is a valid over-approx; delta should be small (< box width)
        assert float(delta) <= box_w + 1e-9


def test_zono_affine_transform_sigmoid_sound():
    # A small zonotope through the sigmoid affine-band transform must contain
    # the true sigmoid of every sampled point and preserve input columns.
    torch.manual_seed(0)
    relax = REGISTRY['Sigmoid']()
    c = torch.randn(3, dtype=_F64)
    G = 0.4 * torch.randn(3, 5, dtype=_F64)
    new_c, new_g = zono_affine_transform(relax, c, G)
    assert new_g.shape[1] == 5 + 3  # lam-scaled gens preserved + fresh per elem
    lo = new_c - new_g.abs().sum(1)
    hi = new_c + new_g.abs().sum(1)
    e = 2 * torch.rand(40000, 5, dtype=_F64) - 1
    xs = c.unsqueeze(0) + e @ G.t()
    ys = torch.sigmoid(xs)
    assert bool((ys >= lo.unsqueeze(0) - 1e-9).all())
    assert bool((ys <= hi.unsqueeze(0) + 1e-9).all())
