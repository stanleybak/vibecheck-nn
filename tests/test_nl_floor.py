"""Soundness tests for the elementwise Floor relaxation (nl_floor.FloorRelax).

We exercise assert_band_sound / assert_interval_sound (which densely sample the
band/interval to TEST a closed-form bound) over the case classes the floor
staircase makes tricky: within a single integer cell, spanning one or many
boundaries, exact integer endpoints, negative ranges, and mixed batched shapes.
The within-cell case is additionally asserted to be EXACT (delta == 0).
"""
import torch

from vibecheck.nl_floor import FloorRelax
from vibecheck.nonlinear_relax import (
    REGISTRY,
    assert_band_sound,
    assert_interval_sound,
)


def _r():
    return FloorRelax()


def test_registered():
    assert REGISTRY['Floor'] is FloorRelax
    assert FloorRelax.onnx_op == 'Floor'


def test_func_matches_torch_floor():
    x = torch.tensor([-2.7, -1.0, -0.3, 0.0, 0.5, 1.9, 2.0, 5.5, 7.0])
    assert torch.equal(_r().func(x), torch.floor(x))


# ---- within a single integer cell: exact, zero-error band ------------------

def test_within_cell_band_exact():
    r = _r()
    lo, hi = torch.tensor([2.1]), torch.tensor([2.7])
    lam, mu, delta = r.affine_band(lo, hi)
    assert float(lam) == 0.0
    assert float(mu) == 2.0
    assert float(delta) == 0.0
    assert_band_sound(r, lo, hi)
    assert_interval_sound(r, lo, hi)


def test_within_cell_negative_exact():
    # floor is constant -2 on [-1.9, -1.1]
    r = _r()
    lo, hi = torch.tensor([-1.9]), torch.tensor([-1.1])
    lam, mu, delta = r.affine_band(lo, hi)
    assert float(lam) == 0.0
    assert float(mu) == -2.0
    assert float(delta) == 0.0
    assert_band_sound(r, lo, hi)
    assert_interval_sound(r, lo, hi)


def test_within_cell_integer_left_endpoint_exact():
    # [2.0, 2.9]: floor(2.0)=floor(2.9)=2, constant cell -> exact.
    r = _r()
    lo, hi = torch.tensor([2.0]), torch.tensor([2.9])
    lam, mu, delta = r.affine_band(lo, hi)
    assert float(lam) == 0.0
    assert float(mu) == 2.0
    assert float(delta) == 0.0
    assert_band_sound(r, lo, hi)
    assert_interval_sound(r, lo, hi)


def test_degenerate_integer_point_exact():
    # [1.0, 1.0]: single integer point, floor == 1, constant -> exact.
    r = _r()
    lo, hi = torch.tensor([1.0]), torch.tensor([1.0])
    lam, mu, delta = r.affine_band(lo, hi)
    assert float(lam) == 0.0
    assert float(mu) == 1.0
    assert float(delta) == 0.0
    assert_band_sound(r, lo, hi)
    assert_interval_sound(r, lo, hi)


# ---- spanning a boundary: lam=1, mu=-0.5, delta=0.5 ------------------------

def test_spanning_one_boundary():
    r = _r()
    lo, hi = torch.tensor([1.8]), torch.tensor([2.3])
    lam, mu, delta = r.affine_band(lo, hi)
    assert float(lam) == 1.0
    assert float(mu) == -0.5
    assert float(delta) == 0.5
    assert_band_sound(r, lo, hi)
    assert_interval_sound(r, lo, hi)


def test_spanning_just_at_boundary():
    # [1.9, 2.0]: floor(1.9)=1 != floor(2.0)=2 -> spanning (2.0 is right-closed)
    r = _r()
    lo, hi = torch.tensor([1.9]), torch.tensor([2.0])
    lam, mu, delta = r.affine_band(lo, hi)
    assert float(lam) == 1.0
    assert float(mu) == -0.5
    assert float(delta) == 0.5
    assert_band_sound(r, lo, hi)
    assert_interval_sound(r, lo, hi)


def test_wide_many_integers():
    r = _r()
    lo, hi = torch.tensor([-3.4]), torch.tensor([7.9])
    lam, mu, delta = r.affine_band(lo, hi)
    assert float(lam) == 1.0
    assert float(delta) == 0.5
    assert_band_sound(r, lo, hi)
    assert_interval_sound(r, lo, hi)


def test_integer_endpoints_span():
    # [2.0, 5.0]: floor(2.0)=2 != floor(5.0)=5 -> spanning band, endpoints int.
    r = _r()
    lo, hi = torch.tensor([2.0]), torch.tensor([5.0])
    lam, mu, delta = r.affine_band(lo, hi)
    assert float(lam) == 1.0
    assert float(mu) == -0.5
    assert float(delta) == 0.5
    assert_band_sound(r, lo, hi)
    assert_interval_sound(r, lo, hi)


# ---- negative ranges (common floor sign bugs) ------------------------------

def test_negative_spanning():
    # floor(-2.7) = -3, floor(-0.3) = -1 ; spanning two boundaries.
    r = _r()
    lo, hi = torch.tensor([-2.7]), torch.tensor([-0.3])
    lam, mu, delta = r.affine_band(lo, hi)
    assert float(lam) == 1.0
    assert float(mu) == -0.5
    assert float(delta) == 0.5
    # interval endpoints land on the right integers
    olo, ohi = r.interval(lo, hi)
    assert float(olo) == -3.0
    assert float(ohi) == -1.0
    assert_band_sound(r, lo, hi)
    assert_interval_sound(r, lo, hi)


def test_negative_crossing_zero():
    # floor(-0.7) = -1, floor(0.7) = 0 ; spans the 0 boundary.
    r = _r()
    lo, hi = torch.tensor([-0.7]), torch.tensor([0.7])
    olo, ohi = r.interval(lo, hi)
    assert float(olo) == -1.0
    assert float(ohi) == 0.0
    assert_band_sound(r, lo, hi)
    assert_interval_sound(r, lo, hi)


# ---- batched vector mixing within-cell and spanning ------------------------

def test_vector_mixed():
    r = _r()
    #              within   spanning  neg-span   int-pt   wide        neg-within
    lo = torch.tensor([2.1, 1.8, -2.7, 1.0, -3.4, -1.9])
    hi = torch.tensor([2.7, 2.3, -0.3, 1.0, 7.9, -1.1])
    lam, mu, delta = r.affine_band(lo, hi)
    # within-cell elements (indices 0, 3, 5) must be exact
    exact = torch.tensor([True, False, False, True, False, True])
    assert torch.equal(delta == 0.0, exact)
    assert torch.equal(lam == 0.0, exact)
    assert_band_sound(r, lo, hi)
    assert_interval_sound(r, lo, hi)


def test_vector_2d_shape():
    r = _r()
    lo = torch.tensor([[-2.7, 2.1], [1.8, 5.0]])
    hi = torch.tensor([[-0.3, 2.7], [2.3, 8.0]])
    lam, mu, delta = r.affine_band(lo, hi)
    assert lam.shape == lo.shape
    assert mu.shape == lo.shape
    assert delta.shape == lo.shape
    assert_band_sound(r, lo, hi)
    assert_interval_sound(r, lo, hi)
