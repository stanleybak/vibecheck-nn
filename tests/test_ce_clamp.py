"""FP-safe clamping of a counterexample witness into the input box.

A box edge like `x >= 9.2` is usually not representable in float32, so a
witness sitting on the edge can round to the OUTSIDE of the box when the
VNNCOMP scorer casts it to the model's float32 input (`float32(9.2) < 9.2`),
making a valid counterexample score as out-of-box. `_clamp_witness_to_box`
must pull such a component back inside so it survives the float32 cast.
"""
import numpy as np

from vibecheck.verify_graph import _clamp_witness_to_box


def _f32(x):
    return np.float32(x).astype(np.float64)


def test_lower_edge_rounds_down_is_pulled_inside():
    # lo = a float64 just ABOVE a float32 value f, so float32(lo) == f < lo:
    # a witness at lo would cast to f, landing under the floor.
    f = np.float32(9.2)
    lo = float(np.nextafter(np.float64(f), np.inf))
    hi = lo + 1.0
    assert _f32(lo) < lo                                 # precondition: rounds out
    c = _clamp_witness_to_box(np.array([lo]), [lo], [hi])
    assert _f32(float(c[0])) >= lo and _f32(float(c[0])) <= hi


def test_upper_edge_rounds_up_is_pulled_inside():
    # hi = a float64 just BELOW a float32 value f, so float32(hi) == f > hi.
    f = np.float32(0.5)
    hi = float(np.nextafter(np.float64(f), -np.inf))
    lo = hi - 1.0
    assert _f32(hi) > hi                                 # precondition: rounds out
    c = _clamp_witness_to_box(np.array([hi]), [lo], [hi])
    assert _f32(float(c[0])) <= hi and _f32(float(c[0])) >= lo


def test_interior_point_unchanged():
    lo = np.array([0.0, -1.0])
    hi = np.array([1.0, 1.0])
    w = np.array([0.5, -0.25])           # float32-exact interior points
    c = _clamp_witness_to_box(w, lo, hi)
    assert np.allclose(c, w, atol=0.0)


def test_out_of_box_witness_clamped_to_edge_and_inside():
    lo = np.array([0.0])
    hi = np.array([1.0])
    c = _clamp_witness_to_box(np.array([5.0]), lo, hi)   # far above hi
    assert _f32(float(c[0])) <= hi[0] and _f32(float(c[0])) >= lo[0]


def test_returned_witness_is_float64_in_box():
    rng = np.random.default_rng(0)
    lo = rng.uniform(-3, 0, 16)
    hi = lo + rng.uniform(0.1, 3, 16)
    w = rng.uniform(-5, 5, 16)
    c = _clamp_witness_to_box(w, lo, hi)
    cf = c.astype(np.float32).astype(np.float64)
    assert np.all(cf >= lo) and np.all(cf <= hi)
    assert c.dtype == np.float64
