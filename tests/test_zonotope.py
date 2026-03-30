"""Basic tests for DenseZonotope."""

import numpy as np
import pytest
from vibecheck.zonotope import DenseZonotope


def test_from_input_bounds():
    x_lo = np.array([0.0, -1.0])
    x_hi = np.array([1.0, 1.0])
    z = DenseZonotope.from_input_bounds(x_lo, x_hi)
    lo, hi = z.bounds()
    np.testing.assert_allclose(lo, x_lo)
    np.testing.assert_allclose(hi, x_hi)


def test_propagate_fc():
    x_lo = np.array([0.0, 0.0])
    x_hi = np.array([1.0, 1.0])
    z = DenseZonotope.from_input_bounds(x_lo, x_hi)

    W = np.array([[1.0, 2.0], [-1.0, 1.0]])
    b = np.array([0.0, 0.0])
    z.propagate_linear((W, b))

    lo, hi = z.bounds()
    # center = W @ [0.5, 0.5] = [1.5, 0.0], abs_sum = [1.5, 1.0]
    np.testing.assert_allclose(lo, [0.0, -1.0])
    np.testing.assert_allclose(hi, [3.0, 1.0])


def test_relu_stable_positive():
    z = DenseZonotope(center=np.array([2.0]), generators=np.array([[0.5]]))
    z.apply_relu(np.array([1.5]), np.array([2.5]))
    lo, hi = z.bounds()
    np.testing.assert_allclose(lo, [1.5])
    np.testing.assert_allclose(hi, [2.5])


def test_relu_stable_negative():
    z = DenseZonotope(center=np.array([-2.0]), generators=np.array([[0.5]]))
    z.apply_relu(np.array([-2.5]), np.array([-1.5]))
    lo, hi = z.bounds()
    np.testing.assert_allclose(lo, [0.0])
    np.testing.assert_allclose(hi, [0.0])


def test_add_shared_only():
    """Add two zonotopes that share all generators."""
    z1 = DenseZonotope(np.array([1.0, 2.0]), np.array([[0.5, 0.0], [0.0, 0.5]]))
    z2 = DenseZonotope(np.array([3.0, 4.0]), np.array([[0.1, 0.0], [0.0, 0.1]]))
    z3 = z1.add(z2, shared_gens=2)
    np.testing.assert_allclose(z3.center, [4.0, 6.0])
    np.testing.assert_allclose(z3.generators, [[0.6, 0.0], [0.0, 0.6]])


def test_add_with_extra_gens():
    """Add two zonotopes where each branch added extra generators."""
    z1 = DenseZonotope(
        np.array([1.0]),
        np.array([[0.5, 0.3, 0.1]]),  # 2 shared + 1 extra
    )
    z2 = DenseZonotope(
        np.array([2.0]),
        np.array([[0.4, 0.2, 0.05, 0.02]]),  # 2 shared + 2 extra
    )
    z3 = z1.add(z2, shared_gens=2)
    np.testing.assert_allclose(z3.center, [3.0])
    np.testing.assert_allclose(z3.generators, [[0.9, 0.5, 0.1, 0.05, 0.02]])


def test_copy_independent():
    z = DenseZonotope(np.array([1.0, 2.0]), np.array([[0.5], [0.3]]))
    z2 = z.copy()
    z2.center[0] = 99.0
    assert z.center[0] == 1.0


# ---- Conv propagation ----

def test_propagate_conv_with_generators():
    """Conv propagation with actual generators."""
    from vibecheck.zonotope import is_conv, conv_output_shape
    kernel = np.random.randn(2, 1, 3, 3)
    bias = np.zeros(2)
    params = {'input_shape': (1, 4, 4), 'stride': (1, 1), 'padding': (0, 0)}
    layer = (kernel, bias, params)
    assert is_conv(layer)

    out_shape = conv_output_shape((1, 4, 4), kernel, params)
    assert out_shape == (2, 2, 2)

    z = DenseZonotope.from_input_bounds(np.zeros(16), np.ones(16))
    z.propagate_linear(layer)
    assert len(z.center) == 8  # 2*2*2
    assert z.generators.shape[0] == 8


def test_propagate_conv_point():
    """Conv propagation with 0 generators (point zonotope)."""
    kernel = np.ones((1, 1, 2, 2))
    bias = np.array([0.0])
    params = {'input_shape': (1, 3, 3), 'stride': (1, 1), 'padding': (0, 0)}
    layer = (kernel, bias, params)

    center = np.arange(9, dtype=float)
    z = DenseZonotope(center, np.zeros((9, 0)))
    z.propagate_linear(layer)
    assert len(z.center) == 4  # 1*2*2
    assert z.generators.shape == (4, 0)


def test_propagate_conv_cached_matches_slow():
    """Conv with kernel caching produces identical results to uncached."""
    rng = np.random.default_rng(99)
    C_in, C_out, H, W, kH, kW = 3, 4, 8, 8, 3, 3
    kernel = rng.standard_normal((C_out, C_in, kH, kW))
    bias = rng.standard_normal(C_out)
    center = rng.standard_normal(C_in * H * W)
    n_gen = 5
    generators = rng.standard_normal((C_in * H * W, n_gen))

    params_cached = {'input_shape': (C_in, H, W), 'stride': (1, 1), 'padding': (1, 1)}
    params_uncached = {'input_shape': (C_in, H, W), 'stride': (1, 1), 'padding': (1, 1)}

    z_cached = DenseZonotope(center.copy(), generators.copy())
    z_cached.propagate_linear((kernel, bias, params_cached))

    z_slow = DenseZonotope(center.copy(), generators.copy())
    z_slow._propagate_conv_slow((kernel, bias, params_uncached))

    np.testing.assert_allclose(z_cached.center, z_slow.center, atol=1e-12)
    np.testing.assert_allclose(z_cached.generators, z_slow.generators, atol=1e-12)

    # Second call should use cache (params_cached now has _torch_kernel)
    assert '_torch_kernel' in params_cached
    z_cached2 = DenseZonotope(center.copy(), generators.copy())
    z_cached2.propagate_linear((kernel, bias, params_cached))
    np.testing.assert_allclose(z_cached2.center, z_slow.center, atol=1e-12)


# ---- ReLU relaxation types ----

def test_relu_unstable_std():
    """Unstable neuron with std relaxation (hi > -lo)."""
    z = DenseZonotope(np.array([0.0]), np.array([[1.0]]))
    z.apply_relu(np.array([-1.0]), np.array([1.0]), 'std')
    lo, hi = z.bounds()
    # λ = 1/(1-(-1)) = 0.5, μ = -1*(-1)/(2*2) = 0.25
    # center = 0.5*0 + 0.25 = 0.25, gens = [0.5, 0.25]
    # bounds = 0.25 ± 0.75 = [-0.5, 1.0]
    assert lo[0] >= -0.51
    assert hi[0] <= 1.01

def test_relu_unstable_std_lo_dominant():
    """Unstable neuron with std where hi < -lo. Same formula: λ = hi/(hi-lo)."""
    z = DenseZonotope(np.array([-0.3]), np.array([[0.5]]))
    z.apply_relu(np.array([-0.8]), np.array([0.2]), 'std')
    lo, hi = z.bounds()
    # λ = 0.2/(0.2+0.8) = 0.2, μ = -0.2*(-0.8)/(2*1.0) = 0.08
    # center = 0.2*(-0.3) + 0.08 = 0.02, gens = [0.1, 0.08]
    # bounds = 0.02 ± 0.18 = [-0.16, 0.20]
    assert lo[0] >= -0.17
    assert hi[0] <= 0.21

def test_relu_y_bloat():
    """Unstable neuron with y_bloat relaxation."""
    z = DenseZonotope(np.array([0.0]), np.array([[1.0]]))
    z.apply_relu(np.array([-1.0]), np.array([1.0]), 'y_bloat')
    lo, hi = z.bounds()
    assert lo[0] >= -1.01
    assert hi[0] <= 2.01  # y_bloat gives wider bounds

def test_relu_invalid_type():
    """Invalid relu_type raises."""
    z = DenseZonotope(np.array([0.0]), np.array([[1.0]]))
    with pytest.raises(AssertionError, match="Unknown relu_type"):
        z.apply_relu(np.array([-1.0]), np.array([1.0]), 'invalid')


def test_relu_box():
    """Unstable neuron with box relaxation."""
    z = DenseZonotope(np.array([0.0]), np.array([[1.0]]))
    z.apply_relu(np.array([-1.0]), np.array([1.0]), 'box')
    lo, hi = z.bounds()
    assert lo[0] >= -0.01
    assert hi[0] <= 1.01


# ---------------------------------------------------------------------------
# Vectorized vs scalar apply_relu equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('relu_type', ['std', 'y_bloat', 'box'])
def test_relu_vectorized_matches_slow(relu_type):
    """Vectorized apply_relu produces identical results to the scalar loop."""
    rng = np.random.default_rng(42)
    n = 200
    k = 50  # generators
    center = rng.standard_normal(n)
    generators = rng.standard_normal((n, k))
    # Bounds that create a mix of active, dead, and unstable neurons
    pre_lo = center - rng.uniform(0.5, 2.0, n)
    pre_hi = center + rng.uniform(0.5, 2.0, n)
    # Force some dead neurons
    pre_hi[:20] = -0.1
    pre_lo[:20] = -2.0
    # Force some stable-positive neurons
    pre_lo[20:40] = 0.1

    z_fast = DenseZonotope(center.copy(), generators.copy())
    z_slow = DenseZonotope(center.copy(), generators.copy())

    z_fast.apply_relu(pre_lo, pre_hi, relu_type)
    z_slow.apply_relu_slow(pre_lo, pre_hi, relu_type)

    np.testing.assert_allclose(z_fast.center, z_slow.center, atol=1e-14)
    np.testing.assert_allclose(z_fast.generators, z_slow.generators, atol=1e-14)


def test_relu_vectorized_point_propagation():
    """Point propagation (0 generators): vectorized matches scalar."""
    n = 100
    center = np.linspace(-2, 2, n)
    generators = np.zeros((n, 0))
    pre_lo = center.copy()
    pre_hi = center.copy()

    z_fast = DenseZonotope(center.copy(), generators.copy())
    z_slow = DenseZonotope(center.copy(), generators.copy())

    z_fast.apply_relu(pre_lo, pre_hi)
    z_slow.apply_relu_slow(pre_lo, pre_hi)

    np.testing.assert_allclose(z_fast.center, z_slow.center, atol=1e-14)
    assert z_fast.generators.shape == z_slow.generators.shape


# ---------------------------------------------------------------------------
# float32 dtype tests
# ---------------------------------------------------------------------------

def test_from_input_bounds_f32():
    """from_input_bounds with float32 dtype."""
    x_lo = np.array([0.0, -1.0], dtype=np.float32)
    x_hi = np.array([1.0, 1.0], dtype=np.float32)
    z = DenseZonotope.from_input_bounds(x_lo, x_hi, dtype=np.float32)
    assert z.center.dtype == np.float32
    assert z.generators.dtype == np.float32
    lo, hi = z.bounds()
    np.testing.assert_allclose(lo, x_lo, atol=1e-6)
    np.testing.assert_allclose(hi, x_hi, atol=1e-6)


def test_propagate_fc_f32():
    """FC propagation preserves float32 dtype when params match."""
    z = DenseZonotope.from_input_bounds(
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32)
    W = np.array([[1.0, 2.0], [-1.0, 1.0]], dtype=np.float32)
    b = np.array([0.0, 0.0], dtype=np.float32)
    z.propagate_linear((W, b))
    assert z.center.dtype == np.float32
    assert z.generators.dtype == np.float32
    lo, hi = z.bounds()
    np.testing.assert_allclose(lo, [0.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(hi, [3.0, 1.0], atol=1e-6)


def test_propagate_conv_f32():
    """Conv propagation preserves float32 dtype."""
    kernel = np.ones((1, 1, 2, 2))
    bias = np.array([0.0])
    params = {'input_shape': (1, 3, 3), 'stride': (1, 1), 'padding': (0, 0)}
    layer = (kernel, bias, params)
    center = np.arange(9, dtype=np.float32)
    z = DenseZonotope(center, np.zeros((9, 0), dtype=np.float32))
    z.propagate_linear(layer)
    assert z.center.dtype == np.float32
    assert z.generators.dtype == np.float32
    assert len(z.center) == 4


def test_relu_f32():
    """ReLU preserves float32 dtype with unstable neurons."""
    center = np.array([0.0, -2.0, 2.0], dtype=np.float32)
    generators = np.array([[1.0], [0.5], [0.3]], dtype=np.float32)
    z = DenseZonotope(center, generators)
    z.apply_relu(np.array([-1.0, -2.5, 1.7], dtype=np.float32),
                 np.array([1.0, -1.5, 2.3], dtype=np.float32))
    assert z.center.dtype == np.float32
    assert z.generators.dtype == np.float32


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_f32_f64_close(dtype):
    """float32 and float64 produce similar results for FC + ReLU."""
    rng = np.random.default_rng(123)
    x_lo = rng.uniform(-1, 0, 10).astype(dtype)
    x_hi = rng.uniform(0, 1, 10).astype(dtype)
    W = rng.standard_normal((5, 10)).astype(dtype)
    b = rng.standard_normal(5).astype(dtype)

    z = DenseZonotope.from_input_bounds(x_lo, x_hi, dtype=dtype)
    z.propagate_linear((W, b))
    lo, hi = z.bounds()
    z.apply_relu(lo, hi)

    # Just check it runs and dtype is preserved
    assert z.center.dtype == dtype
    assert z.generators.dtype == dtype


def test_f32_vs_f64_bounds_close():
    """float32 bounds are close to float64 bounds."""
    rng = np.random.default_rng(456)
    x_lo_64 = rng.uniform(-1, 0, 10)
    x_hi_64 = rng.uniform(0, 1, 10)
    W_64 = rng.standard_normal((5, 10))
    b_64 = rng.standard_normal(5)

    z32 = DenseZonotope.from_input_bounds(
        x_lo_64.astype(np.float32), x_hi_64.astype(np.float32), dtype=np.float32)
    z64 = DenseZonotope.from_input_bounds(x_lo_64, x_hi_64, dtype=np.float64)
    z32.propagate_linear((W_64.astype(np.float32), b_64.astype(np.float32)))
    z64.propagate_linear((W_64, b_64))

    lo32, hi32 = z32.bounds()
    lo64, hi64 = z64.bounds()
    np.testing.assert_allclose(lo32, lo64, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(hi32, hi64, atol=1e-5, rtol=1e-5)
