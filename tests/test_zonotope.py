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


# -------------------------------------------------------------
# TorchZonotope 4D-native representation tests
# -------------------------------------------------------------

import torch
from vibecheck.zonotope import TorchZonotope


def _build_pair(n, K, C, H, W, seed=0):
    """Return (z_old_path, z_new_path) — both (n=C*H*W, K) generators.

    z_old_path forces the 2D representation by touching `.generators`
    after each op, whereas z_new_path keeps the native 4D form.
    """
    assert n == C * H * W
    rng = torch.Generator().manual_seed(seed)
    center = torch.randn(n, dtype=torch.float64, generator=rng)
    gens = torch.randn(n, K, dtype=torch.float64, generator=rng)
    return (TorchZonotope(center.clone(), gens.clone()),
            TorchZonotope(center.clone(), gens.clone()))


def test_torch_zono_4d_conv_matches_2d():
    """After one conv, the 4D-native and 2D-forced paths produce the
    same generators (mod float round-off)."""
    C_in, H_in, W_in = 4, 8, 8
    n = C_in * H_in * W_in
    K = 6
    z_new, z_force2d = _build_pair(n, K, C_in, H_in, W_in, seed=1)
    kernel = torch.randn(3, C_in, 3, 3, dtype=torch.float64)
    bias = torch.randn(3, dtype=torch.float64)
    z_new.propagate_conv(kernel, bias, (C_in, H_in, W_in), (1, 1), (1, 1))
    # 4D should be populated, 2D cache empty
    assert z_new._gen_4d is not None
    assert z_new._gen_2d is None

    z_force2d.propagate_conv(kernel, bias, (C_in, H_in, W_in), (1, 1), (1, 1))
    _ = z_force2d.generators  # force materialization to 2D

    torch.testing.assert_close(z_new.generators, z_force2d.generators,
                                atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(z_new.center, z_force2d.center)


def test_torch_zono_4d_consecutive_convs_stay_in_4d():
    """A conv→conv chain stays 4D internally and matches a 2D-forced
    path bit-for-bit up to rounding."""
    C, H, W = 4, 8, 8
    n = C * H * W
    K = 5
    z_a, z_b = _build_pair(n, K, C, H, W, seed=2)
    k1 = torch.randn(4, C, 3, 3, dtype=torch.float64)
    k2 = torch.randn(4, 4, 3, 3, dtype=torch.float64)
    b1 = torch.randn(4, dtype=torch.float64)
    b2 = torch.randn(4, dtype=torch.float64)
    # z_a: native 4D chain
    z_a.propagate_conv(k1, b1, (C, H, W), (1, 1), (1, 1))
    z_a.propagate_conv(k2, b2, (4, H, W), (1, 1), (1, 1))
    assert z_a._gen_4d is not None  # stayed 4D throughout
    # z_b: force 2D between convs by reading generators
    z_b.propagate_conv(k1, b1, (C, H, W), (1, 1), (1, 1))
    _ = z_b.generators
    z_b.propagate_conv(k2, b2, (4, H, W), (1, 1), (1, 1))
    torch.testing.assert_close(z_a.generators, z_b.generators,
                                atol=1e-9, rtol=1e-9)


def test_torch_zono_4d_relu_matches_2d():
    """apply_relu in 4D mode matches the 2D path."""
    C, H, W = 2, 4, 4
    n = C * H * W
    K = 4
    z_4d, z_2d = _build_pair(n, K, C, H, W, seed=3)
    kernel = torch.randn(C, C, 3, 3, dtype=torch.float64)
    bias = torch.randn(C, dtype=torch.float64)
    z_4d.propagate_conv(kernel, bias, (C, H, W), (1, 1), (1, 1))
    z_2d.propagate_conv(kernel, bias, (C, H, W), (1, 1), (1, 1))
    _ = z_2d.generators  # force 2D
    lo_4d, hi_4d = z_4d.apply_relu()
    lo_2d, hi_2d = z_2d.apply_relu()
    torch.testing.assert_close(z_4d.generators, z_2d.generators,
                                atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(z_4d.center, z_2d.center)
    torch.testing.assert_close(lo_4d, lo_2d)
    torch.testing.assert_close(hi_4d, hi_2d)


def test_torch_zono_4d_bounds_matches_2d():
    """bounds() in 4D mode matches the 2D path."""
    C, H, W = 3, 6, 6
    n = C * H * W
    K = 7
    z_4d, z_2d = _build_pair(n, K, C, H, W, seed=4)
    kernel = torch.randn(C, C, 3, 3, dtype=torch.float64)
    bias = torch.randn(C, dtype=torch.float64)
    z_4d.propagate_conv(kernel, bias, (C, H, W), (1, 1), (1, 1))
    z_2d.propagate_conv(kernel, bias, (C, H, W), (1, 1), (1, 1))
    _ = z_2d.generators
    lo_4d, hi_4d = z_4d.bounds()
    lo_2d, hi_2d = z_2d.bounds()
    torch.testing.assert_close(lo_4d, lo_2d)
    torch.testing.assert_close(hi_4d, hi_2d)


def test_torch_zono_4d_relu_appends_unstable_generators():
    """apply_relu in 4D mode correctly appends new generators for
    unstable neurons (matching 2D behavior)."""
    C, H, W = 2, 3, 3
    n = C * H * W
    # Construct a zonotope where all neurons are unstable — center 0,
    # generators are unit vectors — so ReLU will add 18 new generators.
    center = torch.zeros(n, dtype=torch.float64)
    gens = 0.5 * torch.eye(n, dtype=torch.float64)  # n×n
    z_4d = TorchZonotope(center.clone(), gens.clone())
    z_2d = TorchZonotope(center.clone(), gens.clone())
    # Force z_4d into 4D form by running a passthrough conv (identity 1×1)
    identity = torch.zeros(C, C, 1, 1, dtype=torch.float64)
    for i in range(C):
        identity[i, i, 0, 0] = 1.0
    z_4d.propagate_conv(identity, torch.zeros(C, dtype=torch.float64),
                         (C, H, W), (1, 1), (0, 0))
    _ = z_4d.generators  # materialize to 2D for comparison later
    # Rebuild 4D state so apply_relu uses the 4D path
    z_4d._gen_4d = z_4d._gen_2d.t().contiguous().reshape(n, C, H, W)
    z_4d._gen_2d = None
    z_4d.apply_relu()
    z_2d.apply_relu()
    torch.testing.assert_close(z_4d.generators, z_2d.generators,
                                atol=1e-9, rtol=1e-9)


def test_torch_zono_copy_preserves_mode():
    """copy() preserves whichever form (2D or 4D) is currently active."""
    C, H, W = 2, 4, 4
    n = C * H * W
    z = TorchZonotope(
        torch.randn(n, dtype=torch.float64),
        torch.randn(n, 5, dtype=torch.float64))
    kernel = torch.randn(C, C, 3, 3, dtype=torch.float64)
    z.propagate_conv(kernel, torch.zeros(C, dtype=torch.float64),
                     (C, H, W), (1, 1), (1, 1))
    assert z._gen_4d is not None
    z_copy = z.copy()
    assert z_copy._gen_4d is not None
    assert z_copy._gen_2d is None
    torch.testing.assert_close(z_copy._gen_4d, z._gen_4d)
    # Independence: mutating the copy does not affect the original.
    z_copy._gen_4d.fill_(0.0)
    assert not torch.allclose(z._gen_4d, torch.zeros_like(z._gen_4d))


def test_torch_zono_4d_end_to_end_conv_relu_conv_matches_2d():
    """Full conv→relu→conv chain. End result generators match."""
    C, H, W = 3, 8, 8
    n = C * H * W
    K = 6
    z_a, z_b = _build_pair(n, K, C, H, W, seed=7)
    k1 = torch.randn(4, C, 3, 3, dtype=torch.float64)
    k2 = torch.randn(4, 4, 3, 3, dtype=torch.float64)
    b1 = torch.randn(4, dtype=torch.float64)
    b2 = torch.randn(4, dtype=torch.float64)

    # z_a: native 4D
    z_a.propagate_conv(k1, b1, (C, H, W), (1, 1), (1, 1))
    z_a.apply_relu()
    z_a.propagate_conv(k2, b2, (4, H, W), (1, 1), (1, 1))

    # z_b: force 2D between every op
    z_b.propagate_conv(k1, b1, (C, H, W), (1, 1), (1, 1))
    _ = z_b.generators
    z_b.apply_relu()
    _ = z_b.generators
    z_b.propagate_conv(k2, b2, (4, H, W), (1, 1), (1, 1))

    torch.testing.assert_close(z_a.generators, z_b.generators,
                                atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(z_a.center, z_b.center)


def test_torch_zono_get_gen_row_4d_no_materialization():
    """`get_gen_row` on a 4D zonotope returns the correct row without
    collapsing the 4D cache to 2D."""
    C, H, W = 3, 4, 5
    n = C * H * W
    K = 6
    rng = torch.Generator().manual_seed(7)
    center = torch.randn(n, dtype=torch.float64, generator=rng)
    gens = torch.randn(n, K, dtype=torch.float64, generator=rng)
    z = TorchZonotope(center.clone(), gens.clone())
    # Force 4D state via identity conv.
    identity = torch.zeros(C, C, 1, 1, dtype=torch.float64)
    for i in range(C):
        identity[i, i, 0, 0] = 1.0
    z.propagate_conv(identity, torch.zeros(C, dtype=torch.float64),
                     (C, H, W), (1, 1), (0, 0))
    assert z._gen_4d is not None
    row = z.get_gen_row(7)  # arbitrary flat neuron
    assert z._gen_4d is not None  # still 4D
    assert z._gen_2d is None
    # Cross-check against the 2D materialized form.
    gens_2d = z.generators
    torch.testing.assert_close(row, gens_2d[7, :], atol=0, rtol=0)


def test_torch_zono_get_gen_row_2d():
    """`get_gen_row` returns a row slice when generators are 2D."""
    n, K = 5, 4
    rng = torch.Generator().manual_seed(11)
    center = torch.randn(n, dtype=torch.float64, generator=rng)
    gens = torch.randn(n, K, dtype=torch.float64, generator=rng)
    z = TorchZonotope(center, gens)
    assert z._gen_4d is None
    torch.testing.assert_close(z.get_gen_row(2), gens[2, :], atol=0, rtol=0)


# -------------------------------------------------------------
# TorchZonotope.add — fast (in-place, K_b==shared) vs slow (new alloc).
# Verifies that the in-place fast path produces bit-identical output to
# the slow path used when K_b > shared_gens.
# -------------------------------------------------------------

def _slow_add_reference(a_center, a_gens, b_center, b_gens, shared_gens):
    """Reference implementation: always the slow/allocating path.
    Used to check the fast in-place path agrees column-for-column.
    """
    n = a_gens.shape[0]
    K_a, K_b = a_gens.shape[1], b_gens.shape[1]
    K_out = K_a + K_b - shared_gens
    out = torch.empty(n, K_out, dtype=a_gens.dtype)
    out[:, :shared_gens] = a_gens[:, :shared_gens] + b_gens[:, :shared_gens]
    if K_a > shared_gens:
        out[:, shared_gens:K_a] = a_gens[:, shared_gens:]
    if K_b > shared_gens:
        out[:, K_a:] = b_gens[:, shared_gens:]
    return a_center + b_center, out


def test_torch_zonotope_add_fast_path_Kb_equals_shared():
    """K_b == shared_gens — this triggers the fast in-place path.

    The returned zonotope must match the slow-path reference column-for-column
    AND must BE `self` (in-place mutation: `add` returned the z_a instance).
    """
    rng = torch.Generator().manual_seed(17)
    n, shared = 50, 30
    K_a_extra = 20
    K_a = shared + K_a_extra
    K_b = shared  # skip-branch has no extras — triggers fast path

    c_a = torch.randn(n, dtype=torch.float64, generator=rng)
    c_b = torch.randn(n, dtype=torch.float64, generator=rng)
    G_shared_a = torch.randn(n, shared, dtype=torch.float64, generator=rng)
    G_a_extra = torch.randn(n, K_a_extra, dtype=torch.float64, generator=rng)
    G_a = torch.cat([G_shared_a, G_a_extra], dim=1)
    # Branch B inherits the shared prefix from the fork — *different values*
    # than A in those columns (because A went through convs too, but we
    # simulate that by using arbitrary values), still aligned col-for-col.
    G_shared_b = torch.randn(n, shared, dtype=torch.float64, generator=rng)
    G_b = G_shared_b

    # Reference (slow path).
    ref_center, ref_gens = _slow_add_reference(c_a, G_a, c_b, G_b, shared)

    # Fast path.
    z_a = TorchZonotope(c_a.clone(), G_a.clone())
    z_b = TorchZonotope(c_b.clone(), G_b.clone())
    merged = z_a.add(z_b, shared)

    # Bit-identical to reference.
    torch.testing.assert_close(merged.center, ref_center, atol=0, rtol=0)
    torch.testing.assert_close(merged.generators, ref_gens, atol=0, rtol=0)
    # Output is the mutated self (in-place fast path).
    assert merged is z_a
    # b is untouched.
    torch.testing.assert_close(z_b.generators, G_b, atol=0, rtol=0)


def test_torch_zonotope_add_slow_path_both_branches_have_extras():
    """K_a > shared AND K_b > shared — the slow path must run (new allocation).

    The result must NOT be self (slow path returns a fresh TorchZonotope).
    """
    rng = torch.Generator().manual_seed(19)
    n, shared = 50, 30
    K_a_extra, K_b_extra = 20, 10
    K_a, K_b = shared + K_a_extra, shared + K_b_extra

    c_a = torch.randn(n, dtype=torch.float64, generator=rng)
    c_b = torch.randn(n, dtype=torch.float64, generator=rng)
    G_a = torch.randn(n, K_a, dtype=torch.float64, generator=rng)
    G_b = torch.randn(n, K_b, dtype=torch.float64, generator=rng)

    ref_center, ref_gens = _slow_add_reference(c_a, G_a, c_b, G_b, shared)

    z_a = TorchZonotope(c_a.clone(), G_a.clone())
    z_b = TorchZonotope(c_b.clone(), G_b.clone())
    merged = z_a.add(z_b, shared)

    torch.testing.assert_close(merged.center, ref_center, atol=0, rtol=0)
    torch.testing.assert_close(merged.generators, ref_gens, atol=0, rtol=0)
    # Slow path: result is a fresh object, not `self`.
    assert merged is not z_a
    # a and b unchanged.
    torch.testing.assert_close(z_a.generators, G_a, atol=0, rtol=0)
    torch.testing.assert_close(z_b.generators, G_b, atol=0, rtol=0)


def test_torch_zonotope_add_shared_gens_zero():
    """shared_gens == 0 — both branches are fully branch-specific (disjoint
    noise symbols). K_b == 0 triggers the fast path (K_b == shared); K_b > 0
    with shared=0 triggers slow path."""
    rng = torch.Generator().manual_seed(23)
    n, K_a = 40, 15

    c_a = torch.randn(n, dtype=torch.float64, generator=rng)
    c_b = torch.randn(n, dtype=torch.float64, generator=rng)
    G_a = torch.randn(n, K_a, dtype=torch.float64, generator=rng)

    # Case 1: K_b == 0 == shared (degenerate "empty skip" — fast path).
    G_b = torch.zeros(n, 0, dtype=torch.float64)
    z_a = TorchZonotope(c_a.clone(), G_a.clone())
    z_b = TorchZonotope(c_b.clone(), G_b)
    merged = z_a.add(z_b, 0)
    torch.testing.assert_close(merged.center, c_a + c_b, atol=0, rtol=0)
    torch.testing.assert_close(merged.generators, G_a, atol=0, rtol=0)
    assert merged is z_a

    # Case 2: K_b > 0 with shared==0 (no shared prefix — slow path).
    K_b = 8
    G_b = torch.randn(n, K_b, dtype=torch.float64, generator=rng)
    z_a = TorchZonotope(c_a.clone(), G_a.clone())
    z_b = TorchZonotope(c_b.clone(), G_b.clone())
    merged = z_a.add(z_b, 0)
    ref_center, ref_gens = _slow_add_reference(c_a, G_a, c_b, G_b, 0)
    torch.testing.assert_close(merged.center, ref_center, atol=0, rtol=0)
    torch.testing.assert_close(merged.generators, ref_gens, atol=0, rtol=0)
    assert merged is not z_a


def test_torch_zonotope_add_Ka_equals_shared_Kb_equals_shared():
    """K_a == K_b == shared — both branches trivially match (no extras).

    Fast path mutates a; result center = a+b, gens = a+b elementwise.
    """
    rng = torch.Generator().manual_seed(29)
    n, K = 25, 12

    c_a = torch.randn(n, dtype=torch.float64, generator=rng)
    c_b = torch.randn(n, dtype=torch.float64, generator=rng)
    G_a = torch.randn(n, K, dtype=torch.float64, generator=rng)
    G_b = torch.randn(n, K, dtype=torch.float64, generator=rng)
    expected_gens = G_a + G_b

    z_a = TorchZonotope(c_a.clone(), G_a.clone())
    z_b = TorchZonotope(c_b.clone(), G_b.clone())
    merged = z_a.add(z_b, K)

    torch.testing.assert_close(merged.center, c_a + c_b, atol=0, rtol=0)
    torch.testing.assert_close(merged.generators, expected_gens, atol=0, rtol=0)
    assert merged is z_a


