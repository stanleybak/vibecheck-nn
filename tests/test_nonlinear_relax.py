"""Tests for the ScalarNonlinearRelax base class + registry."""
import pytest
import torch

from vibecheck.nonlinear_relax import (
    ScalarNonlinearRelax, register, REGISTRY,
    assert_band_sound, assert_interval_sound)


def test_base_methods_raise():
    b = ScalarNonlinearRelax()
    with pytest.raises(NotImplementedError):
        b.func(torch.zeros(1))
    with pytest.raises(NotImplementedError):
        b.interval(torch.zeros(1), torch.ones(1))
    with pytest.raises(NotImplementedError):
        b.affine_band(torch.zeros(1), torch.ones(1))


def test_register_and_helpers():
    @register('TestOp')
    class _Identity(ScalarNonlinearRelax):
        def func(self, x):
            return x
        def interval(self, lo, hi):
            return lo, hi
        def affine_band(self, lo, hi):  # y = 1*x + 0 +- 0 (exact)
            z = torch.zeros_like(torch.as_tensor(lo, dtype=torch.float64))
            return torch.ones_like(z), z, z

    try:
        assert REGISTRY['TestOp'] is _Identity
        assert _Identity.onnx_op == 'TestOp'
        assert_band_sound(_Identity(), torch.tensor([0.0]), torch.tensor([3.0]),
                          n_samples=200)
        assert_interval_sound(_Identity(), torch.tensor([0.0]), torch.tensor([3.0]),
                              n_samples=200)
    finally:
        del REGISTRY['TestOp']


def test_zono_affine_transform_sound():
    """The DeepZ affine transformer's output zonotope must over-approximate
    f(z) for every point z in the input zonotope (box-soundness over samples)."""
    import vibecheck.nl_sin  # noqa: F401  registers Sin
    from vibecheck.nonlinear_relax import zono_affine_transform, REGISTRY
    torch.manual_seed(0)
    relax = REGISTRY['Sin']()
    n, k = 6, 5
    center = torch.randn(n, dtype=torch.float64) * 2.0
    gens = 0.3 * torch.randn(n, k, dtype=torch.float64)
    new_c, new_g = zono_affine_transform(relax, center, gens)
    olo = new_c - new_g.abs().sum(1)
    ohi = new_c + new_g.abs().sum(1)
    e = 2 * torch.rand(40000, k, dtype=torch.float64) - 1
    z = center.unsqueeze(0) + e @ gens.T
    fz = torch.sin(z)
    assert bool((fz >= olo - 1e-9).all()) and bool((fz <= ohi + 1e-9).all()), \
        'transformer output zonotope does not over-approximate f(z)'


def test_zono_affine_transform_pad_is_dtype_aware():
    """The rel_pad in zono_affine_transform is a sound inflation covering FLOAT
    ROUNDING of lam*center+mu. A flat ~1e-6 (float32-sized) pad applied in
    float64 is ~10 orders too large: it adds a CONSTANT symmetric bloat that no
    slope/alpha can remove and that dwarfs sub-1e-6 spec margins (this was the
    ml4acopf full prop3 within-tol-sat-instead-of-unsat root cause). The pad
    default is now dtype-aware: ~1e-6 for float32, ~1e-12 for float64. Both stay
    SOUND (over-approximate the exact monotonic range)."""
    import vibecheck.nl_sigmoid_tanh  # noqa: F401  registers Sigmoid
    from vibecheck.nonlinear_relax import zono_affine_transform, REGISTRY
    relax = REGISTRY['Sigmoid']()
    # Deep-saturation interval z in [14, 15]: sigmoid ~ 1, near-flat.
    exact_hi = float(torch.sigmoid(torch.tensor(15.0, dtype=torch.float64)))

    # float64 dtype-aware pad vs an explicit float32-sized pad: the dtype-aware
    # default must be tighter by ~the removed ~1e-6 inflation. The residual
    # overshoot is then only the band's OWN chord-relaxation gap (~6e-8 here),
    # which the spec-time alpha-opt further shrinks by tuning lam — NOT the pad.
    c64 = torch.tensor([14.5], dtype=torch.float64)
    g64 = torch.tensor([[0.5]], dtype=torch.float64)
    nc64, ng64 = zono_affine_transform(relax, c64, g64)               # dtype-aware ~1e-12
    ub64 = float(nc64[0] + ng64[0].abs().sum())
    nc_p6, ng_p6 = zono_affine_transform(relax, c64, g64, rel_pad=1e-6)  # old flat pad
    ub_p6 = float(nc_p6[0] + ng_p6[0].abs().sum())
    assert ub64 >= exact_hi - 1e-15, 'float64 band UB must be sound (>= true max)'
    assert ub_p6 - ub64 > 5e-7, \
        f'dtype-aware pad should drop the ~1e-6 float32 inflation (got {ub_p6 - ub64:.2e})'
    assert ub64 - exact_hi < 5e-7, \
        f'residual overshoot {ub64 - exact_hi:.2e} should be only the band chord gap'

    # float32: the larger (~1e-6) pad is retained (sound for float32 rounding).
    nc32, ng32 = zono_affine_transform(relax, c64.float(), g64.float())
    ub32 = float(nc32[0] + ng32[0].abs().sum())
    assert ub32 >= exact_hi - 1e-6, 'float32 band UB must be sound'
    assert ub32 - exact_hi > 1e-7, 'float32 keeps the larger rounding pad'

    # An explicit rel_pad still overrides the dtype-aware default.
    nc_ex, ng_ex = zono_affine_transform(relax, c64, g64, rel_pad=1e-3)
    ub_ex = float(nc_ex[0] + ng_ex[0].abs().sum())
    assert ub_ex - exact_hi > 1e-4, 'explicit rel_pad must override the default'
