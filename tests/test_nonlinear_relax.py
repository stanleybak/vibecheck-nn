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
