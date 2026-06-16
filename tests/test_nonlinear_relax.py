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
