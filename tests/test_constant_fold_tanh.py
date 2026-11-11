"""Regression: a nonlinearity on a constant subgraph must be folded at load.

``u(x) - u(0)`` exports ``u`` a second time on a constant zero vector. The
Gemm folds to a constant, so the Tanh after it used to be left dangling with
no output shape and the graph failed to load. Tiny CPU net; verdicts below are
hand-checkable (|y| is far below 100, and y0 >= 0 is reachable)."""
import numpy as np
import pytest
import torch
from torch import nn

from vibecheck import Spec, verify
from vibecheck.frontend.onnx_loader import _try_fold_constant


class _Centered(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.u = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
        self.register_buffer('origin', torch.zeros(1, 2))

    def forward(self, x):
        return self.u(x) - self.u(self.origin)


def _run(c, capfd):
    spec = Spec(x_lo=[-1, -1], x_hi=[1, 1]).forbid([[1.0]], [c])
    r = verify(_Centered(), spec, timeout=30, device='cpu',
               example_input=torch.zeros(1, 2))
    err = capfd.readouterr().err
    assert 'graph load failed' not in err
    return r


def test_centered_tanh_unsat(capfd):
    assert _run(-100.0, capfd).verdict == 'unsat'


def test_centered_tanh_sat(capfd):
    r = _run(0.0, capfd)
    assert r.verdict == 'sat'
    assert r.counterexample['Y'].ravel()[0] >= -1e-5


@pytest.mark.parametrize('op,fn', [('Tanh', np.tanh), ('Sin', np.sin),
                                   ('Cos', np.cos), ('Exp', np.exp),
                                   ('Floor', np.floor)])
def test_fold_constant_nonlin(op, fn):
    v = np.array([-1.5, 0.0, 0.7])
    out = _try_fold_constant(op, ['c'], {}, lambda _: v)
    np.testing.assert_allclose(out, fn(v))
