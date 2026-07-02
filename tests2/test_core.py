"""Tier-0 unit tests for the vibecheck2 core (design 4.2).

Run: PYTHONPATH=src <venv>/bin/python -m pytest tests2/ -q
Soundness invariants only need numpy/torch; no benchmark files.
"""
import numpy as np
import pytest
import torch

from vibecheck2.core import linmap as lm
from vibecheck2.core import memory
from vibecheck2.core.relax import REL

torch.manual_seed(0)
_rng = np.random.default_rng(0)


def _linmaps():
    """One instance of every LinMap layout, small random params."""
    W = _rng.normal(size=(7, 5)).astype(np.float32)
    b = _rng.normal(size=7).astype(np.float32)
    k = _rng.normal(size=(4, 3, 3, 3)).astype(np.float32)
    kb = _rng.normal(size=4).astype(np.float32)
    kt = _rng.normal(size=(3, 4, 2, 2)).astype(np.float32)
    yield 'dense', lm.Dense(W, b)
    yield 'dense_nobias', lm.Dense(W, None)
    yield 'conv', lm.Conv2d(k, kb, (3, 6, 6), (4, 4, 4), (1, 1), (0, 0))
    yield 'conv_pad_stride', lm.Conv2d(k, kb, (3, 6, 6), (4, 3, 3), (2, 2), (1, 1))
    yield 'convT', lm.ConvT2d(kt, kb, (3, 5, 5), (4, 11, 11), (2, 2), (0, 0),
                              output_padding=(1, 1))
    yield 'avgpool', lm.AvgPool((3, 6, 6), (3, 3, 3), (2, 2), (2, 2), (0, 0))
    yield 'select', lm.Select(_rng.integers(0, 12, size=9), 12)
    yield 'scale_shift', lm.ScaleShift(_rng.normal(size=11).astype(np.float32),
                                       _rng.normal(size=11).astype(np.float32), 11)
    yield 'sum_axis', lm.SumAxis(3, 4, 5)
    yield 'mean_axis', lm.SumAxis(3, 4, 5, mean=True)


@pytest.mark.parametrize('name,m', list(_linmaps()))
def test_linmap_adjoint_identity(name, m):
    """<lin(x), y> == <x, lin_t(y)> for random x, y (exact adjoint)."""
    X = torch.randn(6, m.n_in)
    Y = torch.randn(6, m.n_out)
    lhs = (m.lin(X) * Y).sum(dim=1)
    rhs = (X * m.lin_t(Y)).sum(dim=1)
    assert torch.allclose(lhs, rhs, atol=1e-4), (name, (lhs - rhs).abs().max())


@pytest.mark.parametrize('name,m', list(_linmaps()))
def test_linmap_abs_dominates(name, m):
    """lin_abs on a nonnegative vector bounds |lin| on any sign pattern:
    |lin(s*r)| <= lin_abs(r) for r >= 0 and any s in {-1,1}^n."""
    r = torch.rand(4, m.n_in)
    bound = m.lin_abs(r)
    for _ in range(8):
        s = torch.where(torch.rand(4, m.n_in) < 0.5, -1.0, 1.0)
        val = m.lin(s * r)
        assert (val.abs() <= bound + 1e-4).all(), name


@pytest.mark.parametrize('name,m', list(_linmaps()))
def test_linmap_point_is_lin_plus_bias(name, m):
    X = torch.randn(3, m.n_in)
    assert torch.allclose(m.point(X), m.lin(X) + m.bias_vec(X), atol=1e-5)


def test_relu_planes_bracket():
    """Sampling VALIDATES the planes (never defines them): dense adversarial
    sampling incl. endpoints must satisfy al*x+bl <= relu(x) <= au*x+bu."""
    lo = torch.tensor([[-3.0, -1e-6, 0.0, 0.5, -2.0]])
    hi = torch.tensor([[2.0, 1e-6, 0.0, 1.5, -0.5]])
    al, bl, au, bu = REL['relu'].planes(lo, hi)
    for t in torch.linspace(0, 1, 101):
        x = lo + t * (hi - lo)
        y = torch.relu(x)
        assert (al * x + bl <= y + 1e-6).all()
        assert (y <= au * x + bu + 1e-6).all()


def test_memory_chunked_matches_unchunked():
    X = torch.randn(37, 4)
    fn = lambda b: b * 2 + 1                                  # noqa: E731
    out = memory.chunked(fn, X, bytes_per_item=1)             # forces chunks
    assert torch.equal(out, fn(X))
    out1 = memory.chunked(fn, X, bytes_per_item=1e12)         # single item/chunk
    assert torch.equal(out1, fn(X))


# --------------------------------------------------------------------------- #
# tiny synthetic net: forward mode soundness on a DAG with a residual merge
# --------------------------------------------------------------------------- #

def _tiny_residual_net(tmp_path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    W1 = numpy_helper.from_array(_rng.normal(size=(4, 4)).astype(np.float32).T, 'W1')
    W2 = numpy_helper.from_array(_rng.normal(size=(4, 4)).astype(np.float32).T, 'W2')
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 4])
    g = helper.make_graph(
        [helper.make_node('MatMul', ['X', 'W1'], ['h1']),
         helper.make_node('Relu', ['h1'], ['r1']),
         helper.make_node('MatMul', ['r1', 'W2'], ['h2']),
         helper.make_node('Add', ['h2', 'X'], ['s']),        # residual merge
         helper.make_node('Relu', ['s'], ['Y'])],
        'g', [X], [Y], [W1, W2])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    p = str(tmp_path / 'res.onnx')
    onnx.save(m, p)
    return p


def test_forward_modes_sound_on_residual_dag(tmp_path):
    from vibecheck2.core import forward as fwd
    from vibecheck2.core import graph as g2
    net = g2.load(_tiny_residual_net(tmp_path))
    lo = torch.full((1, 4), -0.5)
    hi = torch.full((1, 4), 0.7)
    ilo, ihi = fwd.interval(net, lo, hi)
    zlo, zhi = fwd.zono(net, lo, hi)
    # both contain many exact point evaluations (validation sampling)
    xs = torch.rand(256, 4) * (hi - lo) + lo
    ys = fwd.point(net, xs)
    assert (ys >= zlo - 1e-4).all() and (ys <= zhi + 1e-4).all()
    assert (ys >= ilo - 1e-4).all() and (ys <= ihi + 1e-4).all()


def test_forward_batched_boxes(tmp_path):
    """Batched boxes bound their own samples (per-domain isolation)."""
    from vibecheck2.core import forward as fwd
    from vibecheck2.core import graph as g2
    net = g2.load(_tiny_residual_net(tmp_path))
    lo = torch.tensor([[-1.0] * 4, [0.1] * 4, [-0.2] * 4])
    hi = torch.tensor([[-0.5] * 4, [0.9] * 4, [0.3] * 4])
    zlo, zhi = fwd.zono(net, lo, hi)
    for b in range(3):
        xs = torch.rand(128, 4) * (hi[b] - lo[b]) + lo[b]
        ys = fwd.point(net, xs)
        assert (ys >= zlo[b] - 1e-4).all() and (ys <= zhi[b] + 1e-4).all(), b
