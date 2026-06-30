"""Tests for the merged 1-D piecewise-linear lookup table (nl_pwl.PWLRelax) and
the onnx_optimizer.merge_relu_lookup_table rewrite.

Sampling is used ONLY to TEST closed-form bounds (the correct use per the project
soundness rule), via assert_band_sound / assert_interval_sound. The merge is
validated through the production gpu_graph point forward (the path verification
actually runs) against the analytic f, and that the lookup-table ReLU is gone.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import torch
import pytest

from vibecheck.nl_pwl import PWLRelax
from vibecheck.nonlinear_relax import assert_band_sound, assert_interval_sound
from vibecheck.network import ComputeGraph
from vibecheck.onnx_optimizer import merge_relu_lookup_table


# --------------------------------------------------------------------------- #
# PWLRelax: func exactness + soundness of interval / affine_band.
# --------------------------------------------------------------------------- #
def _analytic(x, off, w, b):
    x = torch.as_tensor(x, dtype=torch.float64)
    off = torch.as_tensor(off, dtype=torch.float64)
    w = torch.as_tensor(w, dtype=torch.float64)
    return b + (torch.clamp(x.unsqueeze(-1) - off, min=0.0) * w).sum(-1)


def test_func_matches_relu_sum():
    off = np.array([-2.0, 0.0, 1.5, 3.0])
    w = np.array([1.0, -2.0, 0.5, 2.0])
    r = PWLRelax(off, w, bias=0.7)
    x = torch.linspace(-6, 6, 501, dtype=torch.float64)
    assert torch.allclose(r.func(x), _analytic(x, off, w, 0.7))


def test_slope_at_is_subgradient():
    off = np.array([-1.0, 0.0, 2.0])
    w = np.array([1.0, 1.0, -3.0])
    r = PWLRelax(off, w)
    # left of all offsets -> 0; right of all -> sum(w)
    assert float(r.slope_at(torch.tensor(-5.0))) == 0.0
    assert float(r.slope_at(torch.tensor(5.0))) == pytest.approx(float(w.sum()))


# random tables incl. non-monotone (mixed-sign weights) + adversarial boxes
@pytest.mark.parametrize('seed', range(6))
def test_pwl_interval_and_band_sound(seed):
    g = torch.Generator().manual_seed(seed)
    K = int(torch.randint(3, 12, (1,), generator=g).item())
    off = torch.sort(torch.randn(K, generator=g) * 3).values
    w = torch.randn(K, generator=g) * 2          # mixed signs -> non-monotone
    b = float(torch.randn(1, generator=g) * 1.5)
    r = PWLRelax(off, w, b)
    lo = torch.randn(300, generator=g) * 4 - 2
    hi = lo + torch.rand(300, generator=g) * 6 + 1e-3
    assert_interval_sound(r, lo, hi)             # raises if unsound
    assert_band_sound(r, lo, hi)                 # default chord slope
    # ANY lam must stay sound (the alpha-CROWN hook)
    for lam in (torch.zeros(300), torch.full((300,), 2.0), r.slope_at(lo)):
        L, mu, delta = r.affine_band(lo, hi, lam=lam)
        t = torch.linspace(0, 1, 256, dtype=torch.float64)
        xs = lo.unsqueeze(1) + (hi - lo).unsqueeze(1) * t
        xs = torch.minimum(torch.maximum(xs, lo.unsqueeze(1)), hi.unsqueeze(1))
        dev = (r.func(xs) - (L.unsqueeze(1) * xs + mu.unsqueeze(1))).abs()
        assert float((dev - delta.unsqueeze(1)).max()) <= 1e-6


def test_band_exact_on_breakpoints():
    # f convex-ish PWL: chord band's delta should be the exact half-range.
    off = np.array([0.0, 1.0, 2.0])
    w = np.array([1.0, 1.0, 1.0])                # f(x)=sum relu -> convex
    r = PWLRelax(off, w)
    lo = torch.tensor([-1.0]); hi = torch.tensor([3.0])
    lam, mu, delta = r.affine_band(lo, hi)
    # convex f under its chord -> band is one-sided; verify it brackets exactly
    xs = torch.linspace(-1, 3, 200, dtype=torch.float64).unsqueeze(0)
    dev = (r.func(xs) - (lam.unsqueeze(1) * xs + mu.unsqueeze(1))).abs()
    assert float(dev.max()) <= float(delta) + 1e-9


# --------------------------------------------------------------------------- #
# merge_relu_lookup_table: exact graph rewrite.
# --------------------------------------------------------------------------- #
def _lookup_table_onnx(path, n, off, w, bias):
    """X[1,n] -> Unsqueeze(-1) -> Sub(off) -> Relu -> MatMul(W) -> Add(bias) -> Y.
    i.e. Y[j] = bias + sum_i w_i * ReLU(X[j] - off_i)."""
    K = len(off)
    nodes = [
        helper.make_node('Unsqueeze', ['X', 'axes'], ['unsq']),
        helper.make_node('Sub', ['unsq', 'OFF'], ['sub']),
        helper.make_node('Relu', ['sub'], ['rl']),
        helper.make_node('MatMul', ['rl', 'W'], ['mm']),
        helper.make_node('Add', ['mm', 'B'], ['Y']),
    ]
    inits = [
        numpy_helper.from_array(np.array([-1], np.int64), 'axes'),
        numpy_helper.from_array(np.asarray(off, np.float32), 'OFF'),
        numpy_helper.from_array(np.asarray(w, np.float32), 'W'),
        numpy_helper.from_array(np.asarray(bias, np.float32), 'B'),
    ]
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, n])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, n])
    m = helper.make_model(helper.make_graph(nodes, 'lut', [X], [Y], inits),
                          opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    onnx.save(m, path)


def test_merge_creates_pwl_and_is_exact(tmp_path):
    from vibecheck.verify_zono_bnb import _forward_batch_graph
    n = 5
    off = [-1.5, 0.0, 0.8, 2.0]
    w = [1.0, -2.0, 0.5, 1.5]
    bias = 0.3
    p = str(tmp_path / 'lut.onnx')
    _lookup_table_onnx(p, n, off, w, bias)
    g = ComputeGraph.from_onnx(p, dtype=np.float64)
    assert merge_relu_lookup_table(g) == 1
    # the lookup-table ReLU is gone; a PWLLookup node exists.
    assert not any(nd.op_type == 'Relu' for nd in g.nodes.values())
    assert sum(nd.op_type == 'PWLLookup' for nd in g.nodes.values()) == 1
    assert merge_relu_lookup_table(g) == 0          # second run no-op
    gg = g.gpu_graph(device='cpu', dtype=torch.float64)
    for seed in range(4):
        torch.manual_seed(seed)
        x = torch.randn(1, n, dtype=torch.float64) * 3
        ref = _analytic(x.reshape(-1), off, w, bias)
        out = _forward_batch_graph(x.reshape(1, -1), gg).reshape(-1).double()
        torch.testing.assert_close(out, ref, atol=1e-6, rtol=0)


def test_optimize_merge_gate_by_param_count(tmp_path):
    """optimize() gates merge_relu_lookup_table on the net's parameter count
    (merge_relu_lookup_table_min_params): large nets fold the lookup table to a
    PWL node (the expanded ReLU stack blows the bound up), small nets keep the
    expanded ReLU (its triangle relaxation is tighter). See ml4acopf 14_ieee
    (~6.5K params, merge OFF) vs 118/300 (~292K-1.3M, merge ON)."""
    from vibecheck.settings import default_settings
    n = 5
    off = [-1.5, 0.0, 0.8, 2.0]
    w = [1.0, -2.0, 0.5, 1.5]
    bias = 0.3
    p = str(tmp_path / 'lut.onnx')
    _lookup_table_onnx(p, n, off, w, bias)

    # threshold 0 -> param count clears it -> merge runs -> PWL, ReLU gone.
    g = ComputeGraph.from_onnx(p, dtype=np.float64)
    g.optimize(default_settings(merge_relu_lookup_table=True,
                                merge_relu_lookup_table_min_params=0))
    assert sum(nd.op_type == 'PWLLookup' for nd in g.nodes.values()) == 1
    assert not any(nd.op_type == 'Relu' for nd in g.nodes.values())

    # threshold above this tiny net's param count -> merge skipped -> ReLU kept.
    g2 = ComputeGraph.from_onnx(p, dtype=np.float64)
    g2.optimize(default_settings(merge_relu_lookup_table=True,
                                 merge_relu_lookup_table_min_params=10 ** 9))
    assert not any(nd.op_type == 'PWLLookup' for nd in g2.nodes.values())
    assert any(nd.op_type == 'Relu' for nd in g2.nodes.values())


def test_merge_skips_nonmatching_graph(tmp_path):
    """A plain Gemm->Relu net has no lookup table -> merge is a no-op."""
    nodes = [helper.make_node('MatMul', ['X', 'W'], ['z']),
             helper.make_node('Relu', ['z'], ['Y'])]
    W = np.random.default_rng(0).standard_normal((4, 3)).astype(np.float32)
    inits = [numpy_helper.from_array(W, 'W')]
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3])
    m = helper.make_model(helper.make_graph(nodes, 'g', [X], [Y], inits),
                          opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 9
    p = str(tmp_path / 'g.onnx'); onnx.save(m, p)
    g = ComputeGraph.from_onnx(p, dtype=np.float64)
    assert merge_relu_lookup_table(g) == 0
    assert any(nd.op_type == 'Relu' for nd in g.nodes.values())
