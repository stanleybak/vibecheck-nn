"""Unit tests for onnx_optimizer.min_max_to_relu — exact Min/Max -> ReLU+affine.

Validated through the production gpu_graph point forward (the path verification
actually runs), comparing to torch min/max/clamp; and that no Min/Max node remains.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import torch
import pytest

from vibecheck.network import ComputeGraph
from vibecheck.onnx_optimizer import min_max_to_relu


def _const_clamp_onnx(path, lo, hi):
    """X[1,5] -> Min(X, hi) -> Max(., lo) -> Y  (i.e. clamp(X, lo, hi))."""
    nodes = [
        helper.make_node('Min', ['X', 'HI'], ['cl']),
        helper.make_node('Max', ['cl', 'LO'], ['Y']),
    ]
    inits = [numpy_helper.from_array(np.asarray(hi, np.float32), 'HI'),
             numpy_helper.from_array(np.asarray(lo, np.float32), 'LO')]
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 5])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 5])
    m = helper.make_model(helper.make_graph(nodes, 'clamp', [X], [Y], inits),
                          opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    onnx.save(m, path)


def _twovar_weights():
    rng = np.random.default_rng(0)
    W1 = rng.standard_normal((4, 3)).astype(np.float32)
    W2 = rng.standard_normal((4, 3)).astype(np.float32)   # second draw, same rng
    return W1, W2


def _twovar_onnx(path):
    """X[1,4] -> A=X@W1, B=X@W2 -> Max(A,B) and Min(A,B) -> Add -> Y[1,3]."""
    W1, W2 = _twovar_weights()
    nodes = [
        helper.make_node('MatMul', ['X', 'W1'], ['A']),
        helper.make_node('MatMul', ['X', 'W2'], ['B']),
        helper.make_node('Max', ['A', 'B'], ['mx']),
        helper.make_node('Min', ['A', 'B'], ['mn']),
        helper.make_node('Add', ['mx', 'mn'], ['Y']),
    ]
    inits = [numpy_helper.from_array(W1, 'W1'), numpy_helper.from_array(W2, 'W2')]
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3])
    m = helper.make_model(helper.make_graph(nodes, 'twovar', [X], [Y], inits),
                          opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    onnx.save(m, path)


def test_const_clamp_exact(tmp_path):
    from vibecheck.verify_zono_bnb import _forward_batch_graph
    lo = [-0.5, -0.2, 0.0, 0.1, -1.0]; hi = [0.5, 0.3, 0.2, 0.9, 1.0]
    p = str(tmp_path / 'c.onnx'); _const_clamp_onnx(p, lo, hi)
    g = ComputeGraph.from_onnx(p, dtype=np.float64)
    assert any(n.op_type in ('Min', 'Max') for n in g.nodes.values())
    assert min_max_to_relu(g) is True
    assert not any(n.op_type in ('Min', 'Max') for n in g.nodes.values())
    assert min_max_to_relu(g) is False                    # second run no-op
    gg = g.gpu_graph(device='cpu', dtype=torch.float64)
    lo_t = torch.tensor(lo, dtype=torch.float64); hi_t = torch.tensor(hi, dtype=torch.float64)
    for seed in range(4):
        torch.manual_seed(seed)
        x = torch.randn(1, 5, dtype=torch.float64) * 2
        ref = torch.maximum(torch.minimum(x.reshape(-1), hi_t), lo_t)
        out = _forward_batch_graph(x.reshape(1, -1), gg).reshape(-1).double()
        # atol covers float32-stored clamp bounds (lo/hi) vs the float64 reference
        torch.testing.assert_close(out, ref, atol=1e-6, rtol=0)


def test_twovar_exact(tmp_path):
    from vibecheck.verify_zono_bnb import _forward_batch_graph
    p = str(tmp_path / 't.onnx'); _twovar_onnx(p)
    g = ComputeGraph.from_onnx(p, dtype=np.float64)
    assert min_max_to_relu(g) is True
    assert not any(n.op_type in ('Min', 'Max') for n in g.nodes.values())
    gg = g.gpu_graph(device='cpu', dtype=torch.float64)
    _W1, _W2 = _twovar_weights()
    W1 = torch.tensor(_W1, dtype=torch.float64); W2 = torch.tensor(_W2, dtype=torch.float64)
    for seed in range(4):
        torch.manual_seed(seed)
        x = torch.randn(1, 4, dtype=torch.float64)
        a = x @ W1; b = x @ W2
        ref = (torch.maximum(a, b) + torch.minimum(a, b)).reshape(-1)
        out = _forward_batch_graph(x.reshape(1, -1), gg).reshape(-1).double()
        torch.testing.assert_close(out, ref, atol=1e-9, rtol=0)
