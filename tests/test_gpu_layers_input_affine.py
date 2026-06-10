"""Unit tests for ComputeGraph.gpu_layers input-affine folding + loud raises.

cora's cifar10/svhn nets carry a scalar normalization preamble
``Mul(scale) -> Add(offset)`` before the first MatMul. gpu_layers used to
drop those two ops in its silent catch-all — every consumer (milp_verify's
zono bounds, joint-alpha bbr, the exact MILP encoding, the PGD forward) then
ran on the UNNORMALIZED network. That false-verified known-SAT cora cases
(cifar10-img339 canary, 2026-06-09: claimed joint-alpha lb +1.58 vs true
margin -0.05 at the counterexample).

The fix folds x |-> s*x + t into the first fc (W' = W diag(s),
b' = W t + b) and the catch-all now raises NotImplementedError, per the
no-silent-op-skip rule. Pins:
  1. forward equivalence of the folded net vs reference (scalar Mul+Add)
  2. per-element scale vector also folds correctly
  3. unsupported op (Sigmoid) raises
  4. normalization that never reaches a linear layer (ReLU first) raises
"""
import numpy as np
import onnx
import torch
from onnx import helper, TensorProto
import pytest

from vibecheck.network import ComputeGraph


def _init(name, arr):
    return helper.make_tensor(name, TensorProto.FLOAT, arr.shape, arr.flatten())


def _norm_net(tmp_path, scale, offset, name='norm_net.onnx'):
    """(x*scale + offset) -> 4x5 MatMul+Add -> ReLU -> 5x3 MatMul+Add."""
    rng = np.random.RandomState(0)
    W1 = rng.randn(4, 5).astype(np.float32) * 0.7
    b1 = rng.randn(5).astype(np.float32)
    W2 = rng.randn(5, 3).astype(np.float32) * 0.7
    b2 = rng.randn(3).astype(np.float32)
    nodes = [
        helper.make_node('Mul', ['X', 's'], ['xs']),
        helper.make_node('Add', ['xs', 't'], ['xn']),
        helper.make_node('MatMul', ['xn', 'W1'], ['m1']),
        helper.make_node('Add', ['m1', 'b1'], ['a1']),
        helper.make_node('Relu', ['a1'], ['r1']),
        helper.make_node('MatMul', ['r1', 'W2'], ['m2']),
        helper.make_node('Add', ['m2', 'b2'], ['Y']),
    ]
    inits = [_init('s', scale), _init('t', offset),
             _init('W1', W1), _init('b1', b1),
             _init('W2', W2), _init('b2', b2)]
    g = helper.make_graph(
        nodes, 'norm_net',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3])],
        inits)
    model = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    path = str(tmp_path / name)
    onnx.save(model, path)
    return ComputeGraph.from_onnx(path), (scale, offset, W1, b1, W2, b2)


def _assert_forward_matches(graph, params):
    scale, offset, W1, b1, W2, b2 = (np.asarray(p, np.float64) for p in params)
    layers, _ = graph.gpu_layers(torch.device('cpu'), torch.float64)
    assert [l['type'] for l in layers] == ['fc', 'fc']
    rng = np.random.RandomState(1)
    for _ in range(8):
        x = rng.randn(4)
        xn = x * scale.ravel() + offset.ravel()
        ref = np.maximum(xn @ W1 + b1, 0.0) @ W2 + b2
        h = torch.as_tensor(x, dtype=torch.float64)
        for i, L in enumerate(layers):
            h = L['W'] @ h + L['bias']
            if i < len(layers) - 1:
                h = torch.clamp(h, min=0)
        np.testing.assert_allclose(h.numpy(), ref, rtol=1e-9, atol=1e-9)


def test_scalar_mul_add_preamble_folds(tmp_path):
    scale = np.array([[3.9745631]], dtype=np.float32)
    offset = np.array([[-1.8815581]], dtype=np.float32)
    graph, params = _norm_net(tmp_path, scale, offset)
    _assert_forward_matches(graph, params)


def test_vector_mul_add_preamble_folds(tmp_path):
    scale = np.array([0.5, 2.0, -1.5, 4.0], dtype=np.float32)
    offset = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    graph, params = _norm_net(tmp_path, scale, offset,
                              name='norm_vec.onnx')
    _assert_forward_matches(graph, params)


def test_sub_div_preamble_folds(tmp_path):
    """acasxu-style (x - mean) / range preamble: Sub(sub_val) then Div."""
    rng = np.random.RandomState(2)
    mean = np.array([0.1, -0.3, 0.2, 0.05], dtype=np.float32)
    rngc = np.array([2.0, 0.5, 4.0, 1.0], dtype=np.float32)
    W1 = rng.randn(4, 5).astype(np.float32) * 0.7
    b1 = rng.randn(5).astype(np.float32)
    W2 = rng.randn(5, 3).astype(np.float32) * 0.7
    b2 = rng.randn(3).astype(np.float32)
    nodes = [
        helper.make_node('Sub', ['X', 'mean'], ['xc']),
        helper.make_node('Div', ['xc', 'rng'], ['xn']),
        helper.make_node('MatMul', ['xn', 'W1'], ['m1']),
        helper.make_node('Add', ['m1', 'b1'], ['a1']),
        helper.make_node('Relu', ['a1'], ['r1']),
        helper.make_node('MatMul', ['r1', 'W2'], ['m2']),
        helper.make_node('Add', ['m2', 'b2'], ['Y']),
    ]
    g = helper.make_graph(
        nodes, 'sub_div_net',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3])],
        [_init('mean', mean), _init('rng', rngc),
         _init('W1', W1), _init('b1', b1), _init('W2', W2), _init('b2', b2)])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    path = str(tmp_path / 'sub_div.onnx')
    onnx.save(model, path)
    graph = ComputeGraph.from_onnx(path)
    layers, _ = graph.gpu_layers(torch.device('cpu'), torch.float64)
    assert [l['type'] for l in layers] == ['fc', 'fc']
    rng2 = np.random.RandomState(3)
    for _ in range(8):
        x = rng2.randn(4)
        xn = (x - mean.astype(np.float64)) / rngc.astype(np.float64)
        ref = (np.maximum(xn @ W1.astype(np.float64) + b1, 0.0)
               @ W2.astype(np.float64) + b2)
        h = torch.as_tensor(x, dtype=torch.float64)
        for i, L in enumerate(layers):
            h = L['W'] @ h + L['bias']
            if i < len(layers) - 1:
                h = torch.clamp(h, min=0)
        np.testing.assert_allclose(h.numpy(), ref, rtol=1e-7, atol=1e-7)


def test_unsupported_op_raises(tmp_path):
    rng = np.random.RandomState(0)
    W1 = rng.randn(4, 3).astype(np.float32)
    b1 = rng.randn(3).astype(np.float32)
    nodes = [
        helper.make_node('MatMul', ['X', 'W1'], ['m1']),
        helper.make_node('Add', ['m1', 'b1'], ['a1']),
        helper.make_node('Sigmoid', ['a1'], ['Y']),
    ]
    g = helper.make_graph(
        nodes, 'sig_net',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3])],
        [_init('W1', W1), _init('b1', b1)])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    path = str(tmp_path / 'sig.onnx')
    onnx.save(model, path)
    graph = ComputeGraph.from_onnx(path)
    with pytest.raises(NotImplementedError, match='unsupported op'):
        graph.gpu_layers(torch.device('cpu'), torch.float32)


def test_preamble_hitting_relu_raises(tmp_path):
    rng = np.random.RandomState(0)
    s = np.array([[2.0]], dtype=np.float32)
    W1 = rng.randn(4, 3).astype(np.float32)
    b1 = rng.randn(3).astype(np.float32)
    nodes = [
        helper.make_node('Mul', ['X', 's'], ['xs']),
        helper.make_node('Relu', ['xs'], ['xr']),
        helper.make_node('MatMul', ['xr', 'W1'], ['m1']),
        helper.make_node('Add', ['m1', 'b1'], ['Y']),
    ]
    g = helper.make_graph(
        nodes, 'mul_relu_net',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3])],
        [_init('s', s), _init('W1', W1), _init('b1', b1)])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    path = str(tmp_path / 'mul_relu.onnx')
    onnx.save(model, path)
    graph = ComputeGraph.from_onnx(path)
    with pytest.raises(NotImplementedError, match='ReLU'):
        graph.gpu_layers(torch.device('cpu'), torch.float32)
