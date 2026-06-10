"""Backward CROWN through the attention primitives (exp / reciprocal /
mul_bilinear / matmul_bilinear McCormick) — pins for the step-3 handlers.

Toy: a 2-token, 2-dim single-head attention block built in ONNX
(Q = X W_q, K = X W_k, V = X W_v, S = Q K^T, A = softmax(S),
Y = flatten(A V) W_o). The gg decomposes softmax into
exp/reduce_sum/reciprocal/mul_bilinear, so a backward pass exercises every
new handler. Pins:
  1. point box: backward spec lb == exact network value (all planes
     collapse)
  2. eps box: backward lb is SOUND vs 500 sampled inputs and not (much)
     looser than the forward-zono margin
"""
import os

import numpy as np
import onnx
import pytest
import torch
from onnx import helper, TensorProto


def _init(name, arr):
    return helper.make_tensor(name, TensorProto.FLOAT, arr.shape,
                              arr.flatten())


def _attn_net(tmpdir):
    from vibecheck.onnx_loader import load_onnx
    from vibecheck.settings import default_settings
    rng = np.random.RandomState(0)
    T, D = 2, 2
    Wq = rng.randn(D, D).astype(np.float32) * 0.6
    Wk = rng.randn(D, D).astype(np.float32) * 0.6
    Wv = rng.randn(D, D).astype(np.float32) * 0.6
    Wo = rng.randn(T * D, 3).astype(np.float32) * 0.6
    nodes = [
        # X: (1, T, D)
        helper.make_node('MatMul', ['X', 'Wq'], ['Q']),
        helper.make_node('MatMul', ['X', 'Wk'], ['K']),
        helper.make_node('MatMul', ['X', 'Wv'], ['V']),
        helper.make_node('Transpose', ['K'], ['Kt'], perm=[0, 2, 1]),
        helper.make_node('MatMul', ['Q', 'Kt'], ['S']),
        helper.make_node('Softmax', ['S'], ['A'], axis=-1),
        helper.make_node('MatMul', ['A', 'V'], ['AV']),
        helper.make_node('Reshape', ['AV', 'flat'], ['AVf']),
        helper.make_node('MatMul', ['AVf', 'Wo'], ['Y']),
    ]
    g = helper.make_graph(
        nodes, 'toy_attn',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, T, D])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3])],
        [_init('Wq', Wq), _init('Wk', Wk), _init('Wv', Wv), _init('Wo', Wo),
         helper.make_tensor('flat', TensorProto.INT64, [2],
                            np.array([1, T * D], np.int64))])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    path = os.path.join(tmpdir, 'attn.onnx')
    onnx.save(model, path)
    graph = load_onnx(path, dtype=np.float64, simplify=False)
    graph.optimize(default_settings())
    return graph


def _backward_lb(gg, xl, xh, w, b):
    from vibecheck.verify_zono_bnb import (_forward_zonotope_graph,
                                           _spec_backward_graph)
    dev = torch.device('cpu'); dt = torch.float64
    ob = {}
    sb, zf = _forward_zonotope_graph(xl, xh, gg, dev, dt, op_bounds=ob)
    w_t = torch.as_tensor(w, device=dev, dtype=dt)
    spec_lbs, _ = _spec_backward_graph(
        sb, xl, xh, gg, {0: (w_t, float(b))}, {0}, len(sb), dev, dt,
        op_bounds=ob)
    wg = w_t @ zf.generators
    fwd_lb = float(w_t @ zf.center + b - wg.abs().sum())
    return float(spec_lbs[0]), fwd_lb


def test_point_box_backward_exact(tmp_path):
    from vibecheck.verify_zono_bnb import _forward_batch_graph
    graph = _attn_net(str(tmp_path))
    gg = graph.gpu_graph(torch.device('cpu'), torch.float64)
    rng = np.random.default_rng(1)
    x = rng.uniform(-0.5, 0.5, 4)
    xt = torch.tensor(x, dtype=torch.float64)
    w = np.array([1.0, -2.0, 0.5])
    y = _forward_batch_graph(xt.unsqueeze(0), gg).flatten().numpy()
    exact = float(w @ y) + 0.25
    lb, fwd = _backward_lb(gg, xt.clone(), xt.clone(), w, 0.25)
    assert lb == pytest.approx(exact, abs=1e-7), (lb, exact)
    assert fwd == pytest.approx(exact, abs=1e-7)


def test_eps_box_backward_sound(tmp_path):
    from vibecheck.verify_zono_bnb import _forward_batch_graph
    graph = _attn_net(str(tmp_path))
    gg = graph.gpu_graph(torch.device('cpu'), torch.float64)
    rng = np.random.default_rng(2)
    xc = rng.uniform(-0.5, 0.5, 4)
    eps = 0.05
    xl = torch.tensor(xc - eps, dtype=torch.float64)
    xh = torch.tensor(xc + eps, dtype=torch.float64)
    w = np.array([1.0, -2.0, 0.5])
    lb, fwd = _backward_lb(gg, xl, xh, w, 0.25)
    xs = torch.tensor(rng.uniform(xl.numpy(), xh.numpy(), (500, 4)),
                      dtype=torch.float64)
    ys = _forward_batch_graph(xs, gg)
    vals = ys @ torch.tensor(w, dtype=torch.float64) + 0.25
    true_min = float(vals.min())
    assert lb <= true_min + 1e-9, f'UNSOUND backward: lb {lb} > {true_min}'
    # sanity: the backward shouldn't be wildly looser than the forward
    assert lb > fwd - 5.0, (lb, fwd)
