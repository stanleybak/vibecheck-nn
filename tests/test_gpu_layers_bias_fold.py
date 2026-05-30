"""Unit test for ComputeGraph.gpu_layers folding a standalone Add(bias).

TF/Keras-style exports emit each affine layer as `MatMul` + a SEPARATE
`Add(bias)` node (the MatMul's own bias is zero). gpu_layers flattens the
graph into sequential linear layers for milp_verify's MILP encoding; it must
fold that trailing Add's constant into the preceding linear layer's bias.
Before this was fixed the flattened net was bias-free and milp_verify solved a
DIFFERENT (looser) network — it reported phantom counterexamples on safenlp and
silently verified a bias-free acasxu. The forward-match assertion below pins
both that the bias is present AND that it lands in the correct (pre-activation)
position: a dropped or mis-placed bias would change the output.
"""
import numpy as np
import onnx
import torch
from onnx import helper, TensorProto

from vibecheck.network import ComputeGraph


def _init(name, arr):
    return helper.make_tensor(name, TensorProto.FLOAT, arr.shape, arr.flatten())


def _matmul_add_net(tmp_path, name='matmul_add.onnx'):
    """4 -> 5 (ReLU) -> 5 (ReLU) -> 3, each affine = MatMul + separate Add."""
    rng = np.random.RandomState(0)
    # onnx MatMul semantics: y = x @ W, so W is (in, out).
    W1 = rng.randn(4, 5).astype(np.float32) * 0.7
    b1 = rng.randn(5).astype(np.float32)          # non-zero biases on purpose
    W2 = rng.randn(5, 5).astype(np.float32) * 0.7
    b2 = rng.randn(5).astype(np.float32)
    W3 = rng.randn(5, 3).astype(np.float32) * 0.7
    b3 = rng.randn(3).astype(np.float32)
    nodes = [
        helper.make_node('MatMul', ['X', 'W1'], ['m1']),
        helper.make_node('Add', ['m1', 'b1'], ['a1']),
        helper.make_node('Relu', ['a1'], ['r1']),
        helper.make_node('MatMul', ['r1', 'W2'], ['m2']),
        helper.make_node('Add', ['m2', 'b2'], ['a2']),
        helper.make_node('Relu', ['a2'], ['r2']),
        helper.make_node('MatMul', ['r2', 'W3'], ['m3']),
        helper.make_node('Add', ['m3', 'b3'], ['Y']),
    ]
    inits = [_init('W1', W1), _init('b1', b1),
             _init('W2', W2), _init('b2', b2),
             _init('W3', W3), _init('b3', b3)]
    g = helper.make_graph(
        nodes, 'matmul_add',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3])],
        inits)
    model = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    path = str(tmp_path / name)
    onnx.save(model, path)
    return ComputeGraph.from_onnx(path), (W1, b1, W2, b2, W3, b3)


def _ref_forward(x, params):
    W1, b1, W2, b2, W3, b3 = params
    h = np.maximum(x @ W1 + b1, 0.0)
    h = np.maximum(h @ W2 + b2, 0.0)
    return h @ W3 + b3


def test_gpu_layers_folds_standalone_add_bias(tmp_path):
    graph, params = _matmul_add_net(tmp_path)
    layers, _ = graph.gpu_layers(torch.device('cpu'), torch.float32)

    # Three affine layers, all fc; their biases must be the folded (non-zero)
    # Add constants, not the zero MatMul bias.
    assert [l['type'] for l in layers] == ['fc', 'fc', 'fc']
    for li, b_ref in zip(layers, (params[1], params[3], params[5])):
        assert np.any(li['bias'].numpy() != 0.0), 'bias was dropped (all zero)'
        assert li['bias'].numel() == b_ref.size

    # gpu_layers forward must match the reference forward exactly (proves the
    # bias is folded in the correct pre-activation position).
    rng = np.random.RandomState(1)
    for _ in range(8):
        x = rng.randn(4).astype(np.float32)
        h = torch.as_tensor(x, dtype=torch.float32)
        nh = len(layers) - 1
        for i, L in enumerate(layers):
            h = L['W'] @ h + L['bias']
            if i < nh:
                h = torch.clamp(h, min=0)
        ref = _ref_forward(x.astype(np.float64),
                            tuple(p.astype(np.float64) for p in params))
        np.testing.assert_allclose(h.numpy().astype(np.float64), ref,
                                   rtol=1e-5, atol=1e-5)
