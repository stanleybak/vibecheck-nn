"""gg emission of ONNX Sub forms — pins for the silent-drop audit fixes.

Two latent bugs (commit 9859a7b): plain Subs store their constant as
params['sub_val'] but the gg emission read only params['bias'] (so every
sub_val-form Sub was emitted bias=None and every chain's None-guard turned
it into an IDENTITY — acasxu's all-zero input_Sub was the only reason this
never fired); and negate-form Sub (bias - x) emitted a 'negate' flag that
NO consumer read (silent sign flip), now emitted as canonical mul(-1)+add.

Pins (forward through the gg via _forward_batch_graph vs onnxruntime):
  1. plain  Sub(x, c):  y = x - c with NONZERO c
  2. negate Sub(c, x):  y = c - x with NONZERO c
Plus the basic-zono concat const-scatter (ConcatNode.zonotope_propagate
used to silently DROP constant inputs): a point zonotope through a
Concat(const, live) must place both blocks at the right flat positions.
"""
import numpy as np
import onnx
import pytest
import torch
from onnx import helper, TensorProto

from vibecheck.network import ComputeGraph, DenseZonotope


def _init(name, arr):
    return helper.make_tensor(name, TensorProto.FLOAT, arr.shape,
                              arr.flatten())


def _sub_net(tmp_path, negate, name):
    """x -> 4x3 MatMul+Add -> Sub(*, c) or Sub(c, *) -> Y."""
    rng = np.random.RandomState(0)
    W1 = rng.randn(4, 3).astype(np.float32)
    b1 = rng.randn(3).astype(np.float32)
    c = np.array([0.5, -1.25, 2.0], dtype=np.float32)
    sub_inputs = ['c', 'a1'] if negate else ['a1', 'c']
    nodes = [
        helper.make_node('MatMul', ['X', 'W1'], ['m1']),
        helper.make_node('Add', ['m1', 'b1'], ['a1']),
        helper.make_node('Sub', sub_inputs, ['Y']),
    ]
    g = helper.make_graph(
        nodes, 'sub_net',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3])],
        [_init('W1', W1), _init('b1', b1), _init('c', c)])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    path = str(tmp_path / name)
    onnx.save(model, path)
    return ComputeGraph.from_onnx(path), (W1, b1, c)


@pytest.mark.parametrize('negate', [False, True],
                         ids=['plain_x_minus_c', 'negate_c_minus_x'])
def test_sub_forms_forward_exact(tmp_path, negate):
    from vibecheck.verify_zono_bnb import _forward_batch_graph
    graph, (W1, b1, c) = _sub_net(
        tmp_path, negate, f'sub_{negate}.onnx')
    gg = graph.gpu_graph(torch.device('cpu'), torch.float64)
    rng = np.random.RandomState(1)
    x = rng.randn(5, 4)
    y = _forward_batch_graph(torch.tensor(x, dtype=torch.float64), gg)
    a1 = x @ W1.astype(np.float64) + b1.astype(np.float64)
    ref = (c.astype(np.float64) - a1) if negate else (a1 - c)
    np.testing.assert_allclose(y.numpy(), ref, rtol=1e-9, atol=1e-9)


def test_negate_sub_emits_canonical_mul_add(tmp_path):
    graph, _ = _sub_net(tmp_path, True, 'sub_neg_ops.onnx')
    gg = graph.gpu_graph(torch.device('cpu'), torch.float64)
    types = [op['type'] for op in gg['ops']]
    # no consumer ever implemented a 'negate' flag on 'sub' — the negate
    # form must not be emitted as 'sub' at all
    assert 'sub' not in types
    assert types[-2:] == ['mul', 'add']


def test_concat_const_scatter_basic_zono(tmp_path):
    """ConcatNode.zonotope_propagate places const + live blocks exactly."""
    rng = np.random.RandomState(2)
    cls = rng.randn(1, 1, 3).astype(np.float32)
    W1 = rng.randn(4, 6).astype(np.float32)   # -> reshaped (1, 2, 3)
    nodes = [
        helper.make_node('MatMul', ['X', 'W1'], ['m1']),
        helper.make_node('Reshape', ['m1', 'shape'], ['r1']),
        helper.make_node('Concat', ['cls', 'r1'], ['Y'], axis=1),
    ]
    g = helper.make_graph(
        nodes, 'concat_net',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3, 3])],
        [_init('W1', W1), _init('cls', cls),
         helper.make_tensor('shape', TensorProto.INT64, [3],
                            np.array([1, 2, 3], np.int64))])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    path = str(tmp_path / 'concat.onnx')
    onnx.save(model, path)
    from vibecheck.onnx_loader import load_onnx
    graph = load_onnx(path, dtype=np.float64)

    x = rng.randn(4)
    zono_state = {graph.input_name: DenseZonotope(
        x.astype(np.float64), np.zeros((4, 0)))}
    gen_count = {}
    for name in graph.topo_order:
        graph.nodes[name].zonotope_propagate(
            zono_state, gen_count, lambda nm: zono_state[nm], 'box', graph)
    got = zono_state[graph.output_name].center
    ref = np.concatenate([cls.astype(np.float64).reshape(-1),
                          (x @ W1.astype(np.float64)).reshape(-1)])
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-9)
