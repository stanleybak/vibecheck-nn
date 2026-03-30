"""Tests for ND-aware ops: verify shape inference and data propagation
match numpy/torch reference implementations on small tensors."""

import numpy as np
import pytest
from vibecheck.zonotope import DenseZonotope
from vibecheck.network import (
    GraphNode, PassthroughNode, UnsqueezeNode, SqueezeNode,
    ReshapeNode, TransposeNode, SliceNode, AddNode, SubNode, MulNode,
    GatherNode, ConcatNode, SplitNode, SplitOutputNode, ComputeGraph,
    _prod,
)


def _make_graph(nodes, input_name='input', input_shape=(1, 4),
                output_name=None):
    """Build a minimal ComputeGraph."""
    g = ComputeGraph()
    g.input_name = input_name
    g.input_shape = input_shape
    for n in nodes:
        g.nodes[n.name] = n
    g.output_name = output_name or nodes[-1].name
    g.topological_sort()
    shapes = {input_name: input_shape}
    for name in g.topo_order:
        g.nodes[name].infer_shape(shapes)
        shapes[name] = g.nodes[name].output_shape
    return g


def _run_point(graph, center):
    """Run point zonotope through graph, return output center."""
    zono_state = {graph.input_name: DenseZonotope(
        center, np.zeros((len(center), 0)))}
    gen_count = {graph.input_name: 0}
    forks = graph.fork_points()
    def get(name):
        if name in forks:
            return zono_state[name].copy()
        return zono_state[name]
    for name in graph.topo_order:
        if name in zono_state:
            continue
        graph.nodes[name].zonotope_propagate(
            zono_state, gen_count, get, 'std', graph)
        gen_count[name] = 0
    return zono_state[graph.output_name].center


# ---- Reshape ----

def test_reshape_3d_to_2d():
    """Reshape (1, 2, 3) -> (1, 6)"""
    node = ReshapeNode(name='r', op_type='Reshape', inputs=['input'],
                       params={'shape': (1, 6)})
    g = _make_graph([node], input_shape=(1, 2, 3))
    assert node.output_shape == (1, 6)
    data = np.arange(6, dtype=float)
    np.testing.assert_array_equal(_run_point(g, data), data)


def test_reshape_2d_to_4d():
    """Reshape (1, 12) -> (1, 3, 2, 2)"""
    node = ReshapeNode(name='r', op_type='Reshape', inputs=['input'],
                       params={'shape': (1, 3, 2, 2)})
    g = _make_graph([node], input_shape=(1, 12))
    assert node.output_shape == (1, 3, 2, 2)
    data = np.arange(12, dtype=float)
    np.testing.assert_array_equal(_run_point(g, data), data)


def test_reshape_infer_neg1():
    """Reshape (1, 2, 6) -> (-1, 3, 4) should infer batch=1"""
    node = ReshapeNode(name='r', op_type='Reshape', inputs=['input'],
                       params={'shape': (-1, 3, 4)})
    g = _make_graph([node], input_shape=(1, 2, 6))
    assert node.output_shape == (1, 3, 4)


# ---- Transpose ----

def test_transpose_3d():
    """Transpose (1, 2, 3) with perm [0, 2, 1] -> (1, 3, 2)"""
    node = TransposeNode(name='t', op_type='Transpose', inputs=['input'],
                         params={'perm': [0, 2, 1]})
    g = _make_graph([node], input_shape=(1, 2, 3))
    assert node.output_shape == (1, 3, 2)

    data = np.arange(6, dtype=float)
    out = _run_point(g, data)
    expected = data.reshape(1, 2, 3).transpose(0, 2, 1).flatten()
    np.testing.assert_array_equal(out, expected)


def test_transpose_4d_nhwc_to_nchw():
    """Transpose (1, 2, 3, 4) perm [0, 3, 1, 2] -> (1, 4, 2, 3)"""
    node = TransposeNode(name='t', op_type='Transpose', inputs=['input'],
                         params={'perm': [0, 3, 1, 2]})
    g = _make_graph([node], input_shape=(1, 2, 3, 4))
    assert node.output_shape == (1, 4, 2, 3)

    data = np.arange(24, dtype=float)
    out = _run_point(g, data)
    expected = data.reshape(1, 2, 3, 4).transpose(0, 3, 1, 2).flatten()
    np.testing.assert_array_equal(out, expected)


# ---- Slice ----

def test_slice_axis0_batch():
    """Slice along batch axis (no-op for batch=1)"""
    node = SliceNode(name='s', op_type='Slice', inputs=['input'],
                     params={'starts': [0], 'ends': [1], 'axes': [0]})
    g = _make_graph([node], input_shape=(1, 6))
    assert node.output_shape == (1, 6)

    data = np.arange(6, dtype=float)
    np.testing.assert_array_equal(_run_point(g, data), data)


def test_slice_axis1_3d():
    """Slice (1, 6, 4) along axis 1, [2:5] -> (1, 3, 4)"""
    node = SliceNode(name='s', op_type='Slice', inputs=['input'],
                     params={'starts': [2], 'ends': [5], 'axes': [1]})
    g = _make_graph([node], input_shape=(1, 6, 4))
    assert node.output_shape == (1, 3, 4)

    data = np.arange(24, dtype=float)
    out = _run_point(g, data)
    expected = data.reshape(1, 6, 4)[:, 2:5, :].flatten()
    np.testing.assert_array_equal(out, expected)


def test_slice_axis2():
    """Slice (1, 3, 8) along axis 2, [1:5] -> (1, 3, 4)"""
    node = SliceNode(name='s', op_type='Slice', inputs=['input'],
                     params={'starts': [1], 'ends': [5], 'axes': [2]})
    g = _make_graph([node], input_shape=(1, 3, 8))
    assert node.output_shape == (1, 3, 4)

    data = np.arange(24, dtype=float)
    out = _run_point(g, data)
    expected = data.reshape(1, 3, 8)[:, :, 1:5].flatten()
    np.testing.assert_array_equal(out, expected)


def test_slice_negative_index():
    """Slice with negative end index"""
    node = SliceNode(name='s', op_type='Slice', inputs=['input'],
                     params={'starts': [0], 'ends': [-1], 'axes': [1]})
    g = _make_graph([node], input_shape=(1, 5, 2))
    assert node.output_shape == (1, 4, 2)

    data = np.arange(10, dtype=float)
    out = _run_point(g, data)
    expected = data.reshape(1, 5, 2)[:, 0:-1, :].flatten()
    np.testing.assert_array_equal(out, expected)


# ---- Gather ----

def test_gather_flat():
    """Gather specific indices from flat vector"""
    node = GatherNode(name='g', op_type='Gather', inputs=['input'],
                      params={'indices': np.array([4, 1, 0])})
    g = _make_graph([node], input_shape=(1, 5))
    assert node.output_shape == (3,)

    data = np.array([10, 20, 30, 40, 50], dtype=float)
    out = _run_point(g, data)
    np.testing.assert_array_equal(out, [50, 20, 10])


def test_gather_negative_index():
    """Gather with negative index (last element)"""
    node = GatherNode(name='g', op_type='Gather', inputs=['input'],
                      params={'indices': np.array(-1)})
    g = _make_graph([node], input_shape=(1, 5))

    data = np.array([10, 20, 30, 40, 50], dtype=float)
    out = _run_point(g, data)
    np.testing.assert_array_equal(out, [50])


# ---- Concat ----

def test_concat_flat():
    """Concat two flat vectors"""
    n1 = ReshapeNode(name='a', op_type='Reshape', inputs=['input'],
                     params={'shape': (1, 3)})
    n2 = SliceNode(name='b', op_type='Slice', inputs=['input'],
                   params={'starts': [0], 'ends': [1], 'axes': [0]})
    # Can't easily test Concat without a fork, so test shape only
    node = ConcatNode(name='c', op_type='Concat', inputs=['a', 'b'])
    g = _make_graph([n1, n2, node], input_shape=(1, 3), output_name='c')
    # Both inputs are (1, 3), concat should be (6,) flat
    assert node.output_shape is not None
    assert _prod(node.output_shape) == 6


# ---- Split ----

def test_split_axis1():
    """Split (1, 6, 2) along axis 1 into [3, 3]"""
    split = SplitNode(name='s', op_type='Split', inputs=['input'],
                      params={'axis': 1, 'split': [3, 3]})
    sout = SplitOutputNode(name='s1', op_type='SplitOutput',
                           inputs=['s'], params={'index': 1})
    g = _make_graph([split, sout], input_shape=(1, 6, 2), output_name='s')

    # Manually set SplitOutput shape (normally done by _infer_shapes in onnx_loader)
    sout.output_shape = (1, 3, 2)

    data = np.arange(12, dtype=float)
    # Run propagation
    zono_state = {g.input_name: DenseZonotope(data, np.zeros((12, 0)))}
    gen_count = {g.input_name: 0}
    split.zonotope_propagate(zono_state, gen_count,
                             lambda n: zono_state[n], 'std', g)

    # First split: elements from axis1 [0:3]
    expected_0 = data.reshape(1, 6, 2)[:, 0:3, :].flatten()
    np.testing.assert_array_equal(zono_state['s'].center, expected_0)

    # Second split: elements from axis1 [3:6]
    expected_1 = data.reshape(1, 6, 2)[:, 3:6, :].flatten()
    np.testing.assert_array_equal(zono_state['s1'].center, expected_1)


# ---- Reshape -> Slice pipeline (the pensieve pattern) ----

def test_reshape_then_slice():
    """Reshape (1, 48) -> (1, 6, 8), then Slice axis 1 [2:3] -> (1, 1, 8)"""
    r = ReshapeNode(name='r', op_type='Reshape', inputs=['input'],
                    params={'shape': (1, 6, 8)})
    s = SliceNode(name='s', op_type='Slice', inputs=['r'],
                  params={'starts': [2], 'ends': [3], 'axes': [1]})
    g = _make_graph([r, s], input_shape=(1, 48))
    assert r.output_shape == (1, 6, 8)
    assert s.output_shape == (1, 1, 8)

    data = np.arange(48, dtype=float)
    out = _run_point(g, data)
    expected = data.reshape(1, 6, 8)[:, 2:3, :].flatten()
    np.testing.assert_array_equal(out, expected)
    assert len(out) == 8


# ---- Transpose -> Conv pipeline (traffic signs pattern) ----

def test_transpose_preserves_data():
    """Transpose NHWC->NCHW then back should be identity"""
    t1 = TransposeNode(name='t1', op_type='Transpose', inputs=['input'],
                       params={'perm': [0, 3, 1, 2]})  # NHWC -> NCHW
    t2 = TransposeNode(name='t2', op_type='Transpose', inputs=['t1'],
                       params={'perm': [0, 2, 3, 1]})  # NCHW -> NHWC
    g = _make_graph([t1, t2], input_shape=(1, 2, 3, 4))
    assert t1.output_shape == (1, 4, 2, 3)
    assert t2.output_shape == (1, 2, 3, 4)

    data = np.arange(24, dtype=float)
    out = _run_point(g, data)
    np.testing.assert_array_equal(out, data)


# ---- Unsqueeze / Squeeze (currently PassthroughNode — test shape behavior) ----

def test_unsqueeze_axis2():
    """Unsqueeze (1, 6) with axes=[2] -> (1, 6, 1)"""
    node = UnsqueezeNode(name='u', op_type='Unsqueeze', inputs=['input'],
                         params={'axes': [2]})
    g = _make_graph([node], input_shape=(1, 6))
    assert node.output_shape == (1, 6, 1)
    data = np.arange(6, dtype=float)
    np.testing.assert_array_equal(_run_point(g, data), data)


def test_unsqueeze_axis0():
    """Unsqueeze (1, 3, 4) with axes=[0] -> (1, 1, 3, 4)"""
    node = UnsqueezeNode(name='u', op_type='Unsqueeze', inputs=['input'],
                         params={'axes': [0]})
    g = _make_graph([node], input_shape=(1, 3, 4))
    assert node.output_shape == (1, 1, 3, 4)


def test_unsqueeze_multiple():
    """Unsqueeze (1, 6) with axes=[1, 3] -> (1, 1, 6, 1)"""
    node = UnsqueezeNode(name='u', op_type='Unsqueeze', inputs=['input'],
                         params={'axes': [1, 3]})
    g = _make_graph([node], input_shape=(1, 6))
    assert node.output_shape == (1, 1, 6, 1)


def test_squeeze_remove_all_ones():
    """Squeeze (1, 1, 6) with no axes -> (6,)"""
    node = SqueezeNode(name='sq', op_type='Squeeze', inputs=['input'],
                       params={})
    g = _make_graph([node], input_shape=(1, 1, 6))
    assert node.output_shape == (6,)
    data = np.arange(6, dtype=float)
    np.testing.assert_array_equal(_run_point(g, data), data)


def test_squeeze_specific_axis():
    """Squeeze (1, 1, 6, 1) with axes=[1] -> (1, 6, 1)"""
    node = SqueezeNode(name='sq', op_type='Squeeze', inputs=['input'],
                       params={'axes': [1]})
    g = _make_graph([node], input_shape=(1, 1, 6, 1))
    assert node.output_shape == (1, 6, 1)


# ---- Unsqueeze -> Slice pipeline (ml4acopf pattern) ----

def test_concat_propagation():
    """ConcatNode propagation through a graph with fork."""
    from vibecheck.network import GemmNode, ConcatNode, ReluNode
    W = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.zeros(2)
    # Fork: input -> relu1, input -> relu2, then concat
    r1 = ReluNode(name='r1', op_type='Relu', inputs=['input'])
    r2 = ReluNode(name='r2', op_type='Relu', inputs=['input'])
    cat = ConcatNode(name='cat', op_type='Concat', inputs=['r1', 'r2'])
    g = _make_graph([r1, r2, cat], input_shape=(1, 2), output_name='cat')
    center = np.array([3.0, -1.0])
    out = _run_point(g, center)
    # relu([3, -1]) = [3, 0] for both branches, concat = [3, 0, 3, 0]
    np.testing.assert_array_equal(out, [3, 0, 3, 0])


def test_unsqueeze_then_slice():
    """Unsqueeze (1, 6) -> (1, 6, 1), then Slice axis 1 [2:4] -> (1, 2, 1)"""
    u = UnsqueezeNode(name='u', op_type='Unsqueeze', inputs=['input'],
                      params={'axes': [2]})
    s = SliceNode(name='s', op_type='Slice', inputs=['u'],
                  params={'starts': [2], 'ends': [4], 'axes': [1]})
    g = _make_graph([u, s], input_shape=(1, 6))
    assert u.output_shape == (1, 6, 1)
    assert s.output_shape == (1, 2, 1)

    data = np.arange(6, dtype=float)
    out = _run_point(g, data)
    expected = data.reshape(1, 6, 1)[:, 2:4, :].flatten()
    np.testing.assert_array_equal(out, expected)
    assert len(out) == 2


# ---- Broadcasting arithmetic ----

def test_sub_broadcast():
    """Sub (1, 3, 1) - const (4,) should broadcast to (1, 3, 4)"""
    node = SubNode(name='s', op_type='Sub', inputs=['input'],
                   params={'sub_val': np.array([10, 20, 30, 40])})
    g = _make_graph([node], input_shape=(1, 3, 1))
    assert node.output_shape == (1, 3, 4)

    data = np.array([1, 2, 3], dtype=float)
    out = _run_point(g, data)
    expected = data.reshape(1, 3, 1) - np.array([10, 20, 30, 40])
    np.testing.assert_array_equal(out, expected.flatten())
    assert len(out) == 12


def test_add_broadcast():
    """Add (1, 4, 1) + const (3,) should broadcast to (1, 4, 3)"""
    node = AddNode(name='a', op_type='Add', inputs=['input'],
                   params={'bias': np.array([100, 200, 300])})
    g = _make_graph([node], input_shape=(1, 4, 1))
    assert node.output_shape == (1, 4, 3)

    data = np.array([1, 2, 3, 4], dtype=float)
    out = _run_point(g, data)
    expected = data.reshape(1, 4, 1) + np.array([100, 200, 300])
    np.testing.assert_array_equal(out, expected.flatten())
    assert len(out) == 12


def test_mul_broadcast():
    """Mul (1, 2, 1) * const (3,) should broadcast to (1, 2, 3)"""
    node = MulNode(name='m', op_type='Mul', inputs=['input'],
                   params={'scale': np.array([2, 3, 4])})
    g = _make_graph([node], input_shape=(1, 2, 1))
    assert node.output_shape == (1, 2, 3)

    data = np.array([10, 20], dtype=float)
    out = _run_point(g, data)
    expected = data.reshape(1, 2, 1) * np.array([2, 3, 4])
    np.testing.assert_array_equal(out, expected.flatten())
    assert len(out) == 6


def test_sub_no_broadcast():
    """Sub with same-size const — no broadcast, generators preserved"""
    node = SubNode(name='s', op_type='Sub', inputs=['input'],
                   params={'sub_val': np.array([1, 2, 3])})
    g = _make_graph([node], input_shape=(1, 3))
    assert node.output_shape == (1, 3)

    data = np.array([10, 20, 30], dtype=float)
    out = _run_point(g, data)
    np.testing.assert_array_equal(out, [9, 18, 27])
