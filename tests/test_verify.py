"""Tests for verify.py."""

import numpy as np
from vibecheck.network import ComputeGraph, GraphNode, ReluNode, GemmNode, AddNode, PassthroughNode, _prod
from vibecheck.spec import VNNSpec, Conjunct, Constraint, PairwiseConstraint
from vibecheck.verify import zonotope_verify
from vibecheck.zonotope import DenseZonotope

F = np.float32  # default dtype for test graphs


def _a(vals):
    """Shorthand: create float32 array from list."""
    return np.array(vals, dtype=F)


def _simple_fc_graph(dtype=np.float32):
    """Build a tiny graph: input(1,2) -> Gemm(2,2) -> Relu -> Gemm(2,1)."""
    g = ComputeGraph(dtype=dtype)
    g.input_name = 'input'
    g.input_shape = (1, 2)

    W1 = np.array([[1.0, -1.0], [0.5, 0.5]], dtype=dtype)
    b1 = np.array([0.0, 0.0], dtype=dtype)
    W2 = np.array([[1.0, 1.0]], dtype=dtype)
    b2 = np.array([0.0], dtype=dtype)

    g.nodes['gemm1'] = GemmNode(name='gemm1', op_type='Gemm',
                                 inputs=['input'], params={'W': W1, 'b': b1})
    g.nodes['relu'] = ReluNode(name='relu', op_type='Relu', inputs=['gemm1'])
    g.nodes['gemm2'] = GemmNode(name='gemm2', op_type='Gemm',
                                 inputs=['relu'], params={'W': W2, 'b': b2})
    g.output_name = 'gemm2'
    g.topological_sort()

    shapes = {g.input_name: g.input_shape}
    for name in g.topo_order:
        g.nodes[name].infer_shape(shapes)
        shapes[name] = g.nodes[name].output_shape
    return g


def test_verify_point():
    """Point zonotope (x_lo == x_hi) gives exact output."""
    g = _simple_fc_graph()
    center = _a([1.0, 0.5])
    spec = VNNSpec(center, center,
                   [Conjunct([Constraint(0, '>=', 100.0)])])
    result, details = zonotope_verify(g, spec)
    assert result == 'verified'  # output is far below 100
    assert details['output_lo'].shape == (1,)
    np.testing.assert_allclose(details['output_lo'], details['output_hi'], atol=1e-6)


def test_verify_with_range():
    """Zonotope with actual input range produces bounds."""
    g = _simple_fc_graph()
    x_lo = _a([0.0, 0.0])
    x_hi = _a([1.0, 1.0])
    spec = VNNSpec(x_lo, x_hi,
                   [Conjunct([Constraint(0, '>=', 1000.0)])])
    result, details = zonotope_verify(g, spec)
    # With range [0,1]x[0,1], output bounds should be finite
    assert np.isfinite(details['output_lo']).all()
    assert np.all(details['output_lo'] <= details['output_hi'])


def test_verify_with_fork():
    """Test fork/merge graph (exercises _find_shared_gens and copy)."""
    g = ComputeGraph()
    g.input_name = 'input'
    g.input_shape = (1, 2)

    # Fork: input -> gemm_a and gemm_b, then Add merge
    Wa = _a([[1.0, 0.0], [0.0, 1.0]])
    Wb = _a([[0.0, 1.0], [1.0, 0.0]])
    ba = bb = _a([0.0, 0.0])
    g.nodes['ga'] = GemmNode(name='ga', op_type='Gemm', inputs=['input'],
                              params={'W': Wa, 'b': ba})
    g.nodes['gb'] = GemmNode(name='gb', op_type='Gemm', inputs=['input'],
                              params={'W': Wb, 'b': bb})
    g.nodes['add'] = AddNode(name='add', op_type='Add', inputs=['ga', 'gb'])
    g.output_name = 'add'
    g.topological_sort()
    shapes = {g.input_name: g.input_shape}
    for name in g.topo_order:
        g.nodes[name].infer_shape(shapes)
        shapes[name] = g.nodes[name].output_shape

    x_lo = _a([1.0, 2.0])
    x_hi = _a([3.0, 4.0])
    spec = VNNSpec(x_lo, x_hi, [Conjunct([Constraint(0, '>=', 100.0)])])
    result, details = zonotope_verify(g, spec)
    # With range, output should have bounds
    assert np.all(details['output_lo'] <= details['output_hi'])
    assert np.isfinite(details['output_lo']).all()


def test_verify_with_split():
    """Graph with Split exercises the 'name in zono_state' skip path."""
    g = ComputeGraph()
    g.input_name = 'input'
    g.input_shape = (1, 4)

    from vibecheck.network import SplitNode, SplitOutputNode
    g.nodes['split'] = SplitNode(name='split', op_type='Split',
                                  inputs=['input'],
                                  params={'axis': 1, 'split': [2, 2]})
    g.nodes['s1'] = SplitOutputNode(name='s1', op_type='SplitOutput',
                                     inputs=['split'], params={'index': 1})
    # Output is the second split part
    g.output_name = 's1'
    g.topological_sort()
    shapes = {g.input_name: g.input_shape}
    for name in g.topo_order:
        g.nodes[name].infer_shape(shapes)
        shapes[name] = g.nodes[name].output_shape

    center = _a([1, 2, 3, 4])
    spec = VNNSpec(center, center, [Conjunct([Constraint(0, '>=', 100.0)])])
    result, details = zonotope_verify(g, spec)
    # Second split part should be [3, 4]
    np.testing.assert_allclose(details['output_lo'], [3, 4])


def test_verify_pairwise():
    """Pairwise constraint check."""
    g = ComputeGraph()
    g.input_name = 'input'
    g.input_shape = (1, 3)
    W = np.eye(3, dtype=F)
    b = _a([0.0, 0.0, 0.0])
    g.nodes['gemm'] = GemmNode(name='gemm', op_type='Gemm',
                                inputs=['input'], params={'W': W, 'b': b})
    g.output_name = 'gemm'
    g.topological_sort()
    shapes = {g.input_name: g.input_shape}
    for name in g.topo_order:
        g.nodes[name].infer_shape(shapes)
        shapes[name] = g.nodes[name].output_shape

    center = _a([10.0, 1.0, 1.0])
    spec = VNNSpec(center, center,
                   [Conjunct([PairwiseConstraint(pred=0, comp=1),
                              PairwiseConstraint(pred=0, comp=2)])])
    result, details = zonotope_verify(g, spec)
    assert result == 'verified'
