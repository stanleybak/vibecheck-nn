"""Unit tests for individual op node propagation."""

import numpy as np
import pytest
from vibecheck.zonotope import DenseZonotope
from vibecheck.network import (
    GraphNode, PassthroughNode, UnsqueezeNode, SqueezeNode,
    ReshapeNode, TransposeNode, SliceNode,
    GatherNode, ReluNode, ConvNode, ConvTransposeNode,
    GemmNode, MatMulBilinearNode,
    AddNode, SubNode, MulNode, DivNode, NegNode,
    BatchNormNode, ConcatNode, SplitNode, SplitOutputNode,
    SigmoidNode, SoftmaxNode, SignNode, ClipNode, TanhNode,
    LeakyReluNode, TrigNode, PowNode, FloorNode,
    MaxPoolNode, AveragePoolNode, PadNode, ResizeNode,
    ConstantOfShapeNode, ShapeOpNode, MiscNode,
    ReduceNode, ComputeGraph, _prod,
)


def _make_graph(nodes, input_name='input', input_shape=(1, 4),
                output_name=None):
    """Helper to build a minimal ComputeGraph from a list of nodes."""
    g = ComputeGraph()
    g.input_name = input_name
    g.input_shape = input_shape
    for n in nodes:
        g.nodes[n.name] = n
    g.output_name = output_name or nodes[-1].name
    g.topological_sort()
    # Infer shapes
    shapes = {input_name: input_shape}
    for name in g.topo_order:
        g.nodes[name].infer_shape(shapes)
        shapes[name] = g.nodes[name].output_shape
    return g


def _run_point(graph, center):
    """Run a point zonotope (0 generators) through the graph."""
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
        gen_count[name] = zono_state[name].generators.shape[1]
    return zono_state[graph.output_name].center


# --- Reshape ---

def test_reshape_shape_inference():
    node = ReshapeNode(name='r', op_type='Reshape', inputs=['input'],
                       params={'shape': (-1, 2, 3)})
    g = _make_graph([node], input_shape=(1, 6))
    assert node.output_shape == (1, 2, 3)


def test_reshape_keeps_batch():
    node = ReshapeNode(name='r', op_type='Reshape', inputs=['input'],
                       params={'shape': (1, 3, 2)})
    g = _make_graph([node], input_shape=(1, 6))
    assert node.output_shape == (1, 3, 2)


def test_reshape_propagation():
    node = ReshapeNode(name='r', op_type='Reshape', inputs=['input'],
                       params={'shape': (1, 2, 3)})
    g = _make_graph([node], input_shape=(1, 6))
    center = np.arange(6, dtype=float)
    out = _run_point(g, center)
    np.testing.assert_array_equal(out, center)  # data unchanged


# --- Transpose ---

def test_transpose_nhwc_to_nchw():
    """Transpose (1,H,W,C) -> (1,C,H,W)."""
    node = TransposeNode(name='t', op_type='Transpose', inputs=['input'],
                         params={'perm': [0, 3, 1, 2]})
    g = _make_graph([node], input_shape=(1, 2, 3, 4))
    assert node.output_shape == (1, 4, 2, 3)

    # Data: fill with index to verify permutation
    center = np.arange(24, dtype=float)
    out = _run_point(g, center)
    expected = center.reshape(1, 2, 3, 4).transpose(0, 3, 1, 2).flatten()
    np.testing.assert_array_equal(out, expected)


def test_transpose_reverse():
    """Default transpose (reverse dims)."""
    node = TransposeNode(name='t', op_type='Transpose', inputs=['input'],
                         params={})
    g = _make_graph([node], input_shape=(1, 2, 3))
    assert node.output_shape == (3, 2, 1)

    center = np.arange(6, dtype=float)
    out = _run_point(g, center)
    expected = center.reshape(1, 2, 3).transpose(2, 1, 0).flatten()
    np.testing.assert_array_equal(out, expected)


def test_transpose_with_generators():
    """Transpose permutes generator rows too."""
    node = TransposeNode(name='t', op_type='Transpose', inputs=['input'],
                         params={'perm': [0, 2, 1]})
    g = _make_graph([node], input_shape=(1, 2, 3))

    center = np.arange(6, dtype=float)
    gens = np.eye(6)  # one generator per element
    z_in = DenseZonotope(center, gens)

    zono_state = {'input': z_in}
    gen_count = {'input': 6}
    node.zonotope_propagate(zono_state, gen_count, lambda n: zono_state[n],
                            'std', g)
    z_out = zono_state['t']

    expected_center = center.reshape(1, 2, 3).transpose(0, 2, 1).flatten()
    np.testing.assert_array_equal(z_out.center, expected_center)
    # Generators should be permuted the same way
    assert z_out.generators.shape == (6, 6)


# --- Slice ---

def test_slice_axis1():
    """Slice along axis 1 of (1, 6, 8) -> (1, 1, 8) = 8 elements."""
    node = SliceNode(name='s', op_type='Slice', inputs=['input'],
                     params={'starts': [2], 'ends': [3], 'axes': [1]})
    g = _make_graph([node], input_shape=(1, 6, 8))
    assert node.output_shape == (1, 1, 8)

    center = np.arange(48, dtype=float)
    out = _run_point(g, center)
    expected = center.reshape(1, 6, 8)[:, 2:3, :].flatten()
    np.testing.assert_array_equal(out, expected)


def test_slice_axis2():
    """Slice along last axis."""
    node = SliceNode(name='s', op_type='Slice', inputs=['input'],
                     params={'starts': [1], 'ends': [4], 'axes': [2]})
    g = _make_graph([node], input_shape=(1, 3, 5))
    assert node.output_shape == (1, 3, 3)

    center = np.arange(15, dtype=float)
    out = _run_point(g, center)
    expected = center.reshape(1, 3, 5)[:, :, 1:4].flatten()
    np.testing.assert_array_equal(out, expected)


def test_slice_flat():
    """Slice on a 2D shape (1, N) — flat slice."""
    node = SliceNode(name='s', op_type='Slice', inputs=['input'],
                     params={'starts': [2], 'ends': [5], 'axes': [1]})
    g = _make_graph([node], input_shape=(1, 10))
    assert node.output_shape == (1, 3)

    center = np.arange(10, dtype=float)
    out = _run_point(g, center)
    expected = center.reshape(1, 10)[:, 2:5].flatten()
    np.testing.assert_array_equal(out, expected)


# --- Gather ---

def test_gather_indices():
    node = GatherNode(name='g', op_type='Gather', inputs=['input'],
                      params={'indices': np.array([0, 3, 1])})
    g = _make_graph([node], input_shape=(5,))
    assert node.output_shape == (3,)

    center = np.array([10, 20, 30, 40, 50], dtype=float)
    out = _run_point(g, center)
    np.testing.assert_array_equal(out, [10, 40, 20])


# --- Relu ---

def test_relu_point():
    node = ReluNode(name='r', op_type='Relu', inputs=['input'])
    g = _make_graph([node], input_shape=(1, 4))
    center = np.array([-1, 0, 1, 2], dtype=float)
    out = _run_point(g, center)
    np.testing.assert_array_equal(out, [0, 0, 1, 2])


# --- Sigmoid, Tanh, Sign, Clip, Softmax ---

def test_sigmoid_point():
    node = SigmoidNode(name='s', op_type='Sigmoid', inputs=['input'])
    g = _make_graph([node], input_shape=(1, 3))
    center = np.array([0, 1, -1], dtype=float)
    out = _run_point(g, center)
    expected = 1 / (1 + np.exp(-center))
    np.testing.assert_allclose(out, expected)


def test_tanh_point():
    node = TanhNode(name='t', op_type='Tanh', inputs=['input'])
    g = _make_graph([node], input_shape=(1, 3))
    center = np.array([0, 1, -1], dtype=float)
    out = _run_point(g, center)
    np.testing.assert_allclose(out, np.tanh(center))


def test_sign_point():
    node = SignNode(name='s', op_type='Sign', inputs=['input'])
    g = _make_graph([node], input_shape=(1, 4))
    center = np.array([-2, 0, 0.5, 3], dtype=float)
    out = _run_point(g, center)
    np.testing.assert_array_equal(out, [-1, 0, 1, 1])


def test_clip_point():
    node = ClipNode(name='c', op_type='Clip', inputs=['input'],
                    params={'min': -1.0, 'max': 1.0})
    g = _make_graph([node], input_shape=(1, 4))
    center = np.array([-5, -0.5, 0.5, 5], dtype=float)
    out = _run_point(g, center)
    np.testing.assert_array_equal(out, [-1, -0.5, 0.5, 1])


def test_softmax_point():
    node = SoftmaxNode(name='s', op_type='Softmax', inputs=['input'])
    g = _make_graph([node], input_shape=(1, 3))
    center = np.array([1, 2, 3], dtype=float)
    out = _run_point(g, center)
    e = np.exp(center - center.max())
    np.testing.assert_allclose(out, e / e.sum())


# --- Neg, Add, Sub, Mul, Div ---

def test_neg_point():
    node = NegNode(name='n', op_type='Neg', inputs=['input'])
    g = _make_graph([node], input_shape=(1, 3))
    out = _run_point(g, np.array([1, -2, 3], dtype=float))
    np.testing.assert_array_equal(out, [-1, 2, -3])


def test_add_bias():
    node = AddNode(name='a', op_type='Add', inputs=['input'],
                   params={'bias': np.array([10, 20, 30])})
    g = _make_graph([node], input_shape=(1, 3))
    out = _run_point(g, np.array([1, 2, 3], dtype=float))
    np.testing.assert_array_equal(out, [11, 22, 33])


def test_mul_scale():
    node = MulNode(name='m', op_type='Mul', inputs=['input'],
                   params={'scale': np.array([2, 0.5, -1])})
    g = _make_graph([node], input_shape=(1, 3))
    out = _run_point(g, np.array([4, 6, 8], dtype=float))
    np.testing.assert_array_equal(out, [8, 3, -8])


# --- BatchNorm (unfused) ---

def test_batchnorm_point():
    node = BatchNormNode(name='bn', op_type='BatchNormalization',
                         inputs=['input'],
                         params={
                             'scale': np.array([2.0, 1.0]),
                             'bias': np.array([0.0, 1.0]),
                             'mean': np.array([0.5, 0.5]),
                             'var': np.array([1.0, 1.0]),
                             'epsilon': 0.0,
                         })
    g = _make_graph([node], input_shape=(1, 2))
    center = np.array([1.0, 2.0])
    out = _run_point(g, center)
    # factor = scale / sqrt(var + eps) = [2, 1]
    # offset = -factor * mean + bias = [-1, 0.5]
    # out = factor * center + offset = [2*1 - 1, 1*2 + 0.5] = [1, 2.5]
    np.testing.assert_allclose(out, [1.0, 2.5])


# --- Reduce ---

def test_reduce_sum():
    node = ReduceNode(name='r', op_type='ReduceSum', inputs=['input'])
    g = _make_graph([node], input_shape=(1, 4))
    out = _run_point(g, np.array([1, 2, 3, 4], dtype=float))
    np.testing.assert_array_equal(out, [10])


def test_reduce_mean():
    node = ReduceNode(name='r', op_type='ReduceMean', inputs=['input'])
    g = _make_graph([node], input_shape=(1, 4))
    out = _run_point(g, np.array([1, 2, 3, 4], dtype=float))
    np.testing.assert_array_equal(out, [2.5])


# --- LeakyRelu, Trig, Pow, Floor ---

def test_leakyrelu_point():
    node = LeakyReluNode(name='lr', op_type='LeakyRelu', inputs=['input'],
                         params={'alpha': 0.1})
    g = _make_graph([node], input_shape=(1, 4))
    center = np.array([-2, -1, 0, 1], dtype=float)
    out = _run_point(g, center)
    np.testing.assert_allclose(out, [-0.2, -0.1, 0, 1])


def test_sin_point():
    node = TrigNode(name='s', op_type='Sin', inputs=['input'])
    g = _make_graph([node], input_shape=(1, 3))
    center = np.array([0, np.pi/2, np.pi], dtype=float)
    out = _run_point(g, center)
    np.testing.assert_allclose(out, np.sin(center), atol=1e-10)


def test_cos_point():
    node = TrigNode(name='c', op_type='Cos', inputs=['input'])
    g = _make_graph([node], input_shape=(1, 3))
    center = np.array([0, np.pi/2, np.pi], dtype=float)
    out = _run_point(g, center)
    np.testing.assert_allclose(out, np.cos(center), atol=1e-10)


def test_pow_point():
    node = PowNode(name='p', op_type='Pow', inputs=['input'],
                   params={'exponent': 2.0})
    g = _make_graph([node], input_shape=(1, 3))
    center = np.array([-2, 0, 3], dtype=float)
    out = _run_point(g, center)
    np.testing.assert_array_equal(out, [4, 0, 9])


def test_floor_point():
    node = FloorNode(name='f', op_type='Floor', inputs=['input'])
    g = _make_graph([node], input_shape=(1, 3))
    center = np.array([1.7, -0.3, 2.0], dtype=float)
    out = _run_point(g, center)
    np.testing.assert_array_equal(out, [1, -1, 2])


# --- ConvTranspose ---

def test_conv_transpose_point():
    node = ConvTransposeNode(name='ct', op_type='ConvTranspose', inputs=['input'],
                              params={
                                  'kernel': np.ones((1, 1, 2, 2)),
                                  'bias': np.array([0.0]),
                                  'stride': (2, 2),
                                  'padding': (0, 0),
                                  'output_padding': (0, 0),
                              })
    g = _make_graph([node], input_shape=(1, 1, 2, 2))
    shapes = {g.input_name: g.input_shape}
    node.infer_shape(shapes)
    assert node.output_shape == (1, 1, 4, 4)

    center = np.array([1, 2, 3, 4], dtype=float)
    out = _run_point(g, center)
    assert len(out) == 16  # 1*1*4*4


# --- MaxPool, AvgPool ---

def test_maxpool_point():
    node = MaxPoolNode(name='mp', op_type='MaxPool', inputs=['input'],
                       params={'kernel_shape': (2, 2), 'stride': (2, 2),
                               'padding': (0, 0)})
    g = _make_graph([node], input_shape=(1, 1, 4, 4))
    center = np.arange(16, dtype=float)
    out = _run_point(g, center)
    # Max of each 2x2 block
    expected = center.reshape(1, 1, 4, 4)
    import torch
    import torch.nn.functional as F
    t = torch.tensor(expected, dtype=torch.float64)
    ref = F.max_pool2d(t, 2, 2).flatten().numpy()
    np.testing.assert_array_equal(out, ref)


def test_avgpool_point():
    node = AveragePoolNode(name='ap', op_type='AveragePool', inputs=['input'],
                           params={'kernel_shape': (2, 2), 'stride': (2, 2),
                                   'padding': (0, 0)})
    g = _make_graph([node], input_shape=(1, 1, 4, 4))
    center = np.arange(16, dtype=float)
    out = _run_point(g, center)
    import torch
    import torch.nn.functional as F
    t = torch.tensor(center.reshape(1, 1, 4, 4), dtype=torch.float64)
    ref = F.avg_pool2d(t, 2, 2).flatten().numpy()
    np.testing.assert_array_equal(out, ref)


# --- Pad ---

def test_pad_point():
    node = PadNode(name='p', op_type='Pad', inputs=['input'],
                   params={'pads': [0, 0, 1, 1, 0, 0, 1, 1],
                           'constant_value': 0.0})
    g = _make_graph([node], input_shape=(1, 1, 2, 2))
    center = np.array([1, 2, 3, 4], dtype=float)
    out = _run_point(g, center)
    import torch
    import torch.nn.functional as F
    t = torch.tensor(center.reshape(1, 1, 2, 2), dtype=torch.float64)
    ref = F.pad(t, (1, 1, 1, 1), value=0.0).flatten().numpy()
    np.testing.assert_array_equal(out, ref)


# --- Resize/Upsample ---

def test_resize_nearest():
    node = ResizeNode(name='r', op_type='Resize', inputs=['input'],
                      params={'scales': np.array([1, 1, 2, 2])})
    g = _make_graph([node], input_shape=(1, 1, 2, 2))
    shapes = {g.input_name: g.input_shape}
    node.infer_shape(shapes)
    assert node.output_shape == (1, 1, 4, 4)

    center = np.array([1, 2, 3, 4], dtype=float)
    out = _run_point(g, center)
    assert len(out) == 16
    import torch
    import torch.nn.functional as F
    t = torch.tensor(center.reshape(1, 1, 2, 2), dtype=torch.float64)
    ref = F.interpolate(t, scale_factor=2, mode='nearest').flatten().numpy()
    np.testing.assert_array_equal(out, ref)


# --- ConstantOfShape ---

def test_constant_of_shape():
    node = ConstantOfShapeNode(name='c', op_type='ConstantOfShape',
                                inputs=['input'], params={'value': 7.0})
    g = _make_graph([node], input_shape=(1, 3))
    center = np.array([1, 2, 3], dtype=float)
    out = _run_point(g, center)
    np.testing.assert_array_equal(out, [7, 7, 7])


# --- ShapeOp, Misc ---

def test_shape_op():
    node = ShapeOpNode(name='s', op_type='Shape', inputs=['input'])
    g = _make_graph([node], input_shape=(1, 3, 4))
    shapes = {g.input_name: g.input_shape}
    node.infer_shape(shapes)
    assert node.output_shape == (3,)  # 3 dims

    center = np.arange(12, dtype=float)
    out = _run_point(g, center)
    # ShapeOp is point-only passthrough
    np.testing.assert_array_equal(out, center)


def test_misc_passthrough():
    node = MiscNode(name='m', op_type='Cast', inputs=['input'])
    g = _make_graph([node], input_shape=(1, 3))
    center = np.array([1, 2, 3], dtype=float)
    out = _run_point(g, center)
    np.testing.assert_array_equal(out, center)


# --- MatMulBilinear ---

def test_matmul_bilinear():
    """MatMul with two computed inputs."""
    # Build a graph: input -> Split into two -> MatMul
    split = SplitNode(name='s', op_type='Split', inputs=['input'],
                      params={'axis': 1, 'split': [2, 3]})
    sout = SplitOutputNode(name='s1', op_type='SplitOutput',
                           inputs=['s'], params={'index': 1})
    # Can't easily do bilinear matmul in this framework without two branches
    # Test directly instead
    from vibecheck.network import _bilinear_point_op, _point_zono
    z_a = DenseZonotope(np.array([1.0, 2.0, 3.0]), np.zeros((3, 0)))
    z_b = DenseZonotope(np.array([4.0, 5.0, 6.0]), np.zeros((3, 0)))

    class FakeNode:
        inputs = ['a', 'b']
        name = 'test'
        op_type = 'Mul'
    class FakeGraph:
        input_shape = (1, 3)
        class nodes_cls:
            pass
        nodes = {'a': type('N', (), {'output_shape': (1, 3)})(),
                 'b': type('N', (), {'output_shape': (1, 3)})()}

    result = _bilinear_point_op(z_a, z_b, np.multiply, FakeNode(), FakeGraph())
    np.testing.assert_array_equal(result.center, [4, 10, 18])


# --- Conv with generators ---

def test_conv_with_generators():
    """Conv node with actual generators (not point)."""
    kernel = np.random.randn(2, 1, 3, 3)
    bias = np.zeros(2)
    node = ConvNode(name='c', op_type='Conv', inputs=['input'],
                    params={'kernel': kernel, 'bias': bias,
                            'stride': (1, 1), 'padding': (0, 0)})
    g = _make_graph([node], input_shape=(1, 1, 4, 4))

    x_lo = np.zeros(16)
    x_hi = np.ones(16)
    z = DenseZonotope.from_input_bounds(x_lo, x_hi)

    zono_state = {g.input_name: z}
    gen_count = {g.input_name: z.generators.shape[1]}
    node.zonotope_propagate(zono_state, gen_count,
                            lambda n: zono_state[n], 'std', g)
    z_out = zono_state['c']
    assert z_out.generators.shape[1] == 16  # same number of generators
    lo, hi = z_out.bounds()
    assert np.all(lo <= hi)


# --- Div with const scale ---

def test_div_const():
    node = DivNode(name='d', op_type='Div', inputs=['input'],
                   params={'scale': np.array([0.5, 2.0, 0.25])})
    g = _make_graph([node], input_shape=(1, 3))
    out = _run_point(g, np.array([4, 6, 8], dtype=float))
    np.testing.assert_array_equal(out, [2, 12, 2])


# --- Sub with negate ---

def test_sub_negate():
    node = SubNode(name='s', op_type='Sub', inputs=['input'],
                   params={'negate': True, 'bias': np.array([10, 20, 30])})
    g = _make_graph([node], input_shape=(1, 3))
    out = _run_point(g, np.array([1, 2, 3], dtype=float))
    # negate: -center + bias = [-1, -2, -3] + [10, 20, 30] = [9, 18, 27]
    np.testing.assert_array_equal(out, [9, 18, 27])


# --- BatchNorm with spatial broadcast ---

def test_batchnorm_spatial():
    """BN on a (1, 2, 3) input — broadcast per-channel."""
    node = BatchNormNode(name='bn', op_type='BatchNormalization',
                         inputs=['input'],
                         params={
                             'scale': np.array([1.0, 2.0]),
                             'bias': np.array([0.0, 0.0]),
                             'mean': np.array([0.0, 0.0]),
                             'var': np.array([1.0, 1.0]),
                             'epsilon': 0.0,
                         })
    g = _make_graph([node], input_shape=(1, 2, 3))
    center = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    out = _run_point(g, center)
    # Channel 0 (scale=1): [1,2,3], Channel 1 (scale=2): [8,10,12]
    np.testing.assert_allclose(out, [1, 2, 3, 8, 10, 12])


# --- ReduceSum along axis ---

def test_reduce_sum_axis():
    node = ReduceNode(name='r', op_type='ReduceSum', inputs=['input'],
                      params={'axes': [1], 'keepdims': 0})
    g = _make_graph([node], input_shape=(1, 3, 4))
    center = np.arange(12, dtype=float)
    out = _run_point(g, center)
    expected = center.reshape(1, 3, 4).sum(axis=1, keepdims=False).flatten()
    np.testing.assert_array_equal(out, expected)


# --- Utility functions ---

def test_prod():
    assert _prod((2, 3, 4)) == 24
    assert _prod((1,)) == 1
    assert _prod(()) == 1


def test_infer_conv_input_shape():
    from vibecheck.network import _infer_conv_input_shape
    # Square spatial
    kernel = np.zeros((16, 3, 3, 3))
    shape = _infer_conv_input_shape(3*8*8, kernel)
    assert shape == (3, 8, 8)
    # Non-square
    shape2 = _infer_conv_input_shape(3*4*6, kernel)
    assert shape2[0] == 3 and shape2[1] * shape2[2] == 24
    # Tuple input
    shape3 = _infer_conv_input_shape((3, 8, 8), kernel)
    assert shape3 == (3, 8, 8)
    # Indivisible
    shape4 = _infer_conv_input_shape(7, kernel)
    assert _prod(shape4) == 7
    # Transpose
    kernel_t = np.zeros((3, 16, 3, 3))
    shape5 = _infer_conv_input_shape(3*8*8, kernel_t, transpose=True)
    assert shape5 == (3, 8, 8)


def test_gemm_nd_matmul():
    """GemmNode with 3D input and 2D weight (ND matmul)."""
    W = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # (2, 3)
    b = np.zeros(2)
    node = GemmNode(name='g', op_type='MatMul', inputs=['input'],
                    params={'W': W, 'b': b})
    g = _make_graph([node], input_shape=(1, 4, 3))
    shapes = {g.input_name: (1, 4, 3)}
    node.infer_shape(shapes)
    assert node.output_shape == (1, 4, 2)

    center = np.arange(12, dtype=float)
    out = _run_point(g, center)
    expected = np.matmul(center.reshape(1, 4, 3), W.T).flatten()
    np.testing.assert_allclose(out, expected)


def test_gemm_1d_weight():
    """GemmNode with 1D weight (dot product along last axis)."""
    W = np.array([1.0, 2.0, 3.0])
    b = np.array(0.0)
    node = GemmNode(name='g', op_type='MatMul', inputs=['input'],
                    params={'W': W, 'b': b})
    g = _make_graph([node], input_shape=(1, 2, 3))
    shapes = {g.input_name: (1, 2, 3)}
    node.infer_shape(shapes)
    assert node.output_shape == (1, 2)

    center = np.arange(6, dtype=float)
    out = _run_point(g, center)
    expected = np.matmul(center.reshape(1, 2, 3), W).flatten()
    np.testing.assert_allclose(out, expected)


def test_resize_no_scales():
    """ResizeNode without scales — passthrough."""
    node = ResizeNode(name='r', op_type='Resize', inputs=['input'], params={})
    g = _make_graph([node], input_shape=(1, 2, 3))
    shapes = {g.input_name: (1, 2, 3)}
    node.infer_shape(shapes)
    assert node.output_shape == (1, 2, 3)

    center = np.arange(6, dtype=float)
    out = _run_point(g, center)
    np.testing.assert_array_equal(out, center)


def test_split_nd():
    """SplitNode with ND axis splitting."""
    split = SplitNode(name='sp', op_type='Split', inputs=['input'],
                      params={'axis': 2, 'split': [2, 3]})
    sout = SplitOutputNode(name='sp1', op_type='SplitOutput',
                           inputs=['sp'], params={'index': 1})
    g = _make_graph([split, sout], input_shape=(1, 2, 5), output_name='sp1')
    # Manually set SplitOutput shape
    sout.output_shape = (1, 2, 3)

    center = np.arange(10, dtype=float)
    zono_state = {g.input_name: DenseZonotope(center, np.zeros((10, 0)))}
    gen_count = {g.input_name: 0}
    split.zonotope_propagate(zono_state, gen_count,
                             lambda n: zono_state[n], 'std', g)
    # First part: axis 2 [:2]
    expected_0 = center.reshape(1, 2, 5)[:, :, :2].flatten()
    np.testing.assert_array_equal(zono_state['sp'].center, expected_0)
    # Second part: axis 2 [2:5]
    expected_1 = center.reshape(1, 2, 5)[:, :, 2:5].flatten()
    np.testing.assert_array_equal(zono_state['sp1'].center, expected_1)


def test_conv_1d():
    """1D Conv: kernel (C_out, C_in, kW)."""
    kernel = np.ones((2, 1, 3))  # 1D kernel
    bias = np.zeros(2)
    node = ConvNode(name='c', op_type='Conv', inputs=['input'],
                    params={'kernel': kernel, 'bias': bias,
                            'stride': (1,), 'padding': (0,)})
    g = _make_graph([node], input_shape=(1, 1, 5))
    shapes = {g.input_name: (1, 1, 5)}
    node.infer_shape(shapes)
    assert node.output_shape == (1, 2, 3)


def test_broadcast_const_op_flat_fallback():
    """_broadcast_const_op with shape mismatch falls back to flat."""
    from vibecheck.network import _broadcast_const_op
    z = DenseZonotope(np.array([1.0, 2.0]), np.zeros((2, 0)))

    class FakeNode:
        inputs = ['input']
        name = 'test'
        op_type = 'Add'
    class FakeGraph:
        input_name = 'input'
        input_shape = (1, 99)  # Doesn't match flat size
        nodes = {}

    result = _broadcast_const_op(z, np.array([10.0, 20.0]), np.add,
                                  FakeNode(), FakeGraph())
    np.testing.assert_array_equal(result.center, [11, 22])


def test_reduce_mean_axis_keepdims():
    node = ReduceNode(name='r', op_type='ReduceMean', inputs=['input'],
                      params={'axes': [2], 'keepdims': 1})
    g = _make_graph([node], input_shape=(1, 2, 3))
    center = np.arange(6, dtype=float)
    out = _run_point(g, center)
    expected = center.reshape(1, 2, 3).mean(axis=2, keepdims=True).flatten()
    np.testing.assert_allclose(out, expected)
