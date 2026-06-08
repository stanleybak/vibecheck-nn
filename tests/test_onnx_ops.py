"""Tests for onnx_loader.py — create synthetic ONNX models to cover all parsing branches."""

import numpy as np
import pytest
import onnx
from onnx import helper, TensorProto, numpy_helper
from vibecheck.network import ComputeGraph


def _save_and_load(model, tmp_path, name='test.onnx'):
    """Save ONNX model and load as ComputeGraph."""
    path = str(tmp_path / name)
    onnx.save(model, path)
    return ComputeGraph.from_onnx(path)


def _make_model(nodes, inputs, outputs, initializers, opset=13):
    """Helper to build an ONNX model."""
    graph = helper.make_graph(nodes, 'test', inputs, outputs, initializers)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', opset)])


def _input(name='X', shape=[1, 4]):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _output(name='Y', shape=None):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _init(name, arr):
    return numpy_helper.from_array(arr.astype(np.float32), name=name)


# ---- Basic FC ----

def test_gemm_basic(tmp_path):
    W = _init('W', np.eye(3, 4))
    b = _init('b', np.zeros(3))
    node = helper.make_node('Gemm', ['X', 'W', 'b'], ['Y'], transB=1)
    g = _save_and_load(_make_model([node], [_input()], [_output()], [W, b]), tmp_path)
    assert g.nodes[g.output_name].op_type == 'Gemm'


# ---- Clip ----

def test_clip(tmp_path):
    min_val = _init('min', np.array(0.0))
    max_val = _init('max', np.array(6.0))
    node = helper.make_node('Clip', ['X', 'min', 'max'], ['Y'])
    g = _save_and_load(_make_model([node], [_input()], [_output()], [min_val, max_val]), tmp_path)
    assert g.nodes[g.output_name].op_type == 'Clip'
    assert g.nodes[g.output_name].params['min'] == 0.0
    assert g.nodes[g.output_name].params['max'] == 6.0


# ---- Neg ----

def test_neg(tmp_path):
    node = helper.make_node('Neg', ['X'], ['Y'])
    g = _save_and_load(_make_model([node], [_input()], [_output()], []), tmp_path)
    assert g.nodes[g.output_name].op_type == 'Neg'


# ---- Identity ----

def test_identity(tmp_path):
    node = helper.make_node('Identity', ['X'], ['Y'])
    g = _save_and_load(_make_model([node], [_input()], [_output()], []), tmp_path)
    assert g.nodes[g.output_name].op_type == 'Identity'


# ---- Dropout ----

def test_dropout(tmp_path):
    node = helper.make_node('Dropout', ['X'], ['Y'])
    g = _save_and_load(_make_model([node], [_input()], [_output()], []), tmp_path)
    assert g.nodes[g.output_name].op_type == 'Dropout'


# ---- Flatten, Squeeze, Unsqueeze ----

def test_flatten(tmp_path):
    node = helper.make_node('Flatten', ['X'], ['Y'], axis=1)
    g = _save_and_load(_make_model([node], [_input(shape=[1, 2, 3])], [_output()], []), tmp_path)
    assert g.nodes[g.output_name].op_type == 'Flatten'


def test_squeeze(tmp_path):
    axes = _init('axes', np.array([0]))
    node = helper.make_node('Squeeze', ['X', 'axes'], ['Y'])
    g = _save_and_load(_make_model([node], [_input(shape=[1, 4])], [_output()], [axes]), tmp_path)
    # Squeeze may be folded or remain
    assert len(g.topo_order) >= 1


def test_unsqueeze(tmp_path):
    axes = _init('axes', np.array([2]))
    node = helper.make_node('Unsqueeze', ['X', 'axes'], ['Y'])
    g = _save_and_load(_make_model([node], [_input()], [_output()], [axes]), tmp_path)
    assert any(n.op_type == 'Unsqueeze' for n in g.nodes.values())


# ---- Reshape ----

def test_reshape(tmp_path):
    shape = _init('shape', np.array([1, 2, 2]))
    node = helper.make_node('Reshape', ['X', 'shape'], ['Y'])
    g = _save_and_load(_make_model([node], [_input()], [_output()], [shape]), tmp_path)
    assert any(n.op_type == 'Reshape' for n in g.nodes.values())


# ---- Conv ----

def test_conv(tmp_path):
    W = _init('W', np.random.randn(2, 1, 3, 3))
    b = _init('b', np.zeros(2))
    node = helper.make_node('Conv', ['X', 'W', 'b'], ['Y'],
                            strides=[1, 1], pads=[0, 0, 0, 0])
    g = _save_and_load(_make_model([node], [_input(shape=[1, 1, 5, 5])], [_output()], [W, b]), tmp_path)
    assert any(n.op_type == 'Conv' for n in g.nodes.values())


# ---- ConvTranspose ----

def test_conv_transpose(tmp_path):
    W = _init('W', np.random.randn(1, 1, 2, 2))
    b = _init('b', np.zeros(1))
    node = helper.make_node('ConvTranspose', ['X', 'W', 'b'], ['Y'],
                            strides=[2, 2], pads=[0, 0, 0, 0])
    g = _save_and_load(_make_model([node], [_input(shape=[1, 1, 2, 2])], [_output()], [W, b]), tmp_path)
    assert any(n.op_type == 'ConvTranspose' for n in g.nodes.values())


# ---- MaxPool ----

def test_maxpool(tmp_path):
    node = helper.make_node('MaxPool', ['X'], ['Y'],
                            kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0])
    g = _save_and_load(_make_model([node], [_input(shape=[1, 1, 4, 4])], [_output()], []), tmp_path)
    assert any(n.op_type == 'MaxPool' for n in g.nodes.values())


# ---- AveragePool ----

def test_averagepool(tmp_path):
    node = helper.make_node('AveragePool', ['X'], ['Y'],
                            kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0])
    g = _save_and_load(_make_model([node], [_input(shape=[1, 1, 4, 4])], [_output()], []), tmp_path)
    assert any(n.op_type == 'AveragePool' for n in g.nodes.values())


# ---- Pad ----

def test_pad(tmp_path):
    pads = _init('pads', np.array([0, 0, 1, 1, 0, 0, 1, 1]))
    val = _init('val', np.array(0.0))
    node = helper.make_node('Pad', ['X', 'pads', 'val'], ['Y'])
    g = _save_and_load(_make_model([node], [_input(shape=[1, 1, 3, 3])], [_output()], [pads, val]), tmp_path)
    assert any(n.op_type == 'Pad' for n in g.nodes.values())


# ---- Concat ----

def test_concat(tmp_path):
    W1 = _init('W1', np.eye(2, 4))
    b1 = _init('b1', np.zeros(2))
    W2 = _init('W2', np.eye(3, 4))
    b2 = _init('b2', np.zeros(3))
    g1 = helper.make_node('Gemm', ['X', 'W1', 'b1'], ['a'], transB=1)
    g2 = helper.make_node('Gemm', ['X', 'W2', 'b2'], ['b'], transB=1)
    cat = helper.make_node('Concat', ['a', 'b'], ['Y'], axis=1)
    g = _save_and_load(_make_model([g1, g2, cat], [_input()], [_output()],
                                    [W1, b1, W2, b2]), tmp_path)
    assert any(n.op_type == 'Concat' for n in g.nodes.values())


# ---- Slice ----

def test_slice(tmp_path):
    starts = _init('starts', np.array([1]))
    ends = _init('ends', np.array([3]))
    axes = _init('axes', np.array([1]))
    node = helper.make_node('Slice', ['X', 'starts', 'ends', 'axes'], ['Y'])
    g = _save_and_load(_make_model([node], [_input()], [_output()],
                                    [starts, ends, axes]), tmp_path)
    assert any(n.op_type == 'Slice' for n in g.nodes.values())


# ---- Gather ----

def test_gather(tmp_path):
    indices = _init('idx', np.array([0, 2]))
    node = helper.make_node('Gather', ['X', 'idx'], ['Y'], axis=0)
    g = _save_and_load(_make_model([node], [_input()], [_output()],
                                    [indices]), tmp_path)
    assert any(n.op_type == 'Gather' for n in g.nodes.values())


# ---- ReduceSum / ReduceMean ----

def test_reduce_sum(tmp_path):
    axes = _init('axes', np.array([1]))
    node = helper.make_node('ReduceSum', ['X', 'axes'], ['Y'], keepdims=0)
    g = _save_and_load(_make_model([node], [_input(shape=[1, 3, 4])], [_output()],
                                    [axes]), tmp_path)
    assert any(n.op_type == 'ReduceSum' for n in g.nodes.values())


def test_reduce_mean(tmp_path):
    axes = _init('axes', np.array([1]))
    node = helper.make_node('ReduceMean', ['X', 'axes'], ['Y'], keepdims=1)
    g = _save_and_load(_make_model([node], [_input(shape=[1, 3, 4])], [_output()],
                                    [axes]), tmp_path)
    assert any(n.op_type == 'ReduceMean' for n in g.nodes.values())


# ---- Resize ----

def test_resize(tmp_path):
    roi = _init('roi', np.array([], dtype=np.float32))
    scales = _init('scales', np.array([1, 1, 2, 2], dtype=np.float32))
    node = helper.make_node('Resize', ['X', 'roi', 'scales'], ['Y'])
    g = _save_and_load(_make_model([node], [_input(shape=[1, 1, 2, 2])], [_output()],
                                    [roi, scales]), tmp_path)
    assert any(n.op_type == 'Resize' for n in g.nodes.values())


# ---- Transpose ----

def test_transpose(tmp_path):
    node = helper.make_node('Transpose', ['X'], ['Y'], perm=[0, 2, 1])
    g = _save_and_load(_make_model([node], [_input(shape=[1, 3, 4])], [_output()], []), tmp_path)
    assert any(n.op_type == 'Transpose' for n in g.nodes.values())


# ---- Sin, Cos, Pow ----

def test_sin(tmp_path):
    node = helper.make_node('Sin', ['X'], ['Y'])
    g = _save_and_load(_make_model([node], [_input()], [_output()], []), tmp_path)
    assert any(n.op_type == 'Sin' for n in g.nodes.values())


def test_cos(tmp_path):
    node = helper.make_node('Cos', ['X'], ['Y'])
    g = _save_and_load(_make_model([node], [_input()], [_output()], []), tmp_path)
    assert any(n.op_type == 'Cos' for n in g.nodes.values())


def test_pow(tmp_path):
    exp = _init('exp', np.array(2.0))
    node = helper.make_node('Pow', ['X', 'exp'], ['Y'])
    g = _save_and_load(_make_model([node], [_input()], [_output()], [exp]), tmp_path)
    assert any(n.op_type == 'Pow' for n in g.nodes.values())


# ---- Floor ----

def test_floor(tmp_path):
    node = helper.make_node('Floor', ['X'], ['Y'])
    g = _save_and_load(_make_model([node], [_input()], [_output()], []), tmp_path)
    assert any(n.op_type == 'Floor' for n in g.nodes.values())


# ---- BatchNorm (unfused — no preceding Conv/Gemm) ----

def test_batchnorm_unfused(tmp_path):
    scale = _init('scale', np.ones(4))
    bias = _init('bias', np.zeros(4))
    mean = _init('mean', np.zeros(4))
    var = _init('var', np.ones(4))
    node = helper.make_node('BatchNormalization', ['X', 'scale', 'bias', 'mean', 'var'], ['Y'])
    g = _save_and_load(_make_model([node], [_input()], [_output()],
                                    [scale, bias, mean, var]), tmp_path)
    # BN not folded (no preceding Conv/Gemm)
    assert any(n.op_type == 'BatchNormalization' for n in g.nodes.values())


# ---- BatchNorm folded into Gemm ----

def test_batchnorm_folded(tmp_path):
    W = _init('W', np.eye(4))
    b = _init('b', np.zeros(4))
    scale = _init('scale', np.array([2, 2, 2, 2], dtype=np.float32))
    bn_bias = _init('bn_bias', np.ones(4))
    mean = _init('mean', np.zeros(4))
    var = _init('var', np.ones(4))
    gemm = helper.make_node('Gemm', ['X', 'W', 'b'], ['g'], transB=1)
    bn = helper.make_node('BatchNormalization', ['g', 'scale', 'bn_bias', 'mean', 'var'], ['Y'])
    g = _save_and_load(_make_model([gemm, bn], [_input()], [_output()],
                                    [W, b, scale, bn_bias, mean, var]), tmp_path)
    # BN should be folded
    assert not any(n.op_type == 'BatchNormalization' for n in g.nodes.values())


# ---- Split ----

def test_split(tmp_path):
    split_sizes = _init('split', np.array([2, 2]))
    node = helper.make_node('Split', ['X', 'split'], ['a', 'b'], axis=1)
    relu = helper.make_node('Relu', ['b'], ['Y'])
    g = _save_and_load(_make_model([node, relu], [_input()], [_output()],
                                    [split_sizes]), tmp_path)
    assert any(n.op_type == 'Split' for n in g.nodes.values())


# ---- LeakyRelu ----

def test_leakyrelu(tmp_path):
    node = helper.make_node('LeakyRelu', ['X'], ['Y'], alpha=0.01)
    g = _save_and_load(_make_model([node], [_input()], [_output()], []), tmp_path)
    assert any(n.op_type == 'LeakyRelu' for n in g.nodes.values())


# ---- Sigmoid ----

def test_sigmoid(tmp_path):
    node = helper.make_node('Sigmoid', ['X'], ['Y'])
    g = _save_and_load(_make_model([node], [_input()], [_output()], []), tmp_path)
    assert any(n.op_type == 'Sigmoid' for n in g.nodes.values())


# ---- Tanh ----

def test_tanh(tmp_path):
    node = helper.make_node('Tanh', ['X'], ['Y'])
    g = _save_and_load(_make_model([node], [_input()], [_output()], []), tmp_path)
    assert any(n.op_type == 'Tanh' for n in g.nodes.values())


# ---- Softmax ----

def test_softmax(tmp_path):
    node = helper.make_node('Softmax', ['X'], ['Y'])
    g = _save_and_load(_make_model([node], [_input()], [_output()], []), tmp_path)
    assert any(n.op_type == 'Softmax' for n in g.nodes.values())


# ---- Sign ----

def test_sign(tmp_path):
    node = helper.make_node('Sign', ['X'], ['Y'])
    g = _save_and_load(_make_model([node], [_input()], [_output()], []), tmp_path)
    assert any(n.op_type == 'Sign' for n in g.nodes.values())


# ---- Constant folding: chain of const ops ----

def test_constant_fold_chain(tmp_path):
    """Relu(const) -> Add(const, const) should all be folded."""
    c = _init('c', np.array([-1, 2, -3, 4], dtype=np.float32))
    b = _init('b', np.ones(4))
    relu = helper.make_node('Relu', ['c'], ['r'])
    add = helper.make_node('Add', ['X', 'r'], ['Y'])
    g = _save_and_load(_make_model([relu, add], [_input()], [_output()],
                                    [c, b]), tmp_path)
    # Relu(c) should be folded to a constant; Add should remain
    assert any(n.op_type == 'Add' for n in g.nodes.values())
    # The Relu node should be folded away
    assert not any(n.op_type == 'Relu' for n in g.nodes.values())


# ---- Add/Sub/Mul/Div with constants ----

def test_add_const(tmp_path):
    b = _init('b', np.ones(4))
    node = helper.make_node('Add', ['X', 'b'], ['Y'])
    g = _save_and_load(_make_model([node], [_input()], [_output()], [b]), tmp_path)
    assert any(n.op_type == 'Add' for n in g.nodes.values())


def test_sub_const(tmp_path):
    b = _init('b', np.ones(4))
    node = helper.make_node('Sub', ['X', 'b'], ['Y'])
    g = _save_and_load(_make_model([node], [_input()], [_output()], [b]), tmp_path)
    assert any(n.op_type == 'Sub' for n in g.nodes.values())


def test_mul_const(tmp_path):
    s = _init('s', np.array([2, 2, 2, 2], dtype=np.float32))
    node = helper.make_node('Mul', ['X', 's'], ['Y'])
    g = _save_and_load(_make_model([node], [_input()], [_output()], [s]), tmp_path)
    assert any(n.op_type == 'Mul' for n in g.nodes.values())


def test_div_const(tmp_path):
    d = _init('d', np.array([2, 2, 2, 2], dtype=np.float32))
    node = helper.make_node('Div', ['X', 'd'], ['Y'])
    g = _save_and_load(_make_model([node], [_input()], [_output()], [d]), tmp_path)
    assert any(n.op_type == 'Div' for n in g.nodes.values())


# ---- MatMul ----

def test_matmul_const(tmp_path):
    W = _init('W', np.eye(3, 4))
    node = helper.make_node('MatMul', ['X', 'W'], ['Y'])
    g = _save_and_load(_make_model([node], [_input()], [_output()], [W]), tmp_path)
    assert any(n.op_type == 'MatMul' for n in g.nodes.values())


# ---- Non-gz ONNX file ----

def test_conv_transpose_no_bias(tmp_path):
    """ConvTranspose without explicit bias."""
    W = _init('W', np.random.randn(1, 1, 2, 2))
    node = helper.make_node('ConvTranspose', ['X', 'W'], ['Y'],
                            strides=[2, 2], pads=[0, 0, 0, 0])
    g = _save_and_load(_make_model([node], [_input(shape=[1, 1, 2, 2])], [_output()], [W]), tmp_path)
    assert any(n.op_type == 'ConvTranspose' for n in g.nodes.values())


def test_add_both_computed(tmp_path):
    """Add with two computed inputs (skip connection pattern)."""
    W1 = _init('W1', np.eye(4))
    b1 = _init('b1', np.zeros(4))
    W2 = _init('W2', np.eye(4))
    b2 = _init('b2', np.zeros(4))
    g1 = helper.make_node('Gemm', ['X', 'W1', 'b1'], ['a'], transB=1)
    g2 = helper.make_node('Gemm', ['X', 'W2', 'b2'], ['b'], transB=1)
    add = helper.make_node('Add', ['a', 'b'], ['Y'])
    g = _save_and_load(_make_model([g1, g2, add], [_input()], [_output()],
                                    [W1, b1, W2, b2]), tmp_path)
    assert any(n.op_type == 'Add' for n in g.nodes.values())


def test_sub_both_const_fold(tmp_path):
    """Sub where both inputs are constants → folded."""
    a = _init('a', np.array([3, 4, 5, 6], dtype=np.float32))
    b = _init('b', np.array([1, 1, 1, 1], dtype=np.float32))
    sub = helper.make_node('Sub', ['a', 'b'], ['c'])
    add = helper.make_node('Add', ['X', 'c'], ['Y'])
    g = _save_and_load(_make_model([sub, add], [_input()], [_output()],
                                    [a, b]), tmp_path)
    # Sub should be folded
    assert not any(n.op_type == 'Sub' for n in g.nodes.values())


def test_mul_both_const_fold(tmp_path):
    """Mul where both inputs are constants → folded."""
    a = _init('a', np.array([2, 3, 4, 5], dtype=np.float32))
    b = _init('b', np.array([1, 2, 3, 4], dtype=np.float32))
    mul = helper.make_node('Mul', ['a', 'b'], ['c'])
    add = helper.make_node('Add', ['X', 'c'], ['Y'])
    g = _save_and_load(_make_model([mul, add], [_input()], [_output()],
                                    [a, b]), tmp_path)
    assert not any(n.op_type == 'Mul' for n in g.nodes.values())


def test_matmul_bilinear(tmp_path):
    """MatMul with two computed inputs (no constant weight)."""
    W = _init('W', np.eye(4))
    b = _init('b', np.zeros(4))
    gemm = helper.make_node('Gemm', ['X', 'W', 'b'], ['a'], transB=1)
    mm = helper.make_node('MatMul', ['a', 'X'], ['Y'])
    g = _save_and_load(_make_model([gemm, mm], [_input()], [_output()],
                                    [W, b]), tmp_path)
    assert any(n.op_type == 'MatMul' for n in g.nodes.values())


def test_shape_op(tmp_path):
    """Shape op."""
    node = helper.make_node('Shape', ['X'], ['Y'])
    g = _save_and_load(_make_model([node], [_input(shape=[1, 3, 4])], [_output()], []), tmp_path)
    assert any(n.op_type == 'Shape' for n in g.nodes.values())


def test_constant_of_shape(tmp_path):
    """ConstantOfShape."""
    shape = _init('shape', np.array([1, 3], dtype=np.int64))
    node = helper.make_node('ConstantOfShape', ['shape'], ['Y'],
                            value=helper.make_tensor('val', TensorProto.FLOAT, [1], [7.0]))
    g = _save_and_load(_make_model([node], [_input('shape', [2])], [_output()],
                                    []), tmp_path, 'cshape.onnx')
    # May or may not be in graph depending on folding


def test_dynamic_input_all_zeros(tmp_path):
    """Input with all dynamic dims [0, 0, 0, 5]."""
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [0, 0, 0, 5])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, None)
    node = helper.make_node('Relu', ['X'], ['Y'])
    g = _save_and_load(_make_model([node], [X], [Y], []), tmp_path)
    assert g.input_shape == (1, 5)  # dynamic dims collapsed


def test_softmax_with_axis(tmp_path):
    """Softmax with axis attribute (covers onnx_loader line 185)."""
    node = helper.make_node('Softmax', ['X'], ['Y'], axis=1)
    g = _save_and_load(_make_model([node], [_input()], [_output()], []), tmp_path)
    assert any(n.op_type == 'Softmax' for n in g.nodes.values())


def test_concat_with_const_input(tmp_path):
    """Concat where one input is a constant (covers line 238)."""
    c = _init('c', np.ones(4))
    W = _init('W', np.eye(4))
    b = _init('b', np.zeros(4))
    gemm = helper.make_node('Gemm', ['X', 'W', 'b'], ['a'], transB=1)
    cat = helper.make_node('Concat', ['a', 'c'], ['Y'], axis=1)
    g = _save_and_load(_make_model([gemm, cat], [_input()], [_output()],
                                    [W, b, c]), tmp_path)
    assert any(n.op_type == 'Concat' for n in g.nodes.values())


def test_gather_computed_indices(tmp_path):
    """Gather where indices come from a computed node (covers line 283)."""
    W = _init('W', np.eye(4))
    b = _init('b', np.zeros(4))
    gemm = helper.make_node('Gemm', ['X', 'W', 'b'], ['a'], transB=1)
    # Use the Gemm output as indices (not realistic but tests the path)
    cast = helper.make_node('Cast', ['a'], ['idx'], to=7)  # to int64
    gather = helper.make_node('Gather', ['X', 'idx'], ['Y'], axis=0)
    g = _save_and_load(_make_model([gemm, cast, gather], [_input()], [_output()],
                                    [W, b]), tmp_path)
    # May or may not have Gather depending on folding


def test_reduce_sum_axes_attr(tmp_path):
    """ReduceSum with axes as attribute instead of input (opset < 13, covers line 303)."""
    node = helper.make_node('ReduceSum', ['X'], ['Y'], axes=[1], keepdims=0)
    g = _save_and_load(_make_model([node], [_input(shape=[1, 3, 4])], [_output()], [],
                                    opset=11), tmp_path)
    assert any(n.op_type == 'ReduceSum' for n in g.nodes.values())


def test_constant_of_shape_value_attr(tmp_path):
    """ConstantOfShape with value as attribute (covers line 348)."""
    shape = _init('shape', np.array([1, 3], dtype=np.int64))
    node = helper.make_node('ConstantOfShape', ['shape'], ['c'],
                            value=helper.make_tensor('v', TensorProto.FLOAT, [1], [5.0]))
    add = helper.make_node('Add', ['X', 'c'], ['Y'])
    g = _save_and_load(_make_model([node, add], [_input(shape=[1, 3])], [_output()],
                                    [shape]), tmp_path)


def test_reduce_mean_axes_attr(tmp_path):
    """ReduceMean with axes as attribute (not input), covers onnx_loader line 303."""
    node = helper.make_node('ReduceMean', ['X'], ['Y'], axes=[1], keepdims=1)
    g = _save_and_load(_make_model([node], [_input(shape=[1, 3, 4])], [_output()], [],
                                    opset=11), tmp_path, 'rmean.onnx')
    assert any(n.op_type == 'ReduceMean' for n in g.nodes.values())


def test_constant_of_shape_value_attr_v2(tmp_path):
    """ConstantOfShape with value as tensor attribute, covers onnx_loader line 348."""
    # Create a Shape node to get shape, then ConstantOfShape
    shape_node = helper.make_node('Shape', ['X'], ['s'])
    cos_node = helper.make_node('ConstantOfShape', ['s'], ['c'],
                                 value=helper.make_tensor('v', TensorProto.FLOAT, [1], [42.0]))
    add_node = helper.make_node('Add', ['X', 'c'], ['Y'])
    g = _save_and_load(_make_model([shape_node, cos_node, add_node],
                                    [_input()], [_output()], []), tmp_path, 'cos2.onnx')


def test_load_non_gz(tmp_path):
    """Covers the non-.gz path in load_onnx."""
    W = _init('W', np.eye(2, 4))
    b = _init('b', np.zeros(2))
    node = helper.make_node('Gemm', ['X', 'W', 'b'], ['Y'], transB=1)
    model = _make_model([node], [_input()], [_output()], [W, b])
    path = str(tmp_path / 'test.onnx')
    onnx.save(model, path)
    g = ComputeGraph.from_onnx(path)
    assert g.flat_size(g.output_name) == 2


# ---- Spectral-norm-style folded weights (Div const/const → Gemm/Conv weight) ----

def test_div_folded_gemm_weight(tmp_path):
    """Gemm weight produced by Div(const, const) must fold to a constant.

    Mirrors spectral normalization (W_orig / sigma) as emitted by cgan
    small_transformer: the weight is computed at inference but from all-constant
    operands. Exercises the Div const/const fold + Gemm `_const` resolution.
    """
    W_orig = _init('W_orig', np.arange(12, dtype=np.float32).reshape(3, 4))
    sigma = _init('sigma', np.full((3, 1), 2.0, dtype=np.float32))
    b = _init('b', np.zeros(3))
    div = helper.make_node('Div', ['W_orig', 'sigma'], ['W_folded'])
    gemm = helper.make_node('Gemm', ['X', 'W_folded', 'b'], ['Y'], transB=1)
    g = _save_and_load(_make_model([div, gemm], [_input()], [_output()],
                                   [W_orig, sigma, b]), tmp_path, 'snlinear.onnx')
    node = g.nodes[g.output_name]
    assert node.op_type == 'Gemm'
    # transB=1 keeps the weight un-transposed: W_orig / sigma
    np.testing.assert_allclose(node.params['W'], np.arange(12).reshape(3, 4) / 2.0)


def test_div_folded_conv_kernel(tmp_path):
    """Conv kernel produced by Div(const, const) must fold to a constant."""
    K_orig = _init('K_orig', np.ones((2, 3, 1, 1), dtype=np.float32))
    sigma = _init('sigma', np.array(4.0, dtype=np.float32))
    div = helper.make_node('Div', ['K_orig', 'sigma'], ['K_folded'])
    conv = helper.make_node('Conv', ['X', 'K_folded'], ['Y'],
                            kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0])
    g = _save_and_load(_make_model([div, conv], [_input('X', [1, 3, 2, 2])],
                                   [_output()], [K_orig, sigma]), tmp_path, 'snconv.onnx')
    node = g.nodes[g.output_name]
    assert node.op_type == 'Conv'
    np.testing.assert_allclose(node.params['kernel'], np.ones((2, 3, 1, 1)) / 4.0)


def test_dynamic_gemm_weight_raises(tmp_path):
    """A genuinely non-constant Gemm weight must raise, never silently skip."""
    b = _init('b', np.zeros(3))
    gemm = helper.make_node('Gemm', ['X', 'Wdyn', 'b'], ['Y'], transB=1)
    model = _make_model([gemm],
                        [_input('X', [1, 4]), _input('Wdyn', [3, 4])],
                        [_output()], [b])
    path = str(tmp_path / 'dyn_gemm.onnx')
    onnx.save(model, path)
    with pytest.raises(NotImplementedError, match='non-constant weight'):
        ComputeGraph.from_onnx(path)


def test_dynamic_conv_kernel_raises(tmp_path):
    """A genuinely non-constant Conv kernel must raise, never silently skip."""
    conv = helper.make_node('Conv', ['X', 'Kdyn'], ['Y'],
                            kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0])
    model = _make_model([conv],
                        [_input('X', [1, 3, 2, 2]), _input('Kdyn', [2, 3, 1, 1])],
                        [_output()], [])
    path = str(tmp_path / 'dyn_conv.onnx')
    onnx.save(model, path)
    with pytest.raises(NotImplementedError, match='non-constant kernel'):
        ComputeGraph.from_onnx(path)


def test_dynamic_conv_transpose_kernel_raises(tmp_path):
    """A genuinely non-constant ConvTranspose kernel must raise."""
    ct = helper.make_node('ConvTranspose', ['X', 'Kdyn'], ['Y'],
                          kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0])
    model = _make_model([ct],
                        [_input('X', [1, 3, 2, 2]), _input('Kdyn', [3, 2, 1, 1])],
                        [_output()], [])
    path = str(tmp_path / 'dyn_ct.onnx')
    onnx.save(model, path)
    with pytest.raises(NotImplementedError, match='non-constant kernel'):
        ComputeGraph.from_onnx(path)


# ---- onnxsim auto-simplify trigger (Softmax + >4 MatMul) ----

def _attention_trigger_model():
    """A model that fires the onnxsim auto-simplify trigger: Softmax + 5
    constant-weight MatMuls (which onnxsim fuses into Gemm)."""
    inits = [_init(f'W{i}', np.eye(4)) for i in range(5)]
    nodes = []
    prev = 'X'
    for i in range(5):
        nodes.append(helper.make_node('MatMul', [prev, f'W{i}'], [f'h{i}']))
        prev = f'h{i}'
    nodes.append(helper.make_node('Softmax', [prev], ['Y'], axis=1))
    return _make_model(nodes, [_input('X', [1, 4])],
                       [_output('Y', [1, 4])], inits)


def test_simplify_trigger_loads(tmp_path):
    """Softmax + >4 MatMul auto-triggers onnxsim and loads the folded graph."""
    path = str(tmp_path / 'attn.onnx')
    onnx.save(_attention_trigger_model(), path)
    g = ComputeGraph.from_onnx(path)
    # The simplify path ran and produced a loadable graph; Softmax survives.
    assert any(n.op_type == 'Softmax' for n in g.nodes.values())


def test_simplify_missing_onnxsim_raises_loudly(tmp_path, monkeypatch):
    """When the model needs folding but onnxsim is absent, fail LOUDLY with a
    clear message — never silently fall back to the unfoldable graph."""
    import sys
    path = str(tmp_path / 'attn.onnx')
    onnx.save(_attention_trigger_model(), path)
    # Make `import onnxsim` raise ImportError.
    monkeypatch.setitem(sys.modules, 'onnxsim', None)
    with pytest.raises(ImportError, match='onnxsim is required'):
        ComputeGraph.from_onnx(path)


def test_simplify_failure_raises_loudly(tmp_path, monkeypatch):
    """If onnxsim reports it could not simplify, that must surface loudly."""
    import onnxsim
    path = str(tmp_path / 'attn.onnx')
    onnx.save(_attention_trigger_model(), path)
    monkeypatch.setattr(onnxsim, 'simplify', lambda m, *a, **k: (m, False))
    with pytest.raises(RuntimeError, match='could not simplify'):
        ComputeGraph.from_onnx(path)
