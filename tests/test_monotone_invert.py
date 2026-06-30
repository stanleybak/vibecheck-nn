"""Tests for the gated monotone-output inversion route (converted net/spec).

Covers: structure detection, sound threshold inversion, the truncation, the fp32
numerical gate, the end-to-end converted verification, false-UNSAT guard, and
inertness (gate off / non-matching net). Synthetic ONNX nets (no clone needed).
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import torch

from vibecheck.network import ComputeGraph
from vibecheck.settings import default_settings
from vibecheck.vnnlib_loader import load_vnnlib
from vibecheck import monotone_invert as MI


def _monotone_head_onnx(path, n_in=3, n_z=4, scale=2.0, bias=0.5):
    rng = np.random.default_rng(0)
    W1 = rng.standard_normal((n_in, n_z)).astype(np.float32)
    b1 = rng.standard_normal(n_z).astype(np.float32)
    W2 = rng.standard_normal((n_z, n_z)).astype(np.float32)
    b2 = rng.standard_normal(n_z).astype(np.float32)
    sc = np.full((n_z,), scale, np.float32); bi = np.full((n_z,), bias, np.float32)
    nodes = [helper.make_node('Gemm', ['X', 'W1', 'b1'], ['h']),
             helper.make_node('Relu', ['h'], ['hr']),
             helper.make_node('Gemm', ['hr', 'W2', 'b2'], ['z']),
             helper.make_node('Sigmoid', ['z'], ['sg']),
             helper.make_node('Mul', ['sg', 'SC'], ['m']),
             helper.make_node('Add', ['m', 'BI'], ['Y'])]
    inits = [numpy_helper.from_array(W1, 'W1'), numpy_helper.from_array(b1, 'b1'),
             numpy_helper.from_array(W2, 'W2'), numpy_helper.from_array(b2, 'b2'),
             numpy_helper.from_array(sc, 'SC'), numpy_helper.from_array(bi, 'BI')]
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, n_in])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, n_z])
    m = helper.make_model(helper.make_graph(nodes, 'mh', [X], [Y], inits),
                          opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 9; onnx.checker.check_model(m); onnx.save(m, path)
    return n_z


def _plain_relu_onnx(path, n_in=3, n_out=2):
    W = np.random.default_rng(1).standard_normal((n_in, n_out)).astype(np.float32)
    nodes = [helper.make_node('MatMul', ['X', 'W'], ['z']),
             helper.make_node('Relu', ['z'], ['Y'])]
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, n_in])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, n_out])
    m = helper.make_model(helper.make_graph(nodes, 'pr', [X], [Y],
                          [numpy_helper.from_array(W, 'W')]),
                          opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 9; onnx.save(m, path)


def _spec_vnnlib(path, n_in, n_out, ub_y):
    lines = []
    for i in range(n_in):
        lines.append(f'(declare-const X_{i} Real)')
    for i in range(n_out):
        lines.append(f'(declare-const Y_{i} Real)')
    for i in range(n_in):
        lines += [f'(assert (<= X_{i} 1.0))', f'(assert (>= X_{i} -1.0))']
    for i in range(n_out):
        lines.append(f'(assert (>= Y_{i} {ub_y}))')   # unsafe if Y_i >= ub_y
    open(path, 'w').write('\n'.join(lines) + '\n')


def _g(tmp_path):
    net = str(tmp_path / 'mh.onnx'); nz = _monotone_head_onnx(net)
    g = ComputeGraph.from_onnx(net, dtype=np.float32); g.optimize(default_settings())
    return net, g, g.gpu_graph(device='cpu', dtype=torch.float64), nz


def test_detect_finds_monotone_head(tmp_path):
    _, _, gg, nz = _g(tmp_path)
    det = MI.detect(gg)
    assert det is not None and det['g_type'] == 'sigmoid'
    np.testing.assert_allclose(det['scale'], 2.0)
    np.testing.assert_allclose(det['bias'], 0.5)


def test_detect_none_on_plain_relu(tmp_path):
    net = str(tmp_path / 'pr.onnx'); _plain_relu_onnx(net)
    g = ComputeGraph.from_onnx(net, dtype=np.float32); g.optimize(default_settings())
    assert MI.detect(g.gpu_graph(device='cpu', dtype=torch.float64)) is None


def test_invert_threshold_sound():
    """The inverted threshold must put g(t) on the conservative side."""
    # unsafe g>=c -> z>=t, need g(t) <= c (t <= z*)
    op, t = MI._invert_threshold('sigmoid', None, 0.7, True, -60, 60)
    assert op == '>=' and MI._g_scalar('sigmoid', None, t, np.float32) <= 0.7
    # unsafe g<=c -> z<=t, need g(t) >= c (t >= z*)
    op, t = MI._invert_threshold('sigmoid', None, 0.3, False, -60, 60)
    assert op == '<=' and MI._g_scalar('sigmoid', None, t, np.float32) >= 0.3
    # edge: c above range -> g>=c never -> trivially safe
    op, t = MI._invert_threshold('sigmoid', None, 1.5, True, -60, 60)
    assert t >= 1e29


def test_truncate_at_z_drops_head(tmp_path):
    net, g, gg, nz = _g(tmp_path)
    det = MI.detect(gg)
    gz = MI._truncate_at_z(g, det['z_name'])
    assert gz is not None and gz.output_name == det['z_name']
    types = {gz.nodes[n].op_type for n in gz.nodes}
    assert 'Sigmoid' not in types and 'Mul' not in types   # head dropped


def test_try_verify_certifies_safe(tmp_path):
    net, g, gg, nz = _g(tmp_path)
    sp = str(tmp_path / 's.vnnlib'); _spec_vnnlib(sp, 3, nz, ub_y=100.0)
    spec = load_vnnlib(sp, dtype=np.float64)
    s = default_settings(monotone_output_inversion=True)
    out = MI.try_verify(g, spec, s, device='cpu')
    assert out is not None and out[0] == 'verified' and out[1]['fp32_gate']


def test_gate_off_is_inert(tmp_path):
    net, g, gg, nz = _g(tmp_path)
    sp = str(tmp_path / 's.vnnlib'); _spec_vnnlib(sp, 3, nz, ub_y=100.0)
    spec = load_vnnlib(sp, dtype=np.float64)
    assert MI.try_verify(g, spec, default_settings(
        monotone_output_inversion=False), device='cpu') is None


def test_inert_on_non_matching_net(tmp_path):
    net = str(tmp_path / 'pr.onnx'); _plain_relu_onnx(net)
    g = ComputeGraph.from_onnx(net, dtype=np.float32); g.optimize(default_settings())
    sp = str(tmp_path / 's.vnnlib'); _spec_vnnlib(sp, 3, 2, ub_y=100.0)
    spec = load_vnnlib(sp, dtype=np.float64)
    assert MI.try_verify(g, spec, default_settings(
        monotone_output_inversion=True), device='cpu') is None


def test_tight_threshold_not_falsely_certified(tmp_path):
    """Head = 2*sigmoid(z)+0.5 in (0.5,2.5); threshold 0.6 is exceeded -> must
    NOT verify (guards against false UNSAT)."""
    net, g, gg, nz = _g(tmp_path)
    sp = str(tmp_path / 's.vnnlib'); _spec_vnnlib(sp, 3, nz, ub_y=0.6)
    spec = load_vnnlib(sp, dtype=np.float64)
    out = MI.try_verify(g, spec, default_settings(
        monotone_output_inversion=True), device='cpu')
    assert out is None


def test_fp32_gate_rejects_borderline():
    """_fp32_gate must reject when the fp32 head max is not strictly below rhs."""
    from vibecheck.spec import Constraint, Conjunct, VNNSpec
    det = dict(g_type='sigmoid', relax=None, scale=np.array([1.0]),
               bias=np.array([0.0]))
    zlo = np.array([-1.0]); zhi = np.array([5.0])
    head_range = (0, 1)
    # head = sigmoid(z) in (~0.27, ~0.993); rhs just below the max -> NOT safe
    spec_bad = VNNSpec(np.array([0.0]), np.array([1.0]),
                       [Conjunct([Constraint(0, '>=', 0.9)])])
    assert MI._fp32_gate(spec_bad, det, head_range, zlo, zhi) is False
    # rhs well above the max -> safe
    spec_ok = VNNSpec(np.array([0.0]), np.array([1.0]),
                      [Conjunct([Constraint(0, '>=', 2.0)])])
    assert MI._fp32_gate(spec_ok, det, head_range, zlo, zhi) is True
