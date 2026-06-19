"""Unit tests for the surrogate-attack mode (src/vibecheck/surrogate_pgd.py).

Builds a tiny synthetic INT8-quantized ONNX (QuantizeLinear/DequantizeLinear) so the
fold -> onnx2torch -> PGD -> ORT-CPU-validate path runs end-to-end in ~seconds on CPU.
"""
import gzip
import os

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from vibecheck import surrogate_pgd as sp


# ----------------------------------------------------------------- synthetic models

def _quant_onnx(path, w=(3, 4), per_axis=False):
    """X[1,2] --Q/DQ--> @ DQ(int8 W[2,1]) + b --Sigmoid--> Y[1,1]."""
    W = np.array([[w[0]], [w[1]]], dtype=np.int8)
    xscale = np.array(0.02, np.float32); xzp = np.array(0, np.int8)
    if per_axis:
        wscale = np.array([0.1], np.float32); wzp = np.array([0], np.int8)
        dq_w = helper.make_node('DequantizeLinear', ['W', 'wscale', 'wzp'], ['Wf'], axis=1)
    else:
        wscale = np.array(0.1, np.float32); wzp = np.array(0, np.int8)
        dq_w = helper.make_node('DequantizeLinear', ['W', 'wscale', 'wzp'], ['Wf'])
    b = np.array([0.0], np.float32)
    nodes = [
        helper.make_node('QuantizeLinear', ['X', 'xscale', 'xzp'], ['Xq']),
        helper.make_node('DequantizeLinear', ['Xq', 'xscale', 'xzp'], ['Xdq']),
        dq_w,
        helper.make_node('MatMul', ['Xdq', 'Wf'], ['mm']),
        helper.make_node('Add', ['mm', 'b'], ['pre']),
        helper.make_node('Sigmoid', ['pre'], ['Y']),
    ]
    inits = [numpy_helper.from_array(a, n) for a, n in [
        (W, 'W'), (wscale, 'wscale'), (wzp, 'wzp'),
        (xscale, 'xscale'), (xzp, 'xzp'), (b, 'b')]]
    g = helper.make_graph(nodes, 'q',
                          [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])],
                          [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13),
                                            helper.make_opsetid('com.microsoft', 1)])
    m.ir_version = 8
    onnx.save(m, path)
    return path


def _plain_onnx(path):
    W = np.array([[0.3], [0.4]], np.float32)
    nodes = [helper.make_node('MatMul', ['X', 'W'], ['Y'])]
    g = helper.make_graph(nodes, 'p',
                          [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])],
                          [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1])],
                          [numpy_helper.from_array(W, 'W')])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)]); m.ir_version = 8
    onnx.save(m, path)
    return path


def _v1_spec(path, thr=0.5, op='>'):
    open(path, 'w').write(
        '(declare-const X_0 Real)\n(declare-const X_1 Real)\n(declare-const Y_0 Real)\n'
        '(assert (<= X_0 1.0))\n(assert (>= X_0 0.0))\n'
        '(assert (<= X_1 1.0))\n(assert (>= X_1 0.0))\n'
        f'(assert ({op} Y_0 {thr}))\n')
    return path


def _v2_spec(path, thr=0.5):
    open(path, 'w').write(
        '(vnnlib-version <2.0>)\n'
        '(declare-network f (declare-input X real [1,2]) (declare-output Y real [1,1]))\n'
        '(assert (and (>= X[0,0] 0.0) (<= X[0,0] 1.0)))\n'
        '(assert (and (>= X[0,1] 0.0) (<= X[0,1] 1.0)))\n'
        f'(assert (> Y[0,0] {thr}))\n')
    return path


# --------------------------------------------------------------------------- tests

def test_has_quantized_ops(tmp_path):
    assert sp.has_quantized_ops(_quant_onnx(str(tmp_path / 'q.onnx')))
    assert not sp.has_quantized_ops(_plain_onnx(str(tmp_path / 'p.onnx')))


def test_load_onnx_model_gz(tmp_path):
    p = _plain_onnx(str(tmp_path / 'p.onnx'))
    gz = str(tmp_path / 'p.onnx.gz')
    with open(p, 'rb') as fi, gzip.open(gz, 'wb') as fo:
        fo.write(fi.read())
    assert len(sp._load_onnx_model(gz).graph.node) == 1
    assert sp._decompressed(p) == p
    assert isinstance(sp._decompressed(gz), bytes)


@pytest.mark.parametrize('per_axis', [False, True])
def test_build_float_surrogate_matches_within_quant(tmp_path, per_axis):
    import onnxruntime as ort
    q = _quant_onnx(str(tmp_path / 'q.onnx'), per_axis=per_axis)
    s = sp.build_float_surrogate(q, str(tmp_path / 's.onnx'))
    m = onnx.load(s)
    assert not any(n.op_type in ('DequantizeLinear', 'QuantizeLinear') for n in m.graph.node)
    assert all((o.domain or 'ai.onnx') in ('ai.onnx', '') for o in m.opset_import)
    # surrogate ~= quantized model (differ only by the dropped activation rounding)
    qs = ort.InferenceSession(q, providers=['CPUExecutionProvider'])
    ss = ort.InferenceSession(s, providers=['CPUExecutionProvider'])
    x = np.array([[0.7, 0.3]], np.float32)
    yq = qs.run(None, {'X': x})[0].ravel()[0]
    ys = ss.run(None, {'X': x})[0].ravel()[0]
    assert abs(yq - ys) < 0.05


def test_parse_v1(tmp_path):
    spec = sp.parse_box_and_output(_v1_spec(str(tmp_path / 'v1.vnnlib')))
    assert len(spec.inputs) == 1 and spec.inputs[0][2].tolist() == [0.0, 0.0]
    assert spec.inputs[0][3].tolist() == [1.0, 1.0]
    assert spec.out_dnf == [[(0, 'gt', 0.5)]]


def test_parse_v1_lt(tmp_path):
    spec = sp.parse_box_and_output(_v1_spec(str(tmp_path / 'v1.vnnlib'), op='<'))
    assert spec.out_dnf == [[(0, 'lt', 0.5)]]


def test_parse_v2(tmp_path):
    spec = sp.parse_box_and_output(_v2_spec(str(tmp_path / 'v2.vnnlib')))
    assert len(spec.inputs) == 1
    name, shape, lo, hi = spec.inputs[0]
    assert name == 'X' and shape == (1, 2)
    assert lo.tolist() == [0.0, 0.0] and hi.tolist() == [1.0, 1.0]
    assert spec.out_dnf == [[(0, 'gt', 0.5)]]


def test_parse_gz_spec(tmp_path):
    p = _v1_spec(str(tmp_path / 'v1.vnnlib'))
    gz = str(tmp_path / 'v1.vnnlib.gz')
    with gzip.open(gz, 'wt') as f:
        f.write(open(p).read())
    assert sp.parse_box_and_output(gz).out_dnf == [[(0, 'gt', 0.5)]]


def test_parse_unsupported_raises(tmp_path):
    open(tmp_path / 'bad1.vnnlib', 'w').write('(declare-const X_0 Real)\n')   # no output
    with pytest.raises(NotImplementedError):
        sp.parse_box_and_output(str(tmp_path / 'bad1.vnnlib'))
    open(tmp_path / 'bad2.vnnlib', 'w').write(
        '(vnnlib-version <2.0>)\n(declare-network f (declare-input X real [1,2]) '
        '(declare-output Y real [1,1]))\n(assert (and (>= X[0,0] 0.0) (<= X[0,0] 1.0)))\n')
    with pytest.raises(NotImplementedError):
        sp.parse_box_and_output(str(tmp_path / 'bad2.vnnlib'))


def test_strides_and_flat():
    st = sp._c_strides((1, 3, 4))
    assert st == [12, 4, 1]
    assert sp._flat('0,2,3', st) == 11


def test_violated():
    spec = sp.SurrogateSpec([('X', (2,), np.zeros(2), np.ones(2))], [[(0, 'gt', 0.5)]])
    assert spec.violated(np.array([0.6]))
    assert not spec.violated(np.array([0.4]))
    spec2 = sp.SurrogateSpec([('X', (2,), np.zeros(2), np.ones(2))], [[(0, 'lt', 0.5)]])
    assert spec2.violated(np.array([0.3]))


class _S:
    surrogate_attack_restarts = 2
    surrogate_attack_steps = 25
    sat_validate_atol = 1e-4


def test_surrogate_attack_sat(tmp_path):
    q = _quant_onnx(str(tmp_path / 'q.onnx'))
    v = _v1_spec(str(tmp_path / 'v1.vnnlib'), thr=0.5)        # reachable -> sat
    verdict, wit = sp.surrogate_attack(q, v, _S(), timeout=30,
                                       surrogate_path=str(tmp_path / 's.onnx'), log=lambda _m: None)
    assert verdict == 'sat' and wit is not None
    y = sp._ort_eval(q, wit)
    assert y[0] > 0.5                                        # validated on original


def test_surrogate_attack_unknown(tmp_path):
    q = _quant_onnx(str(tmp_path / 'q.onnx'))
    v = _v1_spec(str(tmp_path / 'v1.vnnlib'), thr=0.99)      # unreachable -> no CE
    verdict, wit = sp.surrogate_attack(q, v, _S(), timeout=30,
                                       surrogate_path=str(tmp_path / 's.onnx'), log=lambda _m: None)
    assert verdict == 'unknown' and wit is None


def test_surrogate_attack_timeout(tmp_path):
    q = _quant_onnx(str(tmp_path / 'q.onnx'))
    v = _v1_spec(str(tmp_path / 'v1.vnnlib'), thr=0.99)
    verdict, _ = sp.surrogate_attack(q, v, _S(), timeout=-1.0,   # already over budget
                                     surrogate_path=str(tmp_path / 's.onnx'), log=lambda _m: None)
    assert verdict == 'timeout'


def test_parse_v2_lt(tmp_path):
    p = str(tmp_path / 'v2lt.vnnlib')
    open(p, 'w').write(
        '(vnnlib-version <2.0>)\n'
        '(declare-network f (declare-input X real [1,2]) (declare-output Y real [1,1]))\n'
        '(assert (and (>= X[0,0] 0.0) (<= X[0,0] 1.0)))\n'
        '(assert (and (>= X[0,1] 0.0) (<= X[0,1] 1.0)))\n'
        '(assert (< Y[0,0] 0.5))\n')
    assert sp.parse_box_and_output(p).out_dnf == [[(0, 'lt', 0.5)]]


def test_surrogate_attack_input_count_mismatch(tmp_path):
    q = _quant_onnx(str(tmp_path / 'q.onnx'))                # model has 1 input
    p = str(tmp_path / 'two.vnnlib')
    open(p, 'w').write(                                      # spec declares 2 inputs
        '(vnnlib-version <2.0>)\n'
        '(declare-network f (declare-input X1 real [1,2]) (declare-input X2 real [1,2]) '
        '(declare-output Y real [1,1]))\n'
        '(assert (and (>= X1[0,0] 0.0) (<= X1[0,0] 1.0)))\n'
        '(assert (and (>= X2[0,0] 0.0) (<= X2[0,0] 1.0)))\n'
        '(assert (> Y[0,0] 0.5))\n')
    with pytest.raises(NotImplementedError):
        sp.surrogate_attack(q, p, _S(), timeout=30,
                            surrogate_path=str(tmp_path / 's.onnx'), log=lambda _m: None)


def test_surrogate_attack_loop_sat(tmp_path):
    q = _quant_onnx(str(tmp_path / 'q.onnx'))
    v = _v1_spec(str(tmp_path / 'v1.vnnlib'), thr=0.6)       # center<0.6, corner>0.6 -> loop SAT
    verdict, wit = sp.surrogate_attack(q, v, _S(), timeout=30,
                                       surrogate_path=str(tmp_path / 's.onnx'), log=lambda _m: None)
    assert verdict == 'sat' and sp._ort_eval(q, wit)[0] > 0.6


def test_surrogate_attack_builds_surrogate_if_missing(tmp_path):
    q = _quant_onnx(str(tmp_path / 'q.onnx'))
    v = _v1_spec(str(tmp_path / 'v1.vnnlib'))
    spath = str(tmp_path / 'auto_surrogate.onnx')
    assert not os.path.exists(spath)
    sp.surrogate_attack(q, v, _S(), timeout=30, surrogate_path=spath, log=lambda _m: None)
    assert os.path.exists(spath)
