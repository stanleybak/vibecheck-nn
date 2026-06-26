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

def _quant_onnx(path, w=(3, 4), per_axis=False, act_uint8=False):
    """X[1,2] --Q/DQ--> @ DQ(int8 W[2,1]) + b --Sigmoid--> Y[1,1].

    w=(0,0) gives a constant model (Y==sigmoid(0)==0.5 for all X). act_uint8 quantizes the
    activation as uint8 (zp 128, qmin/qmax 0/255) instead of int8 (zp 0, -128/127)."""
    W = np.array([[w[0]], [w[1]]], dtype=np.int8)
    if act_uint8:
        xscale = np.array(0.02, np.float32); xzp = np.array(128, np.uint8)
    else:
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


def test_parse_gz_readonly_dir(tmp_path):
    # When only the .gz exists in a NON-writable dir, ensure_decompressed can't materialize
    # a sibling and returns the .gz path -> parse_box_and_output reads it in-memory (gzip).
    import stat
    d = tmp_path / 'ro'
    d.mkdir()
    plain = _v1_spec(str(d / 'v1.vnnlib'))
    gz = str(d / 'v1.vnnlib.gz')
    with gzip.open(gz, 'wt') as f:
        f.write(open(plain).read())
    os.remove(plain)                                   # only the .gz remains
    os.chmod(d, stat.S_IRUSR | stat.S_IXUSR)           # read-only dir
    try:
        assert sp.parse_box_and_output(gz).out_dnf == [[(0, 'gt', 0.5)]]
    finally:
        os.chmod(d, stat.S_IRWXU)                       # restore for tmp cleanup


def test_surrogate_attack_midstep_timeout(tmp_path, monkeypatch):
    # Timeout that elapses AFTER the per-restart check passes but DURING the step loop ->
    # the per-step break fires. Deterministic via a counted fake clock: the surrogate_attack
    # calls time.time() as t0, the load-log, the center validate (x2), the restart check,
    # then the step check (6th call) — trip on that 6th call.
    q = _quant_onnx(str(tmp_path / 'q.onnx'))
    v = _v1_spec(str(tmp_path / 'v.vnnlib'), thr=0.99)            # no CE -> reaches the steps
    spath = str(tmp_path / 's.onnx')
    sp.build_float_surrogate(q, spath)                            # build untimed

    class _Clk:
        def __init__(self):
            self.n = -1

        def __call__(self):
            self.n += 1
            return 0.0 if self.n < 5 else 100.0                  # 6th call (n==5) trips

    monkeypatch.setattr(sp.time, 'time', _Clk())
    verdict, _ = sp.surrogate_attack(q, v, _S(), timeout=1.0,
                                     surrogate_path=spath, log=lambda _m: None)
    assert verdict == 'timeout'


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
    device = 'cpu'   # unit tests stay on CPU (deterministic, no GPU contention)


def test_surrogate_attack_sat(tmp_path):
    q = _quant_onnx(str(tmp_path / 'q.onnx'))
    v = _v1_spec(str(tmp_path / 'v1.vnnlib'), thr=0.5)        # reachable -> sat
    verdict, wit = sp.surrogate_attack(q, v, _S(), timeout=30,
                                       surrogate_path=str(tmp_path / 's.onnx'), log=lambda _m: None)
    assert verdict == 'sat' and wit is not None
    y = sp._ort_eval(q, wit)
    assert y[0] > 0.5                                        # validated on original


def test_surrogate_attack_no_ce_timeout(tmp_path):
    # No reachable CE. Incomplete (attack-only) mode cannot prove unsat, so "didn't
    # find one in the budget" is reported as timeout, not unknown (2026 semantics).
    q = _quant_onnx(str(tmp_path / 'q.onnx'))
    v = _v1_spec(str(tmp_path / 'v1.vnnlib'), thr=0.99)      # unreachable -> no CE
    verdict, wit = sp.surrogate_attack(q, v, _S(), timeout=30,
                                       surrogate_path=str(tmp_path / 's.onnx'), log=lambda _m: None)
    assert verdict == 'timeout' and wit is None


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


# --------------------------------------------------- fake-quant eval oracle (Path B)

@pytest.mark.parametrize('per_axis', [False, True])
def test_build_fakequant_matches_original(tmp_path, per_axis):
    """The fake-quant surrogate reproduces the ORIGINAL quantized output EXACTLY (it IS the
    INT8 rounding), unlike the float/STE surrogate which drops it."""
    import onnxruntime as ort
    q = _quant_onnx(str(tmp_path / 'q.onnx'), per_axis=per_axis)
    fq = sp.build_fakequant_surrogate(q, str(tmp_path / 'fq.onnx'))
    m = onnx.load(fq)
    assert not any(n.op_type in ('DequantizeLinear', 'QuantizeLinear') for n in m.graph.node)
    types = {n.op_type for n in m.graph.node}
    assert {'Round', 'Clip', 'Sub', 'Mul'} <= types          # activation fake-quant emitted
    assert all((o.domain or 'ai.onnx') in ('ai.onnx', '') for o in m.opset_import)
    qs = ort.InferenceSession(q, providers=['CPUExecutionProvider'])
    fs = ort.InferenceSession(fq, providers=['CPUExecutionProvider'])
    for x in ([[0.7, 0.3]], [[0.1, 0.9]], [[0.5, 0.5]]):
        xa = np.array(x, np.float32)
        assert abs(qs.run(None, {'X': xa})[0].ravel()[0]
                   - fs.run(None, {'X': xa})[0].ravel()[0]) < 1e-5


def test_build_fakequant_uint8_activation(tmp_path):
    import onnxruntime as ort
    q = _quant_onnx(str(tmp_path / 'qu.onnx'), act_uint8=True)
    fq = sp.build_fakequant_surrogate(q, str(tmp_path / 'fqu.onnx'))
    qs = ort.InferenceSession(q, providers=['CPUExecutionProvider'])
    fs = ort.InferenceSession(fq, providers=['CPUExecutionProvider'])
    for x in ([[0.7, 0.3]], [[0.2, 0.8]]):
        xa = np.array(x, np.float32)
        assert abs(qs.run(None, {'X': xa})[0].ravel()[0]
                   - fs.run(None, {'X': xa})[0].ravel()[0]) < 1e-5


def _save_graph(path, nodes, inits, in_shape=(1, 2), out_shape=(1, 2)):
    g = helper.make_graph(nodes, 'g',
                          [helper.make_tensor_value_info('X', TensorProto.FLOAT, list(in_shape))],
                          [helper.make_tensor_value_info('Y', TensorProto.FLOAT, list(out_shape))], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)]); m.ir_version = 8
    onnx.save(m, path)
    return path


def test_build_fakequant_raises_quant_init_input(tmp_path):
    # QuantizeLinear on an initializer (weight quant) is unsupported by the fake-quant fold.
    p = _save_graph(str(tmp_path / 'qi.onnx'), [
        helper.make_node('QuantizeLinear', ['C', 'sc', 'zp'], ['cq']),
        helper.make_node('DequantizeLinear', ['cq', 'sc', 'zp'], ['cf']),
        helper.make_node('Add', ['X', 'cf'], ['Y']),
    ], [numpy_helper.from_array(np.array([[0.5, 0.5]], np.float32), 'C'),
        numpy_helper.from_array(np.array(0.02, np.float32), 'sc'),
        numpy_helper.from_array(np.array(0, np.int8), 'zp')])
    with pytest.raises(NotImplementedError):
        sp.build_fakequant_surrogate(p, str(tmp_path / 'o.onnx'))


def test_build_fakequant_raises_per_axis_activation_q(tmp_path):
    # per-axis (non-scalar scale) activation QuantizeLinear needs axis-aware broadcast.
    p = _save_graph(str(tmp_path / 'paq.onnx'), [
        helper.make_node('QuantizeLinear', ['X', 'svec', 'zvec'], ['q'], axis=1),
        helper.make_node('DequantizeLinear', ['q', 'svec', 'zvec'], ['Y'], axis=1),
    ], [numpy_helper.from_array(np.array([0.02, 0.03], np.float32), 'svec'),
        numpy_helper.from_array(np.array([0, 0], np.int8), 'zvec')])
    with pytest.raises(NotImplementedError):
        sp.build_fakequant_surrogate(p, str(tmp_path / 'o.onnx'))


def test_build_fakequant_raises_per_axis_activation_dq(tmp_path):
    # scalar activation Q (builds), then per-axis activation DQ (non-scalar scale) -> raise.
    p = _save_graph(str(tmp_path / 'padq.onnx'), [
        helper.make_node('QuantizeLinear', ['X', 'sc', 'zp'], ['q']),
        helper.make_node('DequantizeLinear', ['q', 'svec', 'zvec'], ['Y'], axis=1),
    ], [numpy_helper.from_array(np.array(0.02, np.float32), 'sc'),
        numpy_helper.from_array(np.array(0, np.int8), 'zp'),
        numpy_helper.from_array(np.array([0.02, 0.03], np.float32), 'svec'),
        numpy_helper.from_array(np.array([0, 0], np.int8), 'zvec')])
    with pytest.raises(NotImplementedError):
        sp.build_fakequant_surrogate(p, str(tmp_path / 'o.onnx'))


# ----------------------------------------- output-strict boundary disposition (2026 rule)

def test_surrogate_attack_boundary_not_sat(tmp_path):
    # Constant model: Y==0.5 for all X; spec `Y > 0.5` is a STRICT constraint never
    # strictly crossed. Under the VNN-COMP 2026 output-strict rule a boundary point
    # (Y == threshold, margin 0) is NOT a counterexample (matches the competition
    # checker, which rejects `0.5 > 0.5` at zero tolerance) — so no clear CE exists
    # and the surrogate returns unknown, NOT a within-tolerance sat. (This is the
    # smart_turn situation: Y pinned at the threshold.)
    q = _quant_onnx(str(tmp_path / 'q0.onnx'), w=(0, 0))
    v = _v1_spec(str(tmp_path / 'v.vnnlib'), thr=0.5)
    verdict, wit = sp.surrogate_attack(q, v, _S(), timeout=30,
                                       surrogate_path=str(tmp_path / 's.onnx'), log=lambda _m: None)
    assert verdict in ('unknown', 'timeout') and wit is None


def test_surrogate_attack_boundary_not_sat_keep_search_off(tmp_path):
    # keep_searching_within_tol is now a no-op (within-tol output is not scorer-
    # accepted); a boundary-only model still yields unknown regardless of the flag.
    class _Simm(_S):
        keep_searching_within_tol = False
    q = _quant_onnx(str(tmp_path / 'q0.onnx'), w=(0, 0))
    v = _v1_spec(str(tmp_path / 'v.vnnlib'), thr=0.5)
    verdict, wit = sp.surrogate_attack(q, v, _Simm(), timeout=30,
                                       surrogate_path=str(tmp_path / 's.onnx'), log=lambda _m: None)
    assert verdict in ('unknown', 'timeout') and wit is None


def test_surrogate_attack_quant_eval_off(tmp_path):
    # surrogate_quant_eval=False -> no fake-quant eval oracle (eval_model None, fq_margin
    # returns None); candidates fall through to the ORT-confirm loop ranked by surrogate
    # loss. thr=0.6: center<0.6 so it reaches that loop, corner clears.
    class _Soff(_S):
        surrogate_quant_eval = False
    q = _quant_onnx(str(tmp_path / 'q.onnx'))
    v = _v1_spec(str(tmp_path / 'v.vnnlib'), thr=0.6)
    verdict, wit = sp.surrogate_attack(q, v, _Soff(), timeout=30,
                                       surrogate_path=str(tmp_path / 's.onnx'), log=lambda _m: None)
    assert verdict == 'sat' and sp._ort_eval(q, wit)[0] > 0.6
