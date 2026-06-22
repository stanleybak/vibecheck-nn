"""Unit tests for the generic onnx2torch PGD attack (src/vibecheck/torch_attack.py).

Tiny synthetic ONNX (MatMul -> Relu -> MatMul, no Sign) so the convert -> PGD ->
ORT-validate path runs end-to-end in ~seconds on CPU.
"""
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from vibecheck import torch_attack as ta


def _net(path, w2=((1.0, -1.0), (-1.0, 1.0)), two_out=False):
    """X[1,2] -MatMul(W1)-> Relu -MatMul(W2)-> Y[1,2] (optionally a 2nd dummy output)."""
    W1 = np.array([[1.0, 0.5], [0.5, 1.0]], np.float32)
    W2 = np.array(w2, np.float32)
    nodes = [
        helper.make_node('MatMul', ['X', 'W1'], ['h']),
        helper.make_node('Relu', ['h'], ['r']),
        helper.make_node('MatMul', ['r', 'W2'], ['Y']),
    ]
    outs = [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])]
    if two_out:
        nodes.append(helper.make_node('Relu', ['Y'], ['Y2']))
        outs.append(helper.make_tensor_value_info('Y2', TensorProto.FLOAT, [1, 2]))
    inits = [numpy_helper.from_array(W1, 'W1'), numpy_helper.from_array(W2, 'W2')]
    g = helper.make_graph(nodes, 'n',
                          [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])],
                          outs, inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)]); m.ir_version = 8
    onnx.save(m, path); return path


def _vnnlib(path, lo=(-1.0, -1.0), hi=(1.0, 1.0)):
    body = (f'(declare-const X_0 Real)\n(declare-const X_1 Real)\n'
            f'(declare-const Y_0 Real)\n(declare-const Y_1 Real)\n'
            f'(assert (>= X_0 {lo[0]}))\n(assert (<= X_0 {hi[0]}))\n'
            f'(assert (>= X_1 {lo[1]}))\n(assert (<= X_1 {hi[1]}))\n'
            f'(assert (<= Y_0 Y_1))\n')
    open(path, 'w').write(body); return path


class _S:
    torch_attack_restarts = 8
    torch_attack_steps = 60
    sat_validate_atol = 1e-4
    keep_searching_within_tol = True
    pgd_seed = 0
    device = 'cpu'


def test_torch_attack_sat(tmp_path):
    q = _net(str(tmp_path / 'n.onnx'))
    v = _vnnlib(str(tmp_path / 'v.vnnlib'))
    verdict, wit = ta.torch_attack(q, v, _S(), timeout=30, log=lambda _m: None)
    assert verdict == 'sat' and wit is not None
    y = ta._ort_eval(q, wit)
    assert y[1] >= y[0] - 1e-4


def test_torch_attack_two_output(tmp_path):
    # a 2-output model -> onnx2torch returns a tuple; flat_out takes out[0]
    q = _net(str(tmp_path / 'n2.onnx'), two_out=True)
    v = _vnnlib(str(tmp_path / 'v.vnnlib'))
    verdict, _ = ta.torch_attack(q, v, _S(), timeout=30, log=lambda _m: None)
    assert verdict == 'sat'


def test_torch_attack_within_tol(tmp_path):
    # equal columns -> Y_0 == Y_1 for every input -> worst_margin == 0 (within-tol CE)
    q = _net(str(tmp_path / 'n.onnx'), w2=((1.0, 1.0), (0.5, 0.5)))
    v = _vnnlib(str(tmp_path / 'v.vnnlib'))
    verdict, wit = ta.torch_attack(q, v, _S(), timeout=30, log=lambda _m: None)
    assert verdict == 'sat' and wit is not None


def test_torch_attack_within_tol_immediate(tmp_path):
    class _Simm(_S):
        keep_searching_within_tol = False
    q = _net(str(tmp_path / 'n.onnx'), w2=((1.0, 1.0), (0.5, 0.5)))
    v = _vnnlib(str(tmp_path / 'v.vnnlib'))
    verdict, wit = ta.torch_attack(q, v, _Simm(), timeout=30, log=lambda _m: None)
    assert verdict == 'sat' and wit is not None


def test_torch_attack_fixed_box_within_tol(tmp_path):
    # fully-fixed box (lo==hi) -> half.max()==0 -> alpha fallback; center is a within-tol CE
    q = _net(str(tmp_path / 'n.onnx'), w2=((1.0, 1.0), (0.5, 0.5)))
    v = _vnnlib(str(tmp_path / 'v.vnnlib'), lo=(0.3, 0.3), hi=(0.3, 0.3))
    verdict, wit = ta.torch_attack(q, v, _S(), timeout=30, log=lambda _m: None)
    assert verdict == 'sat' and wit is not None


def test_torch_attack_unknown(tmp_path):
    # tiny box where class 0 dominates -> no CE reachable
    q = _net(str(tmp_path / 'n.onnx'), w2=((5.0, -5.0), (5.0, -5.0)))
    v = _vnnlib(str(tmp_path / 'v.vnnlib'), lo=(0.4, 0.4), hi=(0.40001, 0.40001))
    verdict, wit = ta.torch_attack(q, v, _S(), timeout=30, log=lambda _m: None)
    assert verdict == 'unknown' and wit is None


def test_torch_attack_timeout(tmp_path):
    q = _net(str(tmp_path / 'n.onnx'), w2=((5.0, -5.0), (5.0, -5.0)))
    v = _vnnlib(str(tmp_path / 'v.vnnlib'), lo=(0.4, 0.4), hi=(0.40001, 0.40001))
    verdict, _ = ta.torch_attack(q, v, _S(), timeout=-1.0, log=lambda _m: None)
    assert verdict == 'timeout'


def test_torch_attack_midstep_timeout(tmp_path, monkeypatch):
    # timeout elapsing DURING the step loop (after the per-restart check) -> per-step break
    q = _net(str(tmp_path / 'n.onnx'), w2=((5.0, -5.0), (5.0, -5.0)))
    v = _vnnlib(str(tmp_path / 'v.vnnlib'), lo=(0.4, 0.4), hi=(0.40001, 0.40001))

    class _Clk:
        def __init__(self):
            self.n = -1

        def __call__(self):
            self.n += 1
            return 0.0 if self.n < 3 else 100.0      # 4th call (the step check) trips

    monkeypatch.setattr(ta.time, 'time', _Clk())
    verdict, _ = ta.torch_attack(q, v, _S(), timeout=1.0, log=lambda _m: None)
    assert verdict == 'timeout'


def test_torch_attack_multi_input_raises(tmp_path):
    g = helper.make_graph(
        [helper.make_node('Add', ['X1', 'X2'], ['Y'], name='add')], 'm',
        [helper.make_tensor_value_info('X1', TensorProto.FLOAT, [1, 2]),
         helper.make_tensor_value_info('X2', TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])], [])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)]); m.ir_version = 8
    p = str(tmp_path / 'two.onnx'); onnx.save(m, p)
    v = _vnnlib(str(tmp_path / 'v.vnnlib'))
    with pytest.raises(NotImplementedError):
        ta.torch_attack(p, v, _S(), timeout=10, log=lambda _m: None)
