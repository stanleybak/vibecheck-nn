"""Unit tests for the Sign-BNN attack mode (src/vibecheck/sign_attack.py).

Builds a tiny synthetic binarized ONNX (Gemm -> Sign -> Add -> Sign_1 -> Gemm) so the
convert -> STE-patch -> PGD -> ORT-validate path runs end-to-end in ~seconds on CPU.
"""
import os

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from vibecheck import sign_attack as sa


def _bnn_onnx(path, w2=((1.0, -1.0), (-1.0, 1.0)), softmax=False):
    """X[1,2] -Gemm(W1)-> Sign -Add(+0.1)-> Sign_1 -Gemm(W2)-> Y[1,2] (optionally Softmax).
    The Sign nodes are named 'Sign'/'Sign_1' so onnx2torch's modules get those leaf names
    (STE / pass-through)."""
    W1 = np.array([[1.0, 0.5], [0.5, 1.0]], np.float32)
    W2 = np.array(w2, np.float32)
    c = np.array([0.1, 0.1], np.float32)
    last = 'logits' if softmax else 'Y'
    nodes = [
        helper.make_node('MatMul', ['X', 'W1'], ['h']),
        helper.make_node('Sign', ['h'], ['s1'], name='Sign'),
        helper.make_node('Add', ['s1', 'c'], ['a'], name='add'),
        helper.make_node('Sign', ['a'], ['s2'], name='Sign_1'),
        helper.make_node('MatMul', ['s2', 'W2'], [last]),
    ]
    if softmax:
        nodes.append(helper.make_node('Softmax', ['logits'], ['Y'], axis=1, name='Softmax'))
    inits = [numpy_helper.from_array(a, n) for a, n in [(W1, 'W1'), (W2, 'W2'), (c, 'c')]]
    g = helper.make_graph(nodes, 'bnn',
                          [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])],
                          [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)]); m.ir_version = 8
    onnx.save(m, path)
    return path


def _plain_onnx(path):
    W = np.array([[0.3, 0.1], [0.1, 0.3]], np.float32)
    g = helper.make_graph([helper.make_node('MatMul', ['X', 'W'], ['Y'])], 'p',
                          [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])],
                          [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])],
                          [numpy_helper.from_array(W, 'W')])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)]); m.ir_version = 8
    onnx.save(m, path); return path


def _vnnlib(path, lo=(-1.0, -1.0), hi=(1.0, 1.0), pairwise=True):
    """Pairwise robustness (unsafe if Y_1 >= Y_0) or a threshold (unsafe if Y_0 <= 0)."""
    body = (f'(declare-const X_0 Real)\n(declare-const X_1 Real)\n'
            f'(declare-const Y_0 Real)\n(declare-const Y_1 Real)\n'
            f'(assert (>= X_0 {lo[0]}))\n(assert (<= X_0 {hi[0]}))\n'
            f'(assert (>= X_1 {lo[1]}))\n(assert (<= X_1 {hi[1]}))\n')
    body += '(assert (<= Y_0 Y_1))\n' if pairwise else '(assert (<= Y_0 0.0))\n'
    open(path, 'w').write(body)
    return path


class _S:
    sign_attack_restarts = 8
    sign_attack_steps = 60
    sign_preact_penalty = 1.0
    sign_per_disjunct = False
    sat_validate_atol = 1e-4
    keep_searching_within_tol = True
    pgd_seed = 0
    device = 'cpu'


def test_has_sign_ops(tmp_path):
    assert sa.has_sign_ops(_bnn_onnx(str(tmp_path / 'b.onnx')))
    assert not sa.has_sign_ops(_plain_onnx(str(tmp_path / 'p.onnx')))


def test_disjunct_loss_and_margin():
    import torch
    from vibecheck.spec import Conjunct, Constraint, PairwiseConstraint
    y = np.array([0.3, 1.2])
    dj_pair = [Conjunct([PairwiseConstraint(pred=0, comp=1)])]          # safe = y0-y1 = -0.9
    assert abs(sa._worst_margin_np(y, dj_pair) - (-0.9)) < 1e-6
    assert abs(float(sa._disjunct_loss(torch.tensor(y), dj_pair, torch)) - 0.9) < 1e-6
    dj_ge = [Conjunct([Constraint(index=0, op='>=', value=1.0)])]       # safe = 1.0 - 0.3 = 0.7
    assert abs(sa._worst_margin_np(y, dj_ge) - 0.7) < 1e-6
    dj_le = [Conjunct([Constraint(index=1, op='<=', value=0.5)])]       # safe = 1.2 - 0.5 = 0.7
    assert abs(sa._worst_margin_np(y, dj_le) - 0.7) < 1e-6
    # two disjuncts: worst = min
    assert abs(sa._worst_margin_np(y, dj_pair + dj_ge) - (-0.9)) < 1e-6
    # the differentiable torch loss = -worst_margin, for each constraint kind
    yt = torch.tensor(y)
    assert abs(float(sa._disjunct_loss(yt, dj_ge, torch)) - (-0.7)) < 1e-6
    assert abs(float(sa._disjunct_loss(yt, dj_le, torch)) - (-0.7)) < 1e-6


def test_sign_attack_within_tol(tmp_path):
    # W2 with equal columns -> Y_0 == Y_1 for every input -> worst_margin == 0 (never < 0):
    # a within-tolerance CE the scorer accepts; emitted after the search finds no clear CE.
    q = _bnn_onnx(str(tmp_path / 'b.onnx'), w2=((1.0, 1.0), (0.5, 0.5)))
    v = _vnnlib(str(tmp_path / 'v.vnnlib'))
    verdict, wit = sa.sign_attack(q, v, _S(), timeout=30, log=lambda _m: None)
    assert verdict == 'sat' and wit is not None
    y = sa._ort_eval(q, wit)
    assert abs(y[0] - y[1]) <= 1e-4 and not (y[1] > y[0])   # within-tol tie, not a clear flip


def test_sign_attack_within_tol_immediate(tmp_path):
    class _Simm(_S):
        keep_searching_within_tol = False
    q = _bnn_onnx(str(tmp_path / 'b.onnx'), w2=((1.0, 1.0), (0.5, 0.5)))
    v = _vnnlib(str(tmp_path / 'v.vnnlib'))
    verdict, wit = sa.sign_attack(q, v, _Simm(), timeout=30, log=lambda _m: None)
    assert verdict == 'sat' and wit is not None


def test_sign_attack_softmax_model(tmp_path):
    # a trailing Softmax -> the pre-softmax hook registers + is used for the loss
    q = _bnn_onnx(str(tmp_path / 'bsm.onnx'), w2=((1.0, -1.0), (-1.0, 1.0)), softmax=True)
    v = _vnnlib(str(tmp_path / 'v.vnnlib'))
    verdict, wit = sa.sign_attack(q, v, _S(), timeout=30, log=lambda _m: None)
    assert verdict in ('sat', 'unknown')


def test_sign_attack_midstep_timeout(tmp_path, monkeypatch):
    # timeout elapsing DURING the step loop (after the per-restart check passes) -> the
    # per-step break fires. Counted fake clock: t0, the load-log, the restart check, then the
    # step check (4th call) trips.
    q = _bnn_onnx(str(tmp_path / 'b.onnx'), w2=((5.0, -5.0), (5.0, -5.0)))
    v = _vnnlib(str(tmp_path / 'v.vnnlib'), lo=(0.4, 0.4), hi=(0.40001, 0.40001))

    class _Clk:
        def __init__(self):
            self.n = -1

        def __call__(self):
            self.n += 1
            return 0.0 if self.n < 3 else 100.0      # 4th call (n==3, the step check) trips

    monkeypatch.setattr(sa.time, 'time', _Clk())
    verdict, _ = sa.sign_attack(q, v, _S(), timeout=1.0, log=lambda _m: None)
    assert verdict == 'timeout'


def test_sign_attack_sat(tmp_path):
    # W2 maps the binarized hidden state so SOME input in the box flips argmax to class 1.
    q = _bnn_onnx(str(tmp_path / 'b.onnx'), w2=((1.0, -1.0), (-1.0, 1.0)))
    v = _vnnlib(str(tmp_path / 'v.vnnlib'))
    verdict, wit = sa.sign_attack(q, v, _S(), timeout=30, log=lambda _m: None)
    assert verdict == 'sat' and wit is not None
    y = sa._ort_eval(q, wit)
    assert y[1] >= y[0] - 1e-4          # validated on the original: class 1 beats class 0


def test_sign_attack_per_disjunct(tmp_path):
    class _SP(_S):
        sign_per_disjunct = True
    q = _bnn_onnx(str(tmp_path / 'b.onnx'))
    v = _vnnlib(str(tmp_path / 'v.vnnlib'))
    verdict, _ = sa.sign_attack(q, v, _SP(), timeout=30, log=lambda _m: None)
    assert verdict in ('sat', 'unknown')


def test_sign_attack_no_ce_timeout(tmp_path):
    # tiny box around a point that stays class 0 -> no CE reachable. Incomplete -> timeout.
    q = _bnn_onnx(str(tmp_path / 'b.onnx'), w2=((5.0, -5.0), (5.0, -5.0)))   # class 0 dominates
    v = _vnnlib(str(tmp_path / 'v.vnnlib'), lo=(0.4, 0.4), hi=(0.40001, 0.40001))
    verdict, wit = sa.sign_attack(q, v, _S(), timeout=30, log=lambda _m: None)
    assert verdict == 'timeout' and wit is None


def test_sign_attack_timeout(tmp_path):
    q = _bnn_onnx(str(tmp_path / 'b.onnx'), w2=((5.0, -5.0), (5.0, -5.0)))
    v = _vnnlib(str(tmp_path / 'v.vnnlib'), lo=(0.4, 0.4), hi=(0.40001, 0.40001))
    verdict, _ = sa.sign_attack(q, v, _S(), timeout=-1.0, log=lambda _m: None)
    assert verdict == 'timeout'


def test_sign_attack_multi_input_raises(tmp_path):
    # a 2-input model is unsupported by sign_attack
    g = helper.make_graph(
        [helper.make_node('Add', ['X1', 'X2'], ['s'], name='add'),
         helper.make_node('Sign', ['s'], ['Y'], name='Sign')], 'm',
        [helper.make_tensor_value_info('X1', TensorProto.FLOAT, [1, 2]),
         helper.make_tensor_value_info('X2', TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])], [])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)]); m.ir_version = 8
    p = str(tmp_path / 'two.onnx'); onnx.save(m, p)
    v = _vnnlib(str(tmp_path / 'v.vnnlib'))
    with pytest.raises(NotImplementedError):
        sa.sign_attack(p, v, _S(), timeout=10, log=lambda _m: None)
