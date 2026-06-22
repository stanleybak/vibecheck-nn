"""Unit tests for the cctsdb_yolo complete-enumeration handler (src/vibecheck/cctsdb_yolo.py).

Synthetic discrete-patch net: X[1,5] = 3 fixed dims + 2 integer "position" dims; Y_0 =
0.1*(X_3 + X_4). The unsafe spec is `Y_0 <= thr`, so the position grid is enumerable and the
verdict is exact.
"""
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from vibecheck import cctsdb_yolo as cy


def _net(path):
    W = np.array([[0.0], [0.0], [0.0], [0.1], [0.1]], np.float32)   # Y_0 = 0.1*(X3+X4)
    g = helper.make_graph([helper.make_node('MatMul', ['X', 'W'], ['Y'])], 'c',
                          [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 5])],
                          [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1])],
                          [numpy_helper.from_array(W, 'W')])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)]); m.ir_version = 8
    onnx.save(m, path); return path


def _vnnlib(path, thr=0.5, p_lo=0, p_hi=5, n_free=2, noninteger=False):
    """Fix X_0..X_2 (and X_4 if n_free==1); vary X_3 (and X_4) over [p_lo,p_hi]; assert Y_0<=thr.
    n_free=5 -> all X free (not a discrete-patch instance)."""
    L = [f'(declare-const X_{i} Real)' for i in range(5)] + ['(declare-const Y_0 Real)']
    hi3 = (p_hi + 0.5) if noninteger else p_hi
    for i in range(5):
        free = (i in (3, 4) and n_free >= 2) or (i == 3 and n_free == 1) or (n_free == 5)
        if free:
            lo, hi = p_lo, (hi3 if i == 3 else p_hi)
        else:
            lo = hi = 0.0
        L.append(f'(assert (>= X_{i} {lo}))')
        L.append(f'(assert (<= X_{i} {hi}))')
    L.append(f'(assert (<= Y_0 {thr}))')
    open(path, 'w').write('\n'.join(L) + '\n')
    return path


class _S:
    sat_validate_atol = 1e-4
    cctsdb_max_positions = 1_000_000


def _two_input_net(path):
    g = helper.make_graph([helper.make_node('Add', ['X1', 'X2'], ['Y'])], 'm',
                          [helper.make_tensor_value_info('X1', TensorProto.FLOAT, [1, 2]),
                           helper.make_tensor_value_info('X2', TensorProto.FLOAT, [1, 2])],
                          [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])], [])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)]); m.ir_version = 8
    onnx.save(m, path); return path


def test_has_cctsdb_structure(tmp_path):
    q = _net(str(tmp_path / 'n.onnx'))
    assert cy.has_cctsdb_structure(q, _vnnlib(str(tmp_path / 'v.vnnlib')))
    assert not cy.has_cctsdb_structure(q, _vnnlib(str(tmp_path / 'v5.vnnlib'), n_free=5))   # >4 free
    assert not cy.has_cctsdb_structure(q, _vnnlib(str(tmp_path / 'vn.vnnlib'), noninteger=True))
    # a multi-input net is not a discrete-patch instance
    assert not cy.has_cctsdb_structure(_two_input_net(str(tmp_path / 't.onnx')),
                                       _vnnlib(str(tmp_path / 'v.vnnlib')))


def test_cctsdb_sat(tmp_path):
    q = _net(str(tmp_path / 'n.onnx'))
    v = _vnnlib(str(tmp_path / 'v.vnnlib'), thr=0.5)   # (0,0)->Y0=0.0 < 0.5 -> clear CE
    verdict, wit = cy.cctsdb_yolo_verify(q, v, _S(), timeout=30, log=lambda _m: None)
    assert verdict == 'sat' and wit is not None
    assert float(np.asarray(wit[0]).ravel()[3] + np.asarray(wit[0]).ravel()[4]) * 0.1 <= 0.5


def test_cctsdb_unsat(tmp_path):
    q = _net(str(tmp_path / 'n.onnx'))
    v = _vnnlib(str(tmp_path / 'v.vnnlib'), thr=-0.01)   # Y0>=0 > -0.01 everywhere -> unsat
    verdict, wit = cy.cctsdb_yolo_verify(q, v, _S(), timeout=30, log=lambda _m: None)
    assert verdict == 'unsat' and wit is None


def test_cctsdb_within_tol(tmp_path):
    q = _net(str(tmp_path / 'n.onnx'))
    v = _vnnlib(str(tmp_path / 'v.vnnlib'), thr=0.0)   # (0,0)->Y0=0.0 -> margin 0 (within-tol)
    verdict, wit = cy.cctsdb_yolo_verify(q, v, _S(), timeout=30, log=lambda _m: None)
    assert verdict == 'sat' and wit is not None


def test_cctsdb_timeout(tmp_path):
    q = _net(str(tmp_path / 'n.onnx'))
    v = _vnnlib(str(tmp_path / 'v.vnnlib'), thr=-0.01)
    verdict, wit = cy.cctsdb_yolo_verify(q, v, _S(), timeout=-1.0, log=lambda _m: None)
    assert verdict == 'timeout' and wit is None


def test_cctsdb_timeout_with_within_tol(tmp_path, monkeypatch):
    # within-tol CE found at (0,0), then the timeout fires on the next iteration's check ->
    # the stashed within-tol CE is emitted as sat.
    q = _net(str(tmp_path / 'n.onnx'))
    v = _vnnlib(str(tmp_path / 'v.vnnlib'), thr=0.0)

    class _Clk:
        def __init__(self):
            self.n = -1

        def __call__(self):
            self.n += 1
            return 0.0 if self.n < 2 else 100.0    # t0, iter0 check ok; iter1 check trips

    monkeypatch.setattr(cy.time, 'time', _Clk())
    verdict, wit = cy.cctsdb_yolo_verify(q, v, _S(), timeout=1.0, log=lambda _m: None)
    assert verdict == 'sat' and wit is not None


def test_cctsdb_refuse_noninteger(tmp_path):
    q = _net(str(tmp_path / 'n.onnx'))
    v = _vnnlib(str(tmp_path / 'v.vnnlib'), noninteger=True)   # X_3 range [0,5.5] -> raise
    with pytest.raises(NotImplementedError):
        cy.cctsdb_yolo_verify(q, v, _S(), timeout=30, log=lambda _m: None)


def test_cctsdb_refuse_too_many(tmp_path):
    class _Scap(_S):
        cctsdb_max_positions = 100
    q = _net(str(tmp_path / 'n.onnx'))
    v = _vnnlib(str(tmp_path / 'v.vnnlib'), p_hi=2000)   # 2000*2000 > 100 -> raise
    with pytest.raises(NotImplementedError):
        cy.cctsdb_yolo_verify(q, v, _Scap(), timeout=30, log=lambda _m: None)
