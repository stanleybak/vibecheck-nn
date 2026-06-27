"""Unit tests for nonlinear_augment: transpile a NONLINEAR v2 spec (degree>=2
polynomial atoms, X*Y coupling) into an augmented ONNX (runs f, then computes each
constraint polynomial as an extra output) + a linear v1 DNF spec. Covers detection,
analysis (feats/cons/xbox), the augmentation graph, the v1 emitter, and the
oracle-gated end-to-end builder, on a tiny synthetic 2-in/1-out net.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import pytest

from vibecheck import nonlinear_augment as nla
from vibecheck.vnnlib_loader import parse_vnnlib_v2, load_vnnlib


# tiny net f(X) = X0 + X1  (Gemm, 2-in -> 1-out)
def _tiny_net(path):
    W = helper.make_tensor('W', TensorProto.FLOAT, [1, 2], [1.0, 1.0])
    b = helper.make_tensor('b', TensorProto.FLOAT, [1], [0.0])
    node = helper.make_node('Gemm', ['X', 'W', 'b'], ['Y'], transB=1)
    g = helper.make_graph(
        [node], 'f',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1])],
        [W, b])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    onnx.save(m, path)


# nonlinear v2 spec: X-box [0,1]^2, two output clauses, a degree-2 monomial
# (X0*X1), an X*Y coupling (X0*Y0), and a STRICT atom (Y0 < 1.0).
_V2 = """
(vnnlib-version <2.0>)
(declare-network f (declare-input X float32 [1,2]) (declare-output Y float32 [1,1]))
(assert (and (>= X[0,0] 0.0) (<= X[0,0] 1.0)))
(assert (and (>= X[0,1] 0.0) (<= X[0,1] 1.0)))
(assert (or (<= (* X[0,0] X[0,1]) 0.5)
            (and (< Y[0,0] 1.0) (<= (* X[0,0] Y[0,0]) 0.3))))
"""

_V1_LINEAR = """
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (<= X_0 1.0))
(assert (>= X_0 0.0))
(assert (<= Y_0 0.0))
"""

_V2_LINEAR = """
(vnnlib-version <2.0>)
(declare-network N (declare-input X float32 [1]) (declare-output Y float32 [1]))
(assert (<= X[0] 1.0))
(assert (>= X[0] 0.0))
(assert (<= Y[0] 0.0))
"""


def test_var_idx():
    assert nla._var_idx('X_0') == ('X', 0)
    assert nla._var_idx('Y_3') == ('Y', 3)


def test_is_nonlinear_v2_spec():
    assert nla.is_nonlinear_v2_spec(_V2) is True
    assert nla.is_nonlinear_v2_spec(_V1_LINEAR) is False     # v1 -> not 2.0
    assert nla.is_nonlinear_v2_spec(_V2_LINEAR) is False     # v2 but degree 1
    # v2-detected but unparseable -> caught (VnnlibParseError) -> False
    assert nla.is_nonlinear_v2_spec(
        "(vnnlib-version <2.0>)\n(declare-network N "
        "(declare-input X float32 [1]) (declare-output Y float32 [1]))\n"
        "(assert (badop X[0] 0))") is False


def test_is_nonlinear_fast_path_skips_parse(monkeypatch):
    """A linear v2 spec (no `(*`/`(/` node) returns False via the fast-path —
    WITHOUT the O(spec-size) `parse_vnnlib_v2` (which is ~37s on smart_turn's
    121 MB input box). Proven by making the parser blow up if reached."""
    def _boom(*a, **k):
        raise AssertionError('parse_vnnlib_v2 must not run for a linear spec')
    monkeypatch.setattr(nla, 'parse_vnnlib_v2', _boom)
    lin = ('(vnnlib-version <2.0>)\n(declare-network N (declare-input X float32 '
           '[1,2]) (declare-output Y float32 [1,1]))\n'
           '(assert (<= X[0,0] 1.0))\n(assert (> Y[0,0] 0.5))\n')
    assert nla.is_nonlinear_v2_spec(lin) is False
    monkeypatch.undo()
    # A spec WITH a multiplication is NOT short-circuited -> parsed -> nonlinear.
    nl = ('(vnnlib-version <2.0>)\n(declare-network N (declare-input X float32 '
          '[1,2]) (declare-output Y float32 [1,1]))\n'
          '(assert (>= (* X[0,0] X[0,0]) 1.0))\n')
    assert nla.is_nonlinear_v2_spec(nl) is True


def test_analyze_feats_cons_xbox():
    prop = parse_vnnlib_v2(_V2)
    feats, cons, clauses, xbox = nla.analyze(prop)
    # degree-2 monomials present
    assert any(len(m) == 2 for m in feats)
    # X-box folded from the single-var linear constraints
    assert xbox[0] == [0.0, 1.0]
    assert xbox[1] == [0.0, 1.0]
    # a STRICT constraint exists (Y0 < 1.0) and a non-strict one
    assert any(strict for (_row, _b, strict) in cons)
    assert any(not strict for (_row, _b, strict) in cons)
    # DNF has two clauses
    assert len(clauses) == 2


def test_build_augmented_instance_oracle_and_threshold(tmp_path):
    net = str(tmp_path / 'f.onnx')
    _tiny_net(net)
    spec_path = str(tmp_path / 'spec.vnnlib')
    open(spec_path, 'w').write(_V2)
    aug_onnx, aug_vnnlib = nla.build_augmented_instance(net, spec_path)

    # the v1 spec uses threshold 0 for BOTH strict and non-strict (sound superset)
    txt = open(aug_vnnlib).read()
    assert '-0.0001' not in txt and '-1e-04' not in txt
    spec = load_vnnlib(aug_vnnlib)
    for dj in spec.disjuncts:
        for c in dj.constraints:
            assert c.value == 0.0 and c.op == '<='

    # augmented ONNX output == the true polynomial values at a sample point
    prop = parse_vnnlib_v2(_V2)
    feats, cons, _clauses, _xbox = nla.analyze(prop)
    sess = ort.InferenceSession(open(aug_onnx, 'rb').read(),
                                providers=['CPUExecutionProvider'])
    x = np.array([0.3, 0.7], np.float32)
    y_net = float(x[0] + x[1])
    ref = nla._poly_eval(cons, x.astype(np.float64), np.array([y_net]), feats)
    got = sess.run(None, {'X': x.reshape(1, 2)})[0].flatten()
    np.testing.assert_allclose(got, ref, atol=1e-4)


def test_build_augmented_instance_oracle_failure(tmp_path, monkeypatch):
    """The oracle gates correctness: a wrong polynomial reference must raise."""
    net = str(tmp_path / 'f.onnx')
    _tiny_net(net)
    spec_path = str(tmp_path / 'spec.vnnlib')
    open(spec_path, 'w').write(_V2)
    # corrupt _poly_eval so augmented output != reference -> oracle assert fires
    monkeypatch.setattr(nla, '_poly_eval',
                        lambda *a, **k: np.full(len(a[0]), 1e9))
    with pytest.raises(AssertionError, match='oracle FAIL'):
        nla.build_augmented_instance(net, spec_path)
