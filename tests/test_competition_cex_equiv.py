"""Equivalence tests: VC's fast counterexample acceptance vs the VENDORED
competition checker (vnncomp_cex_v2, byte-for-byte the VNN-COMP 2026 scorer).

VC uses its own ~4 ms vectorized check in production (the competition's per-
assertion checker is ~24 s on a 1.27M-dim spec). These tests assert VC's
accept/reject VERDICT is identical to the competition checker on representative
v2 instances — most importantly the STRICT-output (`>`/`<`) boundary that
smart_turn hits (Y == threshold must be rejected) and the input-box tolerance.

Requires the `vnnlib` package (pinned dep) — same as the competition checker.
"""
import os
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from vibecheck.vnncomp_cex_v2 import validate_cex_v2, ACCEPTED_RESULTS
from vibecheck.surrogate_pgd import parse_box_and_output


def _onnx_first_coord():
    """ONNX with Y[1,1] = X[0,0] (W = [[1,0]], b = 0) so Y is the first input."""
    W = numpy_helper.from_array(np.array([[1.0, 0.0]], np.float32), name='W')
    b = numpy_helper.from_array(np.zeros(1, np.float32), name='b')
    inp = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])
    out = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1])
    node = helper.make_node('Gemm', ['X', 'W', 'b'], ['Y'], transB=1)
    g = helper.make_graph([node], 'm', [inp], [out], initializer=[W, b])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    p = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
    p.write(m.SerializeToString())
    p.close()
    return p.name


def _v2_spec(out_op, thr, lo=(0.0, 0.0), hi=(1.0, 1.0)):
    """A v2 spec: input box [lo,hi] on X[1,2], output `out_op Y[0,0] thr`."""
    txt = ("(vnnlib-version <2.0>)\n"
           "(declare-network f (declare-input X real [1,2]) "
           "(declare-output Y real [1,1]))\n"
           f"(assert (>= X[0,0] {lo[0]}))\n(assert (<= X[0,0] {hi[0]}))\n"
           f"(assert (>= X[0,1] {lo[1]}))\n(assert (<= X[0,1] {hi[1]}))\n"
           f"(assert ({out_op} Y[0,0] {thr}))\n")
    p = tempfile.NamedTemporaryFile(suffix='.vnnlib', delete=False, mode='w')
    p.write(txt)
    p.close()
    return p.name


def _write_ce(x0, x1, y):
    txt = f"X real [1,2]\n{x0!r}\n{x1!r}\nY real [1,1]\n{y!r}\n"
    p = tempfile.NamedTemporaryFile(suffix='.counterexample', delete=False, mode='w')
    p.write(txt)
    p.close()
    return p.name


def _vc_surrogate_accepts(spec_path, x, y, atol=1e-4):
    """VC's surrogate acceptance gate (surrogate_pgd.ort_consider): the witness is
    in the input box (within atol) AND the output strictly violates (margin > 0)."""
    spec = parse_box_and_output(spec_path)
    inbox = all((np.asarray(x).ravel() >= lo - atol).all()
                and (np.asarray(x).ravel() <= hi + atol).all()
                for _, _, lo, hi in spec.inputs)
    margin = max(min((y[i] - rhs) if op == 'gt' else (rhs - y[i])
                     for i, op, rhs in clause) for clause in spec.out_dnf)
    return bool(inbox and margin > 0.0)


def _mk(out_op, thr, x0, x1, lo=(0.0, 0.0), hi=(1.0, 1.0)):
    """Run BOTH checkers on one (X, Y=X[0,0]) witness; return (vc_accept, comp_accept)."""
    onnx_p = _onnx_first_coord()
    spec_p = _v2_spec(out_op, thr, lo, hi)
    y = float(x0)                                   # Y = X[0,0]
    ce_p = _write_ce(x0, x1, y)
    comp = validate_cex_v2(onnx_p, spec_p, ce_p)[0] in ACCEPTED_RESULTS
    vc = _vc_surrogate_accepts(spec_p, [x0, x1], [y])
    for f in (onnx_p, spec_p, ce_p):
        os.unlink(f)
    return vc, comp


def test_strict_gt_clear_both_accept():
    vc, comp = _mk('>', 0.5, 0.9, 0.3)        # Y=0.9 > 0.5 strict
    assert vc is True and comp is True and vc == comp


def test_strict_gt_boundary_both_reject():
    # smart_turn case: Y == threshold; strict `>` not satisfied at zero tolerance.
    vc, comp = _mk('>', 0.5, 0.5, 0.3)        # Y=0.5, NOT > 0.5
    assert vc is False and comp is False and vc == comp


def test_strict_gt_below_both_reject():
    vc, comp = _mk('>', 0.5, 0.4, 0.3)        # Y=0.4 < 0.5
    assert vc is False and comp is False and vc == comp


def test_strict_lt_clear_both_accept():
    vc, comp = _mk('<', 0.5, 0.1, 0.3)        # Y=0.1 < 0.5 strict
    assert vc is True and comp is True and vc == comp


def test_strict_lt_boundary_both_reject():
    vc, comp = _mk('<', 0.5, 0.5, 0.3)        # Y=0.5, NOT < 0.5
    assert vc is False and comp is False and vc == comp


def test_input_box_within_tol_both_accept():
    # X[0,0] = -5e-5 is just below the floor 0.0 (within atol) AND Y clear.
    # Both accept (input tolerance), Y = -5e-5 with `< 0.5` strict holds.
    vc, comp = _mk('<', 0.5, -5e-5, 0.3)
    assert vc is True and comp is True and vc == comp


def test_input_box_far_outside_both_reject():
    # X[0,0] = 1.5 is well outside [0,1] (beyond atol); both reject even though
    # Y = 1.5 > 0.5 strictly (the input is not in the box).
    vc, comp = _mk('>', 0.5, 1.5, 0.3)
    assert vc is False and comp is False and vc == comp
