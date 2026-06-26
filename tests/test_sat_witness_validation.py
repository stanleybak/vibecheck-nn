"""Tests for `_validate_sat_witness` — the ORT-based defense-in-depth
check that catches spurious SAT verdicts from PGD/MILP/graph-builder
bugs."""

import os
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from vibecheck.verify_graph import _validate_sat_witness, _sat_disposition
from vibecheck.spec import VNNSpec, Constraint, Conjunct
from vibecheck.settings import default_settings


def _identity_onnx(n_in, n_out):
    """Build a 1-Gemm ONNX with `y = W x + b` where W = I[:n_out], b = 0
    (output = first n_out coordinates of input)."""
    W = numpy_helper.from_array(
        np.eye(n_out, n_in, dtype=np.float32), name='W')
    b = numpy_helper.from_array(
        np.zeros(n_out, dtype=np.float32), name='b')
    inp = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, n_in])
    out = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, n_out])
    node = helper.make_node('Gemm', ['x', 'W', 'b'], ['y'], transB=1)
    g = helper.make_graph([node], 'm', [inp], [out], initializer=[W, b])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    return m


def _save(model):
    f = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
    f.write(model.SerializeToString())
    f.close()
    return f.name


def _conjunct_spec(x_lo, x_hi, constraints):
    return VNNSpec(
        x_lo=np.asarray(x_lo, dtype=np.float64),
        x_hi=np.asarray(x_hi, dtype=np.float64),
        disjuncts=[Conjunct(constraints=constraints)])


def test_valid_sat_witness_accepted():
    """Witness inside input box + violates both constraints → accepted."""
    onnx_p = _save(_identity_onnx(2, 2))
    spec = _conjunct_spec(
        x_lo=[-1.0, -1.0], x_hi=[1.0, 1.0],
        constraints=[
            Constraint(index=0, op='<=', value=0.0),  # unsafe: Y_0 <= 0
            Constraint(index=1, op='>=', value=0.0),  # unsafe: Y_1 >= 0
        ])
    # Y = X under identity, so Y_0 = -0.5, Y_1 = 0.5 violates both.
    ok, info = _validate_sat_witness(onnx_p, spec, [-0.5, 0.5])
    assert ok, info
    assert info['spec_check'] == 'unknown'
    os.unlink(onnx_p)


def test_partial_violation_rejected_as_spurious():
    """Witness violates ONE constraint but not the other → rejected
    (this is exactly the cersyve bug — old code returned spurious sat)."""
    onnx_p = _save(_identity_onnx(2, 2))
    spec = _conjunct_spec(
        x_lo=[-1.0, -1.0], x_hi=[1.0, 1.0],
        constraints=[
            Constraint(index=0, op='<=', value=0.0),
            Constraint(index=1, op='>=', value=0.0),
        ])
    # Y_0 = -0.3 satisfies Y_0 <= 0 ✓ but Y_1 = -0.5 fails Y_1 >= 0 ✗
    ok, info = _validate_sat_witness(onnx_p, spec, [-0.3, -0.5])
    assert not ok
    assert info['reason'].startswith('ORT output does not violate spec')
    assert info['worst_margin'] > 0  # spec is verified for this point
    os.unlink(onnx_p)


def test_witness_outside_input_box_rejected():
    """Witness outside [x_lo, x_hi] beyond atol → rejected."""
    onnx_p = _save(_identity_onnx(2, 2))
    spec = _conjunct_spec(
        x_lo=[-1.0, -1.0], x_hi=[1.0, 1.0],
        constraints=[Constraint(index=0, op='<=', value=0.0)])
    ok, info = _validate_sat_witness(onnx_p, spec, [1.5, 0.0], atol=1e-4)
    assert not ok
    assert 'outside input box' in info['reason']
    os.unlink(onnx_p)


def test_witness_at_boundary_passes_with_atol():
    """Witness exactly on input boundary (within atol) → accepted."""
    onnx_p = _save(_identity_onnx(2, 2))
    spec = _conjunct_spec(
        x_lo=[-1.0, -1.0], x_hi=[1.0, 1.0],
        constraints=[Constraint(index=0, op='<=', value=0.0)])
    # X_0 = 1.0 + 1e-5 < 1.0 + atol=1e-4 → in box
    # Y_0 = X_0 = 1.00001, so Y_0 <= 0 NOT satisfied
    ok, info = _validate_sat_witness(onnx_p, spec, [1.0 + 1e-5, 0.0])
    # Box passes but spec doesn't violate
    assert not ok
    assert 'does not violate spec' in info['reason']
    os.unlink(onnx_p)


def test_no_onnx_path_skips_validation():
    """When no onnx_path is provided, skip validation (don't break the
    pipeline on environments without ORT or with un-tracked models)."""
    spec = _conjunct_spec(
        x_lo=[-1.0], x_hi=[1.0],
        constraints=[Constraint(index=0, op='<=', value=0.0)])
    ok, info = _validate_sat_witness(None, spec, [0.5])
    assert ok  # skipped, not failed
    assert 'no onnx_path' in info['reason']


# ---------------------------------------------------------------------------
# VNN-COMP 2026 output-strict rule (evaluation chairs): the 1e-4 tolerance
# applies ONLY to the input box; the replayed OUTPUT must violate with NO
# tolerance. `_validate_sat_witness` defaults `out_atol=0.0` (strict).
# ---------------------------------------------------------------------------

def test_output_atol_is_fixed_not_a_setting():
    """Output tolerance is hard-wired to 0.0 (VNN-COMP 2026 rule) and is NOT a
    config setting — so nothing can loosen it. The validator's default is strict
    and `sat_validate_out_atol` is intentionally absent from the settings."""
    import inspect
    assert inspect.signature(_validate_sat_witness).parameters['out_atol'].default == 0.0
    assert 'sat_validate_out_atol' not in default_settings()


def test_output_within_tol_rejected_by_default():
    """REGRESSION (2026 rule): a witness whose ORT output is within +1e-4 of
    violating but does NOT actually violate (margin > 0) is REJECTED by
    default — output tolerance is no longer scorer-accepted. (Under the old
    both-sided-atol code this point passed via the +/-atol output band.)"""
    onnx_p = _save(_identity_onnx(2, 2))
    spec = _conjunct_spec(
        x_lo=[-1.0, -1.0], x_hi=[1.0, 1.0],
        constraints=[Constraint(index=0, op='<=', value=0.0)])  # unsafe: Y_0 <= 0
    # X_0 = +5e-5 is in-box; Y_0 = +5e-5 > 0 → does NOT violate, but is within
    # atol (1e-4) of the boundary. Default out_atol=0.0 → rejected.
    ok, info = _validate_sat_witness(onnx_p, spec, [5e-5, 0.0])
    assert not ok
    assert 'does not violate spec' in info['reason']
    assert 'out_atol=0' in info['reason']
    assert info['worst_margin'] > 0
    os.unlink(onnx_p)


def test_output_within_tol_accepted_only_when_configured():
    """The within-output-tol band is reachable ONLY by explicitly setting
    out_atol>0 (a non-default config that is NOT scorer-accepted)."""
    onnx_p = _save(_identity_onnx(2, 2))
    spec = _conjunct_spec(
        x_lo=[-1.0, -1.0], x_hi=[1.0, 1.0],
        constraints=[Constraint(index=0, op='<=', value=0.0)])
    ok, info = _validate_sat_witness(onnx_p, spec, [5e-5, 0.0], out_atol=1e-4)
    assert ok, info
    assert info['spec_check'] == 'unknown'
    os.unlink(onnx_p)


def test_output_boundary_violation_accepted():
    """A witness whose output sits exactly on the constraint boundary
    (margin == 0) IS a violation under the official `<=`/`>=` comparison at
    zero tolerance (boundary inclusive) → accepted even with out_atol=0."""
    onnx_p = _save(_identity_onnx(2, 2))
    spec = _conjunct_spec(
        x_lo=[-1.0, -1.0], x_hi=[1.0, 1.0],
        constraints=[Constraint(index=0, op='<=', value=0.0)])
    # X_0 = 0.0 → Y_0 = 0.0 → Y_0 <= 0 holds at the boundary.
    ok, info = _validate_sat_witness(onnx_p, spec, [0.0, 0.0])
    assert ok, info
    assert info['spec_check'] == 'unknown'
    os.unlink(onnx_p)


def test_input_box_tol_still_applies_when_output_strictly_violates():
    """The 1e-4 INPUT-box tolerance is unchanged: a witness up to atol outside
    the box is accepted, provided its output STRICTLY violates."""
    onnx_p = _save(_identity_onnx(2, 2))
    spec = _conjunct_spec(
        x_lo=[-1.0, -1.0], x_hi=[1.0, 1.0],
        constraints=[Constraint(index=0, op='<=', value=0.0)])
    # X_0 = -1.0 - 5e-5 is 5e-5 BELOW the floor (within atol) → in-box-with-tol;
    # after clamp Y_0 ≈ -1.0 < 0 → strict violation. Accepted.
    ok, info = _validate_sat_witness(onnx_p, spec, [-1.0 - 5e-5, 0.0], atol=1e-4)
    assert ok, info
    os.unlink(onnx_p)


def _two_input_add_onnx():
    """ONNX with two inputs A[1,1], B[1,1] and output Y = A + B."""
    inp1 = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 1])
    inp2 = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 1])
    out = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1])
    node = helper.make_node('Add', ['A', 'B'], ['Y'])
    g = helper.make_graph([node], 'm', [inp1, inp2], [out])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    return _save(m)


def test_validate_witness_ort_multi_input():
    """The unified validator handles MULTI-input witnesses (the surrogate/attack
    case): per-input box check + clamp, multi-tensor ORT feed, caller-supplied
    output rule."""
    from vibecheck.verify_graph import _validate_witness_ort
    onnx_p = _two_input_add_onnx()
    box = (np.array([0.0]), np.array([1.0]))

    def _viol(inbox, y):                 # unsafe: Y > 0.5 (strict)
        return (float(y[0]) > 0.5), {'worst_margin': float(y[0]) - 0.5}

    # A=0.4, B=0.4 -> Y=0.8 > 0.5 -> violated -> accept
    ok, info = _validate_witness_ort(
        onnx_p, [np.array([0.4]), np.array([0.4])], [box, box], _viol)
    assert ok and abs(info['out'][0] - 0.8) < 1e-6

    # A=0.1, B=0.1 -> Y=0.2 -> not violated -> reject
    ok2, _ = _validate_witness_ort(
        onnx_p, [np.array([0.1]), np.array([0.1])], [box, box], _viol)
    assert not ok2

    # A=1.5 is outside its box [0,1] beyond atol -> reject (even though Y would violate)
    ok3, info3 = _validate_witness_ort(
        onnx_p, [np.array([1.5]), np.array([0.4])], [box, box], _viol)
    assert not ok3 and 'outside input box' in info3['reason']
    os.unlink(onnx_p)


def test_sat_disposition_boundary_is_real():
    """`_sat_disposition`: a boundary violation (worst margin == 0) is a CLEAR
    counterexample ('real') under the inclusive output rule — it is NOT a
    keep-searching within-tol near-miss."""
    spec = _conjunct_spec(
        x_lo=[-1.0], x_hi=[1.0],
        constraints=[Constraint(index=0, op='<=', value=0.0)])
    info = {'out': np.array([0.0])}            # margin == 0
    s = default_settings()
    assert _sat_disposition(None, spec, s, [0.0], info) == 'real'
    # strict violation is also 'real'
    info2 = {'out': np.array([-0.5])}
    assert _sat_disposition(None, spec, s, [-0.5], info2) == 'real'
