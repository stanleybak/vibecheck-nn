"""Tests for `_validate_sat_witness` — the ORT-based defense-in-depth
check that catches spurious SAT verdicts from PGD/MILP/graph-builder
bugs."""

import os
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from vibecheck.verify_graph import _validate_sat_witness
from vibecheck.spec import VNNSpec, Constraint, Conjunct


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
