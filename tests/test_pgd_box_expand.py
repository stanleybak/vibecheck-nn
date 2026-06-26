"""Tests for the SEARCH-ONLY input-box expansion (`pgd_input_box_expand`).

The 2026 input tolerance accepts a witness up to `sat_validate_atol` OUTSIDE the
box, so every PGD/attack search loosens each input bound by `pgd_input_box_expand`
to reach a counterexample sitting just outside. Two halves are tested:
  1. the search-side helpers (`pgd_box_expand_amount`, `expand_search_box`);
  2. the validation/emit side (`_clamp_witness_to_box` slack + the unified gate's
     "strict first, then ±slack band" logic) — the load-bearing piece: without it
     a just-outside CE gets clamped strictly in-box and LOSES its violation, so
     the whole feature would be a silent no-op.
"""
import os
import tempfile

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from vibecheck.pgd import pgd_box_expand_amount, expand_search_box
from vibecheck.settings import default_settings
from vibecheck.verify_graph import (
    _clamp_witness_to_box, _validate_witness_ort, _validate_sat_witness)
from vibecheck.spec import VNNSpec, Constraint, Conjunct

import pytest


# --------------------------------------------------------------------------- #
# search-side helpers
# --------------------------------------------------------------------------- #

def test_setting_default_off_and_within_atol():
    s = default_settings()
    # DEFAULT 0.0 (OFF): net-zero benefit measured + regresses tiny-eps SAT cases
    # (vggnet spec0 sat->error at expand==atol). Feature is opt-in via --set.
    assert s.pgd_input_box_expand == 0.0
    # the invariant the feature relies on when enabled: never wider than input tol
    assert s.pgd_input_box_expand <= s.sat_validate_atol


def test_amount_default_zero_and_assert():
    s = default_settings()
    assert pgd_box_expand_amount(s) == 0.0           # default OFF -> exactly 0
    s.pgd_input_box_expand = 1e-4                     # opt in at the tolerance
    assert pgd_box_expand_amount(s) == 1e-4
    s.pgd_input_box_expand = 0.0
    assert pgd_box_expand_amount(s) == 0.0           # disabled -> exactly 0
    s.pgd_input_box_expand = 1e-3                     # > sat_validate_atol (1e-4)
    with pytest.raises(AssertionError):
        pgd_box_expand_amount(s)


def test_expand_search_box_torch_and_numpy():
    import torch
    s = default_settings()
    s.pgd_input_box_expand = 1e-4
    lo_t, hi_t = expand_search_box(torch.zeros(2), torch.ones(2), s)
    assert torch.allclose(lo_t, torch.full((2,), -1e-4))
    assert torch.allclose(hi_t, torch.full((2,), 1.0 + 1e-4))
    lo_n, hi_n = expand_search_box(np.zeros(2), np.ones(2), s)
    assert np.allclose(lo_n, -1e-4) and np.allclose(hi_n, 1.0 + 1e-4)


def test_expand_search_box_disabled_is_noop():
    s = default_settings()
    s.pgd_input_box_expand = 0.0
    lo, hi = np.zeros(3), np.ones(3)
    lo2, hi2 = expand_search_box(lo, hi, s)
    assert lo2 is lo and hi2 is hi                    # identity, untouched


# --------------------------------------------------------------------------- #
# clamp slack
# --------------------------------------------------------------------------- #

def test_clamp_strict_vs_band():
    # witness 5e-5 below the floor 0.0
    w = [-5e-5]
    strict = _clamp_witness_to_box(w, [0.0], [1.0])               # slack=0 -> pulled in
    assert strict[0] >= 0.0
    band = _clamp_witness_to_box(w, [0.0], [1.0], slack=1e-4)     # kept just outside
    assert band[0] < 0.0 and band[0] >= -1e-4
    # float32-safe: the band result stays within [lo-slack, hi+slack] after cast
    assert np.float32(band[0]) >= np.float32(-1e-4)


# --------------------------------------------------------------------------- #
# the load-bearing behavioural test: a CE that ONLY holds just outside the box
# --------------------------------------------------------------------------- #

def _identity_onnx(n_in, n_out):
    W = numpy_helper.from_array(np.eye(n_out, n_in, dtype=np.float32), name='W')
    b = numpy_helper.from_array(np.zeros(n_out, dtype=np.float32), name='b')
    inp = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, n_in])
    out = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, n_out])
    node = helper.make_node('Gemm', ['x', 'W', 'b'], ['y'], transB=1)
    g = helper.make_graph([node], 'm', [inp], [out], initializer=[W, b])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    f = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
    f.write(m.SerializeToString()); f.close()
    return f.name


def _spec(x_lo, x_hi, constraints):
    return VNNSpec(x_lo=np.asarray(x_lo, np.float64), x_hi=np.asarray(x_hi, np.float64),
                   disjuncts=[Conjunct(constraints=constraints)])


def test_just_outside_ce_needs_emit_slack():
    """y = x (identity). Unsafe iff Y_0 <= -2e-5. Box x in [0,1]:
      - in-box optimum is x=0 -> y=0, which does NOT violate (0 > -2e-5);
      - x = -5e-5 (within atol of the box) -> y=-5e-5 <= -2e-5 -> violates.
    With emit_slack=0 the witness is clamped strictly to 0 and the violation is
    LOST (reject). With emit_slack=1e-4 it is kept just outside and validates."""
    onnx_p = _identity_onnx(1, 1)
    spec = _spec([0.0], [1.0], [Constraint(index=0, op='<=', value=-2e-5)])

    ok0, info0 = _validate_sat_witness(onnx_p, spec, [-5e-5], atol=1e-4, emit_slack=0.0)
    assert not ok0                       # strict clamp -> y=0 -> not a CE
    assert 'does not violate' in info0['reason']

    ok1, info1 = _validate_sat_witness(onnx_p, spec, [-5e-5], atol=1e-4, emit_slack=1e-4)
    assert ok1, info1                    # band keeps it just outside -> y<0 -> CE
    assert info1['witness_inbox'][0] < 0.0          # emitted point is the outside one
    assert info1['witness_inbox'][0] >= -1e-4       # but within the scorer's tolerance
    os.unlink(onnx_p)


def test_in_box_ce_still_emitted_strictly_in_box():
    """When a strict in-box CE exists, the gate emits the STRICT (in-box) point
    even with emit_slack>0 — the band is only a fallback, so normal cases keep
    scoring CORRECT (witness exactly in [lo,hi])."""
    onnx_p = _identity_onnx(1, 1)
    spec = _spec([0.0], [1.0], [Constraint(index=0, op='<=', value=0.5)])  # y<=0.5 unsafe
    ok, info = _validate_sat_witness(onnx_p, spec, [0.3], atol=1e-4, emit_slack=1e-4)
    assert ok and 0.0 <= info['witness_inbox'][0] <= 1.0    # stays strictly in box
    os.unlink(onnx_p)


def test_emit_slack_default_zero_rejects_outside():
    """Default `_validate_sat_witness` (emit_slack=0) is unchanged: a witness whose
    violation only survives outside the box is rejected (no silent behaviour drift
    for the paths that don't opt into box expansion)."""
    onnx_p = _identity_onnx(1, 1)
    spec = _spec([0.0], [1.0], [Constraint(index=0, op='<=', value=-2e-5)])
    ok, _ = _validate_sat_witness(onnx_p, spec, [-5e-5])    # emit_slack defaults to 0.0
    assert not ok
    os.unlink(onnx_p)


def test_far_outside_still_rejected_with_slack():
    """emit_slack widens the EMIT band, NOT the input-box acceptance: a witness
    beyond `atol` outside is still rejected up front (box check uses atol)."""
    onnx_p = _identity_onnx(1, 1)
    spec = _spec([0.0], [1.0], [Constraint(index=0, op='<=', value=-2e-5)])
    ok, info = _validate_sat_witness(onnx_p, spec, [-2e-4], atol=1e-4, emit_slack=1e-4)
    assert not ok and 'outside input box' in info['reason']
    os.unlink(onnx_p)
