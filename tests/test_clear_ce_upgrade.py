"""Tests for `_try_clear_ce_upgrade` — the top-level chokepoint that upgrades a
NEAR-BOUNDARY closure counterexample (e.g. a network-pair's trivial diagonal,
output diff exactly 0 — a valid `<=` CE the scorer accepts but not a strict
violation) to a CLEAR strict counterexample (margin < -atol) when one exists.
"""
import time

import numpy as np
import onnx
from onnx import TensorProto, helper

from vibecheck import verify_graph as vg
from vibecheck.verify_graph import _try_clear_ce_upgrade, verify_graph
from vibecheck.onnx_loader import load_onnx
from vibecheck.spec import VNNSpec, Constraint, Conjunct
from vibecheck.settings import default_settings


def _identity_onnx(tmp_path):
    """1-Gemm identity Y = X over a single coordinate."""
    W = helper.make_tensor('W', TensorProto.FLOAT, [1, 1], [1.0])
    b = helper.make_tensor('b', TensorProto.FLOAT, [1], [0.0])
    node = helper.make_node('Gemm', ['X', 'W', 'b'], ['Y'])
    g = helper.make_graph(
        [node], 'identity',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1])],
        [W, b])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    p = tmp_path / 'identity.onnx'
    onnx.save(m, str(p))
    return str(p)


class _Graph:
    def __init__(self, onnx_path):
        self.onnx_path = onnx_path


def _spec(lo, hi):
    # Unsafe (closure): Y_0 <= 0.
    return VNNSpec(np.array([lo], np.float64), np.array([hi], np.float64),
                   [Conjunct([Constraint(index=0, op='<=', value=0.0)])])


def test_upgrade_finds_clear_ce(tmp_path):
    """Box [-1, 1]: a clear CE (Y_0 down to -1) exists — the upgrade returns a
    witness strictly inside the unsafe region (margin < -atol)."""
    g = _Graph(_identity_onnx(tmp_path))
    s = default_settings(total_timeout=30.0)
    w = _try_clear_ce_upgrade(g, _spec(-1.0, 1.0), s, time.perf_counter())
    assert w is not None
    assert float(np.asarray(w).flatten()[0]) < -float(s.sat_validate_atol)


def test_upgrade_no_clear_ce_returns_none(tmp_path):
    """Box [0, 1]: the only point with Y_0 <= 0 is the boundary X=0 (Y_0=0);
    no clear CE exists, so the upgrade keeps the boundary fallback (None)."""
    g = _Graph(_identity_onnx(tmp_path))
    s = default_settings(total_timeout=30.0)
    assert _try_clear_ce_upgrade(g, _spec(0.0, 1.0), s, time.perf_counter()) is None


def test_upgrade_no_onnx_path_returns_none():
    s = default_settings(total_timeout=30.0)
    assert _try_clear_ce_upgrade(_Graph(None), _spec(-1.0, 1.0), s,
                                 time.perf_counter()) is None


def test_upgrade_disabled_by_zero_budget(tmp_path):
    g = _Graph(_identity_onnx(tmp_path))
    s = default_settings(total_timeout=30.0, clear_ce_upgrade_budget=0.0)
    assert _try_clear_ce_upgrade(g, _spec(-1.0, 1.0), s,
                                 time.perf_counter()) is None


def test_upgrade_no_budget_left_returns_none(tmp_path):
    """When the global deadline has effectively passed, the upgrade is skipped
    (it must not push the instance into a hard timeout)."""
    g = _Graph(_identity_onnx(tmp_path))
    s = default_settings(total_timeout=0.5)
    # t_start 5 s ago -> remaining budget is negative.
    assert _try_clear_ce_upgrade(g, _spec(-1.0, 1.0), s,
                                 time.perf_counter() - 5.0) is None


def test_upgrade_propagates_pgd_error(tmp_path, monkeypatch):
    """A real error from the PGD attempt is NOT swallowed — it propagates (the
    upgrade does not silently skip; main's crash handler records 'error')."""
    import pytest
    import vibecheck.onnx_torch_runner as otr

    def _boom(*a, **k):
        raise RuntimeError('synthetic pgd failure')

    monkeypatch.setattr(otr, 'pgd_via_onnx', _boom)
    g = _Graph(_identity_onnx(tmp_path))
    s = default_settings(total_timeout=30.0)
    with pytest.raises(RuntimeError, match='synthetic pgd failure'):
        _try_clear_ce_upgrade(g, _spec(-1.0, 1.0), s, time.perf_counter())


def test_chokepoint_keeps_boundary_when_no_clear_ce(tmp_path):
    """End-to-end through verify_graph: box [0, 1] with unsafe Y_0 <= 0 has only
    the boundary X=0 (Y_0=0). The chokepoint runs the clear-CE upgrade, finds
    nothing, and keeps the (valid) boundary witness."""
    g = load_onnx(_identity_onnx(tmp_path))
    s = default_settings(device='cpu', total_timeout=10, print_progress=False)
    g.optimize(s)
    result, details = verify_graph(g, _spec(0.0, 1.0), s)
    assert result == 'sat'
    assert abs(float(np.asarray(details['witness']).flatten()[0])) <= 1e-4


def test_chokepoint_upgrades_witness_to_clear_ce(tmp_path, monkeypatch):
    """When the chokepoint's near-boundary witness CAN be upgraded, the returned
    details carry the CLEAR witness and a `+clear_ce_upgrade` phase tag."""
    g = load_onnx(_identity_onnx(tmp_path))
    s = default_settings(device='cpu', total_timeout=10, print_progress=False)
    g.optimize(s)
    monkeypatch.setattr(vg, '_try_clear_ce_upgrade',
                        lambda *a, **k: np.array([-0.5]))
    result, details = verify_graph(g, _spec(0.0, 1.0), s)
    assert result == 'sat'
    assert float(np.asarray(details['witness']).flatten()[0]) == -0.5
    assert str(details['phase']).endswith('+clear_ce_upgrade')
