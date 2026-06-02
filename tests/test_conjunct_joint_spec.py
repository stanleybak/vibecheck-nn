"""Demonstrates + pins the conjunctive-spec precision bug.

The unsafe region of a conjunct is `c1 AND c2 AND ...`. The cheap closure in
`_zono_spec_lbs_and_open_qis` declared a disjunct CLOSED (safe) iff SOME single
query is provably safe over the zonotope (per-query `any(lb>0)`). That is
sufficient but NOT necessary: a conjunct can be JOINTLY infeasible over the
zonotope even when no single constraint is always-violated — exactly the case on
conjunctive benchmarks (acasxu y0-is-min, cersyve, sat_relu, soundnessbench).

These tests build a correlated 2-output zonotope and check both directions:
  - SAFE case: unsafe conjunct jointly infeasible -> disjunct must be CLOSED.
  - UNSAFE case: unsafe conjunct jointly feasible   -> disjunct must stay OPEN.

`test_safe_conjunct_must_close` FAILS before the joint-check fix (per-query marks
it open) and PASSES after.
"""
import numpy as np

from vibecheck.verify_graph import (
    _zono_spec_lbs_and_open_qis, _conjunct_unsafe_feasible_zono)


# Correlated zonotope: y0 = e, y1 = -e, e in [-1,1]  (perfectly anti-correlated).
# center=(0,0), one generator g=(1,-1).
CENTER = np.array([0.0, 0.0])
GEN = np.array([[1.0], [-1.0]])   # shape (output_dim=2, k=1)


def _queries(pairs):
    """pairs: list of (w, b). All in disjunct 0 (one conjunct). Query convention:
    'safe iff w·y + b > 0'; the unsafe constraint is w·y + b <= 0."""
    return [(0, np.asarray(w, dtype=float), float(b)) for w, b in pairs]


def test_safe_conjunct_must_close():
    # Unsafe conjunct: {y0 >= 0.5 AND y1 >= 0.5}. Safe queries: 0.5 - y0 > 0,
    # 0.5 - y1 > 0  -> w=[-1,0] b=0.5 ; w=[0,-1] b=0.5.
    # y1 = -y0, so y0>=0.5 AND y1>=0.5 (=> y0<=-0.5) is INFEASIBLE -> SAFE.
    # But per query: y0 ranges [-1,1] so neither '0.5-y0>0' nor '0.5-y1>0'
    # holds for the whole zonotope.
    queries = _queries([([-1.0, 0.0], 0.5), ([0.0, -1.0], 0.5)])
    spec_lbs, open_qis = _zono_spec_lbs_and_open_qis(CENTER, GEN, queries)
    # per-query lbs are both negative (this is the trap):
    assert spec_lbs[0] < 0 and spec_lbs[1] < 0
    # the disjunct is genuinely SAFE (jointly infeasible) -> must be CLOSED:
    assert open_qis == [], (
        "joint-infeasible conjunct must be closed, not split "
        f"(open_qis={open_qis})")


def test_unsafe_conjunct_must_stay_open():
    # Unsafe conjunct: {y0 <= 0.5 AND y1 <= 0.5}. Safe queries: y0-0.5>0,
    # y1-0.5>0 -> w=[1,0] b=-0.5 ; w=[0,1] b=-0.5.
    # y0<=0.5 (e<=0.5) AND y1<=0.5 (e>=-0.5) -> e in [-0.5,0.5] FEASIBLE -> UNSAFE.
    queries = _queries([([1.0, 0.0], -0.5), ([0.0, 1.0], -0.5)])
    _spec_lbs, open_qis = _zono_spec_lbs_and_open_qis(CENTER, GEN, queries)
    # must NOT be closed (soundness: a feasible unsafe conjunct stays open):
    assert set(open_qis) == {0, 1}, (
        f"feasible unsafe conjunct must stay open (open_qis={open_qis})")


def test_single_constraint_unchanged():
    # Single-constraint conjunct: per-query == joint, behavior must be unchanged.
    # Safe query y0 - 0.5 > 0 over y0 in [-1,1] -> lb = -1.5 < 0 -> open.
    q_open = _queries([([1.0, 0.0], -0.5)])
    _l, open_qis = _zono_spec_lbs_and_open_qis(CENTER, GEN, q_open)
    assert open_qis == [0]
    # Safe query y0 + 2 > 0 over y0 in [-1,1] -> lb = 1 > 0 -> closed.
    q_closed = _queries([([1.0, 0.0], 2.0)])
    _l2, open_qis2 = _zono_spec_lbs_and_open_qis(CENTER, GEN, q_closed)
    assert open_qis2 == []


def test_point_zonotope_helper():
    # k==0 (no generators): the helper evaluates the conjunct at the point.
    c = np.array([2.0, -3.0])
    G0 = np.zeros((2, 0))
    w = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]  # safe: y0>0, y1>0
    # unsafe {y0<=0 AND y1<=0} at (2,-3): y0=2 not <=0 -> infeasible (safe).
    assert _conjunct_unsafe_feasible_zono(c, G0, w, [0.0, 0.0]) is False
    # at (-1,-1): both <=0 -> unsafe conjunct holds at the point -> feasible.
    assert _conjunct_unsafe_feasible_zono(
        np.array([-1.0, -1.0]), G0, w, [0.0, 0.0]) is True
