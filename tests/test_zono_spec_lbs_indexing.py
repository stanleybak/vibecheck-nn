"""Tests for `_zono_spec_lbs_and_open_qis` — the spec-lb / open-query
indexing helper used by `_phase1_bab_refine`'s pre-cascade PGD hook.

Bug history (the reason this helper was extracted):

(1) Original inline code keyed `spec_lbs_zono_dict` by
    `queries_flat[qi][0]` (the disjunct id) rather than by query index.
    For multi-query conjuncts (cersyve has 2 queries per disjunct), the
    second query silently overwrote the first under the shared disjunct-
    id key. The downstream pre-cascade PGD hook then read
    `spec_lbs.get(qi, 0.0)` expecting query-indexed entries, getting 0.0
    fallback for query indices ≥1 — garbage sort order over disjuncts
    and missed-open disjuncts.

(2) Original "open disjuncts" derivation used per-query OR semantics
    instead of per-disjunct AND semantics (a conjunct is safe iff ANY
    of its constraints is provably violated; thus open iff ALL queries
    have lb ≤ 0).

(3) Bias sign: the helper now uses `wc + bq` to match
    `as_linear_queries`' documented `min(w·y + bias) > 0` convention.
    The inline code had `wc - bq` which is wrong sign for bias≠0 — but
    is silent on cifar/tinyimagenet (pairwise constraints → bias=0) and
    cersyve pendulum (threshold value=0). Catches any future spec with
    non-zero thresholds.
"""

import numpy as np
import pytest

from vibecheck.verify_graph import _zono_spec_lbs_and_open_qis
from vibecheck.spec import VNNSpec, Constraint, PairwiseConstraint, Conjunct


# --- Indexing: query-index keys, not disjunct ids -------------------------

def test_keyed_by_query_index_not_disjunct_id():
    """Multi-query conjunct (2 queries on same disjunct) must produce two
    distinct entries keyed 0 and 1 — NOT a single entry keyed by disjunct
    id (which was the bug)."""
    queries = [
        (0, np.array([1.0, 0.0]), 0.0),
        (0, np.array([0.0, -1.0]), 0.0),
    ]
    c = np.array([0.5, -0.5])
    G = np.array([[0.1, 0.0], [0.0, 0.2]])
    spec_lbs, _ = _zono_spec_lbs_and_open_qis(c, G, queries)
    assert set(spec_lbs.keys()) == {0, 1}, (
        f"expected query-indexed dict {{0,1}}, got {set(spec_lbs.keys())}")
    # qi=0: w·c + bias - |wG|_1 = 0.5 + 0 - 0.1 = 0.4
    assert spec_lbs[0] == pytest.approx(0.4)
    # qi=1: -c[1] + 0 - 0.2 = 0.5 - 0.2 = 0.3
    assert spec_lbs[1] == pytest.approx(0.3)


def test_distinct_lbs_for_multi_query_disjunct():
    """Witness that the pre-fix scheme (key by disjunct id) would have
    silently collapsed both lbs to one entry. Confirms both query
    indices appear AND lbs differ."""
    queries = [
        (0, np.array([1.0, 0.0]), 0.0),
        (0, np.array([0.0, -1.0]), 0.0),
    ]
    c = np.array([0.5, -0.5])
    G = np.array([[0.1, 0.0], [0.0, 0.2]])
    spec_lbs, _ = _zono_spec_lbs_and_open_qis(c, G, queries)
    assert spec_lbs[0] != spec_lbs[1]


# --- Open-disjunct: AND-of-queries semantics ------------------------------

def test_open_disjunct_uses_and_semantics():
    """A disjunct is open iff EVERY query has lb ≤ 0. One provably-safe
    query closes the whole conjunct (unsafe = AND-of-queries; AND fails
    if any operand fails)."""
    # 3 disjuncts:
    #   d=0  q0 lb=-1, q1 lb=-1     → OPEN (both ≤ 0)
    #   d=1  q2 lb=+0.5, q3 lb=-1   → CLOSED via AND (q2 safe)
    #   d=2  q4 lb=+9                → CLOSED
    queries = [
        (0, np.array([1.0, 0.0, 0.0, 0.0]), 0.0),
        (0, np.array([0.0, 1.0, 0.0, 0.0]), 0.0),
        (1, np.array([0.0, 0.0, 1.0, 0.0]), 0.0),
        (1, np.array([0.0, 0.0, 0.0, 1.0]), 0.0),
        (2, np.array([0.0, 0.0, 0.0, 1.0]), 10.0),
    ]
    c = np.array([-0.5, -0.5, 1.0, -0.5])
    G = np.diag([0.5, 0.5, 0.5, 0.5])
    spec_lbs, open_qis = _zono_spec_lbs_and_open_qis(c, G, queries)
    assert spec_lbs[0] == pytest.approx(-1.0)
    assert spec_lbs[1] == pytest.approx(-1.0)
    assert spec_lbs[2] == pytest.approx(0.5)
    assert spec_lbs[3] == pytest.approx(-1.0)
    assert spec_lbs[4] == pytest.approx(9.0)
    assert sorted(open_qis) == [0, 1], (
        f"expected only disjunct-0 queries open (AND semantics); "
        f"got {sorted(open_qis)} — q3 with lb=-1 should NOT be open "
        f"because its sibling q2 in disjunct 1 is provably safe.")


def test_single_query_per_disjunct_regression():
    """cifar100/tinyimagenet path: 1 query per disjunct. Disjunct-id and
    query-index keying coincide here; must still work post-refactor."""
    queries = [
        (0, np.array([1.0, 0.0]), 0.0),
        (1, np.array([0.0, -1.0]), 0.0),
    ]
    c = np.array([0.5, -0.5])
    G = np.array([[0.1, 0.0], [0.0, 0.2]])
    spec_lbs, open_qis = _zono_spec_lbs_and_open_qis(c, G, queries)
    assert spec_lbs[0] == pytest.approx(0.4)
    assert spec_lbs[1] == pytest.approx(0.3)
    assert open_qis == []


# --- Numerical edge cases -------------------------------------------------

def test_empty_generators_pure_center():
    """G with zero columns: lb = w·c + bias for every query, no spread."""
    queries = [(0, np.array([1.0, -1.0]), 0.5)]
    c = np.array([1.0, 0.0])
    G = np.zeros((2, 0))
    spec_lbs, open_qis = _zono_spec_lbs_and_open_qis(c, G, queries)
    # lb = (1-0) + 0.5 - 0 = 1.5
    assert spec_lbs[0] == pytest.approx(1.5)
    assert open_qis == []


def test_bias_sign_convention_matches_as_linear_queries():
    """The helper must use `min(w·y + bias) > 0` (the docstring of
    `as_linear_queries`), i.e. lb = w·c + bias - |w·G|_1. A `w·c - bias`
    sign error would be silent on bias=0 specs (current benchmarks)
    but would produce wrong-signed lbs on any threshold spec with
    non-zero `value`."""
    # Threshold Y[0] <= 0.4 (unsafe). via as_linear_queries:
    #   w = +e_0, bias = -0.4. Safe iff lo[0] > 0.4 ⇔ lb > 0.
    queries = [(0, np.array([1.0]), -0.4)]
    # Zonotope: y[0] ∈ [0.5 ± 0.05] = [0.45, 0.55]. lo=0.45 > 0.4 → safe.
    c = np.array([0.5])
    G = np.array([[0.05]])
    spec_lbs, _ = _zono_spec_lbs_and_open_qis(c, G, queries)
    # Correct lb: c + bias - |G| = 0.5 + (-0.4) - 0.05 = 0.05 (safe, > 0)
    # Buggy lb:   c - bias - |G| = 0.5 - (-0.4) - 0.05 = 0.85 (still > 0,
    #   but matches the *wrong* numeric value — distinguishable).
    assert spec_lbs[0] == pytest.approx(0.05), (
        f"helper bias sign wrong: got {spec_lbs[0]}, expected 0.05. "
        f"This indicates lb = wc - bias - |wG|_1 (wrong) rather than "
        f"wc + bias - |wG|_1 (correct, matches `min(w·y + bias) > 0`).")


def test_bias_sign_flips_unsafe_witness():
    """Construct a case where the wrong bias sign FLIPS the verdict:
    the zono should be lb < 0 (open) under the correct convention,
    but lb > 0 (incorrectly closed) under the buggy one."""
    # Unsafe: Y[0] >= 10. as_linear_queries: w=-e_0, bias=10.
    queries = [(0, np.array([-1.0]), 10.0)]
    # y[0] ∈ [5 ± 0.5] = [4.5, 5.5]. Safe iff hi[0] < 10 ⇔ 5.5 < 10 ✓
    # Correct lb = -5 + 10 - 0.5 = 4.5 (> 0, safe — disjunct closed).
    # Buggy lb   = -5 - 10 - 0.5 = -15.5 (< 0, falsely open).
    c = np.array([5.0])
    G = np.array([[0.5]])
    spec_lbs, open_qis = _zono_spec_lbs_and_open_qis(c, G, queries)
    assert spec_lbs[0] == pytest.approx(4.5)
    assert open_qis == [], (
        "disjunct should be CLOSED (provably safe). open_qis="
        f"{open_qis} indicates bias-sign bug.")


def test_generators_flattened_reshape():
    """Helper must accept G shaped (n_flat, K) for arbitrary leading
    dim and reshape to (c.size, K) when needed. `_phase1_bab_refine`
    flattens the center but the generator tensor may arrive shaped
    differently."""
    c = np.array([0.5, -0.5])
    G_flat = np.array([0.1, 0.0, 0.0, 0.2]).reshape(-1, 2)  # (2, 2)
    queries = [(0, np.array([1.0, 0.0]), 0.0)]
    spec_lbs, _ = _zono_spec_lbs_and_open_qis(c, G_flat, queries)
    assert spec_lbs[0] == pytest.approx(0.4)


# --- End-to-end through VNNSpec.as_linear_queries -------------------------

def test_cersyve_shaped_spec_via_vnnspec_pipeline():
    """Build a cersyve-pendulum-shaped VNNSpec (1 disjunct, 2 threshold
    constraints) and confirm the helper produces 2 distinct query-
    indexed entries with AND-aggregated open-disjunct semantics."""
    # Unsafe: (Y_0 <= 0) AND (Y_1 >= 0)  ← exact cersyve pendulum spec
    conj = Conjunct([
        Constraint(index=0, op='<=', value=0.0),
        Constraint(index=1, op='>=', value=0.0),
    ])
    spec = VNNSpec(x_lo=np.zeros(2), x_hi=np.ones(2), disjuncts=[conj])
    queries = spec.as_linear_queries(2)
    assert len(queries) == 2
    assert {q[0] for q in queries} == {0}, "both queries share disjunct 0"

    # Zono: Y_0 ∈ [+0.1±0.05]=[0.05,0.15] (NOT ≤ 0 ⇒ q0 safe)
    #       Y_1 ∈ [-0.1±0.05]=[-0.15,-0.05] (NOT ≥ 0 ⇒ q1 safe)
    c = np.array([0.1, -0.1])
    G = np.diag([0.05, 0.05])
    spec_lbs, open_qis = _zono_spec_lbs_and_open_qis(c, G, queries)
    assert set(spec_lbs.keys()) == {0, 1}
    # Either query closes the AND-conjunct via and-semantics.
    assert any(spec_lbs[qi] > 0 for qi in [0, 1])
    assert open_qis == [], "at least one query is safe; disjunct CLOSED"


def test_cersyve_shape_both_queries_open():
    """Both threshold constraints unsafe under the bound box → disjunct
    open → both query indices reported."""
    conj = Conjunct([
        Constraint(index=0, op='<=', value=0.0),
        Constraint(index=1, op='>=', value=0.0),
    ])
    spec = VNNSpec(x_lo=np.zeros(2), x_hi=np.ones(2), disjuncts=[conj])
    queries = spec.as_linear_queries(2)
    # Zono straddles 0 on both outputs.
    c = np.array([0.0, 0.0])
    G = np.diag([0.1, 0.1])
    spec_lbs, open_qis = _zono_spec_lbs_and_open_qis(c, G, queries)
    # q0 (Y_0<=0, w=+e_0, bias=0): lb = 0 + 0 - 0.1 = -0.1  (open)
    # q1 (Y_1>=0, w=-e_1, bias=0): lb = 0 + 0 - 0.1 = -0.1  (open)
    assert spec_lbs[0] == pytest.approx(-0.1)
    assert spec_lbs[1] == pytest.approx(-0.1)
    assert sorted(open_qis) == [0, 1]
