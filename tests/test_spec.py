"""Tests for spec.py and vnnlib_loader.py."""

import numpy as np
import pytest
from vibecheck.spec import Constraint, PairwiseConstraint, Conjunct, VNNSpec
from vibecheck.vnnlib_loader import parse_vnnlib_text, load_vnnlib


# ---- Constraint ----

def test_constraint_ge_safe():
    c = Constraint(index=0, op='>=', value=5.0)
    assert c.margin(np.array([0.0]), np.array([3.0])) == 2.0  # safe: hi < val

def test_constraint_ge_unsafe():
    c = Constraint(index=0, op='>=', value=5.0)
    assert c.margin(np.array([0.0]), np.array([6.0])) == -1.0  # unsafe: hi >= val

def test_constraint_le_safe():
    c = Constraint(index=0, op='<=', value=2.0)
    assert c.margin(np.array([3.0]), np.array([5.0])) == 1.0  # safe: lo > val

def test_constraint_le_unsafe():
    c = Constraint(index=0, op='<=', value=5.0)
    assert c.margin(np.array([3.0]), np.array([6.0])) == -2.0  # unsafe: lo <= val

def test_constraint_str():
    assert str(Constraint(0, '>=', 3.5)) == 'Y_0 >= 3.5'
    assert str(Constraint(2, '<=', -1.0)) == 'Y_2 <= -1.0'


# ---- PairwiseConstraint ----

def test_pairwise_safe():
    c = PairwiseConstraint(pred=0, comp=1)
    lo = np.array([5.0, 0.0])
    hi = np.array([6.0, 3.0])
    assert c.margin(lo, hi) == 2.0  # lo[0] - hi[1] = 5 - 3

def test_pairwise_unsafe():
    c = PairwiseConstraint(pred=0, comp=1)
    lo = np.array([1.0, 0.0])
    hi = np.array([2.0, 5.0])
    assert c.margin(lo, hi) == -4.0  # lo[0] - hi[1] = 1 - 5

def test_pairwise_str():
    assert str(PairwiseConstraint(0, 1)) == 'Y_1 >= Y_0'


# ---- Conjunct ----

def test_conjunct_margin():
    c1 = Constraint(0, '>=', 5.0)
    c2 = Constraint(1, '<=', 1.0)
    conj = Conjunct([c1, c2])
    lo = np.array([0.0, 3.0])
    hi = np.array([3.0, 4.0])
    # c1 margin: 5.0 - 3.0 = 2.0, c2 margin: 3.0 - 1.0 = 2.0
    assert conj.margin(lo, hi) == 2.0

def test_conjunct_str():
    c = Conjunct([Constraint(0, '>=', 1.0), Constraint(1, '<=', 0.0)])
    assert 'AND' in str(c)


# ---- VNNSpec ----

def test_vnnspec_check_verified():
    spec = VNNSpec(
        x_lo=np.array([0.0]),
        x_hi=np.array([1.0]),
        disjuncts=[Conjunct([Constraint(0, '>=', 10.0)])])
    result, details = spec.check(np.array([0.0]), np.array([5.0]))
    assert result == 'verified'
    assert details['worst_margin'] == 5.0

def test_vnnspec_check_unknown():
    spec = VNNSpec(
        x_lo=np.array([0.0]),
        x_hi=np.array([1.0]),
        disjuncts=[Conjunct([Constraint(0, '>=', 3.0)])])
    result, details = spec.check(np.array([0.0]), np.array([5.0]))
    assert result == 'unknown'
    assert details['worst_margin'] == -2.0

def test_vnnspec_n_constraints():
    spec = VNNSpec(np.zeros(1), np.ones(1), [
        Conjunct([Constraint(0, '>=', 1.0), Constraint(0, '<=', 0.0)]),
        Conjunct([PairwiseConstraint(0, 1)]),
    ])
    assert spec.n_constraints == 3

def test_vnnspec_str_single():
    spec = VNNSpec(np.zeros(2), np.ones(2),
                   [Conjunct([Constraint(0, '>=', 1.0)])])
    s = str(spec)
    assert 'unsafe if' in s

def test_vnnspec_str_multi():
    spec = VNNSpec(np.zeros(2), np.ones(2), [
        Conjunct([Constraint(0, '>=', 1.0)]),
        Conjunct([Constraint(1, '<=', 0.0)]),
    ])
    s = str(spec)
    assert 'disjuncts' in s


# ---- VNNLIB parsing ----

def test_parse_pairwise_ge():
    text = """
    (declare-const X_0 Real)
    (declare-const Y_0 Real)
    (declare-const Y_1 Real)
    (assert (>= X_0 0))
    (assert (<= X_0 1))
    (assert (>= Y_1 Y_0))
    """
    spec = parse_vnnlib_text(text)
    assert len(spec.x_lo) == 1
    assert len(spec.disjuncts) == 1
    c = spec.disjuncts[0].constraints[0]
    assert isinstance(c, PairwiseConstraint)
    assert c.pred == 0 and c.comp == 1

def test_parse_pairwise_le():
    text = """
    (declare-const X_0 Real)
    (assert (>= X_0 -1))
    (assert (<= X_0 1))
    (assert (<= Y_0 Y_1))
    """
    spec = parse_vnnlib_text(text)
    c = spec.disjuncts[0].constraints[0]
    assert isinstance(c, PairwiseConstraint)
    assert c.pred == 0 and c.comp == 1

def test_parse_threshold_ge():
    text = """
    (assert (>= X_0 0))
    (assert (<= X_0 1))
    (assert (>= Y_0 3.5))
    """
    spec = parse_vnnlib_text(text)
    c = spec.disjuncts[0].constraints[0]
    assert isinstance(c, Constraint)
    assert c.op == '>=' and c.value == 3.5

def test_parse_threshold_le():
    text = """
    (assert (>= X_0 0))
    (assert (<= X_0 1))
    (assert (<= Y_0 -1.0))
    """
    spec = parse_vnnlib_text(text)
    c = spec.disjuncts[0].constraints[0]
    assert c.op == '<=' and c.value == -1.0

def test_parse_mixed_thresholds():
    text = """
    (assert (>= X_0 0))
    (assert (<= X_0 1))
    (assert (>= Y_0 1.0))
    (assert (<= Y_1 0.0))
    """
    spec = parse_vnnlib_text(text)
    assert len(spec.disjuncts[0].constraints) == 2

def test_parse_or_and():
    text = """
    (assert (or
        (and (>= X_0 -1) (<= X_0 1) (>= Y_0 100))
    ))
    """
    spec = parse_vnnlib_text(text)
    assert len(spec.x_lo) == 1
    assert spec.x_lo[0] == -1.0
    assert spec.x_hi[0] == 1.0
    assert len(spec.disjuncts) == 1
    c = spec.disjuncts[0].constraints[0]
    assert isinstance(c, Constraint) and c.value == 100.0

def test_parse_or_and_multiple_disjuncts():
    text = """
    (assert (or
        (and (>= X_0 0) (<= X_0 1) (>= Y_0 10))
        (and (>= X_0 0) (<= X_0 1) (<= Y_0 -10))
    ))
    """
    spec = parse_vnnlib_text(text)
    assert len(spec.disjuncts) == 2

def test_parse_x_bounds_fallback_format():
    """X bounds in X_i lo hi format."""
    text = """
    X_0 0.0 1.0
    X_1 -1.0 1.0
    (assert (>= Y_0 0.5))
    """
    spec = parse_vnnlib_text(text)
    np.testing.assert_array_equal(spec.x_lo, [0, -1])
    np.testing.assert_array_equal(spec.x_hi, [1, 1])


def test_parse_or_and_pairwise_in_block():
    """Pairwise constraints inside (or (and ...)) blocks."""
    text = """
    (assert (or
        (and (>= X_0 0) (<= X_0 1) (<= Y_0 Y_1) (>= Y_2 Y_0))
    ))
    """
    spec = parse_vnnlib_text(text)
    assert len(spec.disjuncts) == 1
    assert len(spec.disjuncts[0].constraints) == 2


def test_parse_or_and_top_level_x_bounds():
    """(or (and ...)) with X bounds outside the or block."""
    text = """
    (assert (>= X_0 -1))
    (assert (<= X_0 1))
    (assert (or
        (and (>= Y_0 10))
    ))
    """
    spec = parse_vnnlib_text(text)
    assert spec.x_lo[0] == -1
    assert spec.x_hi[0] == 1


def test_parse_no_input_bounds():
    with pytest.raises(ValueError, match="No input bounds"):
        parse_vnnlib_text("(assert (>= Y_0 1.0))")

def test_parse_no_output_constraints():
    with pytest.raises(ValueError, match="Cannot parse output"):
        parse_vnnlib_text("""
        (assert (>= X_0 0))
        (assert (<= X_0 1))
        """)

def test_load_vnnlib_gz(vnncomp_benchmarks):
    """Test .gz loading with a real small file."""
    spec = load_vnnlib(str(vnncomp_benchmarks /
        "acasxu_2023/vnnlib/prop_2.vnnlib.gz"))
    assert len(spec.x_lo) == 5
    assert spec.n_constraints > 0


def test_load_vnnlib_plain(tmp_path):
    """Test plain text file loading."""
    f = tmp_path / "test.vnnlib"
    f.write_text("""
    (assert (>= X_0 0))
    (assert (<= X_0 1))
    (assert (>= Y_0 3.5))
    """)
    spec = load_vnnlib(str(f))
    assert len(spec.x_lo) == 1


# ---------------------------------------------------------------------------
# Mixed X/Y conjuncts in (or (and ...)) blocks — regression tests for the
# silent-unsoundness bug caught on nn4sys lindex_* benchmarks. The vnnlib
# pattern is:
#     (assert (or
#         (and (>= X_0 a1) (<= X_0 b1) (<= Y_0 c1))
#         (and (>= X_0 a2) (<= X_0 b2) (>= Y_0 c2))
#     ))
# Each conjunct's unsafe region requires BOTH the X subrange AND the Y
# constraint. Pre-fix bugs:
#   1. `_parse_block_x_bounds` overwrote per-block X bounds into a global
#      dict, so the parsed `x_lo/x_hi` was the LAST block's range (not
#      the UNION). On acasxu prop_6 this halved the input box; on
#      nn4sys lindex it picked a tiny subrange.
#   2. Conjunct only stored Y constraints — `spec.check` treated `Y_0 <= c1`
#      as unsafe-for-ANY-x-in-box, giving false-SAT verdicts when the
#      witness's x wasn't in the conjunct's X subrange.
# Both manifest as soundness bugs (`verified` on a real SAT, or `sat`
# from a witness that doesn't actually violate the full conjunct).
# ---------------------------------------------------------------------------


def test_parse_or_and_x_bounds_unioned_across_disjuncts():
    """X bounds across (and ...) blocks must be UNIONed (min lo, max hi),
    not overwritten by the last block. acasxu prop_6 pattern."""
    text = """
    (assert (or
        (and (>= X_0 0.5) (<= X_0 1.0) (>= Y_0 100))
        (and (>= X_0 -1.0) (<= X_0 -0.5) (<= Y_0 -100))
    ))
    """
    spec = parse_vnnlib_text(text)
    # Bounding box must cover both subranges (UNION).
    assert spec.x_lo[0] == -1.0, (
        f'x_lo should be UNION (-1.0), got {spec.x_lo[0]}')
    assert spec.x_hi[0] == 1.0, (
        f'x_hi should be UNION (1.0), got {spec.x_hi[0]}')


def test_parse_or_and_disjunct_stores_x_constraints():
    """Each (and ...) block's X constraints must be stored on the
    Conjunct so a witness check can validate `X in subrange AND Y violates`.
    Pre-fix: disjuncts contained only Y constraints — a witness with X
    outside the subrange but Y satisfying the Y-constraint was wrongly
    flagged as a counterexample."""
    from vibecheck.spec import Conjunct
    text = """
    (assert (or
        (and (>= X_0 0.5) (<= X_0 1.0) (>= Y_0 100))
    ))
    """
    spec = parse_vnnlib_text(text)
    conj = spec.disjuncts[0]
    # X bounds for the conjunct should be accessible (per-disjunct, not
    # just the global bounding box). Expose via a `input_bounds` attribute
    # or `x_lo`/`x_hi` per Conjunct.
    assert hasattr(conj, 'input_lo') and hasattr(conj, 'input_hi'), (
        'Conjunct must store its X subrange (input_lo, input_hi).')
    assert conj.input_lo[0] == 0.5
    assert conj.input_hi[0] == 1.0


def test_witness_outside_x_subrange_is_not_counterexample():
    """A point x outside the conjunct's X subrange should NOT be flagged
    as violating that conjunct, even if y satisfies the conjunct's Y
    constraints. This is the nn4sys lindex_* false-SAT bug."""
    from vibecheck.spec import VNNSpec
    text = """
    (assert (or
        (and (>= X_0 0.5) (<= X_0 1.0) (<= Y_0 0))
    ))
    """
    spec = parse_vnnlib_text(text)
    # Witness: x=0.0 (outside [0.5, 1.0]), y=-10 (would satisfy Y<=0).
    # Conjunct should NOT consider this a counterexample.
    is_ce, _ = spec.check_witness(
        np.array([0.0]), np.array([-10.0]))
    assert not is_ce, (
        'Witness x=0 outside conjunct X-subrange [0.5, 1.0] must NOT '
        'be flagged as a counterexample.')


def test_witness_inside_x_subrange_is_counterexample():
    """A point x inside the conjunct's X subrange AND y violating Y
    constraints IS a counterexample. Mirror to the previous test."""
    text = """
    (assert (or
        (and (>= X_0 0.5) (<= X_0 1.0) (<= Y_0 0))
    ))
    """
    spec = parse_vnnlib_text(text)
    is_ce, _ = spec.check_witness(
        np.array([0.7]), np.array([-10.0]))
    assert is_ce, (
        'Witness x=0.7 (in [0.5, 1.0]) with y=-10 (≤0) IS a counterexample.')
