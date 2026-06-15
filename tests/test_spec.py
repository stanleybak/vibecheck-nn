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


def test_parse_or_and_trailing_global_y_threshold():
    """Top-level Y asserts beside an (or ...) AND into EVERY disjunct.

    Mirrors lsnc_relu's Lyapunov band: an output disjunction plus a
    trailing `Y_1 in [a,b]` level-set band. Dropping the band enlarges
    the unsafe region -> false-SAT, so the loader must conjoin it.
    """
    text = """
    (assert (>= X_0 -1)) (assert (<= X_0 1))
    (assert (or
        (and (>= Y_0 1e-06))
        (and (<= Y_2 -0.7))
    ))
    (assert (<= Y_1 0.41))
    (assert (>= Y_1 0.35))
    """
    spec = parse_vnnlib_text(text)
    assert len(spec.disjuncts) == 2
    for conj in spec.disjuncts:
        # Each disjunct keeps its own constraint PLUS both band bounds.
        vals = {(c.index, c.op, c.value) for c in conj.constraints
                if isinstance(c, Constraint)}
        assert (1, '<=', 0.41) in vals
        assert (1, '>=', 0.35) in vals
    # disjunct 0 still has its own Y_0 >= 1e-06
    assert any(c.index == 0 and c.op == '>=' for c in spec.disjuncts[0].constraints
               if isinstance(c, Constraint))


def test_parse_or_and_trailing_global_y_pairwise():
    """Trailing top-level PAIRWISE Y assert also conjoins into each disjunct."""
    text = """
    (assert (>= X_0 -1)) (assert (<= X_0 1))
    (assert (or
        (and (>= Y_0 1.0))
        (and (<= Y_2 -1.0))
    ))
    (assert (>= Y_3 Y_4))
    """
    spec = parse_vnnlib_text(text)
    assert len(spec.disjuncts) == 2
    for conj in spec.disjuncts:
        pw = [c for c in conj.constraints if isinstance(c, PairwiseConstraint)]
        assert any(c.comp == 3 and c.pred == 4 for c in pw)

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


def test_per_disjunct_subboxes_uniqued():
    """When multiple disjuncts share the same X subbox, the global
    bounding box still UNIONs over all subranges (so when subranges
    coincide, x_lo/x_hi equals that subrange). Used by the
    `_verify_per_disjunct_subboxes` decomposition."""
    text = """
    (assert (or
        (and (>= X_0 0.1) (<= X_0 0.2) (<= Y_0 -100))
        (and (>= X_0 0.1) (<= X_0 0.2) (>= Y_0 +100))
    ))
    """
    spec = parse_vnnlib_text(text)
    # Both disjuncts have the same X subbox; bounding box = that subbox.
    assert spec.x_lo[0] == pytest.approx(0.1)
    assert spec.x_hi[0] == pytest.approx(0.2)
    # Both disjuncts carry the same X bounds.
    for conj in spec.disjuncts:
        assert conj.input_lo[0] == pytest.approx(0.1)
        assert conj.input_hi[0] == pytest.approx(0.2)


# ===========================================================================
# VNNLIB v2 support (network-header + tensor indexing)
#
# Two layers of tests:
#   (A) Equivalence oracle — a v2 spec and its v1 translation must produce a
#       semantically-identical VNNSpec (the load-bearing correctness gate).
#   (B) Parser/adapter internals — direct coverage of the ported s-expr
#       parser and the VnnlibProperty -> VNNSpec adapter, including every
#       error branch.
# ===========================================================================

from vibecheck.vnnlib_loader import (
    detect_version, parse_vnnlib_v2, _vnnlib_v2_to_spec,
    _tokenize, _parse_sexprs, _as_float, _VarMap, _poly_expr, _atom,
    _to_dnf, _negate, _conjoin_asserts, _parse_shape, _classify_var,
    _adapt_y_constraint, VnnlibParseError, TensorDecl, NetworkDecl,
    PolynomialConstraint, DisjunctiveSpec, VnnlibProperty,
)


def _spec_semantically_equal(s1, s2):
    """Compare two VNNSpecs ignoring the input_lo=None vs ==global-box quirk."""
    if not (np.allclose(s1.x_lo, s2.x_lo) and np.allclose(s1.x_hi, s2.x_hi)):
        return False

    def key(s):
        out = set()
        for c in s.disjuncts:
            cons = frozenset(str(x) for x in c.constraints)
            lo = c.input_lo if c.input_lo is not None else s.x_lo
            hi = c.input_hi if c.input_hi is not None else s.x_hi
            out.add((cons,
                     tuple(np.round(np.asarray(lo, np.float64), 6)),
                     tuple(np.round(np.asarray(hi, np.float64), 6))))
        return out

    return key(s1) == key(s2)


# ---- (A) v1 <-> v2 equivalence ----

def test_v2_equiv_threshold():
    v1 = """
    (declare-const X_0 Real) (declare-const Y_0 Real)
    (assert (>= X_0 0)) (assert (<= X_0 1))
    (assert (>= Y_0 3.5))
    """
    v2 = """
    (vnnlib-version <2.0>)
    (declare-network N (declare-input X float32 [1]) (declare-output Y float32 [1]))
    (assert (>= X[0] 0)) (assert (<= X[0] 1))
    (assert (>= Y[0] 3.5))
    """
    assert _spec_semantically_equal(parse_vnnlib_text(v1), parse_vnnlib_text(v2))


def test_v2_equiv_threshold_le():
    v1 = """
    (declare-const X_0 Real)
    (assert (>= X_0 0)) (assert (<= X_0 1))
    (assert (<= Y_0 -1.0))
    """
    v2 = """
    (declare-network N (declare-input X float32 [1]) (declare-output Y float32 [1]))
    (assert (>= X[0] 0)) (assert (<= X[0] 1))
    (assert (<= Y[0] -1.0))
    """
    s2 = parse_vnnlib_text(v2)
    assert _spec_semantically_equal(parse_vnnlib_text(v1), s2)
    c = s2.disjuncts[0].constraints[0]
    assert isinstance(c, Constraint) and c.op == '<=' and c.value == -1.0


def test_v2_equiv_pairwise():
    v1 = """
    (declare-const X_0 Real)
    (assert (>= X_0 -1)) (assert (<= X_0 1))
    (assert (>= Y_1 Y_0))
    """
    v2 = """
    (declare-network N (declare-input X float32 [1]) (declare-output Y float32 [2]))
    (assert (>= X[0] -1)) (assert (<= X[0] 1))
    (assert (>= Y[1] Y[0]))
    """
    assert _spec_semantically_equal(parse_vnnlib_text(v1), parse_vnnlib_text(v2))


def test_v2_equiv_output_or():
    """Robustness output-OR (relusplitter family): no per-disjunct X box."""
    v1 = """
    (declare-const X_0 Real)
    (assert (>= X_0 -1)) (assert (<= X_0 1))
    (assert (or (and (>= Y_0 Y_1)) (and (>= Y_2 Y_1))))
    """
    v2 = """
    (declare-network N (declare-input X float32 [1]) (declare-output Y float32 [3]))
    (assert (>= X[0] -1)) (assert (<= X[0] 1))
    (assert (or (and (>= Y[0] Y[1])) (and (>= Y[2] Y[1]))))
    """
    s1, s2 = parse_vnnlib_text(v1), parse_vnnlib_text(v2)
    assert _spec_semantically_equal(s1, s2)
    assert len(s2.disjuncts) == 2
    assert all(c.input_lo is None for c in s2.disjuncts)  # output-OR -> no box


def test_v2_equiv_multidim_input_flatten():
    """C-order flatten of a multi-dim input tensor matches v1 flat indices."""
    v1 = """
    (declare-const X_0 Real)
    (assert (<= X_0 1)) (assert (>= X_0 0))
    (assert (<= X_1 1)) (assert (>= X_1 0))
    (assert (<= X_2 1)) (assert (>= X_2 0))
    (assert (<= X_3 1)) (assert (>= X_3 0))
    (assert (>= Y_0 Y_1))
    """
    v2 = """
    (declare-network N (declare-input X float32 [1,2,2]) (declare-output Y float32 [2]))
    (assert (<= X[0,0,0] 1)) (assert (>= X[0,0,0] 0))
    (assert (<= X[0,0,1] 1)) (assert (>= X[0,0,1] 0))
    (assert (<= X[0,1,0] 1)) (assert (>= X[0,1,0] 0))
    (assert (<= X[0,1,1] 1)) (assert (>= X[0,1,1] 0))
    (assert (>= Y[0] Y[1]))
    """
    s1, s2 = parse_vnnlib_text(v1), parse_vnnlib_text(v2)
    assert len(s2.x_lo) == 4
    assert _spec_semantically_equal(s1, s2)


def test_v2_equiv_input_or_per_disjunct_boxes():
    """Input-OR with differing per-disjunct X boxes -> boxes attached (acasxu)."""
    v1 = """
    (declare-const X_0 Real)
    (assert (or
        (and (>= X_0 0.5) (<= X_0 1.0) (>= Y_0 Y_1))
        (and (>= X_0 -1.0) (<= X_0 -0.5) (<= Y_0 Y_1))
    ))
    """
    v2 = """
    (declare-network N (declare-input X float32 [1]) (declare-output Y float32 [2]))
    (assert (or
        (and (>= X[0] 0.5) (<= X[0] 1.0) (>= Y[0] Y[1]))
        (and (>= X[0] -1.0) (<= X[0] -0.5) (<= Y[0] Y[1]))
    ))
    """
    s1, s2 = parse_vnnlib_text(v1), parse_vnnlib_text(v2)
    assert s2.x_lo[0] == pytest.approx(-1.0) and s2.x_hi[0] == pytest.approx(1.0)
    assert all(c.input_lo is not None for c in s2.disjuncts)  # input-OR -> boxes
    assert _spec_semantically_equal(s1, s2)


def test_v2_load_file_matches_v1(tmp_path):
    """End-to-end through load_vnnlib (.vnnlib) for both versions."""
    v1f = tmp_path / "v1.vnnlib"
    v1f.write_text("""
    (declare-const X_0 Real)
    (assert (>= X_0 0)) (assert (<= X_0 1))
    (assert (or (and (>= Y_0 Y_1)) (and (>= Y_2 Y_1))))
    """)
    v2f = tmp_path / "v2.vnnlib"
    v2f.write_text("""
    (vnnlib-version <2.0>)
    (declare-network N (declare-input X float32 [1]) (declare-output Y float32 [3]))
    (assert (>= X[0] 0)) (assert (<= X[0] 1))
    (assert (or (and (>= Y[0] Y[1])) (and (>= Y[2] Y[1]))))
    """)
    assert _spec_semantically_equal(load_vnnlib(str(v1f)), load_vnnlib(str(v2f)))


# ---- detect_version ----

def test_detect_version_v2_marker():
    assert detect_version("(vnnlib-version <2.0>)\n(assert (<= X[0] 1))") == "2.0"

def test_detect_version_declare_network():
    assert detect_version("(declare-network N)") == "2.0"

def test_detect_version_v1():
    assert detect_version("(declare-const X_0 Real)\n(assert (<= X_0 1))") == "1.0"

def test_detect_version_ignores_comments():
    # A '; declare-network ...' comment must NOT trigger v2 detection.
    assert detect_version("; declare-network in a comment\n(declare-const X_0 Real)") == "1.0"


# ---- _tokenize / _parse_sexprs / _as_float ----

def test_tokenize_rejoins_split_brackets():
    assert _tokenize("X[0, 1]") == ["X[0,1]"]

def test_tokenize_rejoins_shape_literal():
    assert _tokenize("[1, 1, 5]") == ["[1,1,5]"]

def test_tokenize_unterminated_bracket():
    with pytest.raises(VnnlibParseError, match="unterminated"):
        _tokenize("X[0,")

def test_parse_sexprs_unbalanced_close():
    with pytest.raises(VnnlibParseError, match="unbalanced"):
        _parse_sexprs([")"])

def test_parse_sexprs_unbalanced_open():
    with pytest.raises(VnnlibParseError, match="unbalanced"):
        _parse_sexprs(["(", "a"])

def test_as_float():
    assert _as_float("1.5") == 1.5
    assert _as_float("foo") is None
    assert _as_float(None) is None


# ---- _VarMap ----

def test_varmap_resolve_scalar_and_indexed():
    vm = _VarMap()
    vm.add(TensorDecl("X", "float32", (2, 3)), is_input=True)
    vm.add(TensorDecl("Y", "float32", ()), is_input=False)
    assert vm.resolve("X[0,0]") == "X_0"
    assert vm.resolve("X[1,2]") == "X_5"   # row-major: 1*3 + 2
    assert vm.resolve("Y") == "Y_0"        # scalar, no index
    assert vm.resolve("Z") is None         # unknown name

def test_varmap_offsets_multiple_tensors():
    vm = _VarMap()
    vm.add(TensorDecl("A", "float32", (2,)), is_input=True)
    vm.add(TensorDecl("B", "float32", (3,)), is_input=True)
    assert vm.resolve("A[1]") == "X_1"
    assert vm.resolve("B[0]") == "X_2"     # offset by A's size

def test_varmap_duplicate_decl():
    vm = _VarMap()
    vm.add(TensorDecl("X", "float32", (1,)), is_input=True)
    with pytest.raises(VnnlibParseError, match="duplicate"):
        vm.add(TensorDecl("X", "float32", (1,)), is_input=True)

def test_varmap_index_rank_mismatch():
    vm = _VarMap()
    vm.add(TensorDecl("X", "float32", (3,)), is_input=True)
    with pytest.raises(VnnlibParseError, match="does not match shape"):
        vm.resolve("X[0,0]")

def test_varmap_index_out_of_bounds():
    vm = _VarMap()
    vm.add(TensorDecl("X", "float32", (3,)), is_input=True)
    with pytest.raises(VnnlibParseError, match="out of bounds"):
        vm.resolve("X[5]")


# ---- _poly_expr ----

def _resolver():
    vm = _VarMap()
    vm.add(TensorDecl("X", "float32", (4,)), is_input=True)
    vm.add(TensorDecl("Y", "float32", (4,)), is_input=False)
    return vm.resolve

def test_poly_expr_constant_and_var():
    r = _resolver()
    assert _poly_expr("2.5", r) == {(): 2.5}
    assert _poly_expr("X[0]", r) == {("X_0",): 1.0}

def test_poly_expr_unknown_var():
    with pytest.raises(VnnlibParseError, match="unknown variable"):
        _poly_expr("Z[0]", _resolver())

def test_poly_expr_empty():
    with pytest.raises(VnnlibParseError, match="empty expression"):
        _poly_expr([], _resolver())

def test_poly_expr_add():
    r = _resolver()
    assert _poly_expr(["+", "X[0]", "X[0]", "3"], r) == {("X_0",): 2.0, (): 3.0}

def test_poly_expr_unary_minus():
    assert _poly_expr(["-", "X[0]"], _resolver()) == {("X_0",): -1.0}

def test_poly_expr_binary_minus():
    r = _resolver()
    assert _poly_expr(["-", "X[0]", "X[1]"], r) == {("X_0",): 1.0, ("X_1",): -1.0}

def test_poly_expr_mul_nonlinear():
    r = _resolver()
    assert _poly_expr(["*", "X[0]", "X[1]"], r) == {("X_0", "X_1"): 1.0}

def test_poly_expr_div_by_constant():
    assert _poly_expr(["/", "X[0]", "2"], _resolver()) == {("X_0",): 0.5}

def test_poly_expr_div_by_variable():
    with pytest.raises(VnnlibParseError, match="division by variable"):
        _poly_expr(["/", "X[0]", "X[1]"], _resolver())

def test_poly_expr_unsupported_operator():
    with pytest.raises(VnnlibParseError, match="unsupported arithmetic"):
        _poly_expr(["%", "X[0]", "2"], _resolver())


# ---- _atom ----

def test_atom_each_operator():
    r = _resolver()
    assert len(_atom("<=", "Y[0]", "1", r)) == 1
    assert len(_atom("<", "Y[0]", "1", r)) == 1
    assert len(_atom(">=", "Y[0]", "1", r)) == 1
    assert len(_atom(">", "Y[0]", "1", r)) == 1
    assert len(_atom("==", "Y[0]", "1", r)[0]) == 2   # two opposing constraints
    assert len(_atom("!=", "Y[0]", "1", r)) == 2      # two disjunctive clauses

def test_atom_strict_flag():
    [[c]] = _atom("<", "Y[0]", "1", _resolver())
    assert c.strict is True

def test_atom_unsupported_operator():
    with pytest.raises(VnnlibParseError, match="unsupported comparison"):
        _atom("~=", "Y[0]", "1", _resolver())


# ---- _to_dnf / _negate ----

def test_to_dnf_infix_atom():
    r = _resolver()
    dnf = _to_dnf(["Y[0]", "<", "Y[1]"], r)
    assert len(dnf) == 1 and dnf[0][0].strict is True

def test_to_dnf_and_or():
    r = _resolver()
    assert len(_to_dnf(["and", ["<=", "Y[0]", "1"], ["<=", "Y[1]", "1"]], r)) == 1
    assert len(_to_dnf(["or", ["<=", "Y[0]", "1"], ["<=", "Y[1]", "1"]], r)) == 2

def test_to_dnf_not():
    # not(Y0 <= 1) == (Y0 > 1)
    [[c]] = _to_dnf(["not", ["<=", "Y[0]", "1"]], _resolver())
    assert c.strict is True

def test_to_dnf_doubled_parens():
    # linearizenn quirk: (<= (0.5 (- Y_0 Y_1)))
    r = _resolver()
    dnf = _to_dnf(["<=", ["0.5", ["-", "Y[0]", "Y[1]"]]], r)
    assert len(dnf) == 1

def test_to_dnf_malformed_comparison():
    with pytest.raises(VnnlibParseError, match="malformed comparison"):
        _to_dnf(["<=", "Y[0]"], _resolver())

def test_to_dnf_unsupported_boolean_op():
    with pytest.raises(VnnlibParseError, match="unsupported boolean"):
        _to_dnf(["xor", "a", "b"], _resolver())

def test_to_dnf_expected_boolean():
    with pytest.raises(VnnlibParseError, match="expected boolean"):
        _to_dnf("Y_0", _resolver())

def test_negate_operators():
    assert _negate(["<=", "Y_0", "1"]) == [">", "Y_0", "1"]
    assert _negate(["not", ["<=", "Y_0", "1"]]) == ["<=", "Y_0", "1"]
    assert _negate(["and", ["<=", "Y_0", "1"], ["<=", "Y_1", "1"]])[0] == "or"
    assert _negate(["or", ["<=", "Y_0", "1"], ["<=", "Y_1", "1"]])[0] == "and"

def test_negate_infix_normalized():
    assert _negate(["Y_0", "<=", "1"]) == [">", "Y_0", "1"]

def test_negate_cannot_negate_nonlist():
    with pytest.raises(VnnlibParseError, match="cannot negate"):
        _negate("foo")

def test_negate_cannot_negate_operator():
    with pytest.raises(VnnlibParseError, match="cannot negate operator"):
        _negate(["xor", "a", "b"])


# ---- _conjoin_asserts (multi-clause + DNF explosion guard) ----

def test_conjoin_asserts_multiclause_and_fastpath():
    r = _resolver()
    # first assert is an OR (2 clauses) -> hits the cross-product branch;
    # second assert is a single clause -> hits the fast-path extend.
    spec = _conjoin_asserts([
        ["or", ["<=", "Y[0]", "1"], ["<=", "Y[1]", "1"]],
        ["<=", "X[0]", "5"],
    ], r)
    assert len(spec.clauses) == 2
    for clause in spec.clauses:
        assert any(c.terms[0][0] == ("X_0",) for c in clause.constraints)

def test_conjoin_asserts_dnf_explosion(monkeypatch):
    import vibecheck.vnnlib_loader as vl
    monkeypatch.setattr(vl, "MAX_DNF_CLAUSES", 2)
    r = _resolver()
    with pytest.raises(VnnlibParseError, match="DNF explosion combining"):
        _conjoin_asserts([
            ["or", ["<=", "Y[0]", "1"], ["<=", "Y[1]", "1"]],
            ["or", ["<=", "Y[2]", "1"], ["<=", "Y[3]", "1"]],
        ], r)

def test_to_dnf_and_explosion(monkeypatch):
    import vibecheck.vnnlib_loader as vl
    monkeypatch.setattr(vl, "MAX_DNF_CLAUSES", 2)
    r = _resolver()
    # Inner ORs stay at 2 clauses (not > cap); the 2x2 and-combine = 4 > cap.
    expr = ["and",
            ["or", ["<=", "Y[0]", "1"], ["<=", "Y[1]", "1"]],
            ["or", ["<=", "Y[0]", "2"], ["<=", "Y[1]", "2"]]]
    with pytest.raises(VnnlibParseError, match="DNF explosion in 'and'"):
        _to_dnf(expr, r)

def test_to_dnf_or_explosion(monkeypatch):
    import vibecheck.vnnlib_loader as vl
    monkeypatch.setattr(vl, "MAX_DNF_CLAUSES", 2)
    r = _resolver()
    expr = ["or", ["<=", "Y[0]", "1"], ["<=", "Y[1]", "1"], ["<=", "Y[2]", "1"]]
    with pytest.raises(VnnlibParseError, match="DNF explosion in 'or'"):
        _to_dnf(expr, r)


# ---- _parse_shape ----

def test_parse_shape_variants():
    assert _parse_shape("[1, 784]") == (1, 784)
    assert _parse_shape("[]") == ()       # scalar

def test_parse_shape_invalid():
    with pytest.raises(VnnlibParseError, match="expected shape literal"):
        _parse_shape("784")


# ---- parse_vnnlib_v2 structural errors / relations ----

def test_parse_vnnlib_v2_version_strip():
    prop = parse_vnnlib_v2(
        "(vnnlib-version <2.0>)\n"
        "(declare-network N (declare-input X float32 [1]) (declare-output Y float32 [1]))\n"
        "(assert (<= X[0] 1)) (assert (>= X[0] 0)) (assert (>= Y[0] 1))")
    assert prop.version == "2.0"
    assert prop.num_inputs == 1 and prop.num_outputs == 1

def test_parse_vnnlib_v2_relations():
    prop = parse_vnnlib_v2(
        "(declare-network N (declare-input X float32 [1]) "
        "(declare-output Y float32 [1]) (isomorphic-to f))\n"
        "(assert (<= X[0] 1)) (assert (>= X[0] 0)) (assert (>= Y[0] 1))")
    assert prop.networks[0].relations == (("isomorphic-to", "f"),)

def test_parse_vnnlib_v2_bad_network_item():
    with pytest.raises(VnnlibParseError, match="bad declare-network item"):
        parse_vnnlib_v2("(declare-network N notalist)")

def test_parse_vnnlib_v2_unsupported_network_item():
    with pytest.raises(VnnlibParseError, match="unsupported declare-network item"):
        parse_vnnlib_v2("(declare-network N (declare-thing X float32 [1]))")

def test_parse_vnnlib_v2_malformed_assert():
    with pytest.raises(VnnlibParseError, match="malformed assert"):
        parse_vnnlib_v2("(declare-network N) (assert)")

def test_parse_vnnlib_v2_unsupported_statement():
    with pytest.raises(VnnlibParseError, match="unsupported v2 statement"):
        parse_vnnlib_v2("(define-fun foo () Real 1.0)")

def test_parse_vnnlib_v2_unexpected_top_level_token():
    with pytest.raises(VnnlibParseError, match="unexpected top-level token"):
        parse_vnnlib_v2("(declare-network N) bareword")


# ---- adapter internals: _classify_var / _adapt_y_constraint ----

def test_classify_var():
    assert _classify_var("X_3") == ("X", 3)
    assert _classify_var("Y_12") == ("Y", 12)

def test_adapt_y_threshold_signs():
    # c>0 -> <= ; c<0 -> >=
    assert _adapt_y_constraint({"Y_0": 1.0}, -3.0) == Constraint(0, '<=', 3.0)
    assert _adapt_y_constraint({"Y_0": -1.0}, 3.0) == Constraint(0, '>=', 3.0)

def test_adapt_y_pairwise_both_orders():
    # negative-coeff var is the comp; positive-coeff var is the pred.
    assert _adapt_y_constraint({"Y_0": -1.0, "Y_1": 1.0}, 0.0) == \
        PairwiseConstraint(pred=1, comp=0)
    assert _adapt_y_constraint({"Y_0": 1.0, "Y_1": -1.0}, 0.0) == \
        PairwiseConstraint(pred=0, comp=1)

def test_adapt_y_unsupported_same_sign():
    with pytest.raises(NotImplementedError, match="unsupported Y output"):
        _adapt_y_constraint({"Y_0": 1.0, "Y_1": 1.0}, 0.0)

def test_adapt_y_unsupported_scaled_pairwise():
    # opposite signs but unequal magnitude -> not a unit pairwise.
    with pytest.raises(NotImplementedError, match="unsupported Y output"):
        _adapt_y_constraint({"Y_0": 1.0, "Y_1": -2.0}, 0.0)

def test_adapt_y_unsupported_nonzero_bias_pairwise():
    with pytest.raises(NotImplementedError, match="unsupported Y output"):
        _adapt_y_constraint({"Y_0": 1.0, "Y_1": -1.0}, 0.5)

def test_adapt_y_unsupported_three_vars():
    with pytest.raises(NotImplementedError, match="unsupported Y output"):
        _adapt_y_constraint({"Y_0": 1.0, "Y_1": 1.0, "Y_2": 1.0}, 0.0)


# ---- adapter internals: _adapt_clause / _vnnlib_v2_to_spec error paths ----

def _v2_to_spec(text):
    return _vnnlib_v2_to_spec(parse_vnnlib_v2(text))

def test_adapter_rejects_nonlinear():
    text = """
    (declare-network N (declare-input X float32 [1]) (declare-output Y float32 [1]))
    (assert (<= X[0] 1)) (assert (>= X[0] 0))
    (assert (<= (* Y[0] Y[0]) 1))
    """
    with pytest.raises(NotImplementedError, match="nonlinear"):
        _v2_to_spec(text)

def test_adapter_rejects_multivar_x():
    text = """
    (declare-network N (declare-input X float32 [2]) (declare-output Y float32 [1]))
    (assert (<= (+ X[0] X[1]) 1))
    (assert (>= Y[0] 1))
    """
    with pytest.raises(NotImplementedError, match="multi-variable X"):
        _v2_to_spec(text)

def test_adapter_rejects_mixed_xy():
    text = """
    (declare-network N (declare-input X float32 [1]) (declare-output Y float32 [1]))
    (assert (<= X[0] 1)) (assert (>= X[0] 0))
    (assert (<= (+ X[0] Y[0]) 1))
    """
    with pytest.raises(NotImplementedError, match="mixed X/Y"):
        _v2_to_spec(text)

def test_adapter_no_input_bounds():
    text = """
    (declare-network N (declare-input X float32 [1]) (declare-output Y float32 [1]))
    (assert (>= Y[0] 1))
    """
    with pytest.raises(ValueError, match="No input bounds"):
        _v2_to_spec(text)

def test_adapter_no_output_constraints():
    text = """
    (declare-network N (declare-input X float32 [1]) (declare-output Y float32 [1]))
    (assert (<= X[0] 1)) (assert (>= X[0] 0))
    """
    with pytest.raises(ValueError, match="no output"):
        _v2_to_spec(text)

def test_adapter_no_disjuncts():
    # Hand-built property with an empty DisjunctiveSpec exercises the guard.
    net = NetworkDecl("N", (TensorDecl("X", "float32", (1,)),),
                      (TensorDecl("Y", "float32", (1,)),))
    prop = VnnlibProperty("2.0", [net], DisjunctiveSpec([]))
    with pytest.raises(ValueError, match="No disjuncts"):
        _vnnlib_v2_to_spec(prop)

def test_adapter_x_intersection_and_missing_bound_fill():
    # X[0] gets two upper bounds (intersect -> min) and only a lower-bound-less
    # X[1] (filled with 0). Exercises the `or 0` / get-default fallbacks.
    text = """
    (declare-network N (declare-input X float32 [2]) (declare-output Y float32 [1]))
    (assert (<= X[0] 1.0)) (assert (<= X[0] 0.5)) (assert (>= X[0] 0.0))
    (assert (<= X[1] 2.0))
    (assert (>= Y[0] 1))
    """
    spec = _v2_to_spec(text)
    assert spec.x_hi[0] == pytest.approx(0.5)   # min of the two upper bounds
    assert spec.x_lo[0] == pytest.approx(0.0)
    assert spec.x_lo[1] == 0.0                  # missing lower bound -> 0

def test_v2_signed_zero_normalized():
    """`make(-1)` yields -0.0 for a zero threshold; the adapter normalizes it
    to +0.0 so the constraint value matches the v1 path (cersyve prop_*)."""
    from vibecheck.vnnlib_loader import _norm_zero
    assert not np.signbit(_norm_zero(-0.0))   # -0.0 normalized to +0.0
    assert _norm_zero(5.0) == 5.0             # nonzero passes through
    text = """
    (declare-network N (declare-input X float32 [1]) (declare-output Y float32 [2]))
    (assert (>= X[0] 0)) (assert (<= X[0] 1))
    (assert (>= Y[1] 0)) (assert (<= Y[0] 0))
    """
    spec = _v2_to_spec(text)
    for c in spec.disjuncts[0].constraints:
        assert not np.signbit(c.value)        # all +0.0
        assert "-0.0" not in str(c)


def test_v2_polynomial_constraint_str_and_is_linear():
    pc = PolynomialConstraint(((("Y_0",), 1.0),), -3.0, strict=False)
    assert "<=" in str(pc)
    assert pc.is_linear is True
    nonlinear = PolynomialConstraint(((("Y_0", "Y_0"), 1.0),), 0.0, strict=True)
    assert "<" in str(nonlinear)
    assert nonlinear.is_linear is False
