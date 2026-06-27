"""Unit tests for the empty-input-region detector (input_feasibility.py) and its
main.py hook. Soundness contract: a non-empty region is NEVER reported empty."""
import os
import types

import pytest

from vibecheck import input_feasibility as inf
from vibecheck.vnnlib_loader import parse_vnnlib_v2


# --------------------------------------------------------------- spec helpers
def _spec(input_asserts, output_assert='(assert (<= Y[0,0] 0.0))', n_in=2, n_out=1):
    """Minimal nonlinear v2 spec text with the given input/output asserts."""
    L = ['(vnnlib-version <2.0>)',
         '(declare-network f',
         f'    (declare-input X real [1,{n_in}])',
         f'    (declare-output Y real [1,{n_out}])',
         ')']
    L += input_asserts
    L.append(output_assert)
    return '\n'.join(L) + '\n'


def _xref(i):
    return f'X[0,{i}]'


# --------------------------------------------------------------- public API
def test_empty_region_quadratic():
    # x0 in [0,1], x1 in [10,20], constraint 200*x0 >= x1^2  ->  x1^2 - 200 x0 <= 0
    # min x1^2 = 100 > max 200 x0 = 200? 100 < 200 -> feasible. Use x1 in [20,30]:
    # min x1^2 = 400 > 200 -> empty.
    asserts = [
        f'(assert (and (>= {_xref(0)} 0.0) (<= {_xref(0)} 1.0)))',
        f'(assert (and (>= {_xref(1)} 20.0) (<= {_xref(1)} 30.0)))',
        f'(assert (>= (* {_xref(0)} 200.0) (* {_xref(1)} {_xref(1)})))',
    ]
    assert inf.empty_input_region(parse_vnnlib_v2(_spec(asserts))) is True


def test_nonempty_region_quadratic():
    # x0 in [0,100], x1 in [0,5]: x1^2 <= 200 x0 satisfiable (x0=1,x1=0) -> not empty
    asserts = [
        f'(assert (and (>= {_xref(0)} 0.0) (<= {_xref(0)} 100.0)))',
        f'(assert (and (>= {_xref(1)} 0.0) (<= {_xref(1)} 5.0)))',
        f'(assert (>= (* {_xref(0)} 200.0) (* {_xref(1)} {_xref(1)})))',
    ]
    assert inf.empty_input_region(parse_vnnlib_v2(_spec(asserts))) is False


def test_nonempty_plain_box():
    # purely linear box, no nonlinear input atom -> feasible
    asserts = [
        f'(assert (and (>= {_xref(0)} 0.0) (<= {_xref(0)} 1.0)))',
        f'(assert (and (>= {_xref(1)} 0.0) (<= {_xref(1)} 1.0)))',
        f'(assert (>= {_xref(1)} (* {_xref(0)} {_xref(0)})))',   # x1 >= x0^2, feasible
    ]
    assert inf.empty_input_region(parse_vnnlib_v2(_spec(asserts))) is False


def test_empty_via_bilinear():
    # x0,x1 in [2,3]; constraint x0*x1 <= 1  -> min x0*x1 = 4 > 1 -> empty
    asserts = [
        f'(assert (and (>= {_xref(0)} 2.0) (<= {_xref(0)} 3.0)))',
        f'(assert (and (>= {_xref(1)} 2.0) (<= {_xref(1)} 3.0)))',
        f'(assert (<= (* {_xref(0)} {_xref(1)}) 1.0))',
    ]
    assert inf.empty_input_region(parse_vnnlib_v2(_spec(asserts))) is True


def test_empty_linear_box_contradiction():
    # x0 >= 5 and x0 <= 1 -> empty by linear contraction alone
    asserts = [
        f'(assert (>= {_xref(0)} 5.0))',
        f'(assert (<= {_xref(0)} 1.0))',
        f'(assert (and (>= {_xref(1)} 0.0) (<= {_xref(1)} 1.0)))',
    ]
    assert inf.empty_input_region(parse_vnnlib_v2(_spec(asserts))) is True


def test_no_x_constraints_not_empty():
    # only an output constraint references variables; no X-only input atoms -> not empty
    prop = parse_vnnlib_v2(_spec([f'(assert (and (>= {_xref(0)} 0.0) (<= {_xref(0)} 1.0)))'],
                                 output_assert='(assert (<= Y[0,0] 0.0))'))
    # force the n_in path + a clause that has X cons
    assert inf.empty_input_region(prop, n_in=2) is False


def test_empty_region_with_n_in_override():
    asserts = [
        f'(assert (and (>= {_xref(0)} 0.0) (<= {_xref(0)} 1.0)))',
        f'(assert (and (>= {_xref(1)} 20.0) (<= {_xref(1)} 30.0)))',
        f'(assert (>= (* {_xref(0)} 200.0) (* {_xref(1)} {_xref(1)})))',
    ]
    assert inf.empty_input_region(parse_vnnlib_v2(_spec(asserts)), n_in=2) is True


def test_empty_region_zero_vars():
    # a parsed prop whose clauses reference no X vars -> nv==0 path
    fake = types.SimpleNamespace(
        spec=types.SimpleNamespace(clauses=[
            types.SimpleNamespace(constraints=[
                types.SimpleNamespace(terms=[(('Y_0',), 1.0)], bias=0.0, strict=False)])]))
    assert inf.empty_input_region(fake) is False


def test_no_clauses():
    fake = types.SimpleNamespace(spec=types.SimpleNamespace(clauses=[]))
    assert inf.empty_input_region(fake) is False
    assert inf.tighten_input_box(fake) == []


# --------------------------------------------------------------- tightener API
def test_tighten_returns_contracted_box():
    # x0 in [0,20], x1 in [-80,-40] (bounded away from 0), 200*x0 >= x1^2
    asserts = [
        f'(assert (and (>= {_xref(0)} 0.0) (<= {_xref(0)} 20.0)))',
        f'(assert (and (>= {_xref(1)} -80.0) (<= {_xref(1)} -40.0)))',
        f'(assert (>= (* {_xref(0)} 200.0) (* {_xref(1)} {_xref(1)})))',
    ]
    box = inf.tighten_input_box(parse_vnnlib_v2(_spec(asserts)),
                                n_in=2, init_box=[(0.0, 20.0), (-80.0, -40.0)])
    assert box is not None
    # x1^2 <= 200*20 = 4000 -> |x1| <= 63.25 -> lo rises from -80 to ~-63.25
    assert -64.0 < box[1][0] < -63.0 and box[1][1] == -40.0
    # x0 >= x1_min^2/200 = 40^2/200 = 8 -> lo rises from 0 to ~8
    assert box[0][0] > 7.0
    # contained in the declared box (sound)
    assert box[0][0] >= 0.0 and box[1][0] >= -80.0


def test_tighten_empty_returns_none():
    asserts = [
        f'(assert (and (>= {_xref(0)} 0.0) (<= {_xref(0)} 1.0)))',
        f'(assert (and (>= {_xref(1)} 20.0) (<= {_xref(1)} 30.0)))',
        f'(assert (>= (* {_xref(0)} 200.0) (* {_xref(1)} {_xref(1)})))',
    ]
    assert inf.tighten_input_box(parse_vnnlib_v2(_spec(asserts)), n_in=2) is None


def test_tighten_unconstrained_clause_keeps_init():
    # only an output constraint -> the clause has no X-only atoms -> box == init
    spec = _spec([], output_assert='(assert (<= Y[0,0] 0.0))')
    box = inf.tighten_input_box(parse_vnnlib_v2(spec), n_in=2,
                                init_box=[(-1.0, 1.0), (-2.0, 2.0)])
    assert box == [[-1.0, 1.0], [-2.0, 2.0]]


def test_tighten_hull_over_two_clauses():
    # disjunctive output -> two clauses sharing the SAME (conjunctive) input box;
    # hull recovers that box (exercises the hull-merge branch)
    asserts = [f'(assert (and (>= {_xref(0)} 1.0) (<= {_xref(0)} 2.0)))',
               f'(assert (and (>= {_xref(1)} 3.0) (<= {_xref(1)} 4.0)))']
    spec = _spec(asserts, output_assert='(assert (or (<= Y[0,0] 0.0) (>= Y[0,0] 1.0)))')
    box = inf.tighten_input_box(parse_vnnlib_v2(spec), n_in=2)
    assert box[0] == [1.0, 2.0] and box[1] == [3.0, 4.0]


def test_tighten_default_init_box():
    # no init_box -> seeded with +-INF; single-var atoms set finite bounds
    asserts = [f'(assert (and (>= {_xref(0)} 1.0) (<= {_xref(0)} 2.0)))',
               f'(assert (and (>= {_xref(1)} 3.0) (<= {_xref(1)} 4.0)))']
    box = inf.tighten_input_box(parse_vnnlib_v2(_spec(asserts)))
    assert box[0] == [1.0, 2.0] and box[1] == [3.0, 4.0]


def test_augment_tighten_xbox_soundness():
    # _tighten_xbox shrinks but stays within the declared box; None/[] -> no-op
    from vibecheck import nonlinear_augment as na
    asserts = [
        f'(assert (and (>= {_xref(0)} 0.0) (<= {_xref(0)} 20.0)))',
        f'(assert (and (>= {_xref(1)} -80.0) (<= {_xref(1)} -40.0)))',
        f'(assert (>= (* {_xref(0)} 200.0) (* {_xref(1)} {_xref(1)})))',
    ]
    prop = parse_vnnlib_v2(_spec(asserts))
    xbox = {0: [0.0, 20.0], 1: [-80.0, -40.0]}
    out = na._tighten_xbox(prop, xbox, 2)
    assert out[0][0] >= 0.0 and out[0][1] <= 20.0
    assert out[1][0] >= -80.0 and out[1][1] <= -40.0
    assert out[0][0] > 7.0          # genuinely tightened
    # indeterminate (no clauses) -> returns the input xbox unchanged
    fake = types.SimpleNamespace(spec=types.SimpleNamespace(clauses=[]))
    assert na._tighten_xbox(fake, xbox, 2) is xbox


# --------------------------------------------------------------- helper units
def test_mono_interval():
    box = [(-2.0, 3.0), (1.0, 4.0)]
    assert inf._mono_interval(('X_0', 'X_1'), box) == (-8.0, 12.0)
    assert inf._mono_interval(('X_1',), box) == (1.0, 4.0)


def test_decompose_branches():
    box = [(-1.0, 2.0), (0.0, 3.0)]
    # terms: x0^2 (A), x0 (L), x0*x1 (bilinear), x1^2 (remainder), bias
    terms = [(('X_0', 'X_0'), 1.0), (('X_0',), 2.0),
             (('X_0', 'X_1'), 1.0), (('X_1', 'X_1'), 1.0)]
    A, Ll, Lh, Rl, Rh = inf._decompose(terms, 5.0, 0, box)
    assert A == 1.0
    # L = 2 (from x0) + coef*[x1] = [0,3] -> [2, 5]
    assert (Ll, Lh) == (2.0, 5.0)
    # R = bias 5 + x1^2 over [0,3] = [0,9] -> [5,14]
    assert (Rl, Rh) == (5.0, 14.0)
    # bilinear j-selection when xs[1]==i (mono ordered (X_1,X_0))
    A2, Ll2, Lh2, _, _ = inf._decompose([(('X_1', 'X_0'), 1.0)], 0.0, 0, box)
    assert (Ll2, Lh2) == (0.0, 3.0)


def test_decompose_y_term_makes_remainder_unbounded():
    # a Y in a monomial -> remainder spans [-INF, INF] (defensive; public API filters)
    A, Ll, Lh, Rl, Rh = inf._decompose([(('X_0', 'Y_0'), 1.0)], 0.0, 0, [(0.0, 1.0)])
    assert Rl == -inf.INF and Rh == inf.INF


def test_solve_quad_le_branches():
    sq = inf._solve_quad_le
    assert sq(0, 0, 0, 1.0, 0.0) is None             # dlo>dhi
    assert sq(0.0, 0.0, -1.0, 0.0, 5.0) == (0.0, 5.0)  # const <=0 -> full
    assert sq(0.0, 0.0, 1.0, 0.0, 5.0) is None         # const >0 -> empty
    assert sq(0.0, 2.0, -2.0, 0.0, 5.0) == (0.0, 1.0)  # 2x-2<=0 -> x<=1
    assert sq(0.0, -2.0, -2.0, 0.0, 5.0) == (0.0, 5.0) # -2x-2<=0 -> x>=-1 -> [0,5]
    assert sq(1.0, 0.0, 1.0, -5.0, 5.0) is None        # x^2+1<=0 never
    assert sq(1.0, 0.0, -4.0, -5.0, 5.0) == (-2.0, 2.0)  # x^2-4<=0 -> [-2,2]
    assert sq(1.0, 0.0, -4.0, 3.0, 5.0) is None        # roots [-2,2], domain [3,5] -> empty
    assert sq(-1.0, 0.0, -1.0, -3.0, 3.0) == (-3.0, 3.0)  # -x^2-1<=0 always
    # concave -x^2+1<=0 -> |x|>=1: left part only / right part only / both
    assert sq(-1.0, 0.0, 1.0, -5.0, -2.0) == (-5.0, -2.0)   # left only
    assert sq(-1.0, 0.0, 1.0, 2.0, 5.0) == (2.0, 5.0)       # right only
    assert sq(-1.0, 0.0, 1.0, -5.0, 5.0) == (-5.0, 5.0)     # both -> hull
    assert sq(-1.0, 0.0, 1.0, -0.5, 0.5) is None            # strictly inside roots -> empty


def test_contract_var_empty_and_split():
    # x^2 - 4 <= 0 -> [-2,2]
    assert inf._contract_var(1.0, 0.0, 0.0, -4.0, -5.0, 5.0) == (-2.0, 2.0)
    # x^2 + 1 <= 0 -> empty
    assert inf._contract_var(1.0, 0.0, 0.0, 1.0, -5.0, 5.0) is None


# --------------------------------------------------------------- main.py hook
def test_main_hook_empty_emits_unsat(tmp_path):
    from vibecheck import main
    asserts = [
        f'(assert (and (>= {_xref(0)} 0.0) (<= {_xref(0)} 1.0)))',
        f'(assert (and (>= {_xref(1)} 20.0) (<= {_xref(1)} 30.0)))',
        f'(assert (>= (* {_xref(0)} 200.0) (* {_xref(1)} {_xref(1)})))',
    ]
    spec = tmp_path / 'empty.vnnlib'
    spec.write_text(_spec(asserts))
    rf = tmp_path / 'rf.txt'
    args = types.SimpleNamespace(spec=str(spec), results_file=str(rf))
    assert main._maybe_empty_input(args) == 0
    assert rf.read_text().splitlines()[0] == 'unsat'


def test_main_hook_nonempty_returns_none(tmp_path):
    from vibecheck import main
    asserts = [
        f'(assert (and (>= {_xref(0)} 0.0) (<= {_xref(0)} 100.0)))',
        f'(assert (and (>= {_xref(1)} 0.0) (<= {_xref(1)} 5.0)))',
        f'(assert (>= (* {_xref(0)} 200.0) (* {_xref(1)} {_xref(1)})))',
    ]
    spec = tmp_path / 'ne.vnnlib'
    spec.write_text(_spec(asserts))
    args = types.SimpleNamespace(spec=str(spec), results_file=None)
    assert main._maybe_empty_input(args) is None


def test_main_hook_linear_spec_skipped(tmp_path):
    from vibecheck import main
    # a linear v2 spec is not nonlinear -> hook returns None
    spec = tmp_path / 'lin.vnnlib'
    spec.write_text(_spec([f'(assert (and (>= {_xref(0)} 0.0) (<= {_xref(0)} 1.0)))'],
                          output_assert='(assert (<= Y_0 0.0))'))
    args = types.SimpleNamespace(spec=str(spec), results_file=None)
    assert main._maybe_empty_input(args) is None


def test_main_hook_missing_spec_returns_none():
    from vibecheck import main
    args = types.SimpleNamespace(spec='/nonexistent/path.vnnlib', results_file=None)
    assert main._maybe_empty_input(args) is None
