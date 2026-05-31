"""Soundness invariant for the gen-LP / spec-MILP construction.

There was NO test exercising the gen-LP builders (`solve_spec` /
`build_gen_lp_from_state` / `state_from_*`) with a sigmoid/tanh net — every
gen-LP test was pure-ReLU. For pure-ReLU the ReLU `e_new` columns are
consecutive, so the `_unused` gap padding never triggers; the bug only appears
once a sigmoid interleaves γ-slack generator columns between them. That blind
spot is why an unsound `[0,0]` padding shipped.

`test_genlp_relaxation_equals_zonotope` is the functional guard: with NO ReLU
constraints (`unstable_list = []`), the gen-LP's LP relaxation of `qw·y` must
exactly equal the underlying zonotope's range `qw·c ± ||qw·G||_1`. Every
generator column is a noise symbol e ∈ [-1,1]; if the builder pins any column to
[0,0] (the `_unused` padding), the relaxation's `min` is too HIGH — it
under-approximates the zonotope, which lets the spec MILP certify a false UNSAT.
"""
import numpy as np
import scipy.sparse as sp

from vibecheck import verify_gen_lp


def test_genlp_relaxation_equals_zonotope():
    n_input, n_gens = 2, 5
    # ReLU e_new at column 4 -> columns 2,3 are a GAP (would be `_unused` padded),
    # standing in for sigmoid γ-slack columns. Give them nonzero objective weight.
    G = np.array([[1.0, -2.0, 3.0, -1.5, 0.7]])   # (1 output, 5 gens)
    c = np.array([0.5])
    state = {
        'formulation': 'phase1', 'n_input': n_input, 'n_gens': n_gens,
        'unstable_list': [],          # NO relu constraints -> LP == bare zonotope
        'stable_list': [],
        'obj_c_out': c, 'obj_G_out_csr': sp.csr_matrix(G),
        'x_lo': np.zeros(n_input), 'x_hi': np.ones(n_input),
        'gg_ops_ser': [], 'input_name': 'x', 'output_op_name': 'y',
    }
    qw = np.array([1.0])
    m, env, _, _ = verify_gen_lp.build_gen_lp_from_state(
        state, qw, 0.0, milp_set=set(), unsafe_halfspace='none')
    m.setParam('OutputFlag', 0)
    m.optimize()
    lp_min = float(m.ObjVal)

    # True zonotope min of qw·y over e in [-1,1]^n_gens.
    zono_min = float(qw @ c) - float(np.sum(np.abs(qw @ G)))   # 0.5 - 8.2 = -7.7
    env.dispose()

    assert lp_min <= zono_min + 1e-6, (
        f"gen-LP relaxation min {lp_min:.4f} > true zonotope min {zono_min:.4f}: "
        f"the builder pinned some generator columns to [0,0] (the `_unused` gap "
        f"padding), under-approximating the zonotope -> the spec MILP can certify "
        f"a false UNSAT on sigmoid/tanh nets. Generator columns must be free "
        f"e in [-1,1].")


def test_genlp_relaxation_equals_zonotope_alpha_zono():
    """Same invariant for `formulation='alpha_zono'` — the blind spot that
    shipped the dist_shift index4312 false-verify.

    `_build_alpha_zono_lp` defaulted every generator column to [0,0] and only
    opened input symbols + this query's *listed* unstable e_new columns to
    [-1,1]. But `state_from_alpha_zono` reserves e_new columns for unstable
    neurons it skips (no pre-ReLU snapshot — e.g. mnist_concat's encoder /
    generator ReLUs), which are NOT in `unstable_list`. Those columns carry
    real objective weight; fixing them at 0 collapses each neuron's
    parallelogram to its center line — an UNSOUND ReLU enclosure that
    under-approximates the output zonotope. The α-CROWN spec LB (closed-form
    `qw·c - Σ|qw·G|` over ALL columns) then disagreed with the bin-0 LP, and
    binarising the classifier ReLUs cut off a real CEX. Every column is a
    noise symbol e ∈ [-1,1].
    """
    n_input, n_gens = 2, 5
    # unstable_list = [] -> NO relu constraints, pure zonotope. Columns 2,3,4
    # are 'orphan' e_new columns (skipped-unstable reservations) carrying
    # objective weight — the exact shape that false-verified index4312.
    G = np.array([[1.0, -2.0, 3.0, -1.5, 0.7]])
    c = np.array([0.5])
    state = {
        'formulation': 'alpha_zono', 'n_input': n_input, 'n_gens': n_gens,
        'unstable_list': [],
        'stable_list': [],
        'obj_c_out': c, 'obj_G_out_csr': sp.csr_matrix(G),
        'x_lo': np.zeros(n_input), 'x_hi': np.ones(n_input),
        'gg_ops_ser': [], 'input_name': 'x', 'output_op_name': 'y',
    }
    qw = np.array([1.0])
    m, env, _, _ = verify_gen_lp.build_gen_lp_from_state(
        state, qw, 0.0, milp_set=set(), unsafe_halfspace='none')
    m.setParam('OutputFlag', 0)
    m.optimize()
    lp_min = float(m.ObjVal)
    zono_min = float(qw @ c) - float(np.sum(np.abs(qw @ G)))   # 0.5 - 8.2 = -7.7
    env.dispose()

    assert lp_min <= zono_min + 1e-6, (
        f"alpha_zono LP relaxation min {lp_min:.4f} > true zonotope min "
        f"{zono_min:.4f}: orphan e_new columns pinned to [0,0] -> the relaxation "
        f"under-approximates the output zonotope and false-verifies SAT cases "
        f"(dist_shift index4312). Every generator column must be free e in [-1,1].")


def test_optimize_checked_zero_fixed_obj_var_guard():
    """The `optimize_checked` guard (enabled session-wide via conftest) raises
    on a [0,0]-fixed variable that carries nonzero objective weight — the
    universal backstop against the gen-LP orphan-column bug — but does NOT
    flag a [0,0] variable with zero objective weight (a legitimately
    fixed-at-zero intermediate, e.g. a dead-neuron bias output)."""
    import pytest
    try:
        import gurobipy as grb
    except ImportError:
        pytest.skip("gurobipy not installed")
    from vibecheck import gurobi_util
    from vibecheck.gurobi_util import (
        optimize_checked, GurobiZeroFixedObjVar, set_zero_fixed_obj_var_check)

    def _model(obj_on_fixed):
        env = grb.Env(empty=True); env.setParam('OutputFlag', 0); env.start()
        m = grb.Model(env=env)
        free = m.addVar(lb=-1.0, ub=1.0, name='free')
        fixed0 = m.addVar(lb=0.0, ub=0.0, name='fixed0')   # pinned to [0,0]
        # objective weight on the [0,0] var iff obj_on_fixed
        m.setObjective(free + (3.0 if obj_on_fixed else 0.0) * fixed0,
                       grb.GRB.MINIMIZE)
        m.update()
        return m, env

    # Guard is enabled by the autouse conftest fixture for the whole session.
    assert gurobi_util._CHECK_ZERO_FIXED_OBJ_VARS is True

    # [0,0] var WITH objective weight -> raise.
    m, env = _model(obj_on_fixed=True)
    with pytest.raises(GurobiZeroFixedObjVar):
        optimize_checked(m)
    m.dispose(); env.dispose()

    # [0,0] var WITHOUT objective weight -> no false positive.
    m, env = _model(obj_on_fixed=False)
    optimize_checked(m)
    assert m.Status == grb.GRB.OPTIMAL
    m.dispose(); env.dispose()

    # Guard OFF -> the offending model solves without raising.
    prev = set_zero_fixed_obj_var_check(False)
    try:
        m, env = _model(obj_on_fixed=True)
        optimize_checked(m)
        assert m.Status == grb.GRB.OPTIMAL
        m.dispose(); env.dispose()
    finally:
        set_zero_fixed_obj_var_check(prev)
