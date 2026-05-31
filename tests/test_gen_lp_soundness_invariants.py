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
