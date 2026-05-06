"""Unit tests for vibecheck.box_halfspace."""
import numpy as np
import pytest

from vibecheck.box_halfspace import (
    lagrangian_min,
    lagrangian_max,
    tighten_layer,
    tighten_all_layers,
)


def _gurobi_reference(d, c0, a, beta, sense):
    """Solve the same LP with Gurobi for a reference answer.

    sense: 'min' or 'max'.
    """
    import gurobipy as grb
    d = np.asarray(d, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    n = d.size
    env = grb.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = grb.Model(env=env)
    m.setParam('OutputFlag', 0)
    m.setParam('Threads', 1)
    m.setParam('NumericFocus', 3)
    evars = m.addMVar(n, lb=-1.0, ub=1.0)
    m.addConstr(a @ evars <= float(beta))
    if sense == 'min':
        m.setObjective(d @ evars + float(c0), grb.GRB.MINIMIZE)
    else:
        m.setObjective(d @ evars + float(c0), grb.GRB.MAXIMIZE)
    m.optimize()
    status = m.Status
    val = None
    if status == grb.GRB.OPTIMAL:
        val = float(m.ObjVal)
    elif status == grb.GRB.INFEASIBLE:
        val = float('inf') if sense == 'min' else float('-inf')
    m.dispose()
    env.dispose()
    return val, status


def _check_matches_gurobi(d, c0, a, beta, tol=1e-7):
    cf_lo = lagrangian_min(d, c0, a, beta)
    cf_hi = lagrangian_max(d, c0, a, beta)
    g_lo, st_lo = _gurobi_reference(d, c0, a, beta, 'min')
    g_hi, st_hi = _gurobi_reference(d, c0, a, beta, 'max')
    # Both should either be finite and match, or both flag infeasible.
    if np.isfinite(cf_lo) and g_lo is not None and np.isfinite(g_lo):
        assert abs(cf_lo - g_lo) < tol, (
            f"min mismatch: cf={cf_lo} gurobi={g_lo} "
            f"(d={d!r}, a={a!r}, beta={beta!r})")
    if np.isfinite(cf_hi) and g_hi is not None and np.isfinite(g_hi):
        assert abs(cf_hi - g_hi) < tol, (
            f"max mismatch: cf={cf_hi} gurobi={g_hi} "
            f"(d={d!r}, a={a!r}, beta={beta!r})")
    return cf_lo, cf_hi, g_lo, g_hi


def test_halfspace_redundant_yields_box_bounds():
    # If a·e ≤ β is satisfied at the box minimum of a·e (i.e. −||a||_1 ≤ β),
    # the halfspace is redundant and the LP reduces to box min/max of d·e.
    d = np.array([1.0, -2.0, 3.0])
    a = np.array([1.0, 1.0, 1.0])
    beta = 3.0  # ||a||_1 = 3, so box max of a·e = 3 equals beta — redundant.
    _check_matches_gurobi(d, 0.0, a, beta)
    assert abs(lagrangian_min(d, 0.0, a, beta) - (-6.0)) < 1e-12
    assert abs(lagrangian_max(d, 0.0, a, beta) - 6.0) < 1e-12


def test_halfspace_tight_at_box_corner():
    # a = d; a·e ≤ β cuts below the box min for d·e.
    d = np.array([2.0, -1.0, 3.0])
    a = d.copy()
    beta = -2.0  # box min of a·e = -6, so halfspace picks portion with a·e ≤ -2
    cf_lo, cf_hi, g_lo, g_hi = _check_matches_gurobi(d, 0.5, a, beta)
    # Min of d·e over box ∩ {d·e ≤ -2} is just box min = -6 (halfspace contains it).
    assert abs(cf_lo - (0.5 + -6.0)) < 1e-9
    # Max of d·e is beta = -2 (constrained).
    assert abs(cf_hi - (0.5 + -2.0)) < 1e-9


def test_mixed_signs_and_breakpoints():
    rng = np.random.default_rng(42)
    for _ in range(20):
        n = rng.integers(2, 30)
        d = rng.normal(size=n)
        a = rng.normal(size=n)
        # Pick β between the box min and max of a·e so halfspace is non-trivial.
        lo_a = -np.sum(np.abs(a))
        hi_a = np.sum(np.abs(a))
        beta = rng.uniform(lo_a + 0.1 * (hi_a - lo_a),
                           lo_a + 0.9 * (hi_a - lo_a))
        c0 = rng.normal()
        _check_matches_gurobi(d, c0, a, beta, tol=1e-7)


def test_zero_coefficients_in_a_and_d():
    # a_i = 0 or d_i = 0 must be handled (no breakpoint).
    d = np.array([1.0, 0.0, -2.0, 3.0, 0.0])
    a = np.array([0.0, 1.0, -1.0, 0.0, 0.5])
    beta = 0.5
    _check_matches_gurobi(d, 1.0, a, beta, tol=1e-7)


def test_all_zero_d_returns_c0():
    # d is all zero: LP min/max over any non-empty polytope is just c0.
    d = np.zeros(5)
    a = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    beta = 0.0  # feasible (e = 0 is inside)
    assert abs(lagrangian_min(d, 1.7, a, beta) - 1.7) < 1e-12
    assert abs(lagrangian_max(d, 1.7, a, beta) - 1.7) < 1e-12


def test_infeasible_halfspace_returns_inf():
    # min a·e over box = -||a||_1. If beta < -||a||_1, halfspace infeasible.
    d = np.array([1.0, -1.0])
    a = np.array([1.0, 1.0])
    beta = -3.0   # box min of a·e is -2, so beta=-3 is infeasible
    assert lagrangian_min(d, 0.0, a, beta) == float('inf')
    # lagrangian_max calls min on -d / -c0 so also infeasible.
    assert lagrangian_max(d, 0.0, a, beta) == float('-inf')


def test_single_neuron_halfspace_active():
    # One dim. a_1 = 1, constraint e_1 ≤ 0.3. Min d·e = d · -1 if d > 0 else ...
    d = np.array([2.0])
    a = np.array([1.0])
    beta = 0.3
    # min 2e subject to e ∈ [-1,1] ∧ e ≤ 0.3  →  e=-1, val=-2
    assert abs(lagrangian_min(d, 0.0, a, beta) - (-2.0)) < 1e-12
    # max 2e: e=0.3 (halfspace active), val=0.6
    assert abs(lagrangian_max(d, 0.0, a, beta) - 0.6) < 1e-12


def test_tighten_layer_pads_generator_matrix():
    # G_L has fewer columns than the full generator space; tighten_layer
    # must zero-pad to match `a`'s length without changing results.
    c_L = np.array([0.0, 0.1, -0.2])
    # 3 neurons, 2 existing gens
    G_L = np.array([[1.0, 0.0],
                    [0.0, 1.0],
                    [0.5, -0.5]])
    lo = np.array([-1.0, -1.0, -1.0])
    hi = np.array([1.0, 1.0, 1.0])
    a = np.array([1.0, 1.0, 0.5, 0.5])   # 4 total gens
    beta = 0.5
    new_lo, new_hi = tighten_layer(c_L, G_L, lo, hi, a, beta, n_gens=4)
    # All 3 neurons are unstable (lo<0<hi), so all get updated.
    # Check they never loosened.
    assert np.all(new_lo >= lo - 1e-12)
    assert np.all(new_hi <= hi + 1e-12)
    assert np.all(new_lo <= new_hi + 1e-9)


def test_tighten_layer_skips_stable_neurons():
    c_L = np.array([0.5, -0.5, 0.0])
    G_L = np.array([[0.1, 0.0],
                    [0.0, 0.1],
                    [0.0, 0.0]])
    # neuron 0 stable-on (lo=0.4), neuron 1 stable-off (hi=-0.4), neuron 2 unstable
    lo = np.array([0.4, -0.6, -0.1])
    hi = np.array([0.6, -0.4, 0.1])
    a = np.array([1.0, 1.0])
    beta = 0.5
    new_lo, new_hi = tighten_layer(c_L, G_L, lo, hi, a, beta)
    # Stable neurons untouched
    assert new_lo[0] == lo[0] and new_hi[0] == hi[0]
    assert new_lo[1] == lo[1] and new_hi[1] == hi[1]
    # Unstable neuron 2 may tighten (G row is all zero → center only,
    # so it doesn't tighten but must not loosen).
    assert new_lo[2] >= lo[2] - 1e-12
    assert new_hi[2] <= hi[2] + 1e-12


def test_tighten_all_layers_structure():
    # Exercise tighten_all_layers with a trivial 1-layer setup on CPU.
    import torch
    device = torch.device('cpu')
    dtype = torch.float64
    c_out = np.array([0.1])
    # 3 generators
    G_out = np.array([[1.0, -1.0, 0.5]])
    w_q = np.array([1.0])
    b_q = 0.0
    # pre_relu at layer 0 — 2 neurons, 3 generators
    c_L = torch.tensor([0.0, 0.1], dtype=dtype, device=device)
    G_L = torch.tensor([[0.2, -0.2, 0.1],
                        [0.1, 0.3, -0.1]], dtype=dtype, device=device)
    pre_relu = {0: (c_L, G_L)}
    bbr = {0: (np.array([-0.5, -0.6]), np.array([0.4, 0.5]))}
    result, stats = tighten_all_layers(
        pre_relu, c_out, G_out, w_q, b_q, bbr, layers=[0],
        device=device, dtype=dtype)
    assert 0 in result
    lo_n, hi_n = result[0]
    # Sanity: bounds never loosen
    assert np.all(lo_n >= bbr[0][0] - 1e-12)
    assert np.all(hi_n <= bbr[0][1] + 1e-12)
    # Stats populated
    assert 'per_layer' in stats and 0 in stats['per_layer']
    assert stats['per_layer'][0]['un'] == 2
    assert stats['per_layer'][0]['t_xfer'] >= 0.0
    assert stats['per_layer'][0]['t_LP'] >= 0.0


def test_tighten_all_layers_empty_layer():
    # Layer with no unstable neurons should short-circuit cleanly.
    import torch
    device = torch.device('cpu')
    dtype = torch.float64
    c_out = np.array([0.0])
    G_out = np.array([[1.0, 0.5]])
    w_q = np.array([1.0])
    b_q = 0.0
    c_L = torch.tensor([0.5], dtype=dtype, device=device)
    G_L = torch.tensor([[0.1, 0.0]], dtype=dtype, device=device)
    pre_relu = {0: (c_L, G_L)}
    bbr = {0: (np.array([0.4]), np.array([0.6]))}   # stable-on
    result, stats = tighten_all_layers(
        pre_relu, c_out, G_out, w_q, b_q, bbr, layers=[0],
        device=device, dtype=dtype)
    assert stats['per_layer'][0]['un'] == 0
    assert stats['per_layer'][0]['flipped'] == 0
    # Bounds unchanged
    assert np.all(result[0][0] == bbr[0][0])
    assert np.all(result[0][1] == bbr[0][1])


def test_all_a_zero_with_infeasible_beta():
    # When a is all zero, the halfspace is "0 ≤ β". If β < 0, infeasible.
    # gprime0_plus = -β > 0 so we enter the breakpoint loop, but there are
    # no valid breakpoints (a_i = 0 everywhere) — must return +inf.
    d = np.array([1.0, -2.0, 3.0])
    a = np.zeros(3)
    beta = -0.5   # infeasible
    assert lagrangian_min(d, 0.0, a, beta) == float('inf')


def test_tighten_all_layers_pads_when_g_cols_less_than_n_gens():
    # Forward zono at an earlier layer has fewer generators than at the
    # output: tighten_all_layers must zero-pad G rows to match n_gens.
    import torch
    device = torch.device('cpu')
    dtype = torch.float64
    c_out = np.array([0.0])
    # n_gens = 4 at the output
    G_out = np.array([[0.3, -0.2, 0.1, 0.05]])
    w_q = np.array([1.0])
    b_q = 0.0
    # pre-ReLU at layer L has only 2 generators (earlier in the network)
    c_L = torch.tensor([0.0, 0.1], dtype=dtype, device=device)
    G_L_short = torch.tensor([[0.4, -0.3],
                               [0.1, 0.2]], dtype=dtype, device=device)
    pre_relu = {0: (c_L, G_L_short)}
    bbr = {0: (np.array([-0.6, -0.4]), np.array([0.5, 0.5]))}
    result, stats = tighten_all_layers(
        pre_relu, c_out, G_out, w_q, b_q, bbr, layers=[0],
        device=device, dtype=dtype)
    lo_n, hi_n = result[0]
    # Correctness sanity: bounds never loosen
    assert np.all(lo_n >= bbr[0][0] - 1e-12)
    assert np.all(hi_n <= bbr[0][1] + 1e-12)
    assert stats['per_layer'][0]['un'] == 2


def test_large_random_instance_matches_gurobi():
    # Scale test: n=500, random d/a/β. Confirms closed form doesn't drift
    # numerically on realistic sizes (this is the common path).
    rng = np.random.default_rng(7)
    n = 500
    d = rng.normal(size=n)
    a = rng.normal(size=n)
    lo_a = -np.sum(np.abs(a)); hi_a = np.sum(np.abs(a))
    beta = lo_a + 0.5 * (hi_a - lo_a)
    c0 = 0.3
    _check_matches_gurobi(d, c0, a, beta, tol=1e-6)
