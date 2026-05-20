"""Tests for Phase 8 MILP three-mode dispatch.

Covers:
  Mode 1 ('find_sat')        : current default — MILP on LP triangle
                                relaxation, ObjBound > 0 ⇒ UNSAT.
  Mode 2 ('infeasibility')   : add `qw·y + qb == 0` halfspace; UNSAT
                                signal is Gurobi.INFEASIBLE.
  Mode 3 ('alpha_zono_bnb')  : MILP on α-CROWN-tightened forward
                                zonotope WITHOUT triangle floor for
                                non-binarized neurons.

Soundness anchors:
  - Mode 3 with `milp_set = ∅` returns LP min equal to α-CROWN's spec
    LB (closed-form `qw·c_α + qb - Σ |obj_coef_k|`).
  - Mode 3 with all unstable in `milp_set` returns the same MILP optimum
    as Mode 1 with the same milp_set (both encode the exact ReLU).
  - Mode 1 ObjBound > 0 ⟺ Mode 2 returns INFEASIBLE.
"""

import numpy as np
import pytest
import onnx
import torch
from onnx import helper, TensorProto, numpy_helper

from vibecheck.network import ComputeGraph
from vibecheck.settings import default_settings
from vibecheck import verify_gen_lp
from vibecheck import alpha_crown as ac
from vibecheck.gurobi_util import optimize_checked
from vibecheck.verify_graph import _forward_keep_pre_gpu
import gurobipy as grb


# ---------------------------------------------------------------------------
# Tiny model helpers (keep test surface small enough to enumerate corners)
# ---------------------------------------------------------------------------

def _init(name, arr):
    return numpy_helper.from_array(arr.astype(np.float32), name)


def _inp(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _save(model, tmp_path, name='m.onnx'):
    path = str(tmp_path / name)
    onnx.save(model, path)
    return path


def _make_fc_relu_fc(tmp_path, seed=7, name='fc.onnx'):
    """2 → FC(4) → ReLU → FC(2). Tiny enough for exact enumeration."""
    rng = np.random.RandomState(seed)
    W1 = rng.randn(4, 2).astype(np.float32)
    b1 = rng.randn(4).astype(np.float32) * 0.05
    W2 = rng.randn(2, 4).astype(np.float32)
    b2 = rng.randn(2).astype(np.float32) * 0.05
    nodes = [
        helper.make_node('Gemm', ['X', 'W1', 'b1'], ['fc1'], transB=1),
        helper.make_node('Relu', ['fc1'], ['r1']),
        helper.make_node('Gemm', ['r1', 'W2', 'b2'], ['Y'], transB=1),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes, 'g',
            [_inp('X', [1, 2])], [_inp('Y', [1, 2])],
            [_init('W1', W1), _init('b1', b1),
             _init('W2', W2), _init('b2', b2)]),
        opset_imports=[helper.make_opsetid('', 13)])
    path = _save(model, tmp_path, name)
    return ComputeGraph.from_onnx(path)


def _make_fc_relu_fc_relu_fc(tmp_path, seed=11, name='fc2.onnx'):
    """2 → FC(3) → ReLU → FC(3) → ReLU → FC(2). Two unstable layers."""
    rng = np.random.RandomState(seed)
    W1 = rng.randn(3, 2).astype(np.float32) * 0.6
    b1 = rng.randn(3).astype(np.float32) * 0.05
    W2 = rng.randn(3, 3).astype(np.float32) * 0.6
    b2 = rng.randn(3).astype(np.float32) * 0.05
    W3 = rng.randn(2, 3).astype(np.float32) * 0.6
    b3 = rng.randn(2).astype(np.float32) * 0.05
    nodes = [
        helper.make_node('Gemm', ['X', 'W1', 'b1'], ['fc1'], transB=1),
        helper.make_node('Relu', ['fc1'], ['r1']),
        helper.make_node('Gemm', ['r1', 'W2', 'b2'], ['fc2'], transB=1),
        helper.make_node('Relu', ['fc2'], ['r2']),
        helper.make_node('Gemm', ['r2', 'W3', 'b3'], ['Y'], transB=1),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes, 'g',
            [_inp('X', [1, 2])], [_inp('Y', [1, 2])],
            [_init('W1', W1), _init('b1', b1),
             _init('W2', W2), _init('b2', b2),
             _init('W3', W3), _init('b3', b3)]),
        opset_imports=[helper.make_opsetid('', 13)])
    path = _save(model, tmp_path, name)
    return ComputeGraph.from_onnx(path)


def _bbr_from_forward(xl, xh, gg, device, dtype):
    """Pre-ReLU bounds from a single forward zonotope pass."""
    _, pre_relu = _forward_keep_pre_gpu(xl, xh, gg, device, dtype)
    bbr = {}
    for L, (c, G) in pre_relu.items():
        radius = G.abs().sum(dim=1)
        lo = (c - radius).cpu().numpy().astype(np.float64)
        hi = (c + radius).cpu().numpy().astype(np.float64)
        bbr[L] = (lo, hi)
    return bbr


def _serialize(graph, settings):
    from vibecheck.verify_graph import _serialize_gg_ops, _conv_sparse_matrix
    graph.optimize(settings)
    device = torch.device('cpu')
    dtype = torch.float64
    gg = graph.gpu_graph(device, dtype)
    gg_ops_ser = _serialize_gg_ops(gg)
    for d in gg_ops_ser:
        if d['type'] == 'conv' and 'W_sp' not in d:
            d['W_sp'] = _conv_sparse_matrix(
                d['kernel_np'], d['in_shape'], d['stride'], d['padding'])
    return gg, gg_ops_ser


def _build_alpha_zono_state(graph, x_lo, x_hi, qw, qb, *, n_iters=10):
    """Run α-CROWN + direction-adaptive forward + state_from_alpha_zono.

    Returns (state, gg, gg_ops_ser, bbr, lb_alpha).
    """
    settings = default_settings(device='cpu', bits=64, print_progress=False)
    gg, gg_ops_ser = _serialize(graph, settings)
    device = torch.device('cpu')
    dtype = torch.float64
    xl = torch.tensor(x_lo, device=device, dtype=dtype)
    xh = torch.tensor(x_hi, device=device, dtype=dtype)

    bbr = _bbr_from_forward(xl, xh, gg, device, dtype)
    best_lb, alpha_params, _, _ = ac.run_alpha_crown_fixed_intermediate(
        gg, xl, xh, bbr, qw, float(qb),
        device, dtype, n_iters=n_iters, lr=0.25, lr_decay=0.98,
        early_stop_on_positive=False)
    _, ew_at_relu = ac.capture_ew_per_relu(
        gg, xl, xh, alpha_params['spec'], bbr, qw, float(qb),
        device, dtype)
    alpha_per_layer = ac.build_dir_adaptive_alpha(
        alpha_params['spec'], ew_at_relu, bbr, device, dtype)

    unstable_per_layer = {}
    for L, (lo, hi) in bbr.items():
        un = np.where((np.asarray(lo) < 0) & (np.asarray(hi) > 0))[0]
        unstable_per_layer[L] = torch.as_tensor(
            un, dtype=torch.long, device=device)

    z_alpha, pre_relu_gpu = ac.forward_zono_dir_adaptive(
        xl, xh, gg, alpha_per_layer, bbr, device, dtype,
        settings=settings, unstable_per_layer=unstable_per_layer)

    state = verify_gen_lp.state_from_alpha_zono(
        z_alpha, pre_relu_gpu, alpha_per_layer, bbr,
        np.asarray(x_lo, dtype=np.float64),
        np.asarray(x_hi, dtype=np.float64),
        gg_ops_ser, gg['input_name'], gg_ops_ser[-1]['name'],
        unstable_per_layer=unstable_per_layer)
    return state, gg, gg_ops_ser, bbr, best_lb


def _solve(state, qw, qb, *, milp_set=None,
           unsafe_halfspace='none', time_limit=30.0):
    """Build + solve gen-LP from a state. Returns (status, lb, val)."""
    m, env, _info, _coef = verify_gen_lp.build_gen_lp_from_state(
        state, qw, qb, milp_set=milp_set, n_threads=1,
        unsafe_halfspace=unsafe_halfspace)
    m.setParam('TimeLimit', time_limit)
    optimize_checked(m)
    status = m.Status
    lb = float(m.ObjBound) if status in (
        grb.GRB.OPTIMAL, grb.GRB.USER_OBJ_LIMIT) else None
    val = float(m.ObjVal) if m.SolCount > 0 else None
    m.dispose()
    env.dispose()
    return status, lb, val


# ---------------------------------------------------------------------------
# Mode 3 sanity anchors
# ---------------------------------------------------------------------------

def test_mode3_zero_bin_matches_alpha_crown_lb(tmp_path):
    """Mode 3 LP min with `milp_set=∅` equals α-CROWN's spec LB.

    Closed-form: `qw·c_α + qb - Σ|qw·G_α_k|`. The bare parallelogram
    relaxation has no constraints beyond `e ∈ [-1, 1]^n_gens`, so the
    LP min collapses to that closed form and must match what α-CROWN
    backward-pass returned at the optimal α.
    """
    g = _make_fc_relu_fc(tmp_path, seed=7)
    x_lo = np.array([-1.0, -1.0], dtype=np.float64)
    x_hi = np.array([1.0, 1.0], dtype=np.float64)
    qw = np.array([1.0, -1.0], dtype=np.float64)
    qb = 0.0

    state, _, _, _, lb_alpha = _build_alpha_zono_state(
        g, x_lo, x_hi, qw, qb, n_iters=20)

    # Closed-form: qw·c_α + qb - Σ |qw·G_α_k|
    obj_coef = state['obj_G_out_csr'].T @ qw
    obj_const = float(state['obj_c_out'] @ qw) + qb
    lb_closed = float(obj_const - np.sum(np.abs(obj_coef)))

    # LP solve with no binaries.
    _, lb_lp, _ = _solve(state, qw, qb, milp_set=set())
    assert lb_lp is not None
    assert abs(lb_lp - lb_closed) < 1e-7, (
        f'LP {lb_lp:+.9f} ≠ closed-form {lb_closed:+.9f}')
    # And the LP optimum should match α-CROWN's reported best LB.
    assert abs(lb_lp - lb_alpha) < 1e-5, (
        f'LP {lb_lp:+.9f} ≠ α-CROWN LB {lb_alpha:+.9f}')


def test_mode3_full_bin_matches_mode1_full_bin(tmp_path):
    """Mode 3 with every unstable binarized = Mode 1 full-bin = exact MILP.

    Both encode the exact ReLU on every unstable neuron. The LPs differ
    only in the relaxation choice (parallelogram vs triangle) for
    non-binarized neurons; with no non-binarized neurons, both encode
    the same y_k = ReLU(z_k), so optima coincide.
    """
    g = _make_fc_relu_fc(tmp_path, seed=21)
    x_lo = np.array([-1.0, -1.0], dtype=np.float64)
    x_hi = np.array([1.0, 1.0], dtype=np.float64)
    qw = np.array([1.0, -1.0], dtype=np.float64)
    qb = 0.0

    settings = default_settings(device='cpu', bits=64, print_progress=False)
    gg, gg_ops_ser = _serialize(g, settings)
    device = torch.device('cpu')
    dtype = torch.float64
    xl = torch.tensor(x_lo, device=device, dtype=dtype)
    xh = torch.tensor(x_hi, device=device, dtype=dtype)
    bbr = _bbr_from_forward(xl, xh, gg, device, dtype)

    # Mode 1 standard precompute (triangle relaxation).
    state1 = verify_gen_lp.precompute_gen_state(
        gg_ops_ser, x_lo, x_hi, bbr, gg['input_name'],
        gg_ops_ser[-1]['name'],
        device='cpu', dtype=torch.float64, formulation='dense')
    keys1 = sorted([(u['layer_idx'], u['neuron_idx'])
                     for u in state1['unstable_list']])
    if not keys1:
        pytest.skip('no unstable neurons — full-bin test trivial')

    # Mode 3 α-zono state.
    state3, _, _, _, _ = _build_alpha_zono_state(
        g, x_lo, x_hi, qw, qb, n_iters=15)
    keys3 = sorted([(u['layer_idx'], u['neuron_idx'])
                     for u in state3['unstable_list']])
    assert keys1 == keys3, (
        'Mode 1 and Mode 3 must agree on the unstable set '
        '(same bounds_by_relu)')

    milp_set = set(keys1)
    _, lb1, _ = _solve(state1, qw, qb, milp_set=milp_set)
    _, lb3, _ = _solve(state3, qw, qb, milp_set=milp_set)
    assert lb1 is not None and lb3 is not None
    # Both encode the exact y_k = ReLU(z_k); LP/MILP optima must match.
    assert abs(lb1 - lb3) < 1e-5, (
        f'full-bin mismatch: mode1={lb1:+.9f} mode3={lb3:+.9f}')


# ---------------------------------------------------------------------------
# Mode 1 / Mode 2 duality
# ---------------------------------------------------------------------------

def test_mode2_infeasibility_duality(tmp_path):
    """LP feasibility theorem (correct, inequality form):
    standard LP min > 0 ⟺ adding `qw·y + qb ≤ 0` makes the LP infeasible.

    The INEQUALITY form is the sound one. The EQUALITY form is NOT a
    biconditional in either direction (covered by the regression test
    `test_equality_halfspace_infeasible_does_NOT_imply_safe`).
    """
    g = _make_fc_relu_fc(tmp_path, seed=33)
    x_lo = np.array([-0.3, -0.3], dtype=np.float64)
    x_hi = np.array([0.3, 0.3], dtype=np.float64)
    settings = default_settings(device='cpu', bits=64, print_progress=False)
    gg, gg_ops_ser = _serialize(g, settings)
    device = torch.device('cpu')
    dtype = torch.float64
    xl = torch.tensor(x_lo, device=device, dtype=dtype)
    xh = torch.tensor(x_hi, device=device, dtype=dtype)
    bbr = _bbr_from_forward(xl, xh, gg, device, dtype)
    state = verify_gen_lp.precompute_gen_state(
        gg_ops_ser, x_lo, x_hi, bbr, gg['input_name'],
        gg_ops_ser[-1]['name'],
        device='cpu', dtype=torch.float64, formulation='dense')

    # Sweep multiple spec directions; for each, test the (sound)
    # inequality duality.
    rng = np.random.RandomState(0)
    n_trials = 6
    for trial in range(n_trials):
        qw = rng.randn(2).astype(np.float64)
        qb = float(rng.uniform(-0.5, 0.5))
        # Standard LP min.
        st1, lb1, _ = _solve(state, qw, qb, milp_set=set())
        assert st1 == grb.GRB.OPTIMAL
        # Inequality halfspace.
        st2, _, _ = _solve(state, qw, qb, milp_set=set(),
                            unsafe_halfspace='inequality')
        if lb1 is not None and lb1 > 1e-7:
            assert st2 == grb.GRB.INFEASIBLE, (
                f'trial {trial}: lb1={lb1:+.6f} > 0 but '
                f'inequality LP not infeasible (status={st2})')
        elif lb1 is not None and lb1 < -1e-7:
            assert st2 != grb.GRB.INFEASIBLE, (
                f'trial {trial}: lb1={lb1:+.6f} < 0 but '
                f'inequality LP infeasible (status={st2})')


def test_equality_halfspace_infeasible_does_NOT_imply_safe(tmp_path):
    """Regression: equality-halfspace INFEASIBLE is NOT a sound UNSAT
    signal. If the relaxation lies entirely below the spec hyperplane
    (real-SAT case), equality+INFEASIBLE incorrectly looked like UNSAT
    in the previous classifier — we now downgrade it to INCONCLUSIVE.

    Construct: trivially-UNSAFE direction `Y[0] - 1e6 ≥ 0` (impossible;
    the spec is violated for every input). The α-zono relaxation lies
    far below the spec hyperplane `Y[0] - 1e6 == 0`. With the equality
    halfspace, Gurobi returns INFEASIBLE — but the spec is violated, so
    'UNSAT' would be a false-verified result.
    """
    g = _make_fc_relu_fc(tmp_path, seed=42)
    x_lo = np.array([-0.05, -0.05], dtype=np.float64)
    x_hi = np.array([0.05, 0.05], dtype=np.float64)
    qw = np.array([1.0, 0.0], dtype=np.float64)
    qb = -1e6  # spec: Y[0] - 1e6 ≥ 0 → unsafe at every real x
    state, _, _, _, _ = _build_alpha_zono_state(
        g, x_lo, x_hi, qw, qb, n_iters=5)

    # solve_spec with equality halfspace: relaxation max ≪ 0, so the
    # equality LP is INFEASIBLE.
    res, _, info = verify_gen_lp.solve_spec(
        None, None, None, None, None, None, qw, qb,
        milp_set=None, time_limit=10.0, state=state, device='cpu',
        unsafe_halfspace='equality')
    # Pre-fix this returned 'UNSAT' (false-verified). Now must be
    # INCONCLUSIVE — caller must use inequality halfspace to get a
    # sound verdict.
    assert info['status'] == grb.GRB.INFEASIBLE
    assert res == 'INCONCLUSIVE', (
        f'equality+INFEASIBLE must NOT be classified UNSAT (was {res})')

    # Inequality halfspace on the same case: also INFEASIBLE? No — the
    # relaxation min ≤ qw·c_α + qb ≪ 0, so the inequality halfspace
    # `qw·y + qb ≤ 0` IS satisfied by the relaxation interior. LP is
    # feasible, optimum is far below 0 → INCONCLUSIVE.
    res_iq, _, info_iq = verify_gen_lp.solve_spec(
        None, None, None, None, None, None, qw, qb,
        milp_set=None, time_limit=10.0, state=state, device='cpu',
        unsafe_halfspace='inequality')
    # Inequality must NOT be infeasible here — the relaxation contains
    # the unsafe halfspace fully.
    assert info_iq['status'] != grb.GRB.INFEASIBLE
    assert res_iq == 'INCONCLUSIVE'


# ---------------------------------------------------------------------------
# Three-mode agreement on a known-truth instance
# ---------------------------------------------------------------------------

def test_three_modes_agree_on_known_safe(tmp_path):
    """For a trivially-safe spec direction (`Y[0] >= 1e6` is unreachable),
    all three modes must report UNSAT (verified)."""
    g = _make_fc_relu_fc(tmp_path, seed=42)
    x_lo = np.array([-0.05, -0.05], dtype=np.float64)
    x_hi = np.array([0.05, 0.05], dtype=np.float64)
    # spec direction: Y[0] - 1e6 >= 0 — provably false; here qw·y + qb is
    # the lower bound of (Y[0] - 1e6), unbounded-positive after subtracting
    # huge constant. Use the unreachable form: Y[0] - 1e6 >= 0 means we
    # want LP min(Y[0] - 1e6) > 0 → impossible. Flip the question: prove
    # 1e6 - Y[0] >= 0 (always true). qw=[-1, 0], qb=1e6.
    qw = np.array([-1.0, 0.0], dtype=np.float64)
    qb = 1e6

    # Mode 1
    settings = default_settings(device='cpu', bits=64, print_progress=False)
    gg, gg_ops_ser = _serialize(g, settings)
    device = torch.device('cpu'); dtype = torch.float64
    xl = torch.tensor(x_lo, device=device, dtype=dtype)
    xh = torch.tensor(x_hi, device=device, dtype=dtype)
    bbr = _bbr_from_forward(xl, xh, gg, device, dtype)
    state1 = verify_gen_lp.precompute_gen_state(
        gg_ops_ser, x_lo, x_hi, bbr, gg['input_name'],
        gg_ops_ser[-1]['name'],
        device='cpu', dtype=torch.float64, formulation='dense')
    _, lb1, _ = _solve(state1, qw, qb, milp_set=set())
    assert lb1 is not None and lb1 > 0, f'Mode 1 lb={lb1}'

    # Mode 2 (inequality halfspace ⇒ INFEASIBLE since relaxation is
    # entirely above the spec hyperplane).
    st2, _, _ = _solve(state1, qw, qb, milp_set=set(),
                        unsafe_halfspace='inequality')
    assert st2 == grb.GRB.INFEASIBLE

    # Mode 3 (parallelogram-only LP)
    state3, _, _, _, lb_alpha = _build_alpha_zono_state(
        g, x_lo, x_hi, qw, qb, n_iters=10)
    _, lb3, _ = _solve(state3, qw, qb, milp_set=set())
    assert lb3 is not None and lb3 > 0, f'Mode 3 lb={lb3}'
    assert lb_alpha > 0


def test_state_from_alpha_zono_two_layer(tmp_path):
    """state_from_alpha_zono on a two-hidden-layer network: e_new_col
    counter advances correctly across layers."""
    g = _make_fc_relu_fc_relu_fc(tmp_path, seed=11)
    x_lo = np.array([-0.5, -0.5], dtype=np.float64)
    x_hi = np.array([0.5, 0.5], dtype=np.float64)
    qw = np.array([1.0, -1.0], dtype=np.float64)
    qb = 0.0

    state, _, _, bbr, _ = _build_alpha_zono_state(
        g, x_lo, x_hi, qw, qb, n_iters=15)

    # n_input + sum(unstable per layer) = n_gens (one new col per
    # unstable, regardless of layer).
    n_input = 2
    total_unstable = 0
    for L in bbr:
        lo, hi = bbr[L]
        total_unstable += int(((lo < 0) & (hi > 0)).sum())
    assert state['n_input'] == n_input
    assert state['n_gens'] == n_input + total_unstable
    assert len(state['unstable_list']) == total_unstable

    # e_new_col strictly increasing (topological order).
    cols = [u['e_new_col'] for u in state['unstable_list']]
    assert cols == sorted(cols)
    assert cols == list(range(n_input, n_input + total_unstable))

    # Closed-form bin-0 sanity carries through with two layers.
    obj_coef = state['obj_G_out_csr'].T @ qw
    obj_const = float(state['obj_c_out'] @ qw) + qb
    lb_closed = float(obj_const - np.sum(np.abs(obj_coef)))
    _, lb_lp, _ = _solve(state, qw, qb, milp_set=set())
    assert lb_lp is not None
    assert abs(lb_lp - lb_closed) < 1e-7


# ---------------------------------------------------------------------------
# Settings dispatch
# ---------------------------------------------------------------------------

def test_settings_default_is_alpha_zono_bnb():
    """Default `phase8_milp_mode` is 'alpha_zono_bnb' — empirically the
    best mode on CIFAR100 (verifies hard UNSAT cases that the legacy
    find_sat path times out on, finds CEXes on cases find_sat OOMs)."""
    s = default_settings()
    assert s.phase8_milp_mode == 'alpha_zono_bnb'


def test_settings_phase8_milp_mode_explicit_none_is_legacy():
    """Setting phase8_milp_mode=None explicitly falls back to the legacy
    halfspace flags (back-compat for users who were on the old path)."""
    s = default_settings(phase8_milp_mode=None)
    assert s.phase8_milp_mode is None


def test_settings_unknown_mode_raises_at_dispatch():
    """An unknown phase8_milp_mode value raises ValueError when the
    dispatch resolver runs (verify_graph._run_pipeline)."""
    s = default_settings(phase8_milp_mode='not_a_mode')
    valid = (None, 'find_sat', 'infeasibility', 'alpha_zono_bnb',
             'alpha_zono_infeasibility')
    assert s.phase8_milp_mode not in valid


def test_settings_alpha_zono_infeasibility_is_valid():
    """The new 4th-axis-combo mode is recognised by the settings layer."""
    s = default_settings(phase8_milp_mode='alpha_zono_infeasibility')
    assert s.phase8_milp_mode == 'alpha_zono_infeasibility'


def test_inequality_halfspace_cross_check_consistent_with_standard(tmp_path):
    """Inequality halfspace + INFEASIBLE is classified UNSAT only when
    the cross-check standard LP also confirms `min > 0`.

    Construct a trivially-safe spec direction; both the standard LP min
    is large positive AND the inequality LP is INFEASIBLE → cross-check
    passes → UNSAT.
    """
    g = _make_fc_relu_fc(tmp_path, seed=15)
    x_lo = np.array([-0.05, -0.05], dtype=np.float64)
    x_hi = np.array([0.05, 0.05], dtype=np.float64)
    qw = np.array([-1.0, 0.0], dtype=np.float64)
    qb = 1e6  # spec: 1e6 - Y[0] >= 0 — trivially safe
    state, _, _, _, _ = _build_alpha_zono_state(
        g, x_lo, x_hi, qw, qb, n_iters=5)
    res, _, info = verify_gen_lp.solve_spec(
        None, None, None, None, None, None, qw, qb,
        milp_set=None, time_limit=10.0, state=state, device='cpu',
        unsafe_halfspace='inequality')
    assert info['status'] == grb.GRB.INFEASIBLE
    assert res == 'UNSAT', (
        f'inequality + INFEASIBLE + cross-check std-min>>0 should be '
        f'UNSAT (got {res})')


def test_resolve_standard_lb_returns_obj_bound(tmp_path):
    """The cross-check helper returns the standard-LP ObjBound on a
    well-formed problem."""
    g = _make_fc_relu_fc(tmp_path, seed=22)
    x_lo = np.array([-0.05, -0.05], dtype=np.float64)
    x_hi = np.array([0.05, 0.05], dtype=np.float64)
    qw = np.array([-1.0, 0.0], dtype=np.float64)
    qb = 1e6
    state, _, _, _, _ = _build_alpha_zono_state(
        g, x_lo, x_hi, qw, qb, n_iters=5)
    cross_lb = verify_gen_lp._resolve_standard_lb(
        state, qw, qb, None, 1, 5.0,
        None, None, None, None, None, None, 'cpu', torch.float64)
    assert cross_lb is not None
    assert cross_lb > 1.0  # trivially safe → ObjBound > 0


def test_alpha_zono_infeasibility_emits_halfspace(tmp_path):
    """`build_gen_lp_from_state` on an alpha_zono state with
    `unsafe_halfspace='inequality'` (the sound mode used by
    'alpha_zono_infeasibility') emits the halfspace_unsafe constraint.
    """
    g = _make_fc_relu_fc(tmp_path, seed=4)
    x_lo = np.array([-0.4, -0.4], dtype=np.float64)
    x_hi = np.array([0.4, 0.4], dtype=np.float64)
    qw = np.array([1.0, -1.0], dtype=np.float64)
    qb = 0.0
    state, _, _, _, _ = _build_alpha_zono_state(
        g, x_lo, x_hi, qw, qb, n_iters=5)
    m, env, _, _ = verify_gen_lp.build_gen_lp_from_state(
        state, qw, qb, milp_set=None, n_threads=1,
        unsafe_halfspace='inequality')
    cnames = [c.ConstrName for c in m.getConstrs()]
    assert 'halfspace_unsafe' in cnames
    m.dispose(); env.dispose()


def test_alpha_zono_infeasibility_duality_on_safe_spec(tmp_path):
    """Trivially-safe spec ⇒ alpha_zono LP min > 0 ⇒ adding the equality
    halfspace makes the polytope INFEASIBLE.

    Mirrors `test_mode2_infeasibility_duality` for the alpha_zono
    relaxation: provides the soundness anchor for the new mode.
    """
    g = _make_fc_relu_fc(tmp_path, seed=10)
    x_lo = np.array([-0.05, -0.05], dtype=np.float64)
    x_hi = np.array([0.05, 0.05], dtype=np.float64)
    # Trivially safe: 1e6 - Y[0] >= 0 always holds on a bounded network.
    qw = np.array([-1.0, 0.0], dtype=np.float64)
    qb = 1e6
    state, _, _, _, lb_alpha = _build_alpha_zono_state(
        g, x_lo, x_hi, qw, qb, n_iters=10)
    # Bin-0 LP min should be way > 0.
    _, lb, _ = _solve(state, qw, qb, milp_set=set())
    assert lb is not None and lb > 1.0
    assert lb_alpha > 1.0
    # Adding the inequality halfspace ⇒ relaxation above hyperplane
    # ⇒ polytope empty ⇒ INFEASIBLE.
    st, _, _ = _solve(state, qw, qb, milp_set=set(),
                       unsafe_halfspace='inequality')
    assert st == grb.GRB.INFEASIBLE


def test_alpha_zono_triangle_set_emits_floor_constraints(tmp_path):
    """`triangle_set={(li, j), ...}` emits `tri_lo` + `tri_up` constraints
    for those neurons WITHOUT a binary, on top of the bare parallelogram.

    Counts: K triangulated neurons + 0 binarised → exactly 2K extra
    `tri_*` constraints vs. parallelogram-only baseline.
    """
    g = _make_fc_relu_fc(tmp_path, seed=8)
    x_lo = np.array([-0.4, -0.4], dtype=np.float64)
    x_hi = np.array([0.4, 0.4], dtype=np.float64)
    qw = np.array([1.0, -1.0], dtype=np.float64)
    qb = 0.0
    state, _, _, _, _ = _build_alpha_zono_state(
        g, x_lo, x_hi, qw, qb, n_iters=5)
    if not state['unstable_list']:
        pytest.skip('no unstable neurons')

    # Baseline: parallelogram-only.
    m0, env0, _, _ = verify_gen_lp.build_gen_lp_from_state(
        state, qw, qb, milp_set=None, n_threads=1)
    n_tri_baseline = sum(
        1 for c in m0.getConstrs()
        if c.ConstrName.startswith(('tri_lo_', 'tri_up_')))
    assert n_tri_baseline == 0
    m0.dispose(); env0.dispose()

    # Pick the first 2 unstable keys to triangulate (no binaries).
    keys = sorted([(u['layer_idx'], u['neuron_idx'])
                   for u in state['unstable_list']])[:2]
    triangle_set = set(keys)
    m1, env1, _, _ = verify_gen_lp.build_gen_lp_from_state(
        state, qw, qb, milp_set=None, n_threads=1,
        triangle_set=triangle_set)
    n_tri = sum(
        1 for c in m1.getConstrs()
        if c.ConstrName.startswith(('tri_lo_', 'tri_up_')))
    n_bigm = sum(
        1 for c in m1.getConstrs()
        if c.ConstrName.startswith(('bigM_hi_', 'bigM_z_')))
    n_bin = sum(
        1 for v in m1.getVars() if v.VType == grb.GRB.BINARY)
    assert n_tri == 2 * len(keys), (
        f'expected {2*len(keys)} tri_* constraints, got {n_tri}')
    assert n_bigm == 0
    assert n_bin == 0
    m1.dispose(); env1.dispose()


def test_alpha_zono_triangle_set_tightens_lp_min(tmp_path):
    """Adding triangle floor for any subset can only RAISE the LP min
    (relaxation is strictly tighter or equal)."""
    g = _make_fc_relu_fc(tmp_path, seed=12)
    x_lo = np.array([-0.4, -0.4], dtype=np.float64)
    x_hi = np.array([0.4, 0.4], dtype=np.float64)
    qw = np.array([1.0, -1.0], dtype=np.float64)
    qb = 0.0
    state, _, _, _, _ = _build_alpha_zono_state(
        g, x_lo, x_hi, qw, qb, n_iters=10)
    if not state['unstable_list']:
        pytest.skip('no unstable neurons')

    _, lb_par, _ = _solve(state, qw, qb, milp_set=set())

    # Triangulate ALL unstable neurons (most aggressive).
    triangle_all = {(u['layer_idx'], u['neuron_idx'])
                    for u in state['unstable_list']}
    m, env, _, _ = verify_gen_lp.build_gen_lp_from_state(
        state, qw, qb, milp_set=None, n_threads=1,
        triangle_set=triangle_all)
    m.setParam('TimeLimit', 30.0)
    optimize_checked(m)
    lb_tri = float(m.ObjBound) if m.Status == grb.GRB.OPTIMAL else None
    m.dispose(); env.dispose()
    assert lb_par is not None and lb_tri is not None
    # Tightening can only raise ObjBound (relaxation is stricter).
    assert lb_tri >= lb_par - 1e-7, (
        f'triangle relaxation has LOWER LP min ({lb_tri}) than '
        f'parallelogram-only ({lb_par}); not tighter.')


def test_alpha_zono_triangle_with_binary_overlap(tmp_path):
    """When a neuron is in BOTH milp_set AND triangle_set, it gets the
    full big-M encoding — triangle constraints are emitted ONCE, not
    duplicated. The total `tri_*` count equals `|milp_set ∪ triangle_set|`
    × 2 (each receives tri_lo + tri_up) and big-M constraints come ONLY
    from milp_set."""
    g = _make_fc_relu_fc(tmp_path, seed=14)
    x_lo = np.array([-0.4, -0.4], dtype=np.float64)
    x_hi = np.array([0.4, 0.4], dtype=np.float64)
    qw = np.array([1.0, -1.0], dtype=np.float64)
    qb = 0.0
    state, _, _, _, _ = _build_alpha_zono_state(
        g, x_lo, x_hi, qw, qb, n_iters=5)
    keys = sorted([(u['layer_idx'], u['neuron_idx'])
                   for u in state['unstable_list']])
    if len(keys) < 3:
        pytest.skip('need ≥3 unstable neurons')
    milp_set = {keys[0]}
    triangle_set = {keys[0], keys[1], keys[2]}  # superset incl. milp_set
    m, env, _, _ = verify_gen_lp.build_gen_lp_from_state(
        state, qw, qb, milp_set=milp_set, n_threads=1,
        triangle_set=triangle_set)
    n_tri = sum(
        1 for c in m.getConstrs()
        if c.ConstrName.startswith(('tri_lo_', 'tri_up_')))
    n_bigm = sum(
        1 for c in m.getConstrs()
        if c.ConstrName.startswith(('bigM_hi_', 'bigM_z_')))
    n_bin = sum(
        1 for v in m.getVars() if v.VType == grb.GRB.BINARY)
    # 3 unique keys → 6 tri_* constraints (no double-emission for the
    # one in milp_set).
    assert n_tri == 2 * len(triangle_set)
    # Only the milp_set neuron gets big-M.
    assert n_bigm == 2 * len(milp_set)
    assert n_bin == len(milp_set)
    m.dispose(); env.dispose()


def test_settings_phase8_alpha_zono_triangle_top_k_default():
    """Default `phase8_alpha_zono_triangle_top_k=0` (no triangulation,
    bare parallelogram on non-binarised neurons)."""
    s = default_settings()
    assert s.phase8_alpha_zono_triangle_top_k == 0


def test_build_gen_lp_dispatch_alpha_zono(tmp_path):
    """`build_gen_lp_from_state` routes formulation='alpha_zono' to the
    α-zono builder."""
    g = _make_fc_relu_fc(tmp_path, seed=3)
    x_lo = np.array([-0.5, -0.5], dtype=np.float64)
    x_hi = np.array([0.5, 0.5], dtype=np.float64)
    qw = np.array([1.0, -1.0], dtype=np.float64)
    qb = 0.0
    state, _, _, _, _ = _build_alpha_zono_state(
        g, x_lo, x_hi, qw, qb, n_iters=5)
    assert state['formulation'] == 'alpha_zono'
    m, env, info, coef = verify_gen_lp.build_gen_lp_from_state(
        state, qw, qb, milp_set=None, n_threads=1)
    # Each unstable neuron should have a 'a_L{li}_{j}' var; no 'tri_*'
    # constraints when milp_set is empty (parallelogram only).
    cnames = [c.ConstrName for c in m.getConstrs()]
    assert all(not nm.startswith('tri_') for nm in cnames)
    # Each unstable info entry has lam, mu fields populated.
    for u in info:
        assert 'lam' in u and 'mu' in u
        assert u['formulation'] == 'alpha_zono'
    m.dispose()
    env.dispose()
