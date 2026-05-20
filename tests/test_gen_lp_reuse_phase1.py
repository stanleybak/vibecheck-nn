"""Equality between `precompute_gen_state` (re-forward) and
`state_from_phase1` (reuse Phase-1 zonotope) for spec-LP construction.

Both paths must produce the same LP optimum because:
- They encode the same triangle-relaxation polytope.
- Same input box, same Phase-1 (lo, hi) bounds for ReLU.
- Phase-1 form: y_k = λ·z + μ·(1+e_new) with e_new ∈ [-1, 1] plus
  added y≥0, y≥z constraints.
- Direct form: y_k = e_new ∈ [0, hi_j] with z-e_new ≤ -c (tri_lo) and
  e_new - λ·z ≤ λ·(c-lo) (tri_up).
The first restricts a parallelogram; the second is the standard triangle
LP. Their feasible (z, y) sets coincide, so LP optima match exactly
(modulo Gurobi numerical tolerance).
"""

import numpy as np
import pytest
import onnx
from onnx import helper, TensorProto, numpy_helper
import torch

from vibecheck.network import ComputeGraph
from vibecheck.spec import VNNSpec, Conjunct, PairwiseConstraint
from vibecheck.settings import default_settings
from vibecheck import verify_gen_lp
from vibecheck.gurobi_util import optimize_checked
from vibecheck.verify_graph import (
    _build_reference, _serialize_gg_ops, _forward_zonotope_interleaved,
    _conv_sparse_matrix,
)


def _save(model, tmp_path, name='m.onnx'):
    path = str(tmp_path / name)
    onnx.save(model, path)
    return path


def _init(name, arr):
    return numpy_helper.from_array(arr.astype(np.float32), name)


def _inp(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _out(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _make_fc_relu_fc(tmp_path, seed=7):
    """2 -> FC(4) -> ReLU -> FC(2). Tiny enough to enumerate corners."""
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
            [_inp('X', [1, 2])], [_out('Y', [1, 2])],
            [_init('W1', W1), _init('b1', b1),
             _init('W2', W2), _init('b2', b2)]),
        opset_imports=[helper.make_opsetid('', 13)])
    path = _save(model, tmp_path, 'fc.onnx')
    g = ComputeGraph.from_onnx(path)

    x_lo = np.array([-1.0, -1.0], dtype=np.float64)
    x_hi = np.array([1.0, 1.0], dtype=np.float64)
    spec = VNNSpec(x_lo, x_hi, [Conjunct([PairwiseConstraint(0, 1)])])
    return g, spec


def _make_conv_relu_conv_relu_fc(tmp_path, seed=11):
    """1×4×4 -> Conv(2,3,pad=1) -> ReLU -> Conv(2,3,pad=1) -> ReLU
        -> Flatten -> FC(2). Two ReLU layers => exercises multi-layer
    rec_zono cache + Phase-1 propagation through chained convs.
    """
    rng = np.random.RandomState(seed)
    k1 = rng.randn(2, 1, 3, 3).astype(np.float32) * 0.4
    b_c1 = rng.randn(2).astype(np.float32) * 0.05
    k2 = rng.randn(2, 2, 3, 3).astype(np.float32) * 0.4
    b_c2 = rng.randn(2).astype(np.float32) * 0.05
    W_fc = rng.randn(2, 32).astype(np.float32) * 0.25
    b_fc = rng.randn(2).astype(np.float32) * 0.05

    nodes = [
        helper.make_node('Conv', ['X', 'k1', 'bc1'], ['c1'],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node('Relu', ['c1'], ['r1']),
        helper.make_node('Conv', ['r1', 'k2', 'bc2'], ['c2'],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node('Relu', ['c2'], ['r2']),
        helper.make_node('Flatten', ['r2'], ['f'], axis=1),
        helper.make_node('Gemm', ['f', 'Wfc', 'bfc'], ['Y'], transB=1),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes, 'g',
            [_inp('X', [1, 1, 4, 4])], [_out('Y', [1, 2])],
            [_init('k1', k1), _init('bc1', b_c1),
             _init('k2', k2), _init('bc2', b_c2),
             _init('Wfc', W_fc), _init('bfc', b_fc)]),
        opset_imports=[helper.make_opsetid('', 13)])
    path = _save(model, tmp_path, 'conv.onnx')
    g = ComputeGraph.from_onnx(path)

    x_lo = np.full(16, -0.4, dtype=np.float64)
    x_hi = np.full(16, 0.4, dtype=np.float64)
    spec = VNNSpec(x_lo, x_hi, [Conjunct([PairwiseConstraint(0, 1)])])
    return g, spec


def _run_phase1_capture(graph, spec):
    """Run Phase-1 interleaved zonotope forward and capture
    (z_final, rec_zono, bounds_by_relu, gg, gg_ops_ser, x_lo, x_hi).
    """
    device = torch.device('cpu')
    dtype = torch.float64
    settings = default_settings(
        device='cpu', bits=64, total_timeout=30,
        print_progress=False,
        tighten_formulation='gen_cone',
        tighten_solver='lp',
        zono_impl='dense',  # PatchesZonotope is broken on tiny conv shapes
                            # without the input-shape inference Phase 1
                            # already needs; dense covers the equality test
                            # — same code path inside `state_from_phase1`.
    )
    settings.graph_impl = 'reference'

    gg = graph.gpu_graph(device, dtype)
    gg_ops_ser = _serialize_gg_ops(gg)
    for d in gg_ops_ser:
        if d['type'] == 'conv' and 'W_sp' not in d:
            d['W_sp'] = _conv_sparse_matrix(
                d['kernel_np'], d['in_shape'], d['stride'], d['padding'])

    x_lo = spec.x_lo.astype(np.float64)
    x_hi = spec.x_hi.astype(np.float64)
    xl_g = torch.tensor(x_lo, dtype=dtype, device=device)
    xh_g = torch.tensor(x_hi, dtype=dtype, device=device)

    import time
    t_start = time.perf_counter()
    rec_zono = {}
    sb, bounds_by_relu, z_final, _ = _forward_zonotope_interleaved(
        xl_g, xh_g, gg, gg_ops_ser, x_lo, x_hi,
        _build_reference, 5.0, 1,
        lambda: 30.0 - (time.perf_counter() - t_start),
        device, dtype, settings, rec_zono=rec_zono)

    return (z_final, rec_zono, bounds_by_relu, gg, gg_ops_ser, x_lo, x_hi)


def _solve_lp(state, qw, qb, *, milp_set=None):
    """Build and solve the spec LP from a state dict; return (lb, status)."""
    m, env, _info, _coef = verify_gen_lp.build_gen_lp_from_state(
        state, qw, qb, milp_set=milp_set, n_threads=1)
    m.setParam('TimeLimit', 30.0)
    optimize_checked(m)
    import gurobipy as grb
    status = m.Status
    lb = float(m.ObjBound) if status in (grb.GRB.OPTIMAL,) else None
    val = float(m.ObjVal) if m.SolCount > 0 else None
    m.dispose()
    env.dispose()
    return lb, val, status


@pytest.mark.parametrize('builder,seed', [
    (_make_fc_relu_fc, 7),
    (_make_fc_relu_fc, 21),
    (_make_conv_relu_conv_relu_fc, 11),
    (_make_conv_relu_conv_relu_fc, 33),
])
def test_lp_equality_phase1_vs_precompute(tmp_path, builder, seed):
    """LP optimum from `state_from_phase1` matches `precompute_gen_state`."""
    graph, spec = builder(tmp_path, seed=seed)
    z_final, rec_zono, bbr, gg, gg_ops_ser, x_lo, x_hi = \
        _run_phase1_capture(graph, spec)

    last_name = gg_ops_ser[-1]['name']
    state_old = verify_gen_lp.precompute_gen_state(
        gg_ops_ser, x_lo, x_hi, bbr, gg['input_name'], last_name,
        device='cpu', dtype=torch.float64, formulation='dense')
    state_new = verify_gen_lp.state_from_phase1(
        z_final, rec_zono, x_lo, x_hi, gg_ops_ser,
        gg['input_name'], last_name)

    assert state_old['n_input'] == state_new['n_input']
    # Both states must agree on n_gens (one new gen per unstable, in
    # topological order). If Phase 1's apply_relu and precompute_gen_state
    # disagree on the unstable count at any layer, this will fail.
    assert state_old['n_gens'] == state_new['n_gens'], (
        f'n_gens mismatch: old={state_old["n_gens"]} new={state_new["n_gens"]}')
    assert (len(state_old['unstable_list'])
            == len(state_new['unstable_list']))

    rng = np.random.RandomState(seed + 1)
    for trial in range(3):
        qw = rng.randn(2).astype(np.float64)
        qb = float(rng.randn() * 0.1)
        lb_old, val_old, _ = _solve_lp(state_old, qw, qb)
        lb_new, val_new, _ = _solve_lp(state_new, qw, qb)
        assert lb_old is not None and lb_new is not None
        # Both LPs minimize over the same triangle-relaxation polytope;
        # tolerance accounts for fp + Gurobi simplex slack.
        assert abs(lb_old - lb_new) < 1e-6, (
            f'trial {trial}: LP opt mismatch '
            f'old={lb_old:+.9f} new={lb_new:+.9f}')


def test_milp_equivalence_phase1_vs_precompute(tmp_path):
    """With a non-empty milp_set, the MILP from Phase-1 form must produce
    the same ObjBound as the standard MILP — both encode the exact ReLU
    via big-M binaries on the same neurons.
    """
    graph, spec = _make_fc_relu_fc(tmp_path, seed=42)
    z_final, rec_zono, bbr, gg, gg_ops_ser, x_lo, x_hi = \
        _run_phase1_capture(graph, spec)

    last_name = gg_ops_ser[-1]['name']
    state_old = verify_gen_lp.precompute_gen_state(
        gg_ops_ser, x_lo, x_hi, bbr, gg['input_name'], last_name,
        device='cpu', dtype=torch.float64, formulation='dense')
    state_new = verify_gen_lp.state_from_phase1(
        z_final, rec_zono, x_lo, x_hi, gg_ops_ser,
        gg['input_name'], last_name)

    if not state_old['unstable_list']:
        pytest.skip('no unstable neurons — MILP test trivial')

    # Binarize the first 2 unstable keys (keys are (li, neuron_idx)).
    keys = sorted([(u['layer_idx'], u['neuron_idx'])
                   for u in state_old['unstable_list']])[:2]
    milp_set = set(keys)

    qw = np.array([1.0, -1.0], dtype=np.float64)
    qb = 0.0
    lb_old, _, _ = _solve_lp(state_old, qw, qb, milp_set=milp_set)
    lb_new, _, _ = _solve_lp(state_new, qw, qb, milp_set=milp_set)
    assert lb_old is not None and lb_new is not None
    assert abs(lb_old - lb_new) < 1e-5, (
        f'MILP opt mismatch: old={lb_old:+.9f} new={lb_new:+.9f}')


def test_solve_spec_phase1_state_via_dispatch(tmp_path):
    """`solve_spec(state=...)` accepts a Phase-1 state and produces the
    same verdict as the precompute path on a small instance."""
    graph, spec = _make_fc_relu_fc(tmp_path, seed=3)
    z_final, rec_zono, bbr, gg, gg_ops_ser, x_lo, x_hi = \
        _run_phase1_capture(graph, spec)
    last_name = gg_ops_ser[-1]['name']

    state_new = verify_gen_lp.state_from_phase1(
        z_final, rec_zono, x_lo, x_hi, gg_ops_ser,
        gg['input_name'], last_name)
    state_old = verify_gen_lp.precompute_gen_state(
        gg_ops_ser, x_lo, x_hi, bbr, gg['input_name'], last_name,
        device='cpu', dtype=torch.float64, formulation='dense')

    qw = np.array([0.5, -1.0], dtype=np.float64)
    qb = 0.0
    res_old, _, info_old = verify_gen_lp.solve_spec(
        None, None, None, None, None, None, qw, qb,
        milp_set=None, time_limit=10.0, state=state_old, device='cpu')
    res_new, _, info_new = verify_gen_lp.solve_spec(
        None, None, None, None, None, None, qw, qb,
        milp_set=None, time_limit=10.0, state=state_new, device='cpu')
    assert res_old == res_new
    assert abs(info_old['lb'] - info_new['lb']) < 1e-6


def test_state_from_phase1_empty_unstable(tmp_path):
    """Edge case: rec_zono with no unstable neurons (all-stable network).

    Constraints loop is empty; LP collapses to a pure linear objective
    over `e_in ∈ [-1, 1]^n_input`.
    """
    rng = np.random.RandomState(1)
    # FC -> ReLU -> FC where pre-activation is biased strongly positive,
    # so all ReLU neurons are stable-on (no unstable).
    W1 = rng.randn(3, 2).astype(np.float32) * 0.1
    b1 = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    W2 = rng.randn(2, 3).astype(np.float32) * 0.1
    b2 = np.zeros(2, dtype=np.float32)

    nodes = [
        helper.make_node('Gemm', ['X', 'W1', 'b1'], ['fc1'], transB=1),
        helper.make_node('Relu', ['fc1'], ['r1']),
        helper.make_node('Gemm', ['r1', 'W2', 'b2'], ['Y'], transB=1),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes, 'g',
            [_inp('X', [1, 2])], [_out('Y', [1, 2])],
            [_init('W1', W1), _init('b1', b1),
             _init('W2', W2), _init('b2', b2)]),
        opset_imports=[helper.make_opsetid('', 13)])
    path = _save(model, tmp_path, 'all_stable.onnx')
    graph = ComputeGraph.from_onnx(path)
    x_lo = np.array([-1.0, -1.0], dtype=np.float64)
    x_hi = np.array([1.0, 1.0], dtype=np.float64)
    spec = VNNSpec(x_lo, x_hi, [Conjunct([PairwiseConstraint(0, 1)])])

    z_final, rec_zono, bbr, gg, gg_ops_ser, x_lo, x_hi = \
        _run_phase1_capture(graph, spec)
    state_new = verify_gen_lp.state_from_phase1(
        z_final, rec_zono, x_lo, x_hi, gg_ops_ser,
        gg['input_name'], gg_ops_ser[-1]['name'])
    assert state_new['unstable_list'] == []

    qw = np.array([1.0, -1.0], dtype=np.float64)
    lb_new, _, _ = _solve_lp(state_new, qw, 0.0)
    assert lb_new is not None
