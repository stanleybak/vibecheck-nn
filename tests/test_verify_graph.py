"""Tests for vibecheck.verify_graph (graph verification mode)."""

import os
import numpy as np
import onnx
import pytest
import torch
from onnx import helper, TensorProto, numpy_helper

from vibecheck.network import ComputeGraph
from vibecheck.settings import default_settings
from vibecheck.spec import (
    VNNSpec, Conjunct, Constraint, PairwiseConstraint,
)
from vibecheck.verify_graph import (
    verify_graph,
    _build_reference, _build_optimized,
    _run_pipeline, _serialize_gg_ops,
    _BUILDERS, _adaptive_spec_lb,
)


# ---------------------------------------------------------------------------
# Local helpers (minimal ONNX builders)
# ---------------------------------------------------------------------------

def _init(name, arr):
    return numpy_helper.from_array(arr.astype(np.float32), name=name)


def _input_val(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _save_and_load(model, tmp_path, name='m.onnx'):
    path = str(tmp_path / name)
    onnx.save(model, path)
    return ComputeGraph.from_onnx(path)


def _build_identity_skip_model(rng):
    """Simple identity-skip residual block.

    Input(1,2,4,4) -> Conv -> Relu -> Conv + input -> Add -> Relu -> FC -> Y
    """
    k1 = rng.randn(2, 2, 3, 3).astype(np.float32) * 0.3
    b1 = np.zeros(2, dtype=np.float32)
    k2 = rng.randn(2, 2, 3, 3).astype(np.float32) * 0.3
    b2 = np.zeros(2, dtype=np.float32)
    Wo = rng.randn(2, 32).astype(np.float32) * 0.3
    bo = np.zeros(2, dtype=np.float32)
    nodes = [
        helper.make_node('Conv', ['X', 'k1', 'b1'], ['c1'],
                         kernel_shape=[3, 3], strides=[1, 1],
                         pads=[1, 1, 1, 1]),
        helper.make_node('Relu', ['c1'], ['r1']),
        helper.make_node('Conv', ['r1', 'k2', 'b2'], ['c2'],
                         kernel_shape=[3, 3], strides=[1, 1],
                         pads=[1, 1, 1, 1]),
        helper.make_node('Add', ['c2', 'X'], ['add']),
        helper.make_node('Relu', ['add'], ['r2']),
        helper.make_node('Flatten', ['r2'], ['flat'], axis=1),
        helper.make_node('Gemm', ['flat', 'Wo', 'bo'], ['Y'], transB=1),
    ]
    inits = [_init('k1', k1), _init('b1', b1),
             _init('k2', k2), _init('b2', b2),
             _init('Wo', Wo), _init('bo', bo)]
    graph = helper.make_graph(
        nodes, 'id_skip',
        [_input_val('X', [1, 2, 4, 4])],
        [_input_val('Y', [1, 2])],
        inits)
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid('', 13)])


def _build_dead_branch_resnet():
    """Two-branch residual net where branch A is entirely dead.

    Layout:
        X ─┬─ FC_a (b=-1)  → Relu_a (DEAD) → FC_a2 (W=0, b=[10,0])
           │                                       │
           └─ FC_b (b=[1,5]) → Relu_b (ACTIVE) → FC_b2 (I)
                                                     │
                                       Add ─────────┘
                                        │
                                    FC_out (I) → Y

    Input bounds: X ∈ [-0.1, 0.1]^2. Branch A produces values in
    [-1.02, -0.98] → ReLU fully dead → FC_a2 takes all-dead input and
    outputs its bias [10, 0]. Branch B outputs around [1, 5]. Total Y ≈
    [11, 5], so Y[0] > Y[1] is provably true.

    The buggy `_solve_spec_graph_worker` drops the [10, 0] contribution
    (dead-input conv/fc → `out.append(None)`), making the LP believe
    Y ≈ [1, 5] and therefore fail to verify.
    """
    W_a = np.array([[0.1, 0.1], [0.1, 0.1]], dtype=np.float32)
    b_a = np.array([-1.0, -1.0], dtype=np.float32)
    W_a2 = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    b_a2 = np.array([10.0, 0.0], dtype=np.float32)

    W_b = np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float32)
    b_b = np.array([1.0, 5.0], dtype=np.float32)
    W_b2 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    b_b2 = np.array([0.0, 0.0], dtype=np.float32)

    W_out = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    b_out = np.array([0.0, 0.0], dtype=np.float32)

    nodes = [
        helper.make_node('Gemm', ['X', 'Wa', 'ba'], ['ga'], transB=1),
        helper.make_node('Relu', ['ga'], ['ra']),
        helper.make_node('Gemm', ['ra', 'Wa2', 'ba2'], ['ga2'], transB=1),
        helper.make_node('Gemm', ['X', 'Wb', 'bb'], ['gb'], transB=1),
        helper.make_node('Relu', ['gb'], ['rb']),
        helper.make_node('Gemm', ['rb', 'Wb2', 'bb2'], ['gb2'], transB=1),
        helper.make_node('Add', ['ga2', 'gb2'], ['add']),
        helper.make_node('Gemm', ['add', 'Wout', 'bout'], ['Y'], transB=1),
    ]
    inits = [
        _init('Wa', W_a), _init('ba', b_a),
        _init('Wa2', W_a2), _init('ba2', b_a2),
        _init('Wb', W_b), _init('bb', b_b),
        _init('Wb2', W_b2), _init('bb2', b_b2),
        _init('Wout', W_out), _init('bout', b_out),
    ]
    graph = helper.make_graph(
        nodes, 'dead_branch_resnet',
        [_input_val('X', [1, 2])],
        [_input_val('Y', [1, 2])],
        inits)
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid('', 13)])


def _build_dead_branch_model():
    """Two-layer FC where the first layer's bias is very negative, making
    the entire ReLU output dead. The second layer then has all-dead inputs
    but a nonzero bias — Bug #1 regression fixture.

    Input(1,2) -> FC1 (dead after relu) -> FC2 -> Y
    """
    W1 = np.eye(2, dtype=np.float32)
    b1 = np.array([-100.0, -100.0], dtype=np.float32)
    W2 = np.array([[1.0, 0.5], [0.3, -0.7]], dtype=np.float32)
    b2 = np.array([2.0, 4.0], dtype=np.float32)
    nodes = [
        helper.make_node('Gemm', ['X', 'W1', 'b1'], ['g1'], transB=1),
        helper.make_node('Relu', ['g1'], ['r1']),
        helper.make_node('Gemm', ['r1', 'W2', 'b2'], ['Y'], transB=1),
    ]
    inits = [_init('W1', W1), _init('b1', b1),
             _init('W2', W2), _init('b2', b2)]
    graph = helper.make_graph(
        nodes, 'dead_branch',
        [_input_val('X', [1, 2])],
        [_input_val('Y', [1, 2])],
        inits)
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid('', 13)])


def _fixed_spec(n_in, high_threshold):
    """Construct a VNNSpec whose unsafe region (Y[0] >= threshold) is
    unreachable, so the correct result is 'verified'."""
    x = np.zeros(n_in, dtype=np.float32)
    eps = 0.01
    return VNNSpec(
        x_lo=x - eps, x_hi=x + eps,
        disjuncts=[Conjunct(
            [Constraint(index=0, op='>=', value=high_threshold)])])


# ---------------------------------------------------------------------------
# 1. Reference vs optimized equivalence
# ---------------------------------------------------------------------------

def test_reference_optimized_equivalence_small(tmp_path):
    """Both builders on a residual block must give identical verify results."""
    rng = np.random.RandomState(42)
    model = _build_identity_skip_model(rng)
    g = _save_and_load(model, tmp_path, 'id_skip.onnx')
    spec = _fixed_spec(32, 1e6)

    s_ref = default_settings(device='cpu', total_timeout=20,
                              print_progress=False, graph_impl='reference')
    s_opt = default_settings(device='cpu', total_timeout=20,
                              print_progress=False, graph_impl='optimized')
    r_ref, d_ref = verify_graph(g, spec, s_ref)
    r_opt, d_opt = verify_graph(g, spec, s_opt)

    assert r_ref == r_opt == 'verified'
    assert sorted(d_ref['timing'].keys()) == sorted(d_opt['timing'].keys())
    assert d_ref['n_splits'].keys() == d_opt['n_splits'].keys()


def test_reference_optimized_lp_minima_match(tmp_path):
    """Build full LP via each builder on a residual block with unstable
    neurons, optimize the output neuron, and assert the minima match."""
    import gurobipy as grb

    rng = np.random.RandomState(0)
    model = _build_identity_skip_model(rng)
    g = _save_and_load(model, tmp_path, 'id_skip.onnx')
    gg = g.gpu_graph(torch.device('cpu'), torch.float32)
    gg_ops = _serialize_gg_ops(gg)

    # Pick wide input bounds to create unstable neurons
    x_lo = np.full(32, -0.2, dtype=np.float64)
    x_hi = np.full(32, 0.2, dtype=np.float64)

    # Forward through zonotope to get bounds_by_relu
    from vibecheck.verify_zono_bnb import _forward_zonotope_graph
    xl_t = torch.tensor(x_lo, dtype=torch.float32)
    xh_t = torch.tensor(x_hi, dtype=torch.float32)
    sb, _ = _forward_zonotope_graph(
        xl_t, xh_t, gg, torch.device('cpu'), torch.float32)
    bounds_by_relu = {li: (lo.cpu().numpy().astype(np.float64),
                           hi.cpu().numpy().astype(np.float64))
                      for li, (lo, hi) in sb.items()}

    objs_ref = []
    objs_opt = []
    for build_fn, objs in ((_build_reference, objs_ref),
                            (_build_optimized, objs_opt)):
        m, env, op_var_refs, _ = build_fn(
            gg_ops, x_lo, x_hi, bounds_by_relu, gg['input_name'])
        m.setParam('DualReductions', 0)
        last_name = gg_ops[-1]['name']
        last_vars = op_var_refs[last_name]
        for v in last_vars:
            if v is None:
                continue
            m.setObjective(v, grb.GRB.MINIMIZE)
            m.optimize()
            assert m.Status in (grb.GRB.OPTIMAL, grb.GRB.TIME_LIMIT)
            if m.Status == grb.GRB.OPTIMAL:
                objs.append(m.ObjVal)
        m.dispose()
        env.dispose()

    assert len(objs_ref) == len(objs_opt)
    for a, b in zip(objs_ref, objs_opt):
        assert abs(a - b) < 1e-4, f'LP minima differ: ref={a} opt={b}'


# ---------------------------------------------------------------------------
# 2. Result details shape
# ---------------------------------------------------------------------------

def test_result_details_shape_verbose(tmp_path):
    """With print_progress=True, details includes verbose fields."""
    rng = np.random.RandomState(1)
    model = _build_identity_skip_model(rng)
    g = _save_and_load(model, tmp_path, 'id_skip.onnx')
    spec = _fixed_spec(32, 1e6)
    s = default_settings(device='cpu', total_timeout=20,
                         print_progress=True, graph_impl='optimized')
    result, details = verify_graph(g, spec, s)

    assert result == 'verified'
    assert 'phase' in details
    assert 'time' in details
    assert isinstance(details['timing'], dict)
    assert isinstance(details['n_splits'], dict)
    # n_splits keys match relu_names exactly
    gg = g.gpu_graph(torch.device('cpu'), torch.float32)
    assert set(details['n_splits'].keys()) == set(gg['relu_names'])
    for v in details['n_splits'].values():
        assert isinstance(v, int) and v >= 0
    # Verbose-only keys
    assert 'per_layer_timing' in details
    assert 'avg_layer_width' in details
    assert 'build_time_total' in details
    assert set(details['avg_layer_width'].keys()) == set(gg['relu_names'])


def test_result_details_shape_quiet(tmp_path):
    """With print_progress=False, verbose fields are absent."""
    rng = np.random.RandomState(2)
    model = _build_identity_skip_model(rng)
    g = _save_and_load(model, tmp_path, 'id_skip.onnx')
    spec = _fixed_spec(32, 1e6)
    s = default_settings(device='cpu', total_timeout=20,
                         print_progress=False, graph_impl='reference')
    result, details = verify_graph(g, spec, s)

    assert 'per_layer_timing' not in details
    assert 'avg_layer_width' not in details
    assert 'build_time_total' not in details
    assert 'timing' in details
    assert 'n_splits' in details


# ---------------------------------------------------------------------------
# 3. Bug #1: dead branch bias preserved
# ---------------------------------------------------------------------------

def test_dead_branch_bias_preserved_reference(tmp_path):
    """Reference builder preserves bias for all-dead conv/fc outputs."""
    _check_dead_branch_bias(tmp_path, _build_reference, 'ref.onnx')


def test_dead_branch_bias_preserved_optimized(tmp_path):
    """Optimized builder preserves bias for all-dead conv/fc outputs."""
    _check_dead_branch_bias(tmp_path, _build_optimized, 'opt.onnx')


def _check_dead_branch_bias(tmp_path, build_fn, filename):
    """Construct a model where FC1+ReLU is entirely dead, so FC2 sees
    all-dead inputs. FC2's output should equal its bias in the LP."""
    import gurobipy as grb

    model = _build_dead_branch_model()
    g = _save_and_load(model, tmp_path, filename)
    gg = g.gpu_graph(torch.device('cpu'), torch.float32)
    gg_ops = _serialize_gg_ops(gg)

    x_lo = np.array([0.0, 0.0], dtype=np.float64)
    x_hi = np.array([1.0, 1.0], dtype=np.float64)

    # bounds_by_relu: r1 is fully dead (hi <= 0)
    bounds_by_relu = {
        0: (np.array([-100.0, -100.0]), np.array([-99.0, -99.0])),
    }

    m, env, op_var_refs, _ = build_fn(
        gg_ops, x_lo, x_hi, bounds_by_relu, gg['input_name'])
    m.setParam('DualReductions', 0)

    last_name = gg_ops[-1]['name']
    last_vars = op_var_refs[last_name]
    assert last_vars[0] is not None, 'Bug #1: var must not be None'
    assert last_vars[1] is not None, 'Bug #1: var must not be None'

    # Minimize Y[0] — should be exactly bias[0] = 2.0
    m.setObjective(last_vars[0], grb.GRB.MINIMIZE)
    m.optimize()
    assert m.Status == grb.GRB.OPTIMAL
    assert abs(m.ObjVal - 2.0) < 1e-6, f'Y[0] min = {m.ObjVal}, want 2.0'

    m.setObjective(last_vars[1], grb.GRB.MINIMIZE)
    m.optimize()
    assert m.Status == grb.GRB.OPTIMAL
    assert abs(m.ObjVal - 4.0) < 1e-6, f'Y[1] min = {m.ObjVal}, want 4.0'

    m.dispose()
    env.dispose()


# ---------------------------------------------------------------------------
# 4. Three-way racing: SAT must not become verified
# ---------------------------------------------------------------------------

def test_racing_sat_not_verified(tmp_path):
    """Even with very short timeout, a SAT instance must never verify."""
    rng = np.random.RandomState(3)
    model = _build_identity_skip_model(rng)
    g = _save_and_load(model, tmp_path, 'id_skip.onnx')
    # A pairwise constraint Y[0] > Y[1] with wide input bounds — likely SAT
    x = np.zeros(32, dtype=np.float32)
    eps = 0.5
    spec = VNNSpec(
        x_lo=x - eps, x_hi=x + eps,
        disjuncts=[Conjunct([PairwiseConstraint(pred=0, comp=1)])])
    s = default_settings(device='cpu', total_timeout=1.0,
                         print_progress=False, graph_impl='optimized')
    result, _ = verify_graph(g, spec, s)
    assert result in ('sat', 'unknown'), (
        f'got {result} — must never be verified on a likely-SAT instance')


# ---------------------------------------------------------------------------
# 5. main.py --mode choices
# ---------------------------------------------------------------------------

def test_main_mode_choices_include_graph():
    """The 'graph' mode is registered in the CLI parser."""
    from vibecheck import main as main_module
    import argparse
    import io
    import contextlib
    # argparse's parser object isn't exposed publicly; capture --help text
    buf = io.StringIO()
    parser = argparse.ArgumentParser()
    # Re-parse from the module's source to find the choices — simpler:
    # just invoke main() with --help in a subprocess would be slow.
    # Instead, check that verify_graph is importable from main's namespace.
    assert hasattr(main_module, 'main')
    # Run the parser creation logic by calling main() with a helper that
    # inspects argv — easiest: read the source and check 'graph' is a choice.
    src = open(main_module.__file__).read()
    assert "'graph'" in src and "choices=['zonotope', 'bnb', 'milp', 'graph']" in src


# ---------------------------------------------------------------------------
# 6. CIFAR100 flagship
# ---------------------------------------------------------------------------

_CIFAR_ONNX = ('/home/stan/repositories/vnncomp2025_benchmarks/benchmarks/'
               'cifar100_2024/onnx/CIFAR100_resnet_medium.onnx.gz')
_CIFAR_VNNLIB = ('/home/stan/repositories/vnncomp2025_benchmarks/benchmarks/'
                 'cifar100_2024/vnnlib/'
                 'CIFAR100_resnet_medium_prop_idx_7258_sidx_3539_eps_0.0039'
                 '.vnnlib.gz')


# ---------------------------------------------------------------------------
# Adaptive-zonotope-for-spec == CROWN backward (identical algorithm)
# ---------------------------------------------------------------------------
# Both use the same min-area triangle slopes (`lo_s`, `up_s`, `up_t`), so
# they must agree bit-exact up to float-precision rounding.

def _spec_equiv_setup(tmp_path, build_model_fn, name):
    model = build_model_fn()
    g = _save_and_load(model, tmp_path, name)
    gg = g.gpu_graph(torch.device('cpu'), torch.float64)
    from vibecheck.verify_zono_bnb import _forward_zonotope_graph
    in_n = gg['input_n']
    xl = torch.full((in_n,), -0.1, dtype=torch.float64)
    xh = torch.full((in_n,), 0.1, dtype=torch.float64)
    sb, _ = _forward_zonotope_graph(
        xl, xh, gg, torch.device('cpu'), torch.float64)
    bounds_by_relu = {li: (lo.cpu().numpy().astype(np.float64),
                           hi.cpu().numpy().astype(np.float64))
                      for li, (lo, hi) in sb.items()}
    return gg, xl, xh, sb, bounds_by_relu


def _assert_spec_lb_equal(gg, xl, xh, sb, bounds_by_relu, tag):
    """For each of several random spec weight vectors, assert that
    `_adaptive_spec_lb` and `_spec_backward_graph` give the same
    lower bound."""
    from vibecheck.verify_zono_bnb import _spec_backward_graph

    # Find output size
    n_out = None
    for op in reversed(gg['ops']):
        if op['type'] == 'fc':
            n_out = op['W'].shape[0]
            break
        if op['type'] == 'conv':
            n_out = op['n_out']
            break
    assert n_out is not None

    rng = np.random.RandomState(0xc0ffee)
    for trial in range(6):
        w = rng.randn(n_out).astype(np.float64)
        bias = float(rng.randn() * 0.5)
        ew_t = torch.tensor(w, dtype=torch.float64)
        spec_ew = {0: (ew_t, bias)}
        crown_lbs, _ = _spec_backward_graph(
            sb, xl, xh, gg, spec_ew, {0}, gg['n_relu'],
            torch.device('cpu'), torch.float64)
        adapt_lb = _adaptive_spec_lb(
            gg, xl, xh, bounds_by_relu, ew_t, bias,
            torch.device('cpu'), torch.float64)
        # Both are scalar CROWN lower bounds with min-area slopes;
        # the two implementations should match to float precision.
        diff = abs(adapt_lb - crown_lbs[0])
        assert diff < 1e-10, (
            f'{tag} trial {trial}: adaptive={adapt_lb} '
            f'crown={crown_lbs[0]} diff={diff:.3e}')


def test_adaptive_spec_lb_equals_crown_sequential(tmp_path):
    """On a sequential FC network, `_adaptive_spec_lb` must match
    `_spec_backward_graph` exactly."""
    def _build():
        rng = np.random.RandomState(42)
        W1 = rng.randn(5, 4).astype(np.float32) * 0.5
        b1 = rng.randn(5).astype(np.float32) * 0.1
        W2 = rng.randn(5, 5).astype(np.float32) * 0.5
        b2 = rng.randn(5).astype(np.float32) * 0.1
        W3 = rng.randn(3, 5).astype(np.float32) * 0.5
        b3 = rng.randn(3).astype(np.float32) * 0.1
        nodes = [
            helper.make_node('Gemm', ['X', 'W1', 'b1'], ['g1'], transB=1),
            helper.make_node('Relu', ['g1'], ['r1']),
            helper.make_node('Gemm', ['r1', 'W2', 'b2'], ['g2'], transB=1),
            helper.make_node('Relu', ['g2'], ['r2']),
            helper.make_node('Gemm', ['r2', 'W3', 'b3'], ['Y'], transB=1),
        ]
        inits = [_init('W1', W1), _init('b1', b1),
                 _init('W2', W2), _init('b2', b2),
                 _init('W3', W3), _init('b3', b3)]
        graph = helper.make_graph(
            nodes, 'seq',
            [_input_val('X', [1, 4])],
            [_input_val('Y', [1, 3])], inits)
        return helper.make_model(
            graph, opset_imports=[helper.make_opsetid('', 13)])

    gg, xl, xh, sb, bounds_by_relu = \
        _spec_equiv_setup(tmp_path, _build, 'seq.onnx')
    _assert_spec_lb_equal(gg, xl, xh, sb, bounds_by_relu, 'sequential')


def test_adaptive_spec_lb_equals_crown_identity_skip(tmp_path):
    """On an identity-skip residual block (conv+merge DAG),
    `_adaptive_spec_lb` must match `_spec_backward_graph` exactly."""
    gg, xl, xh, sb, bounds_by_relu = _spec_equiv_setup(
        tmp_path,
        lambda: _build_identity_skip_model(np.random.RandomState(7)),
        'id_skip.onnx')
    _assert_spec_lb_equal(gg, xl, xh, sb, bounds_by_relu, 'identity_skip')


def test_adaptive_spec_lb_equals_crown_dead_branch(tmp_path):
    """On the dead-branch resnet (has all-dead relu + bias-only FC),
    `_adaptive_spec_lb` must match `_spec_backward_graph` exactly."""
    gg, xl, xh, sb, bounds_by_relu = _spec_equiv_setup(
        tmp_path, _build_dead_branch_resnet, 'dead_branch.onnx')
    _assert_spec_lb_equal(gg, xl, xh, sb, bounds_by_relu, 'dead_branch')


# ---------------------------------------------------------------------------
# Bug #1 regression: verify_milp worker must not drop dead-branch bias
# ---------------------------------------------------------------------------

def _dead_branch_resnet_setup(tmp_path, filename):
    """Load the dead-branch resnet and precompute everything a spec worker
    needs: serialized ops, float64 bounds_by_relu, and the query."""
    model = _build_dead_branch_resnet()
    g = _save_and_load(model, tmp_path, filename)
    gg = g.gpu_graph(torch.device('cpu'), torch.float64)
    gg_ops = _serialize_gg_ops(gg)
    x_lo = np.array([-0.1, -0.1], dtype=np.float64)
    x_hi = np.array([0.1, 0.1], dtype=np.float64)
    # Run zonotope forward to get bounds_by_relu.
    from vibecheck.verify_zono_bnb import _forward_zonotope_graph
    xl_t = torch.tensor(x_lo, dtype=torch.float64)
    xh_t = torch.tensor(x_hi, dtype=torch.float64)
    sb, _ = _forward_zonotope_graph(
        xl_t, xh_t, gg, torch.device('cpu'), torch.float64)
    bounds_by_relu = {li: (lo.cpu().numpy().astype(np.float64),
                            hi.cpu().numpy().astype(np.float64))
                       for li, (lo, hi) in sb.items()}
    # PairwiseConstraint(pred=0, comp=1) → linear query for Y[0] - Y[1] > 0.
    spec = VNNSpec(
        x_lo=x_lo.astype(np.float32),
        x_hi=x_hi.astype(np.float32),
        disjuncts=[Conjunct([PairwiseConstraint(pred=0, comp=1)])])
    queries = spec.as_linear_queries(2)
    _, q_w, q_bias = queries[0]
    return g, gg, gg_ops, bounds_by_relu, spec, q_w, q_bias


def test_verify_milp_worker_drops_dead_bias_regression(tmp_path):
    """Directly calls _solve_spec_graph_worker on a dead-branch resnet.

    With a sound LP, min(Y[0] - Y[1]) ≈ 11 - 5 ≈ 5.98 > 0, which means
    adding `Y[0] - Y[1] <= 0` as a constraint should be INFEASIBLE. The
    buggy worker drops FC_a2's bias [10, 0] (dead-branch path) so it
    thinks min ≈ 1 - 5 = -4, finds the LP feasible, and returns SAT.
    """
    from vibecheck.verify_milp import _solve_spec_graph_worker

    g, gg, gg_ops, bounds_by_relu, spec, q_w, q_bias = \
        _dead_branch_resnet_setup(tmp_path, 'dead_branch.onnx')

    x_lo = spec.x_lo.astype(np.float64)
    x_hi = spec.x_hi.astype(np.float64)

    feas_args = (
        'feasibility', gg_ops, x_lo, x_hi, bounds_by_relu,
        q_w, q_bias, [], 0, 1, 30.0, gg['input_name'], gg['fork_points'])
    res, _, _ = _solve_spec_graph_worker(feas_args)
    assert res == 'UNSAT', (
        f'verify_milp worker Bug #1 regression: feasibility returned {res}, '
        f'expected UNSAT (LP must be infeasible because Y[0]-Y[1] >= 5.98 > 0)')

    opt_args = (
        'optimize', gg_ops, x_lo, x_hi, bounds_by_relu,
        q_w, q_bias, [], 0, 1, 30.0, gg['input_name'], gg['fork_points'])
    opt_res, _, opt_lb = _solve_spec_graph_worker(opt_args)
    assert opt_lb is not None and opt_lb > 4.9, (
        f'verify_milp worker Bug #1 regression: optimize lb={opt_lb}, '
        f'expected ~5.98 (contribution from FC_a2 bias [10, 0])')


def test_verify_graph_worker_on_dead_branch(tmp_path):
    """verify_graph's own worker must also report LP-infeasible on the
    same instance (it already has the Bug #1 fix). This pins the expected
    behavior so both workers agree after the verify_milp fix.
    """
    from vibecheck.verify_graph import _solve_spec_worker_graph

    g, gg, gg_ops, bounds_by_relu, spec, q_w, q_bias = \
        _dead_branch_resnet_setup(tmp_path, 'dead_branch_g.onnx')
    x_lo = spec.x_lo.astype(np.float64)
    x_hi = spec.x_hi.astype(np.float64)

    feas_args = (
        'feasibility', 'optimized', gg_ops, x_lo, x_hi, bounds_by_relu,
        q_w, q_bias, [], 0, 1, 30.0, gg['input_name'])
    res, _, _ = _solve_spec_worker_graph(feas_args)
    assert res == 'UNSAT', f'verify_graph worker returned {res}, expected UNSAT'

    opt_args = (
        'optimize', 'optimized', gg_ops, x_lo, x_hi, bounds_by_relu,
        q_w, q_bias, [], 0, 1, 30.0, gg['input_name'])
    _, _, opt_lb = _solve_spec_worker_graph(opt_args)
    assert opt_lb is not None and opt_lb > 4.9, (
        f'verify_graph worker lb={opt_lb}, expected ~5.98')


def test_milp_and_graph_agree_on_dead_branch(tmp_path):
    """End-to-end cross-check: milp_verify and verify_graph must both
    report 'verified' on the dead-branch resnet."""
    from vibecheck.verify_milp import milp_verify

    model = _build_dead_branch_resnet()
    g = _save_and_load(model, tmp_path, 'dead_branch_e2e.onnx')
    spec = VNNSpec(
        x_lo=np.array([-0.1, -0.1], dtype=np.float32),
        x_hi=np.array([0.1, 0.1], dtype=np.float32),
        disjuncts=[Conjunct([PairwiseConstraint(pred=0, comp=1)])])
    s = default_settings(device='cpu', total_timeout=30, print_progress=False)

    r_milp, _ = milp_verify(g, spec, s)
    r_graph, _ = verify_graph(g, spec, s)
    assert r_milp == 'verified', f'milp_verify = {r_milp}'
    assert r_graph == 'verified', f'verify_graph = {r_graph}'


@pytest.mark.skipif(not os.path.exists(_CIFAR_ONNX),
                    reason='CIFAR100 benchmark files not available')
def test_cifar100_runs_without_crash():
    """End-to-end smoke on the flagship CIFAR100 instance.

    With the Bug #1 fix in the spec worker and the sound per-neuron
    adaptive bounds (separate EW_lb/EW_ub) the pipeline legitimately
    reports 'unknown' for this instance in 180 s — the LP relaxation
    cannot prove 3 of the 99 disjuncts and MILP escalation does not
    finish in the remaining budget. Upgrading to 'verified' requires
    either branch-and-bound / alpha-CROWN-style tightening or a bigger
    time budget. The previous 'verified' result relied on two unsound
    shortcuts (Bug #1 in verify_milp and a wrong upper-bound
    computation in the adaptive pass) and was therefore wrong.
    """
    from vibecheck.vnnlib_loader import load_vnnlib
    g = ComputeGraph.from_onnx(_CIFAR_ONNX)
    spec = load_vnnlib(_CIFAR_VNNLIB)
    s = default_settings(device='cpu', total_timeout=180,
                         print_progress=False, graph_impl='optimized')
    result, details = verify_graph(g, spec, s)
    assert result in ('verified', 'unknown'), (
        f'got {result} (phase={details.get("phase")})')


@pytest.mark.skipif(not os.path.exists(_CIFAR_ONNX),
                    reason='CIFAR100 benchmark files not available')
def test_cifar100_reference_agrees():
    """Reference builder should also make progress on CIFAR100."""
    from vibecheck.vnnlib_loader import load_vnnlib
    g = ComputeGraph.from_onnx(_CIFAR_ONNX)
    spec = load_vnnlib(_CIFAR_VNNLIB)
    s = default_settings(device='cpu', total_timeout=180,
                         print_progress=False, graph_impl='reference')
    result, _ = verify_graph(g, spec, s)
    assert result in ('verified', 'unknown')
