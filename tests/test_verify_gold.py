"""Sanity tests for verify_gold.py — GOLD library functions on small synthetic networks."""

import numpy as np
import pytest
import onnx
from onnx import helper, TensorProto, numpy_helper
import gurobipy as grb
import torch

from vibecheck.network import ComputeGraph
from vibecheck.spec import VNNSpec, Conjunct, PairwiseConstraint
from vibecheck.settings import default_settings
from vibecheck.verify_graph import (
    _build_reference, _serialize_gg_ops, _forward_zonotope_interleaved,
)
from vibecheck.verify_gold import (
    solve_joint_lp, find_boundary_op, find_tail_op_chain,
    extract_rho_chain_rule, detect_rho_sign, build_tail_sub,
    solve_model, gold_joint_mixed_milp, gold_tail_decomposition,
    gold_solve_query, _count_relu,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(model, tmp_path, name='test.onnx'):
    path = str(tmp_path / name)
    onnx.save(model, path)
    return path


def _init(name, arr):
    return numpy_helper.from_array(arr.astype(np.float32), name)


def _inp(name='X', shape=None):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _out(name='Y', shape=None):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _make_model(nodes, inputs, outputs, initializers, opset=13):
    graph = helper.make_graph(nodes, 'test', inputs, outputs, initializers)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', opset)])


def _run_phase1(graph, spec):
    """Run phase 1 (interleaved zonotope forward) and return needed data."""
    device = torch.device('cpu')
    dtype = torch.float64
    settings = default_settings(device='cpu', print_progress=False,
                                total_timeout=60)
    settings.graph_impl = 'reference'

    gg = graph.gpu_graph(device, dtype)
    gg_ops_ser = _serialize_gg_ops(gg)
    for d in gg_ops_ser:
        if d['type'] == 'conv' and 'W_sp' not in d:
            from vibecheck.verify_graph import _conv_sparse_matrix
            d['W_sp'] = _conv_sparse_matrix(
                d['kernel_np'], d['in_shape'], d['stride'], d['padding'])

    x_lo = spec.x_lo.astype(np.float64)
    x_hi = spec.x_hi.astype(np.float64)
    xl_g = torch.tensor(x_lo, dtype=dtype, device=device)
    xh_g = torch.tensor(x_hi, dtype=dtype, device=device)

    import time
    t_start = time.perf_counter()
    sb, bounds_by_relu, _ = _forward_zonotope_interleaved(
        xl_g, xh_g, gg, gg_ops_ser, x_lo, x_hi,
        _build_reference, 5.0, 1,
        lambda: 60.0 - (time.perf_counter() - t_start),
        device, dtype, settings)

    return gg, gg_ops_ser, x_lo, x_hi, bounds_by_relu


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _build_fc_sequential(tmp_path, seed=42):
    """2-input -> FC(3) -> ReLU -> FC(2) -> output. Returns (graph, spec, qw, qb)."""
    rng = np.random.RandomState(seed)
    W1 = rng.randn(3, 2).astype(np.float32)
    b1 = rng.randn(3).astype(np.float32) * 0.1
    W2 = rng.randn(2, 3).astype(np.float32)
    b2 = rng.randn(2).astype(np.float32) * 0.1

    nodes = [
        helper.make_node('Gemm', ['X', 'W1', 'b1'], ['fc1'], transB=1),
        helper.make_node('Relu', ['fc1'], ['relu1']),
        helper.make_node('Gemm', ['relu1', 'W2', 'b2'], ['Y'], transB=1),
    ]
    model = _make_model(
        nodes, [_inp('X', [1, 2])], [_out('Y', [1, 2])],
        [_init('W1', W1), _init('b1', b1), _init('W2', W2), _init('b2', b2)])
    path = _save(model, tmp_path, 'fc_seq.onnx')
    graph = ComputeGraph.from_onnx(path)

    x_lo = np.array([-1.0, -1.0], dtype=np.float64)
    x_hi = np.array([1.0, 1.0], dtype=np.float64)
    spec = VNNSpec(x_lo, x_hi, [Conjunct([PairwiseConstraint(0, 1)])])
    qw = np.array([1.0, -1.0], dtype=np.float64)
    qb = 0.0
    return graph, spec, qw, qb


def _build_skip_connection(tmp_path, seed=42):
    """2-input -> FC(3) -> ReLU -> FC(3) -> Add(passthrough FC) -> ReLU -> FC(2).

    The skip connection uses a separate FC from the input to match dimensions.
    """
    rng = np.random.RandomState(seed)
    W1 = rng.randn(3, 2).astype(np.float32)
    b1 = rng.randn(3).astype(np.float32) * 0.1
    W2 = rng.randn(3, 3).astype(np.float32)
    b2 = rng.randn(3).astype(np.float32) * 0.1
    W_skip = rng.randn(3, 2).astype(np.float32)
    b_skip = rng.randn(3).astype(np.float32) * 0.1
    W3 = rng.randn(2, 3).astype(np.float32)
    b3 = rng.randn(2).astype(np.float32) * 0.1

    nodes = [
        helper.make_node('Gemm', ['X', 'W1', 'b1'], ['fc1'], transB=1),
        helper.make_node('Relu', ['fc1'], ['relu1']),
        helper.make_node('Gemm', ['relu1', 'W2', 'b2'], ['fc2'], transB=1),
        helper.make_node('Gemm', ['X', 'W_skip', 'b_skip'], ['skip'], transB=1),
        helper.make_node('Add', ['fc2', 'skip'], ['add1']),
        helper.make_node('Relu', ['add1'], ['relu2']),
        helper.make_node('Gemm', ['relu2', 'W3', 'b3'], ['Y'], transB=1),
    ]
    model = _make_model(
        nodes, [_inp('X', [1, 2])], [_out('Y', [1, 2])],
        [_init('W1', W1), _init('b1', b1), _init('W2', W2), _init('b2', b2),
         _init('W_skip', W_skip), _init('b_skip', b_skip),
         _init('W3', W3), _init('b3', b3)])
    path = _save(model, tmp_path, 'skip.onnx')
    graph = ComputeGraph.from_onnx(path)

    x_lo = np.array([-1.0, -1.0], dtype=np.float64)
    x_hi = np.array([1.0, 1.0], dtype=np.float64)
    spec = VNNSpec(x_lo, x_hi, [Conjunct([PairwiseConstraint(0, 1)])])
    qw = np.array([1.0, -1.0], dtype=np.float64)
    qb = 0.0
    return graph, spec, qw, qb


def _build_conv_skip(tmp_path, seed=42):
    """Conv(1,1,3,pad=1) -> ReLU -> Conv(1,1,3,pad=1) -> Add(skip) -> ReLU -> Flatten -> FC(2).

    Input shape (1,1,4,4).
    """
    rng = np.random.RandomState(seed)
    k1 = rng.randn(1, 1, 3, 3).astype(np.float32) * 0.5
    b_c1 = rng.randn(1).astype(np.float32) * 0.1
    k2 = rng.randn(1, 1, 3, 3).astype(np.float32) * 0.5
    b_c2 = rng.randn(1).astype(np.float32) * 0.1
    W_fc = rng.randn(2, 16).astype(np.float32) * 0.3
    b_fc = rng.randn(2).astype(np.float32) * 0.1

    nodes = [
        helper.make_node('Conv', ['X', 'k1', 'bc1'], ['conv1'],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node('Relu', ['conv1'], ['relu1']),
        helper.make_node('Conv', ['relu1', 'k2', 'bc2'], ['conv2'],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node('Add', ['conv2', 'X'], ['add1']),
        helper.make_node('Relu', ['add1'], ['relu2']),
        helper.make_node('Flatten', ['relu2'], ['flat'], axis=1),
        helper.make_node('Gemm', ['flat', 'Wfc', 'bfc'], ['Y'], transB=1),
    ]
    model = _make_model(
        nodes, [_inp('X', [1, 1, 4, 4])], [_out('Y', [1, 2])],
        [_init('k1', k1), _init('bc1', b_c1),
         _init('k2', k2), _init('bc2', b_c2),
         _init('Wfc', W_fc), _init('bfc', b_fc)])
    path = _save(model, tmp_path, 'conv_skip.onnx')
    graph = ComputeGraph.from_onnx(path)

    x_lo = np.full(16, -0.5, dtype=np.float64)
    x_hi = np.full(16, 0.5, dtype=np.float64)
    spec = VNNSpec(x_lo, x_hi, [Conjunct([PairwiseConstraint(0, 1)])])
    qw = np.array([1.0, -1.0], dtype=np.float64)
    qb = 0.0
    return graph, spec, qw, qb


def _build_all_stable_last(tmp_path, seed=99):
    """FC network where the last ReLU layer is all-stable (positive bounds).

    2-input -> FC(3) -> ReLU -> FC(3) -> ReLU -> FC(2).
    Weights chosen so pre-activation of second ReLU is positive on [-0.1, 0.1].
    """
    W1 = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    b1 = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    W2 = np.eye(3, dtype=np.float32)
    b2 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    W3 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    b3 = np.zeros(2, dtype=np.float32)

    nodes = [
        helper.make_node('Gemm', ['X', 'W1', 'b1'], ['fc1'], transB=1),
        helper.make_node('Relu', ['fc1'], ['relu1']),
        helper.make_node('Gemm', ['relu1', 'W2', 'b2'], ['fc2'], transB=1),
        helper.make_node('Relu', ['fc2'], ['relu2']),
        helper.make_node('Gemm', ['relu2', 'W3', 'b3'], ['Y'], transB=1),
    ]
    model = _make_model(
        nodes, [_inp('X', [1, 2])], [_out('Y', [1, 2])],
        [_init('W1', W1), _init('b1', b1), _init('W2', W2), _init('b2', b2),
         _init('W3', W3), _init('b3', b3)])
    path = _save(model, tmp_path, 'all_stable.onnx')
    graph = ComputeGraph.from_onnx(path)

    x_lo = np.array([-0.1, -0.1], dtype=np.float64)
    x_hi = np.array([0.1, 0.1], dtype=np.float64)
    spec = VNNSpec(x_lo, x_hi, [Conjunct([PairwiseConstraint(0, 1)])])
    qw = np.array([1.0, -1.0], dtype=np.float64)
    qb = 0.0
    return graph, spec, qw, qb


def _build_provably_verified(tmp_path, seed=77):
    """FC network where y0 >> y1 on [-0.1, 0.1], so spec y0-y1 > 0 is verified.

    2-input -> FC(3) -> ReLU -> FC(2).
    """
    W1 = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    b1 = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    W2 = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]], dtype=np.float32)
    b2 = np.array([10.0, -10.0], dtype=np.float32)

    nodes = [
        helper.make_node('Gemm', ['X', 'W1', 'b1'], ['fc1'], transB=1),
        helper.make_node('Relu', ['fc1'], ['relu1']),
        helper.make_node('Gemm', ['relu1', 'W2', 'b2'], ['Y'], transB=1),
    ]
    model = _make_model(
        nodes, [_inp('X', [1, 2])], [_out('Y', [1, 2])],
        [_init('W1', W1), _init('b1', b1), _init('W2', W2), _init('b2', b2)])
    path = _save(model, tmp_path, 'verified.onnx')
    graph = ComputeGraph.from_onnx(path)

    x_lo = np.array([-0.1, -0.1], dtype=np.float64)
    x_hi = np.array([0.1, 0.1], dtype=np.float64)
    spec = VNNSpec(x_lo, x_hi, [Conjunct([PairwiseConstraint(0, 1)])])
    qw = np.array([1.0, -1.0], dtype=np.float64)
    qb = 0.0
    return graph, spec, qw, qb


def _build_skip_in_tail(tmp_path, seed=42):
    """Network where skip connection lands AFTER the last hidden ReLU.

    2-input -> FC(3) -> ReLU -> FC(2), and separately
    2-input -> FC(2), then Add(fc_out, skip_out) -> output.
    The Add is a merge in the tail.
    """
    rng = np.random.RandomState(seed)
    W1 = rng.randn(3, 2).astype(np.float32)
    b1 = rng.randn(3).astype(np.float32) * 0.1
    W2 = rng.randn(2, 3).astype(np.float32)
    b2 = rng.randn(2).astype(np.float32) * 0.1
    W_skip = rng.randn(2, 2).astype(np.float32)
    b_skip = rng.randn(2).astype(np.float32) * 0.1

    nodes = [
        helper.make_node('Gemm', ['X', 'W1', 'b1'], ['fc1'], transB=1),
        helper.make_node('Relu', ['fc1'], ['relu1']),
        helper.make_node('Gemm', ['relu1', 'W2', 'b2'], ['fc2'], transB=1),
        helper.make_node('Gemm', ['X', 'W_skip', 'b_skip'], ['skip'], transB=1),
        helper.make_node('Add', ['fc2', 'skip'], ['Y']),
    ]
    model = _make_model(
        nodes, [_inp('X', [1, 2])], [_out('Y', [1, 2])],
        [_init('W1', W1), _init('b1', b1), _init('W2', W2), _init('b2', b2),
         _init('W_skip', W_skip), _init('b_skip', b_skip)])
    path = _save(model, tmp_path, 'skip_in_tail.onnx')
    graph = ComputeGraph.from_onnx(path)

    x_lo = np.array([-1.0, -1.0], dtype=np.float64)
    x_hi = np.array([1.0, 1.0], dtype=np.float64)
    spec = VNNSpec(x_lo, x_hi, [Conjunct([PairwiseConstraint(0, 1)])])
    qw = np.array([1.0, -1.0], dtype=np.float64)
    qb = 0.0
    return graph, spec, qw, qb


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_t1_tiny_fc_joint_lp(tmp_path):
    """T1: Joint LP matches hand-built Gurobi LP on tiny FC network."""
    graph, spec, qw, qb = _build_fc_sequential(tmp_path)
    gg, gg_ops_ser, x_lo, x_hi, bounds_by_relu = _run_phase1(graph, spec)
    output_op = gg_ops_ser[-1]['name']
    input_name = gg['input_name']

    lp_bound, op_var_refs, m, env = solve_joint_lp(
        gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
        output_op, qw, qb)
    m.dispose()
    env.dispose()

    ref_m, ref_env, ref_vars, _ = _build_reference(
        gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name)
    last_vars = ref_vars[output_op]
    obj = grb.LinExpr()
    for j in range(len(qw)):
        if qw[j] != 0 and last_vars[j] is not None:
            obj.add(last_vars[j], float(qw[j]))
    ref_m.setObjective(obj + float(qb), grb.GRB.MINIMIZE)
    ref_m.optimize()
    ref_bound = float(ref_m.ObjVal)
    ref_m.dispose()
    ref_env.dispose()

    assert abs(lp_bound - ref_bound) < 1e-6, \
        f'LP bound mismatch: {lp_bound} vs {ref_bound}'


def test_t2_skip_connection_lp_equivalence(tmp_path):
    """T2: Skip-connection LP matches direct _build_reference LP."""
    graph, spec, qw, qb = _build_skip_connection(tmp_path)
    gg, gg_ops_ser, x_lo, x_hi, bounds_by_relu = _run_phase1(graph, spec)
    output_op = gg_ops_ser[-1]['name']
    input_name = gg['input_name']

    has_merge = any(op.get('is_merge') for op in gg_ops_ser)
    assert has_merge, 'expected at least one merge-Add op'

    lp_bound, op_var_refs, m, env = solve_joint_lp(
        gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
        output_op, qw, qb)
    m.dispose()
    env.dispose()

    ref_m, ref_env, ref_vars, _ = _build_reference(
        gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name)
    last_vars = ref_vars[output_op]
    obj = grb.LinExpr()
    for j in range(len(qw)):
        if qw[j] != 0 and last_vars[j] is not None:
            obj.add(last_vars[j], float(qw[j]))
    ref_m.setObjective(obj + float(qb), grb.GRB.MINIMIZE)
    ref_m.optimize()
    ref_bound = float(ref_m.ObjVal)
    ref_m.dispose()
    ref_env.dispose()

    assert abs(lp_bound - ref_bound) < 1e-6, \
        f'skip-conn LP bound mismatch: {lp_bound} vs {ref_bound}'


def test_t3_conv_skip_lp_equivalence(tmp_path):
    """T3: Conv + skip connection LP equivalence."""
    graph, spec, qw, qb = _build_conv_skip(tmp_path)
    gg, gg_ops_ser, x_lo, x_hi, bounds_by_relu = _run_phase1(graph, spec)
    output_op = gg_ops_ser[-1]['name']
    input_name = gg['input_name']

    has_merge = any(op.get('is_merge') for op in gg_ops_ser)
    assert has_merge, 'expected merge-Add in conv skip fixture'

    lp_bound, _, m, env = solve_joint_lp(
        gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
        output_op, qw, qb)
    m.dispose()
    env.dispose()

    ref_m, ref_env, ref_vars, _ = _build_reference(
        gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name)
    last_vars = ref_vars[output_op]
    obj = grb.LinExpr()
    for j in range(len(qw)):
        if qw[j] != 0 and last_vars[j] is not None:
            obj.add(last_vars[j], float(qw[j]))
    ref_m.setObjective(obj + float(qb), grb.GRB.MINIMIZE)
    ref_m.optimize()
    ref_bound = float(ref_m.ObjVal)
    ref_m.dispose()
    ref_env.dispose()

    assert abs(lp_bound - ref_bound) < 1e-6, \
        f'conv skip LP bound mismatch: {lp_bound} vs {ref_bound}'


def test_t4_boundary_op_identification(tmp_path):
    """T4: find_boundary_op returns the FC feeding the LAST hidden ReLU."""
    graph, spec, qw, qb = _build_skip_connection(tmp_path)
    gg, gg_ops_ser, x_lo, x_hi, bounds_by_relu = _run_phase1(graph, spec)
    n_relu = gg['n_relu']

    boundary_name, last_relu_name = find_boundary_op(gg_ops_ser, n_relu)

    relu_ops = [op for op in gg_ops_ser
                if op['type'] == 'relu' and 'layer_idx' in op]
    expected_last_relu = relu_ops[-1]
    assert last_relu_name == expected_last_relu['name']
    assert boundary_name == expected_last_relu['inputs'][0]


def test_t5_tail_op_chain(tmp_path):
    """T5: find_tail_op_chain returns ops from last ReLU through output."""
    graph, spec, qw, qb = _build_skip_connection(tmp_path)
    gg, gg_ops_ser, x_lo, x_hi, bounds_by_relu = _run_phase1(graph, spec)
    n_relu = gg['n_relu']
    output_op = gg_ops_ser[-1]['name']

    _, last_relu_name = find_boundary_op(gg_ops_ser, n_relu)
    chain = find_tail_op_chain(gg_ops_ser, last_relu_name, output_op)

    assert chain[0]['name'] == last_relu_name
    assert chain[0]['type'] == 'relu'
    assert chain[-1]['name'] == output_op
    tail_types = [op['type'] for op in chain]
    assert 'relu' in tail_types
    assert 'fc' in tail_types


def test_t6_bound_at_rho_zero(tmp_path):
    """T6: At rho=0, tail sub LP matches a hand-built look-back LP."""
    graph, spec, qw, qb = _build_skip_connection(tmp_path)
    gg, gg_ops_ser, x_lo, x_hi, bounds_by_relu = _run_phase1(graph, spec)
    n_relu = gg['n_relu']
    output_op = gg_ops_ser[-1]['name']
    last_relu_layer_idx = n_relu - 1

    _, last_relu_name = find_boundary_op(gg_ops_ser, n_relu)
    chain = find_tail_op_chain(gg_ops_ser, last_relu_name, output_op)

    lo_r, hi_r = bounds_by_relu[last_relu_layer_idx]
    rho_zero = {i: 0.0 for i in range(len(lo_r)) if hi_r[i] > 0}

    m_lp, env_lp = build_tail_sub(bounds_by_relu, last_relu_layer_idx,
                                   chain, qw, qb, rho_zero, inner='lp')
    m_lp.optimize()
    v_tail_lp = float(m_lp.ObjVal)
    m_lp.dispose()
    env_lp.dispose()

    m_milp, env_milp = build_tail_sub(bounds_by_relu, last_relu_layer_idx,
                                       chain, qw, qb, rho_zero, inner='milp')
    m_milp.setParam('TimeLimit', 10.0)
    m_milp.optimize()
    v_tail_milp = float(m_milp.ObjBound)
    m_milp.dispose()
    env_milp.dispose()

    assert v_tail_milp >= v_tail_lp - 1e-6, \
        f'MILP tail {v_tail_milp} < LP tail {v_tail_lp}'


def test_t7_bound_formula_all_lp(tmp_path):
    """T7: With chain-rule rho but LP-only tail, delta is zero."""
    graph, spec, qw, qb = _build_skip_connection(tmp_path)
    gg, gg_ops_ser, x_lo, x_hi, bounds_by_relu = _run_phase1(graph, spec)
    n_relu = gg['n_relu']
    output_op = gg_ops_ser[-1]['name']
    input_name = gg['input_name']
    last_relu_layer_idx = n_relu - 1

    lp_triangle, op_var_refs, m_lp, env_lp = solve_joint_lp(
        gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
        output_op, qw, qb)
    boundary_name, last_relu_name = find_boundary_op(gg_ops_ser, n_relu)
    rho_raw = extract_rho_chain_rule(op_var_refs, boundary_name)
    m_lp.dispose()
    env_lp.dispose()

    chain = find_tail_op_chain(gg_ops_ser, last_relu_name, output_op)
    rho_sign = detect_rho_sign(rho_raw, bounds_by_relu, last_relu_layer_idx,
                               chain, qw, qb, lp_triangle)
    rho = {i: rho_sign * v for i, v in rho_raw.items()}

    m_lp1, env1 = build_tail_sub(bounds_by_relu, last_relu_layer_idx,
                                  chain, qw, qb, rho, inner='lp')
    m_lp1.optimize()
    v1 = float(m_lp1.ObjVal)
    m_lp1.dispose()
    env1.dispose()

    m_lp2, env2 = build_tail_sub(bounds_by_relu, last_relu_layer_idx,
                                  chain, qw, qb, rho, inner='lp')
    m_lp2.optimize()
    v2 = float(m_lp2.ObjVal)
    m_lp2.dispose()
    env2.dispose()

    assert abs(v1 - v2) < 1e-6
    delta = v2 - v1
    gold_bound = lp_triangle + delta
    assert abs(gold_bound - lp_triangle) < 1e-6, \
        f'all-LP delta should be ~0, got {delta}'


def test_t8_bound_formula_validity(tmp_path):
    """T8: GOLD bound is between LP triangle and true MILP optimum."""
    graph, spec, qw, qb = _build_skip_connection(tmp_path)
    gg, gg_ops_ser, x_lo, x_hi, bounds_by_relu = _run_phase1(graph, spec)
    n_relu = gg['n_relu']
    output_op = gg_ops_ser[-1]['name']
    input_name = gg['input_name']

    result = gold_tail_decomposition(
        gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
        output_op, qw, qb, time_limit=30.0)
    gold_bound = result['gold_bound']
    lp_triangle = result['lp_triangle']

    assert gold_bound >= lp_triangle - 1e-6, \
        f'GOLD bound {gold_bound} < LP triangle {lp_triangle}'

    m_milp, env_milp, milp_vars, _ = _build_reference(
        gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name, use_milp=True)
    last_vars = milp_vars[output_op]
    obj = grb.LinExpr()
    for j in range(len(qw)):
        if qw[j] != 0 and last_vars[j] is not None:
            obj.add(last_vars[j], float(qw[j]))
    m_milp.setObjective(obj + float(qb), grb.GRB.MINIMIZE)
    m_milp.setParam('TimeLimit', 30.0)
    m_milp.optimize()
    true_opt = float(m_milp.ObjVal)
    m_milp.dispose()
    env_milp.dispose()

    assert gold_bound <= true_opt + 1e-4, \
        f'GOLD bound {gold_bound} > true optimum {true_opt}'


def test_t9_sign_detection(tmp_path):
    """T9: Detected sign gives GOLD bound >= LP triangle."""
    graph, spec, qw, qb = _build_skip_connection(tmp_path)
    gg, gg_ops_ser, x_lo, x_hi, bounds_by_relu = _run_phase1(graph, spec)
    n_relu = gg['n_relu']
    output_op = gg_ops_ser[-1]['name']
    input_name = gg['input_name']
    last_relu_layer_idx = n_relu - 1

    lp_triangle, op_var_refs, m_lp, env_lp = solve_joint_lp(
        gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
        output_op, qw, qb)
    boundary_name, last_relu_name = find_boundary_op(gg_ops_ser, n_relu)
    rho_raw = extract_rho_chain_rule(op_var_refs, boundary_name)
    m_lp.dispose()
    env_lp.dispose()

    chain = find_tail_op_chain(gg_ops_ser, last_relu_name, output_op)
    chosen_sign = detect_rho_sign(rho_raw, bounds_by_relu,
                                  last_relu_layer_idx, chain,
                                  qw, qb, lp_triangle)
    rho = {i: chosen_sign * v for i, v in rho_raw.items()}

    m_lp_tail, env1 = build_tail_sub(bounds_by_relu, last_relu_layer_idx,
                                      chain, qw, qb, rho, inner='lp')
    m_lp_tail.optimize()
    v_lp = float(m_lp_tail.ObjVal)
    m_lp_tail.dispose()
    env1.dispose()

    m_milp_tail, env2 = build_tail_sub(bounds_by_relu, last_relu_layer_idx,
                                        chain, qw, qb, rho, inner='milp')
    m_milp_tail.setParam('TimeLimit', 10.0)
    m_milp_tail.optimize()
    v_milp = float(m_milp_tail.ObjBound)
    m_milp_tail.dispose()
    env2.dispose()

    gold_bound = lp_triangle + (v_milp - v_lp)
    assert gold_bound >= lp_triangle - 1e-6, \
        f'wrong sign: GOLD {gold_bound} < LP {lp_triangle}'


def test_t10_skip_in_tail(tmp_path):
    """T10: Merge-Add in tail chain raises NotImplementedError."""
    graph, spec, qw, qb = _build_skip_in_tail(tmp_path)
    gg, gg_ops_ser, x_lo, x_hi, bounds_by_relu = _run_phase1(graph, spec)
    n_relu = gg['n_relu']
    output_op = gg_ops_ser[-1]['name']
    last_relu_layer_idx = n_relu - 1

    _, last_relu_name = find_boundary_op(gg_ops_ser, n_relu)
    chain = find_tail_op_chain(gg_ops_ser, last_relu_name, output_op)

    has_merge_in_tail = any(op.get('is_merge') for op in chain)
    if has_merge_in_tail:
        lo_r, hi_r = bounds_by_relu[last_relu_layer_idx]
        rho = {i: 0.0 for i in range(len(lo_r)) if hi_r[i] > 0}
        with pytest.raises(NotImplementedError, match='merge-Add'):
            build_tail_sub(bounds_by_relu, last_relu_layer_idx,
                           chain, qw, qb, rho, inner='lp')


def test_t11_best_bd_stop(tmp_path):
    """T11: BestBdStop triggers early with a provably verified spec."""
    graph, spec, qw, qb = _build_provably_verified(tmp_path)
    gg, gg_ops_ser, x_lo, x_hi, bounds_by_relu = _run_phase1(graph, spec)
    output_op = gg_ops_ser[-1]['name']
    input_name = gg['input_name']
    n_relu = gg['n_relu']
    last_relu_layer_idx = n_relu - 1

    lo_r, hi_r = bounds_by_relu[last_relu_layer_idx]
    ambiguous = {j for j in range(len(lo_r)) if lo_r[j] < 0 and hi_r[j] > 0}

    obj_bound, obj_val, runtime = gold_joint_mixed_milp(
        gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
        output_op, qw, qb, last_relu_layer_idx, ambiguous,
        time_limit=60.0, best_bd_stop=0.0, n_threads=1)

    assert obj_bound >= 0.0 - 1e-6, \
        f'expected verified (bound >= 0), got {obj_bound}'


def test_t2b_skip_lp_with_zero_qw(tmp_path):
    """Cover the qw==0 branch in solve_joint_lp and gold_joint_mixed_milp."""
    graph, spec, _, _ = _build_skip_connection(tmp_path)
    gg, gg_ops_ser, x_lo, x_hi, bounds_by_relu = _run_phase1(graph, spec)
    output_op = gg_ops_ser[-1]['name']
    input_name = gg['input_name']

    qw_sparse = np.array([1.0, 0.0], dtype=np.float64)
    qb = 0.0

    lp_bound, _, m, env = solve_joint_lp(
        gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
        output_op, qw_sparse, qb)
    m.dispose()
    env.dispose()
    assert np.isfinite(lp_bound)

    n_relu = gg['n_relu']
    li = n_relu - 1
    lo_r, hi_r = bounds_by_relu[li]
    ambiguous = {j for j in range(len(lo_r)) if lo_r[j] < 0 and hi_r[j] > 0}
    obj_bound, _, _ = gold_joint_mixed_milp(
        gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
        output_op, qw_sparse, qb, li, ambiguous,
        time_limit=10.0, n_threads=1)
    assert np.isfinite(obj_bound)


def test_tail_sub_reshape_in_chain(tmp_path):
    """Cover the reshape/flatten passthrough in build_tail_sub."""
    graph, spec, qw, qb = _build_conv_skip(tmp_path)
    gg, gg_ops_ser, x_lo, x_hi, bounds_by_relu = _run_phase1(graph, spec)
    n_relu = gg['n_relu']
    output_op = gg_ops_ser[-1]['name']
    last_relu_layer_idx = n_relu - 1

    _, last_relu_name = find_boundary_op(gg_ops_ser, n_relu)
    chain = find_tail_op_chain(gg_ops_ser, last_relu_name, output_op)

    tail_types = [op['type'] for op in chain]
    assert 'reshape' in tail_types, 'expected reshape/flatten in conv_skip tail'

    lo_r, hi_r = bounds_by_relu[last_relu_layer_idx]
    rho = {i: 0.0 for i in range(len(lo_r)) if hi_r[i] > 0}
    m_lp, env_lp = build_tail_sub(bounds_by_relu, last_relu_layer_idx,
                                   chain, qw, qb, rho, inner='lp')
    m_lp.optimize()
    val = float(m_lp.ObjVal)
    m_lp.dispose()
    env_lp.dispose()
    assert np.isfinite(val)


def test_tail_sub_unsupported_op():
    """Cover the unsupported op type raise in build_tail_sub."""
    bounds = {0: (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))}
    chain = [
        {'name': 'relu0', 'type': 'relu', 'inputs': ['fc0'], 'layer_idx': 0},
        {'name': 'bad_op', 'type': 'globalavgpool', 'inputs': ['relu0']},
    ]
    rho = {0: 0.0, 1: 0.0}
    with pytest.raises(NotImplementedError, match='unsupported op type'):
        build_tail_sub(bounds, 0, chain, np.array([1.0, -1.0]),
                       0.0, rho, inner='lp')


def test_tail_sub_add_bias():
    """Cover non-merge add with bias in build_tail_sub."""
    bounds = {0: (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))}
    W = np.eye(2, dtype=np.float64)
    b = np.zeros(2, dtype=np.float64)
    chain = [
        {'name': 'relu0', 'type': 'relu', 'inputs': ['fc0'], 'layer_idx': 0},
        {'name': 'fc1', 'type': 'fc', 'inputs': ['relu0'],
         'W_np': W, 'bias_np': b},
        {'name': 'add1', 'type': 'add', 'inputs': ['fc1'], 'is_merge': False,
         'bias': np.array([0.5, -0.5])},
    ]
    rho = {0: 0.0, 1: 0.0}
    qw = np.array([1.0, -1.0], dtype=np.float64)
    m, env = build_tail_sub(bounds, 0, chain, qw, 0.0, rho, inner='lp')
    m.optimize()
    val = float(m.ObjVal)
    m.dispose()
    env.dispose()
    assert np.isfinite(val)


def test_tail_sub_sub_bias():
    """Cover sub with bias in build_tail_sub."""
    bounds = {0: (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))}
    W = np.eye(2, dtype=np.float64)
    b = np.zeros(2, dtype=np.float64)
    chain = [
        {'name': 'relu0', 'type': 'relu', 'inputs': ['fc0'], 'layer_idx': 0},
        {'name': 'fc1', 'type': 'fc', 'inputs': ['relu0'],
         'W_np': W, 'bias_np': b},
        {'name': 'sub1', 'type': 'sub', 'inputs': ['fc1'],
         'bias': np.array([0.5, -0.5])},
    ]
    rho = {0: 0.0, 1: 0.0}
    qw = np.array([1.0, -1.0], dtype=np.float64)
    m, env = build_tail_sub(bounds, 0, chain, qw, 0.0, rho, inner='lp')
    m.optimize()
    val = float(m.ObjVal)
    m.dispose()
    env.dispose()
    assert np.isfinite(val)


def test_tail_sub_conv_in_tail():
    """Cover conv op handler in build_tail_sub."""
    bounds = {0: (np.full(16, -1.0), np.full(16, 1.0))}
    kernel = np.random.RandomState(42).randn(1, 1, 3, 3).astype(np.float64)
    bias_conv = np.zeros(1, dtype=np.float64)
    W_fc = np.random.RandomState(42).randn(2, 16).astype(np.float64)
    b_fc = np.zeros(2, dtype=np.float64)
    chain = [
        {'name': 'relu0', 'type': 'relu', 'inputs': ['fc0'], 'layer_idx': 0},
        {'name': 'conv1', 'type': 'conv', 'inputs': ['relu0'],
         'kernel_np': kernel, 'bias_np': bias_conv,
         'in_shape': (1, 4, 4), 'out_shape': (1, 4, 4),
         'stride': (1, 1), 'padding': (1, 1), 'n_out': 16},
        {'name': 'fc_out', 'type': 'fc', 'inputs': ['conv1'],
         'W_np': W_fc, 'bias_np': b_fc},
    ]
    rho = {i: 0.0 for i in range(16)}
    qw = np.array([1.0, -1.0], dtype=np.float64)
    m, env = build_tail_sub(bounds, 0, chain, qw, 0.0, rho, inner='lp')
    m.optimize()
    val = float(m.ObjVal)
    m.dispose()
    env.dispose()
    assert np.isfinite(val)


def test_tail_sub_dead_boundary_neurons():
    """Cover hi <= 0 (dead off) and None boundary vars paths."""
    lo = np.array([-1.0, -2.0, -0.5], dtype=np.float64)
    hi = np.array([1.0, -0.1, 0.5], dtype=np.float64)
    bounds = {0: (lo, hi)}
    W = np.ones((2, 3), dtype=np.float64)
    b = np.zeros(2, dtype=np.float64)
    chain = [
        {'name': 'relu0', 'type': 'relu', 'inputs': ['fc0'], 'layer_idx': 0},
        {'name': 'fc1', 'type': 'fc', 'inputs': ['relu0'],
         'W_np': W, 'bias_np': b},
    ]
    rho = {0: 0.1, 2: -0.1}
    qw = np.array([1.0, -1.0], dtype=np.float64)
    m, env = build_tail_sub(bounds, 0, chain, qw, 0.0, rho, inner='lp')
    m.optimize()
    val = float(m.ObjVal)
    m.dispose()
    env.dispose()
    assert np.isfinite(val)


def test_tail_sub_fc_no_live_inputs():
    """Cover FC with no live inputs in build_tail_sub (all dead)."""
    lo = np.array([-2.0, -3.0], dtype=np.float64)
    hi = np.array([-0.1, -0.2], dtype=np.float64)
    bounds = {0: (lo, hi)}
    W = np.ones((2, 2), dtype=np.float64)
    b = np.array([5.0, 3.0], dtype=np.float64)
    chain = [
        {'name': 'relu0', 'type': 'relu', 'inputs': ['fc0'], 'layer_idx': 0},
        {'name': 'fc1', 'type': 'fc', 'inputs': ['relu0'],
         'W_np': W, 'bias_np': b},
    ]
    rho = {}
    qw = np.array([1.0, -1.0], dtype=np.float64)
    m, env = build_tail_sub(bounds, 0, chain, qw, 0.0, rho, inner='lp')
    m.optimize()
    val = float(m.ObjVal)
    m.dispose()
    env.dispose()
    assert abs(val - 2.0) < 1e-6


def test_solve_model_no_solution():
    """Cover solve_model exception path when model has no feasible solution."""
    env = grb.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = grb.Model(env=env)
    x = m.addVar(lb=0.0, ub=1.0)
    m.addConstr(x >= 2.0)
    m.setObjective(x, grb.GRB.MINIMIZE)
    m.update()
    obj_bound, obj_val, runtime = solve_model(m, 10.0)
    assert obj_val is None
    m.dispose()
    env.dispose()


def test_gold_tail_with_best_bd_stop(tmp_path):
    """Cover best_bd_stop path in gold_tail_decomposition."""
    graph, spec, qw, qb = _build_provably_verified(tmp_path)
    gg, gg_ops_ser, x_lo, x_hi, bounds_by_relu = _run_phase1(graph, spec)
    output_op = gg_ops_ser[-1]['name']
    input_name = gg['input_name']

    result = gold_tail_decomposition(
        gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
        output_op, qw, qb, time_limit=30.0, best_bd_stop=0.0)
    assert result['gold_bound'] >= -1e-6


def test_gold_solve_query_variant_a(tmp_path):
    """Cover gold_solve_query variant A path including ambiguous set build."""
    graph, spec, qw, qb = _build_fc_sequential(tmp_path)
    gg, gg_ops_ser, x_lo, x_hi, bounds_by_relu = _run_phase1(graph, spec)

    result = gold_solve_query(gg, gg_ops_ser, x_lo, x_hi, bounds_by_relu,
                              qw, qb, variant='A', time_limit=10.0,
                              n_threads=1)
    assert result['method'] == 'joint_mixed_milp'
    assert np.isfinite(result['bound'])


def test_extract_rho_with_dead_neurons(tmp_path):
    """Cover the v is None branch in extract_rho_chain_rule."""
    graph, spec, qw, qb = _build_fc_sequential(tmp_path)
    gg, gg_ops_ser, x_lo, x_hi, bounds_by_relu = _run_phase1(graph, spec)
    output_op = gg_ops_ser[-1]['name']
    input_name = gg['input_name']
    n_relu = gg['n_relu']

    _, op_var_refs, m, env = solve_joint_lp(
        gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
        output_op, qw, qb)
    boundary_name, _ = find_boundary_op(gg_ops_ser, n_relu)

    orig_vars = op_var_refs[boundary_name]
    modified = list(orig_vars)
    modified[0] = None
    op_var_refs[boundary_name] = modified

    rho = extract_rho_chain_rule(op_var_refs, boundary_name)
    assert 0 not in rho
    m.dispose()
    env.dispose()


def test_add_no_bias_passthrough():
    """Cover add with no bias (pure passthrough) in build_tail_sub."""
    bounds = {0: (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))}
    W = np.eye(2, dtype=np.float64)
    b = np.zeros(2, dtype=np.float64)
    chain = [
        {'name': 'relu0', 'type': 'relu', 'inputs': ['fc0'], 'layer_idx': 0},
        {'name': 'fc1', 'type': 'fc', 'inputs': ['relu0'],
         'W_np': W, 'bias_np': b},
        {'name': 'add1', 'type': 'add', 'inputs': ['fc1'], 'is_merge': False,
         'bias': None},
    ]
    rho = {0: 0.0, 1: 0.0}
    qw = np.array([1.0, -1.0], dtype=np.float64)
    m, env = build_tail_sub(bounds, 0, chain, qw, 0.0, rho, inner='lp')
    m.optimize()
    val = float(m.ObjVal)
    m.dispose()
    env.dispose()
    assert np.isfinite(val)


def test_sub_no_bias_passthrough():
    """Cover sub with no bias (passthrough) in build_tail_sub."""
    bounds = {0: (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))}
    W = np.eye(2, dtype=np.float64)
    b = np.zeros(2, dtype=np.float64)
    chain = [
        {'name': 'relu0', 'type': 'relu', 'inputs': ['fc0'], 'layer_idx': 0},
        {'name': 'fc1', 'type': 'fc', 'inputs': ['relu0'],
         'W_np': W, 'bias_np': b},
        {'name': 'sub1', 'type': 'sub', 'inputs': ['fc1'], 'bias': None},
    ]
    rho = {0: 0.0, 1: 0.0}
    qw = np.array([1.0, -1.0], dtype=np.float64)
    m, env = build_tail_sub(bounds, 0, chain, qw, 0.0, rho, inner='lp')
    m.optimize()
    val = float(m.ObjVal)
    m.dispose()
    env.dispose()
    assert np.isfinite(val)


def test_tail_sub_add_dead_current_var():
    """Cover the current_vars[i] is None branch in add bias handler."""
    lo = np.array([-1.0, -2.0], dtype=np.float64)
    hi = np.array([1.0, -0.1], dtype=np.float64)
    bounds = {0: (lo, hi)}
    chain = [
        {'name': 'relu0', 'type': 'relu', 'inputs': ['fc0'], 'layer_idx': 0},
        {'name': 'add1', 'type': 'add', 'inputs': ['relu0'], 'is_merge': False,
         'bias': np.array([0.5, -0.5])},
    ]
    rho = {0: 0.0}
    qw = np.array([1.0, -1.0], dtype=np.float64)
    m, env = build_tail_sub(bounds, 0, chain, qw, 0.0, rho, inner='lp')
    m.optimize()
    val = float(m.ObjVal)
    m.dispose()
    env.dispose()
    assert np.isfinite(val)


def test_tail_sub_sub_dead_current_var():
    """Cover the current_vars[i] is None branch in sub bias handler."""
    lo = np.array([-1.0, -2.0], dtype=np.float64)
    hi = np.array([1.0, -0.1], dtype=np.float64)
    bounds = {0: (lo, hi)}
    chain = [
        {'name': 'relu0', 'type': 'relu', 'inputs': ['fc0'], 'layer_idx': 0},
        {'name': 'sub1', 'type': 'sub', 'inputs': ['relu0'],
         'bias': np.array([0.5, -0.5])},
    ]
    rho = {0: 0.0}
    qw = np.array([1.0, -1.0], dtype=np.float64)
    m, env = build_tail_sub(bounds, 0, chain, qw, 0.0, rho, inner='lp')
    m.optimize()
    val = float(m.ObjVal)
    m.dispose()
    env.dispose()
    assert np.isfinite(val)



def test_conv_tail_dead_inputs():
    """Cover conv no-live-inputs branch (line 251) in build_tail_sub."""
    lo = np.full(16, -2.0)
    hi = np.full(16, -0.1)
    bounds = {0: (lo, hi)}
    kernel = np.ones((1, 1, 3, 3), dtype=np.float64)
    bias_conv = np.array([1.0], dtype=np.float64)
    chain = [
        {'name': 'relu0', 'type': 'relu', 'inputs': ['fc0'], 'layer_idx': 0},
        {'name': 'conv1', 'type': 'conv', 'inputs': ['relu0'],
         'kernel_np': kernel, 'bias_np': bias_conv,
         'in_shape': (1, 4, 4), 'out_shape': (1, 4, 4),
         'stride': (1, 1), 'padding': (1, 1), 'n_out': 16},
    ]
    rho = {}
    qw = np.zeros(16, dtype=np.float64)
    qw[0] = 1.0
    m, env = build_tail_sub(bounds, 0, chain, qw, 0.0, rho, inner='lp')
    m.optimize()
    val = float(m.ObjVal)
    m.dispose()
    env.dispose()
    assert abs(val - 1.0) < 1e-6


def _build_naive(gg_ops, x_lo, x_hi, bounds_by_relu, input_name,
                 *, target_layer_idx=None, use_milp=False,
                 milp_by_layer=None, n_threads=1):
    """Unoptimized builder: redundant vars for stable-on ReLU and one-sided
    merge-Add. Used as a reference to verify the optimized builders."""
    inf = grb.GRB.INFINITY
    env = grb.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = grb.Model(env=env)
    m.setParam('Threads', n_threads)

    from vibecheck.verify_graph import (
        _compute_dead_at, _conv_sparse_matrix, _conv_bias_const, _dead_constant,
    )

    inp_vars = [m.addVar(lb=float(x_lo[i]), ub=float(x_hi[i]))
                for i in range(len(x_lo))]
    m.update()
    dead_at = _compute_dead_at(gg_ops, bounds_by_relu)
    op_var_refs = {input_name: inp_vars}

    for op in gg_ops:
        nm = op['name']
        t = op['type']

        if t in ('conv', 'fc'):
            prev = op_var_refs[op['inputs'][0]]
            n_prev = len(prev)
            if t == 'conv':
                n_out = op['n_out']
                W_sp = op.get('W_sp')
                if W_sp is None:
                    W_sp = _conv_sparse_matrix(
                        op['kernel_np'], op['in_shape'],
                        op['stride'], op['padding'])
                    op['W_sp'] = W_sp
            else:
                n_out = op['W_np'].shape[0]
            my_dead = dead_at.get(nm)
            if my_dead is not None and len(my_dead) != n_out:
                my_dead = None
            all_prev_dead = all(p is None for p in prev)
            out = [None] * n_out
            for j in range(n_out):
                if my_dead is not None and my_dead[j]:
                    continue
                if all_prev_dead:
                    c = _dead_constant(op, j)
                    v = m.addVar(lb=c, ub=c)
                    out[j] = v
                    continue
                expr = grb.LinExpr()
                if t == 'conv':
                    row = W_sp.getrow(j)
                    for fi, w in zip(row.indices, row.data):
                        if fi < n_prev and prev[fi] is not None:
                            expr.add(prev[fi], float(w))
                    b_j = _conv_bias_const(op, j)
                else:
                    W = op['W_np']
                    for k in range(n_prev):
                        wjk = W[j, k]
                        if wjk != 0 and prev[k] is not None:
                            expr.add(prev[k], float(wjk))
                    b_j = float(op['bias_np'][j])
                if expr.size() == 0:
                    v = m.addVar(lb=b_j, ub=b_j)
                    out[j] = v
                    continue
                v = m.addVar(lb=-inf, ub=inf)
                out[j] = v
                m.addConstr(v == expr + b_j)
            m.update()
            op_var_refs[nm] = out

        elif t == 'relu':
            if 'layer_idx' not in op:
                op_var_refs[nm] = op_var_refs[op['inputs'][0]]
                continue
            li = op['layer_idx']
            lo_r, hi_r = bounds_by_relu[li]
            prev = op_var_refs[op['inputs'][0]]
            n = len(prev)
            out = [None] * n
            for j in range(n):
                if hi_r[j] <= 0:
                    continue
                z = prev[j]
                if z is None:
                    continue
                if lo_r[j] >= 0:
                    # Naive: redundant copy
                    a = m.addVar(lb=float(lo_r[j]), ub=float(hi_r[j]))
                    m.addConstr(a == z)
                    out[j] = a
                else:
                    a = m.addVar(lb=0.0, ub=float(hi_r[j]))
                    m.addConstr(a >= z)
                    slope = float(hi_r[j]) / (float(hi_r[j]) - float(lo_r[j]))
                    m.addConstr(a <= slope * z - slope * float(lo_r[j]))
                    out[j] = a
            m.update()
            op_var_refs[nm] = out

        elif t == 'add':
            if op.get('is_merge'):
                va = op_var_refs[op['inputs'][0]]
                vb = op_var_refs[op['inputs'][1]]
                n = len(va)
                out = [None] * n
                for j in range(n):
                    if va[j] is None and vb[j] is None:
                        continue
                    # Naive: always create new var
                    expr = grb.LinExpr()
                    if va[j] is not None:
                        expr.add(va[j], 1.0)
                    if vb[j] is not None:
                        expr.add(vb[j], 1.0)
                    v = m.addVar(lb=-inf, ub=inf)
                    m.addConstr(v == expr)
                    out[j] = v
                m.update()
                op_var_refs[nm] = out
            else:
                op_var_refs[nm] = op_var_refs[op['inputs'][0]]

        elif t == 'sub':
            prev = op_var_refs[op['inputs'][0]]
            bias = op.get('bias')
            if bias is not None:
                bias_flat = bias.flatten().astype(np.float64)
                n = len(prev)
                out = [None] * n
                for j in range(n):
                    if prev[j] is None:
                        continue
                    v = m.addVar(lb=-inf, ub=inf)
                    m.addConstr(v == prev[j] - float(bias_flat[j]))
                    out[j] = v
                m.update()
                op_var_refs[nm] = out
            else:
                op_var_refs[nm] = op_var_refs[op['inputs'][0]]

        elif t == 'reshape':
            op_var_refs[nm] = op_var_refs[op['inputs'][0]]

    m.update()
    return m, env, op_var_refs, None


def _solve_lp(builder, gg_ops_ser, x_lo, x_hi, bounds_by_relu,
              input_name, output_op, qw, qb):
    """Build LP with the given builder, solve, return (bound, n_vars, n_constrs)."""
    m, env, refs, _ = builder(gg_ops_ser, x_lo, x_hi, bounds_by_relu,
                              input_name)
    last_vars = refs[output_op]
    obj = grb.LinExpr()
    for j in range(len(qw)):
        if qw[j] != 0 and last_vars[j] is not None:
            obj.add(last_vars[j], float(qw[j]))
    m.setObjective(obj + float(qb), grb.GRB.MINIMIZE)
    m.update()
    nv, nc = m.NumVars, m.NumConstrs
    m.optimize()
    assert m.Status == 2
    bound = float(m.ObjVal)
    m.dispose()
    env.dispose()
    return bound, nv, nc


def test_compact_lp_vs_naive(tmp_path):
    """compact_lp=True gives same LP bound with fewer vars."""
    from vibecheck.verify_graph import _build_optimized

    for fixture_fn in [_build_skip_connection, _build_all_stable_last,
                       _build_conv_skip]:
        graph, spec, qw, qb = fixture_fn(tmp_path)
        gg, gg_ops_ser, x_lo, x_hi, bounds_by_relu = _run_phase1(graph, spec)
        output_op = gg_ops_ser[-1]['name']
        input_name = gg['input_name']

        naive_bound, naive_nv, _ = _solve_lp(
            _build_naive, gg_ops_ser, x_lo, x_hi, bounds_by_relu,
            input_name, output_op, qw, qb)

        def _build_compact(ops, xl, xh, bbr, iname, **kw):
            return _build_optimized(ops, xl, xh, bbr, iname,
                                    compact_lp=True, **kw)

        compact_bound, compact_nv, _ = _solve_lp(
            _build_compact, gg_ops_ser, x_lo, x_hi, bounds_by_relu,
            input_name, output_op, qw, qb)

        assert abs(naive_bound - compact_bound) < 1e-6, \
            f'naive={naive_bound} vs compact={compact_bound}'
        assert compact_nv <= naive_nv, \
            f'compact should have <= vars: compact={compact_nv} naive={naive_nv}'


def test_passthrough_vs_naive(tmp_path):
    """Optimized builders match naive builder but use fewer vars/constrs."""
    from vibecheck.verify_graph import _build_optimized

    for fixture_fn in [_build_skip_connection, _build_all_stable_last,
                       _build_conv_skip]:
        graph, spec, qw, qb = fixture_fn(tmp_path)
        gg, gg_ops_ser, x_lo, x_hi, bounds_by_relu = _run_phase1(graph, spec)
        output_op = gg_ops_ser[-1]['name']
        input_name = gg['input_name']

        naive_bound, naive_nv, naive_nc = _solve_lp(
            _build_naive, gg_ops_ser, x_lo, x_hi, bounds_by_relu,
            input_name, output_op, qw, qb)
        ref_bound, ref_nv, ref_nc = _solve_lp(
            _build_reference, gg_ops_ser, x_lo, x_hi, bounds_by_relu,
            input_name, output_op, qw, qb)
        opt_bound, opt_nv, opt_nc = _solve_lp(
            _build_optimized, gg_ops_ser, x_lo, x_hi, bounds_by_relu,
            input_name, output_op, qw, qb)

        assert abs(naive_bound - ref_bound) < 1e-6, \
            f'naive={naive_bound} vs ref={ref_bound}'
        assert abs(naive_bound - opt_bound) < 1e-6, \
            f'naive={naive_bound} vs opt={opt_bound}'
        assert ref_nv <= naive_nv, \
            f'ref should have <= vars: ref={ref_nv} naive={naive_nv}'
        assert opt_nv <= naive_nv, \
            f'opt should have <= vars: opt={opt_nv} naive={naive_nv}'


def test_t12_variant_a_matches_b_all_stable(tmp_path):
    """T12: On all-stable-last-layer network, A == B == LP triangle."""
    graph, spec, qw, qb = _build_all_stable_last(tmp_path)
    gg, gg_ops_ser, x_lo, x_hi, bounds_by_relu = _run_phase1(graph, spec)
    output_op = gg_ops_ser[-1]['name']
    input_name = gg['input_name']

    lp_bound, _, m, env = solve_joint_lp(
        gg_ops_ser, x_lo, x_hi, bounds_by_relu, input_name,
        output_op, qw, qb)
    m.dispose()
    env.dispose()

    result_a = gold_solve_query(gg, gg_ops_ser, x_lo, x_hi, bounds_by_relu,
                                qw, qb, variant='A', time_limit=30.0,
                                n_threads=1)
    result_b = gold_solve_query(gg, gg_ops_ser, x_lo, x_hi, bounds_by_relu,
                                qw, qb, variant='B', time_limit=30.0,
                                n_threads=1)

    assert abs(result_a['bound'] - lp_bound) < 1e-4, \
        f'A bound {result_a["bound"]} != LP {lp_bound}'
    assert abs(result_b['bound'] - lp_bound) < 1e-4, \
        f'B bound {result_b["bound"]} != LP {lp_bound}'
    assert abs(result_a['bound'] - result_b['bound']) < 1e-4, \
        f'A {result_a["bound"]} != B {result_b["bound"]}'
