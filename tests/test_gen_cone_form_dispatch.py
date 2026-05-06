"""Regression tests for the gen-cone LP/MILP form dispatch.

This file guards against the bug observed on mnist_fc_256x6 prop_5
where the rec_zono *piggyback* path produced rows in **zono** form
(e_new ∈ [-1, 1] with μ-scaled e_new column) but the per-target-neuron
LP/MILP builder ``_build_gen_cone_lp`` interpreted them in **alpha**
form (a ∈ [0, hi] with literal-weight column). The mismatch was sound
but materially looser — a single L=1 neuron's MILP optimum changed
from −11.531 (correct) to −14.886 (mixed-form artefact).

The tests below pin down the contract:

  1. Rows produced by ``precompute_gen_state`` are tagged ``form='alpha'``
     and rows produced by ``_record_zono_pre_relu_rows`` are tagged
     ``form='phase1'``.
  2. ``_build_gen_cone_lp`` rejects entries with a non-alpha tag.
  3. ``_build_gen_cone_lp_phase1`` rejects entries with the alpha tag.
  4. Both consistent dispatches (alpha rows + alpha builder; phase1 rows
     + phase1 builder) produce the SAME MILP optimum on a small network
     when every upstream unstable is binarised.
"""
import numpy as np
import pytest
import torch

from vibecheck.network import ComputeGraph
from vibecheck.zonotope import TorchZonotope
from vibecheck.verify_graph import (
    _build_gen_cone_lp,
    _build_gen_cone_lp_phase1,
    _build_gen_rows_reverse_map,
    _dependency_cone,
    _gen_cone_state,
    _record_zono_pre_relu_rows,
)


@pytest.fixture
def tiny_two_relu_state():
    """Small FC2 → ReLU → FC2 → ReLU → FC1 with both upstream unstable."""
    import os, tempfile, onnx
    from onnx import helper, TensorProto, numpy_helper

    W1 = np.array([[1.0, -1.0], [-1.0, -2.0]], dtype=np.float32)
    b1 = np.array([0.5, 1.0], dtype=np.float32)
    W2 = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float32)
    b2 = np.array([0.0, 0.0], dtype=np.float32)
    W3 = np.array([[2.0, -3.0]], dtype=np.float32)
    b3 = np.array([0.0], dtype=np.float32)

    inp = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2])
    out = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1])
    inits = [
        numpy_helper.from_array(W1.T, name='W1'),
        numpy_helper.from_array(b1, name='b1'),
        numpy_helper.from_array(W2.T, name='W2'),
        numpy_helper.from_array(b2, name='b2'),
        numpy_helper.from_array(W3.T, name='W3'),
        numpy_helper.from_array(b3, name='b3'),
    ]
    nodes = [
        helper.make_node('Gemm', ['input', 'W1', 'b1'], ['z1'],
                         alpha=1.0, beta=1.0),
        helper.make_node('Relu', ['z1'], ['y1']),
        helper.make_node('Gemm', ['y1', 'W2', 'b2'], ['z2'],
                         alpha=1.0, beta=1.0),
        helper.make_node('Relu', ['z2'], ['y2']),
        helper.make_node('Gemm', ['y2', 'W3', 'b3'], ['output'],
                         alpha=1.0, beta=1.0),
    ]
    graph = helper.make_graph(nodes, 'tiny', [inp], [out], initializer=inits)
    m = helper.make_model(graph,
                          opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 7
    fd, p = tempfile.mkstemp(suffix='.onnx')
    os.close(fd)
    onnx.save(m, p)

    g = ComputeGraph.from_onnx(p, dtype=np.float32)
    os.remove(p)

    x_lo = np.array([-1.0, -1.0], dtype=np.float32)
    x_hi = np.array([1.0, 1.0], dtype=np.float32)
    return g, x_lo, x_hi


def test_precompute_state_entries_tagged_alpha(tiny_two_relu_state):
    """precompute_gen_state must tag every unstable entry with form='alpha'."""
    g, x_lo, x_hi = tiny_two_relu_state
    gg = g.gpu_graph(device=torch.device('cpu'), dtype=torch.float64)
    gg_ops = list(gg['ops'])

    # Run a forward to populate bounds_by_relu.
    from vibecheck.verify_zono_bnb import _forward_zonotope_graph
    xl = torch.tensor(x_lo, dtype=torch.float64).flatten()
    xh = torch.tensor(x_hi, dtype=torch.float64).flatten()
    sb, _ = _forward_zonotope_graph(xl, xh, gg, torch.device('cpu'),
                                     torch.float64)
    bbr = {L: (lo.cpu().numpy(), hi.cpu().numpy()) for L, (lo, hi) in sb.items()}

    # Find last relu op.
    last_relu = max((op for op in gg_ops if op['type'] == 'relu'),
                    key=lambda o: o.get('layer_idx', -1))
    state = _gen_cone_state(
        gg_ops, x_lo, x_hi, bbr, gg['input_name'],
        last_relu['name'], device=torch.device('cpu'),
        dtype=torch.float64)
    assert len(state['unstable_list']) > 0
    assert all(e['form'] == 'alpha' for e in state['unstable_list'])


def test_record_zono_entries_tagged_phase1(tiny_two_relu_state):
    """_record_zono_pre_relu_rows must tag every entry with form='phase1'."""
    g, x_lo, x_hi = tiny_two_relu_state
    rec_zono = {'gen_rows_by_layer': {}, 'col_origin': {}, 'n_input': 2}

    # Build a zonotope at the input box, propagate through one FC + ReLU.
    n = 2
    half = (x_hi - x_lo) / 2
    z = TorchZonotope(
        center=torch.tensor((x_lo + x_hi) / 2, dtype=torch.float64),
        generators=torch.diag(torch.tensor(half, dtype=torch.float64)),
    )
    W1 = torch.tensor([[1.0, -1.0], [-1.0, -2.0]], dtype=torch.float64)
    b1 = torch.tensor([0.5, 1.0], dtype=torch.float64)
    z.propagate_fc(W1, b1)
    lo_z, hi_z = z.bounds()
    bounds = (lo_z.cpu().numpy(), hi_z.cpu().numpy())
    _record_zono_pre_relu_rows(z, 0, bounds, rec_zono)

    layer = rec_zono['gen_rows_by_layer'][0]
    assert len(layer) > 0
    for j, entry in layer.items():
        assert entry['form'] == 'phase1', (
            f'neuron {j} entry tagged {entry["form"]!r}, '
            'expected "phase1"')


def test_alpha_builder_rejects_phase1_entries():
    """_build_gen_cone_lp must refuse phase1-tagged entries."""
    bogus = [{
        'layer_idx': 0, 'neuron_idx': 0,
        'c_in': 0.0, 'lo': -1.0, 'hi': 1.0, 'e_new_col': 2,
        'row_indices': np.array([0, 1], dtype=np.int32),
        'row_values': np.array([1.0, -1.0], dtype=np.float64),
        'form': 'phase1',
    }]
    with pytest.raises(AssertionError, match='form='):
        _build_gen_cone_lp(
            target_c=0.0,
            target_row_indices=np.array([0, 1, 2], dtype=np.int32),
            target_row_values=np.array([1.0, 1.0, 1.0],
                                        dtype=np.float64),
            input_cols=[0, 1],
            upstream_topo=bogus,
            milp_neurons=frozenset(),
            sense='min')


def test_phase1_builder_rejects_alpha_entries():
    """_build_gen_cone_lp_phase1 must refuse alpha-tagged entries."""
    bogus = [{
        'layer_idx': 0, 'neuron_idx': 0,
        'c_in': 0.0, 'lo': -1.0, 'hi': 1.0, 'e_new_col': 2,
        'row_indices': np.array([0, 1], dtype=np.int32),
        'row_values': np.array([1.0, -1.0], dtype=np.float64),
        'form': 'alpha',
    }]
    with pytest.raises(AssertionError, match='form='):
        _build_gen_cone_lp_phase1(
            target_c=0.0,
            target_row_indices=np.array([0, 1, 2], dtype=np.int32),
            target_row_values=np.array([1.0, 1.0, 1.0],
                                        dtype=np.float64),
            input_cols=[0, 1],
            upstream_topo=bogus,
            milp_neurons=frozenset(),
            sense='min')


def test_two_consistent_paths_match_when_all_binarized(tiny_two_relu_state):
    """Critical soundness/correctness invariant.

    With every upstream unstable in ``milp_set``, both consistent
    dispatch paths (alpha rows + alpha builder; phase1 rows + phase1
    builder) MUST give the SAME MILP optimum — they encode the exact
    ReLU on the same network. If this test fails, one of the two
    builders has a bug in its algebraic derivation.
    """
    import gurobipy as grb
    g, x_lo, x_hi = tiny_two_relu_state
    gg = g.gpu_graph(device=torch.device('cpu'), dtype=torch.float64)
    gg_ops = list(gg['ops'])

    # --- Get bbr from a plain forward.
    from vibecheck.verify_zono_bnb import _forward_zonotope_graph
    xl = torch.tensor(x_lo, dtype=torch.float64).flatten()
    xh = torch.tensor(x_hi, dtype=torch.float64).flatten()
    sb, _ = _forward_zonotope_graph(xl, xh, gg, torch.device('cpu'),
                                     torch.float64)
    bbr = {L: (lo.cpu().numpy(), hi.cpu().numpy())
           for L, (lo, hi) in sb.items()}

    # --- Alpha-form rows: precompute_gen_state up to L=1's relu.
    last_relu = max((op for op in gg_ops if op['type'] == 'relu'),
                    key=lambda o: o.get('layer_idx', -1))
    alpha_state = _gen_cone_state(
        gg_ops, x_lo, x_hi, bbr, gg['input_name'],
        last_relu['name'], device=torch.device('cpu'),
        dtype=torch.float64)
    alpha_rows, alpha_co = _build_gen_rows_reverse_map(alpha_state)
    alpha_n_input = alpha_state['n_input']

    # --- Phase1-form rows: live zono + _record_zono_pre_relu_rows.
    z = TorchZonotope(
        center=torch.tensor((x_lo + x_hi) / 2, dtype=torch.float64),
        generators=torch.diag(torch.tensor((x_hi - x_lo) / 2,
                                            dtype=torch.float64)),
    )
    rec_zono = {'gen_rows_by_layer': {}, 'col_origin': {}, 'n_input': 2}
    # Walk the ops, recording pre-ReLU rows just before each relu.
    state_z = {gg['input_name']: z}
    for op in gg_ops:
        nm = op['name']
        t = op['type']
        if t == 'fc':
            zz = state_z[op['inputs'][0]].copy()
            zz.propagate_fc(op['W'], op['bias'])
            state_z[nm] = zz
        elif t == 'relu':
            zz = state_z[op['inputs'][0]].copy()
            li = op.get('layer_idx')
            if li is not None:
                lo_b, hi_b = bbr[li]
                _record_zono_pre_relu_rows(zz, li, (lo_b, hi_b), rec_zono)
            zz.apply_relu()
            state_z[nm] = zz
        elif t == 'conv':
            zz = state_z[op['inputs'][0]].copy()
            zz.propagate_conv(op['kernel'], op['bias'], op['in_shape'],
                               op['stride'], op['padding'])
            state_z[nm] = zz
    phase1_rows = rec_zono['gen_rows_by_layer']
    phase1_co = rec_zono['col_origin']
    phase1_n_input = rec_zono['n_input']

    # Pick the last hidden layer's first unstable (alpha and phase1
    # share neuron indices since they're bound to the same network).
    target_li = max(phase1_rows.keys())
    common = sorted(set(alpha_rows.get(target_li, {}).keys())
                    & set(phase1_rows.get(target_li, {}).keys()))
    assert common, 'no common unstable neuron at target layer'
    j = common[0]

    def _solve(rows, co, ni, j, builder):
        entry = rows[target_li][j]
        nz = entry['row_indices']
        vals = entry['row_values']
        c_in = float(entry['c_in'])
        input_cols, upstream_topo = _dependency_cone(
            nz, vals, rows, co, ni)
        milp_set = frozenset(
            (e['layer_idx'], e['neuron_idx']) for e in upstream_topo)
        out = {}
        for sense in ('min', 'max'):
            m, env = builder(c_in, nz, vals, input_cols, upstream_topo,
                             milp_neurons=milp_set, sense=sense)
            m.setParam('OutputFlag', 0)
            m.setParam('TimeLimit', 60.0)
            m.setParam('MIPGap', 0.0)
            m.setParam('MIPGapAbs', 1e-9)
            m.optimize()
            assert m.Status == grb.GRB.OPTIMAL, (
                f'sense={sense} did not solve to optimality '
                f'(status={m.Status})')
            out[sense] = float(m.ObjVal)
            m.dispose()
            env.dispose()
        return out

    a = _solve(alpha_rows, alpha_co, alpha_n_input, j, _build_gen_cone_lp)
    p = _solve(phase1_rows, phase1_co, phase1_n_input, j,
               _build_gen_cone_lp_phase1)
    for sense in ('min', 'max'):
        assert abs(a[sense] - p[sense]) < 1e-6, (
            f'alpha {sense}={a[sense]:.6f} ≠ '
            f'phase1 {sense}={p[sense]:.6f} — '
            'two consistent paths must agree on exact MILP optimum')
