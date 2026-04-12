"""Tests for vibecheck.verify_ld (Lagrangian-decomposition solver)."""

import numpy as np
import onnx
import pytest
import torch
from onnx import helper, TensorProto, numpy_helper

from vibecheck.network import ComputeGraph
from vibecheck.settings import default_settings
from vibecheck.spec import VNNSpec, Conjunct, Constraint
from vibecheck.verify_graph import verify_graph, _serialize_gg_ops
from vibecheck import verify_ld


def _init(name, arr):
    return numpy_helper.from_array(arr.astype(np.float32), name=name)


def _input_val(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _save_and_load(model, tmp_path, name='m.onnx'):
    path = str(tmp_path / name)
    onnx.save(model, path)
    return ComputeGraph.from_onnx(path)


def _build_tiny_sequential_fc():
    """3-layer FC: 2 -> 3 -> 3 -> 2."""
    rng = np.random.RandomState(0)
    W1 = rng.randn(3, 2).astype(np.float32) * 0.5
    b1 = np.zeros(3, dtype=np.float32)
    W2 = rng.randn(3, 3).astype(np.float32) * 0.5
    b2 = np.zeros(3, dtype=np.float32)
    W3 = rng.randn(2, 3).astype(np.float32) * 0.5
    b3 = np.zeros(2, dtype=np.float32)
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
        nodes, 'tiny_fc',
        [_input_val('X', [1, 2])],
        [_input_val('Y', [1, 2])],
        inits)
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid('', 13)])


def _build_tiny_conv_fc():
    """1ch -> Conv(2ch,3x3) -> Relu -> Conv(4ch,3x3) -> Relu -> FC(2)."""
    rng = np.random.RandomState(0)
    k1 = rng.randn(2, 1, 3, 3).astype(np.float32) * 0.3
    b1 = np.zeros(2, dtype=np.float32)
    k2 = rng.randn(4, 2, 3, 3).astype(np.float32) * 0.3
    b2 = np.zeros(4, dtype=np.float32)
    Wo = rng.randn(2, 64).astype(np.float32) * 0.3
    bo = np.zeros(2, dtype=np.float32)
    nodes = [
        helper.make_node('Conv', ['X', 'k1', 'b1'], ['c1'],
                         kernel_shape=[3, 3], strides=[1, 1],
                         pads=[1, 1, 1, 1]),
        helper.make_node('Relu', ['c1'], ['r1']),
        helper.make_node('Conv', ['r1', 'k2', 'b2'], ['c2'],
                         kernel_shape=[3, 3], strides=[1, 1],
                         pads=[1, 1, 1, 1]),
        helper.make_node('Relu', ['c2'], ['r2']),
        helper.make_node('Flatten', ['r2'], ['flat'], axis=1),
        helper.make_node('Gemm', ['flat', 'Wo', 'bo'], ['Y'], transB=1),
    ]
    inits = [_init('k1', k1), _init('b1', b1),
             _init('k2', k2), _init('b2', b2),
             _init('Wo', Wo), _init('bo', bo)]
    graph = helper.make_graph(
        nodes, 'tiny_conv_fc',
        [_input_val('X', [1, 1, 4, 4])],
        [_input_val('Y', [1, 2])],
        inits)
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid('', 13)])


def _gg_and_ops(model, tmp_path, name='m.onnx'):
    """Load onnx model and return (gg, gg_ops, input_name) for inspection."""
    g = _save_and_load(model, tmp_path, name)
    gg = g.gpu_graph(torch.device('cpu'), torch.float64)
    return gg, _serialize_gg_ops(gg), gg['input_name']


def _easy_spec(n_in):
    """Verifiable spec: Y[0] >= 1e6 is unreachable on a small FC."""
    x = np.zeros(n_in, dtype=np.float32)
    eps = 0.01
    return VNNSpec(
        x_lo=x - eps, x_hi=x + eps,
        disjuncts=[Conjunct(
            [Constraint(index=0, op='>=', value=1e6)])])


def test_ld_settings_default_off():
    """Default settings must have ld_enabled=False so the new path is
    dormant until explicitly turned on."""
    s = default_settings()
    assert s.ld_enabled is False


def test_ld_enabled_gate_reaches_ld_phase(monkeypatch, tmp_path):
    """With ld_enabled=True and a forced CROWN-miss, the gated call
    fires and phase_ld telemetry appears in details."""
    from vibecheck import verify_graph as vg

    model = _build_tiny_sequential_fc()
    g = _save_and_load(model, tmp_path, 'tiny.onnx')
    spec = _easy_spec(2)

    original_backward = vg._spec_backward_graph

    def fake_backward(sb, xl, xh, gg, spec_ew, qids, nh, device, dtype,
                     return_ew=False):
        if return_ew:
            return original_backward(sb, xl, xh, gg, spec_ew, qids, nh,
                                     device, dtype, return_ew=True)
        lbs = {qi: -1.0 for qi in qids}
        return lbs, None

    monkeypatch.setattr(vg, '_spec_backward_graph', fake_backward)

    def fake_pgd(*a, **k):
        return False, None

    monkeypatch.setattr(vg, '_pgd_attack_general', fake_pgd)

    s = default_settings(device='cpu', total_timeout=30,
                         print_progress=False, ld_enabled=True,
                         ld_num_iterations=0)
    result, details = verify_graph(g, spec, s)
    assert 'phase_ld' in details['timing']
    assert details['ld']['ld_ran'] is True
    assert result == 'verified'


def test_classify_neurons_trivial():
    """Hand-built (l, u) with known stable-on / stable-off / ambiguous."""
    l = np.array([0.5, -1.0, -0.5, -2.0, 0.0, 0.0, -0.1])
    u = np.array([1.0,  0.5, -0.1,  0.0, 1.0, 0.0,  0.1])
    on, off, amb = verify_ld._classify_neurons(l, u)
    # partition check
    assert np.all(on.astype(int) + off.astype(int) + amb.astype(int) == 1)
    assert list(on)  == [True,  False, False, False, True,  False, False]
    assert list(off) == [False, False, True,  True,  False, True,  False]
    assert list(amb) == [False, True,  False, False, False, False, True]


def test_classify_neurons_empty():
    """Empty input returns empty masks."""
    l = np.array([], dtype=np.float64)
    u = np.array([], dtype=np.float64)
    on, off, amb = verify_ld._classify_neurons(l, u)
    assert on.shape == (0,) and off.shape == (0,) and amb.shape == (0,)


def test_extract_look_back_1_layers_sequential_fc(tmp_path):
    """Tiny 3-layer FC: expect three blocks (g1, g2, Y) with correct
    prev_layer_idx and layer_idx wiring."""
    model = _build_tiny_sequential_fc()
    gg, gg_ops, input_name = _gg_and_ops(model, tmp_path, 'tiny_fc.onnx')
    bounds_by_relu = {
        0: (np.full(3, -1.0), np.full(3, 1.0)),
        1: (np.full(3, -1.0), np.full(3, 1.0)),
    }
    blocks = verify_ld._extract_look_back_1_layers(
        gg_ops, bounds_by_relu, input_name)
    assert len(blocks) == 3
    # Block 0: g1 — first layer, no preceding ReLU
    assert blocks[0]['linear_op']['name'] == 'g1'
    assert blocks[0]['prev_layer_idx'] is None
    assert blocks[0]['input_source'] == 'x'
    assert blocks[0]['layer_idx'] == 0
    assert blocks[0]['is_merge'] is False
    assert blocks[0]['prev_bounds'] == (None, None)
    # Block 1: g2 — second layer, prev ReLU is layer 0
    assert blocks[1]['linear_op']['name'] == 'g2'
    assert blocks[1]['prev_layer_idx'] == 0
    assert blocks[1]['input_source'] == 'relu'
    assert blocks[1]['layer_idx'] == 1
    lo, hi = blocks[1]['prev_bounds']
    assert lo.shape == (3,) and hi.shape == (3,)
    assert np.allclose(lo, -1.0) and np.allclose(hi, 1.0)
    # Block 2: Y — output layer, no following ReLU -> layer_idx None
    assert blocks[2]['linear_op']['name'] == 'Y'
    assert blocks[2]['prev_layer_idx'] == 1
    assert blocks[2]['layer_idx'] is None


def test_extract_look_back_1_layers_passthrough_between_linear_and_relu():
    """Hand-built gg_ops where a non-merge Add(bias) sits between a
    linear op and its consuming ReLU. The extractor must walk through
    the Add to associate layer_idx with the linear op, and walk through
    the Add when tracing the next linear op's input backward."""
    gg_ops = [
        {'name': 'fc1', 'type': 'fc', 'inputs': ['X'],
         'W_np': np.eye(2, dtype=np.float32),
         'bias_np': np.zeros(2, dtype=np.float32)},
        {'name': 'add1', 'type': 'add', 'inputs': ['fc1'],
         'is_merge': False,
         'bias': np.array([0.1, -0.1], dtype=np.float32)},
        {'name': 'r1', 'type': 'relu', 'inputs': ['add1'], 'layer_idx': 0},
        {'name': 'fc2', 'type': 'fc', 'inputs': ['r1'],
         'W_np': np.eye(2, dtype=np.float32),
         'bias_np': np.zeros(2, dtype=np.float32)},
    ]
    bounds = {0: (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))}
    blocks = verify_ld._extract_look_back_1_layers(gg_ops, bounds, 'X')
    assert len(blocks) == 2
    # r1's layer_idx must map to fc1, not add1
    assert blocks[0]['linear_op']['name'] == 'fc1'
    assert blocks[0]['layer_idx'] == 0
    # fc2's prev walk must skip over add1 and find r1's layer_idx
    assert blocks[1]['linear_op']['name'] == 'fc2'
    assert blocks[1]['prev_layer_idx'] == 0
    assert blocks[1]['input_source'] == 'relu'


def _build_identity_skip_fc():
    """Tiny FC residual: Input(1,2) -> FC1 -> Relu -> FC2 -> Add(X) -> Relu -> FC3."""
    W1 = np.eye(2, dtype=np.float32) * 0.8
    b1 = np.array([0.1, -0.2], dtype=np.float32)
    W2 = np.eye(2, dtype=np.float32) * 0.5
    b2 = np.array([0.0, 0.0], dtype=np.float32)
    W3 = np.array([[1.0, -1.0]], dtype=np.float32)
    b3 = np.array([0.0], dtype=np.float32)
    nodes = [
        helper.make_node('Gemm', ['X', 'W1', 'b1'], ['g1'], transB=1),
        helper.make_node('Relu', ['g1'], ['r1']),
        helper.make_node('Gemm', ['r1', 'W2', 'b2'], ['g2'], transB=1),
        helper.make_node('Add', ['g2', 'X'], ['add']),
        helper.make_node('Relu', ['add'], ['r2']),
        helper.make_node('Gemm', ['r2', 'W3', 'b3'], ['Y'], transB=1),
    ]
    inits = [_init('W1', W1), _init('b1', b1),
             _init('W2', W2), _init('b2', b2),
             _init('W3', W3), _init('b3', b3)]
    graph = helper.make_graph(
        nodes, 'id_skip_fc',
        [_input_val('X', [1, 2])],
        [_input_val('Y', [1, 1])],
        inits)
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid('', 13)])


def test_extract_look_back_1_layers_resnet_identity_skip(tmp_path):
    """Identity-skip FC residual: the extractor must emit three blocks
    (layer 0 sequential, layer 1 MERGE with 2 branches, Y spec)."""
    model = _build_identity_skip_fc()
    gg, gg_ops, input_name = _gg_and_ops(model, tmp_path, 'res.onnx')
    bounds_by_relu = {
        0: (np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
        1: (np.array([-1.5, -1.5]), np.array([1.5, 1.5])),
    }
    blocks = verify_ld._extract_look_back_1_layers(
        gg_ops, bounds_by_relu, input_name)
    assert len(blocks) == 3
    # Block 0: sequential layer 0 (g1)
    assert blocks[0]['layer_idx'] == 0
    assert blocks[0]['is_merge'] is False
    assert len(blocks[0]['branches']) == 1
    assert blocks[0]['branches'][0]['linear_op']['name'] == 'g1'
    # Block 1: MERGE layer 1 (g2 + identity of X)
    assert blocks[1]['layer_idx'] == 1
    assert blocks[1]['is_merge'] is True
    assert len(blocks[1]['branches']) == 2
    main_branch = blocks[1]['branches'][0]
    skip_branch = blocks[1]['branches'][1]
    assert main_branch['linear_op']['name'] == 'g2'
    assert main_branch['prev_layer_idx'] == 0
    assert skip_branch['linear_op'] is None
    assert skip_branch['prev_layer_idx'] is None
    # Block 2: spec (Y), sequential
    assert blocks[2]['layer_idx'] is None
    assert blocks[2]['is_merge'] is False
    assert blocks[2]['branches'][0]['linear_op']['name'] == 'Y'
    assert blocks[2]['branches'][0]['prev_layer_idx'] == 1


def test_build_per_neuron_milp_merge_identity_skip():
    """Hand-built merge block with a main FC branch + identity@input
    skip. Solving at producer_coef=1 must match the sum c2[j] + X[j]
    min under the box constraints."""
    # Main branch: W=[[2, -1]], b=[0.3], prev layer bounds [-0.5, 0.5]^2
    main_op = {
        'name': 'main',
        'type': 'fc',
        'inputs': ['r_prev'],
        'W_np': np.array([[2.0, -1.0]], dtype=np.float64),
        'bias_np': np.array([0.3], dtype=np.float64),
    }
    block = {
        'layer_idx': 1,
        'is_merge': True,
        'linear_op': main_op,
        'prev_layer_idx': 0,
        'prev_bounds': (np.array([-0.5, -0.5]), np.array([0.5, 0.5])),
        'input_source': 'relu',
        'branches': [
            {
                'linear_op': main_op,
                'prev_layer_idx': 0,
                'prev_bounds': (np.array([-0.5, -0.5]),
                                np.array([0.5, 0.5])),
            },
            {
                'linear_op': None,  # identity from input
                'prev_layer_idx': None,
                'prev_bounds': (None, None),
            },
        ],
    }
    x_lo = np.array([-0.2, -0.3])
    x_hi = np.array([0.2, 0.3])
    refs = verify_ld._build_per_neuron_milp(block, 0, x_lo, x_hi)
    try:
        assert refs['is_merge'] is True
        # Main branch's vars (two prev-layer neurons, both ambiguous)
        # go into z_in_vars for coupling; skip's input vars (x[0])
        # go into aux_vars.
        assert 0 in refs['z_in_vars']
        assert 1 in refs['z_in_vars']
        # Identity skip at index 0 maps input[0] to aux_vars
        assert 0 in refs['aux_vars']
        verify_ld._set_objective_per_neuron(
            refs, producer_coef=1.0, rho_prev=np.zeros(2))
        res = verify_ld._solve_subproblem(refs, timeout=5.0)
    finally:
        refs['model'].dispose()
        refs['env'].dispose()
    # z_out = 2*a_prev[0] - 1*a_prev[1] + 0.3 + x[0]
    #       where a_prev[i] = relu(z_prev[i]), z_prev in [-0.5, 0.5]
    #       x[0] in [-0.2, 0.2]
    # min: a_prev[0]=0, a_prev[1]=0.5 (max), x[0]=-0.2
    #   -> 0 - 0.5 + 0.3 - 0.2 = -0.4
    assert res['obj_val'] == pytest.approx(-0.4, abs=1e-6)


def test_extract_look_back_1_layers_conv(tmp_path):
    """Tiny conv+conv+fc: expect three blocks with the reshape/flatten
    in between properly skipped."""
    model = _build_tiny_conv_fc()
    gg, gg_ops, input_name = _gg_and_ops(model, tmp_path, 'tiny_conv.onnx')
    bounds_by_relu = {
        0: (np.full(32, -0.5), np.full(32, 0.5)),
        1: (np.full(64, -0.3), np.full(64, 0.3)),
    }
    blocks = verify_ld._extract_look_back_1_layers(
        gg_ops, bounds_by_relu, input_name)
    assert len(blocks) == 3
    # Block 0: c1 — first layer
    assert blocks[0]['linear_op']['name'] == 'c1'
    assert blocks[0]['linear_op']['type'] == 'conv'
    assert blocks[0]['prev_layer_idx'] is None
    assert blocks[0]['input_source'] == 'x'
    assert blocks[0]['layer_idx'] == 0
    # Block 1: c2 — prev relu is layer 0 (32 neurons)
    assert blocks[1]['linear_op']['name'] == 'c2'
    assert blocks[1]['prev_layer_idx'] == 0
    assert blocks[1]['layer_idx'] == 1
    lo, hi = blocks[1]['prev_bounds']
    assert lo.shape == (32,) and hi.shape == (32,)
    # Block 2: Y (fc) — prev relu is layer 1 (64 neurons), pass through reshape
    assert blocks[2]['linear_op']['name'] == 'Y'
    assert blocks[2]['linear_op']['type'] == 'fc'
    assert blocks[2]['prev_layer_idx'] == 1
    assert blocks[2]['layer_idx'] is None
    lo, hi = blocks[2]['prev_bounds']
    assert lo.shape == (64,) and hi.shape == (64,)


def _fake_fc_block(W, b, prev_lo, prev_hi):
    """Hand-build a look-back-1 block for an FC op with given weights,
    bias, and previous-layer bounds. No gg_ops round-trip."""
    return {
        'linear_op': {
            'name': 'fc',
            'type': 'fc',
            'inputs': ['prev'],
            'W_np': np.asarray(W, dtype=np.float64),
            'bias_np': np.asarray(b, dtype=np.float64),
        },
        'layer_idx': None,
        'prev_layer_idx': 0,
        'prev_bounds': (np.asarray(prev_lo, dtype=np.float64),
                        np.asarray(prev_hi, dtype=np.float64)),
        'input_source': 'relu',
        'is_merge': False,
    }


def _fake_conv_block(kernel, bias, in_shape, out_shape, prev_lo, prev_hi,
                    stride=(1, 1), padding=(1, 1)):
    """Hand-build a look-back-1 block for a conv op."""
    return {
        'linear_op': {
            'name': 'conv',
            'type': 'conv',
            'inputs': ['prev'],
            'kernel_np': np.asarray(kernel, dtype=np.float64),
            'bias_np': np.asarray(bias, dtype=np.float64),
            'in_shape': in_shape,
            'out_shape': out_shape,
            'stride': stride,
            'padding': padding,
            'n_out': int(np.prod(out_shape)),
        },
        'layer_idx': None,
        'prev_layer_idx': 0,
        'prev_bounds': (np.asarray(prev_lo, dtype=np.float64),
                        np.asarray(prev_hi, dtype=np.float64)),
        'input_source': 'relu',
        'is_merge': False,
    }


def test_build_per_neuron_milp_matches_exact_small():
    """1-hidden-neuron network. Spec subproblem on the last layer with
    producer_coef=1 solves for min z_out, which we compare against a
    brute-force enumeration over the ReLU on/off cases."""
    # z_in[0] in [-0.5, 0.5] (ambiguous). W_last = 2, b_last = -0.3.
    block = _fake_fc_block(
        W=[[2.0]], b=[-0.3], prev_lo=[-0.5], prev_hi=[0.5])
    refs = verify_ld._build_per_neuron_milp(block, 0, None, None)
    try:
        verify_ld._set_objective_per_neuron(
            refs, producer_coef=1.0, rho_prev=np.array([0.0]))
        res = verify_ld._solve_subproblem(refs, timeout=5.0)
    finally:
        refs['model'].dispose()
        refs['env'].dispose()
    # Brute force:
    #   off: a=0, z_out=-0.3; on: a=z in [0, 0.5], z_out=2z-0.3 in [-0.3, 0.7]
    # min = -0.3
    assert res['obj_val'] == pytest.approx(-0.3, abs=1e-6)
    # Negative-weight variant
    block2 = _fake_fc_block(
        W=[[-1.0]], b=[0.1], prev_lo=[-0.5], prev_hi=[0.5])
    refs2 = verify_ld._build_per_neuron_milp(block2, 0, None, None)
    try:
        verify_ld._set_objective_per_neuron(
            refs2, producer_coef=1.0, rho_prev=np.array([0.0]))
        res2 = verify_ld._solve_subproblem(refs2, timeout=5.0)
    finally:
        refs2['model'].dispose()
        refs2['env'].dispose()
    # off: z_out=0.1; on: z in [0, 0.5], z_out=-z+0.1, min=-0.4
    assert res2['obj_val'] == pytest.approx(-0.4, abs=1e-6)


def test_build_per_neuron_milp_stable_handling():
    """Stable_on, stable_off, and ambiguous neurons in the same rf.
    Verifies variable counts and classification sets, and that the
    solved optimum matches a hand-computed minimum."""
    # Three hidden neurons:
    #   0: [0.2, 0.8]  (stable_on)
    #   1: [-0.3, -0.1] (stable_off)
    #   2: [-0.2, 0.3]  (ambiguous)
    block = _fake_fc_block(
        W=[[1.0, 1.0, 1.0]], b=[0.0],
        prev_lo=[0.2, -0.3, -0.2],
        prev_hi=[0.8, -0.1, 0.3])
    refs = verify_ld._build_per_neuron_milp(block, 0, None, None)
    try:
        # Variables and classification
        assert set(refs['z_in_vars'].keys()) == {0, 2}  # stable_off dropped
        assert refs['stable_on'] == {0}
        assert refs['ambiguous'] == {2}
        assert refs['rf_indices'] == [0, 2]
        # Solve at producer_coef=1, rho=0
        verify_ld._set_objective_per_neuron(
            refs, producer_coef=1.0, rho_prev=np.zeros(3))
        res = verify_ld._solve_subproblem(refs, timeout=5.0)
    finally:
        refs['model'].dispose()
        refs['env'].dispose()
    # z_out = z_in[0] + a[2]. min when z_in[0]=0.2 and a[2]=0 -> 0.2
    assert res['obj_val'] == pytest.approx(0.2, abs=1e-6)


def test_build_per_neuron_milp_conv_receptive_field():
    """1-channel 4x4 input, 1-channel 3x3 conv, pad=1. MILP for output
    neuron (1,1) flat idx=5 must have exactly 9 z_in vars (its rf)."""
    rng = np.random.RandomState(0)
    kernel = rng.randn(1, 1, 3, 3).astype(np.float64)
    bias = np.array([0.1], dtype=np.float64)
    block = _fake_conv_block(
        kernel=kernel, bias=bias,
        in_shape=(1, 4, 4), out_shape=(1, 4, 4),
        # All 16 inputs are ambiguous so they'd all get vars IF included
        prev_lo=[-1.0] * 16, prev_hi=[1.0] * 16,
        stride=(1, 1), padding=(1, 1))
    target_j = 1 * 4 + 1  # output neuron at (1, 1)
    refs = verify_ld._build_per_neuron_milp(block, target_j, None, None)
    try:
        # rf of (1,1) in a 4x4 with 3x3 kernel + pad=1 = the 3x3 window
        # covering rows 0..2, cols 0..2.
        expected_rf = sorted({r * 4 + c for r in range(3) for c in range(3)})
        assert refs['rf_indices'] == expected_rf
        assert refs['ambiguous'] == set(expected_rf)
        # No stable neurons here
        assert refs['stable_on'] == set()
    finally:
        refs['model'].dispose()
        refs['env'].dispose()


def test_build_per_neuron_milp_first_layer_input_box():
    """First-layer block: input vars take the input box bounds; no
    ReLU encoding or stability classification."""
    W = np.array([[1.0, 2.0]], dtype=np.float64)
    b = np.array([0.5], dtype=np.float64)
    block = {
        'linear_op': {'type': 'fc', 'W_np': W, 'bias_np': b},
        'layer_idx': 0, 'prev_layer_idx': None,
        'prev_bounds': (None, None),
        'input_source': 'x', 'is_merge': False,
    }
    x_lo = np.array([-1.0, -2.0])
    x_hi = np.array([1.0, 2.0])
    refs = verify_ld._build_per_neuron_milp(block, 0, x_lo, x_hi)
    try:
        assert refs['is_first_layer'] is True
        assert set(refs['z_in_vars'].keys()) == {0, 1}
        assert refs['stable_on'] == set()
        assert refs['ambiguous'] == set()
        verify_ld._set_objective_per_neuron(
            refs, producer_coef=1.0, rho_prev=None)
        res = verify_ld._solve_subproblem(refs, timeout=5.0)
    finally:
        refs['model'].dispose()
        refs['env'].dispose()
    # min (x0 + 2*x1 + 0.5) over x in [-1,1] x [-2,2]
    # = -1 + 2*(-2) + 0.5 = -4.5
    assert res['obj_val'] == pytest.approx(-4.5, abs=1e-6)


def test_set_objective_consumer_term_applied():
    """With non-zero rho_prev the consumer term must influence the
    optimum: obj = 1*z_out - rho*z_in must move the minimum from the
    rho=0 optimum toward whatever minimizes the -rho*z_in contribution."""
    # Single ambiguous neuron, W=1, b=0 -> z_out = a in [0, u]
    block = _fake_fc_block(
        W=[[1.0]], b=[0.0], prev_lo=[-0.5], prev_hi=[0.5])
    refs = verify_ld._build_per_neuron_milp(block, 0, None, None)
    try:
        # producer_coef=1, rho_prev=[2.0] on z_in which is in [-0.5, 0.5]
        # obj = z_out - 2*z_in. Minimized when z_in is max (0.5) and
        #   ReLU on (so a = 0.5 -> z_out = 0.5). obj = 0.5 - 1 = -0.5.
        # Alternative: z_in anything, ReLU off (a = 0 -> z_out = 0). Then
        #   z_in free; min of -2*z_in over [-0.5, 0] (ReLU off requires
        #   z_in <= 0 from the big-M encoding). min at z_in=0 -> obj=0.
        # So global min is -0.5.
        verify_ld._set_objective_per_neuron(
            refs, producer_coef=1.0, rho_prev=np.array([2.0]))
        res = verify_ld._solve_subproblem(refs, timeout=5.0)
    finally:
        refs['model'].dispose()
        refs['env'].dispose()
    assert res['obj_val'] == pytest.approx(-0.5, abs=1e-6)


def test_build_per_neuron_milp_empty_rf():
    """FC row of all zeros gives a bias-only z_out."""
    block = _fake_fc_block(
        W=[[0.0, 0.0]], b=[1.5],
        prev_lo=[-1.0, -1.0], prev_hi=[1.0, 1.0])
    refs = verify_ld._build_per_neuron_milp(block, 0, None, None)
    try:
        assert refs['z_in_vars'] == {}
        assert refs['rf_indices'] == []
        verify_ld._set_objective_per_neuron(
            refs, producer_coef=1.0, rho_prev=np.zeros(2))
        res = verify_ld._solve_subproblem(refs, timeout=5.0)
    finally:
        refs['model'].dispose()
        refs['env'].dispose()
    assert res['obj_val'] == pytest.approx(1.5, abs=1e-6)


def _exact_spec_min_via_milp(gg_ops, bounds_by_relu, x_lo, x_hi,
                             input_name, query_w, query_bias):
    """Solve the full-network MILP for `min query_w @ y + query_bias`.
    Reuses verify_graph's `_build_sparse_neuron_graph` on the final
    output neuron, then manually sums weighted copies for multi-output
    queries. For single-output queries this is a direct sparse build.
    """
    from vibecheck.verify_graph import _build_sparse_neuron_graph
    import gurobipy as grb

    last_op = None
    for op in gg_ops:
        if op['type'] in ('conv', 'fc'):
            last_op = op
    assert last_op is not None
    # Build a combined objective across output neurons with w[j] != 0.
    nz = np.nonzero(np.asarray(query_w))[0]
    total_lb = float(query_bias)
    for j in nz:
        m, env, tv = _build_sparse_neuron_graph(
            gg_ops, x_lo, x_hi, bounds_by_relu, last_op['name'],
            int(j), input_name, use_milp=True, n_threads=1)
        w = float(query_w[j])
        sense = grb.GRB.MINIMIZE if w > 0 else grb.GRB.MAXIMIZE
        m.setObjective(tv, sense)
        m.setParam('TimeLimit', 30.0)
        m.optimize()
        assert m.Status in (grb.GRB.OPTIMAL, grb.GRB.SUBOPTIMAL,
                            grb.GRB.USER_OBJ_LIMIT)
        val = float(m.ObjVal) if m.SolCount > 0 else float(m.ObjBound)
        total_lb += w * val
        m.dispose()
        env.dispose()
    return total_lb


def _ld_settings(**overrides):
    """Minimal LD-tuned settings for fast iteration tests."""
    base = dict(
        device='cpu', total_timeout=60, print_progress=False,
        ld_enabled=True, ld_num_iterations=0, ld_early_stop=False,
        ld_initial_step=1e-2, ld_final_step=1e-4,
        ld_step_schedule='linear_decay', ld_log_interval=0,
        ld_subproblem_timeout=5.0,
    )
    base.update(overrides)
    return default_settings(**base)


def _prep_tiny_fc(tmp_path, seed=0):
    """Load the tiny 3-layer FC, compute bounds_by_relu via the
    existing pipeline, return (gg_ops, bounds_by_relu, x_lo, x_hi,
    input_name)."""
    import torch
    from vibecheck.verify_zono_bnb import _forward_zonotope_graph
    model = _build_tiny_sequential_fc()
    g = _save_and_load(model, tmp_path, f'tiny_fc_{seed}.onnx')
    gg = g.gpu_graph(torch.device('cpu'), torch.float64)
    gg_ops = _serialize_gg_ops(gg)
    x_lo = np.full(2, -1.0)
    x_hi = np.full(2, 1.0)
    xl = torch.tensor(x_lo, dtype=torch.float64)
    xh = torch.tensor(x_hi, dtype=torch.float64)
    sb, _ = _forward_zonotope_graph(xl, xh, gg, torch.device('cpu'), torch.float64)
    bounds_by_relu = {}
    for li in range(gg['n_relu']):
        lo_t, hi_t = sb[li]
        bounds_by_relu[li] = (
            lo_t.cpu().numpy().astype(np.float64),
            hi_t.cpu().numpy().astype(np.float64))
    return gg_ops, bounds_by_relu, x_lo, x_hi, gg['input_name']


def test_ld_dual_bound_at_zero_rho_valid(tmp_path):
    """At rho=0 the LD dual bound must be <= the exact MILP minimum."""
    gg_ops, bounds_by_relu, x_lo, x_hi, input_name = _prep_tiny_fc(tmp_path)
    # Query: min Y[0] + 2*Y[1]
    query_w = np.array([1.0, 2.0])
    query_bias = 0.0
    exact = _exact_spec_min_via_milp(
        gg_ops, bounds_by_relu, x_lo, x_hi, input_name,
        query_w, query_bias)
    s = _ld_settings(ld_num_iterations=0)
    best, n_iter = verify_ld._ld_iterate_one_query(
        gg_ops, bounds_by_relu, x_lo, x_hi,
        query_w, query_bias, input_name, s, lambda: 10.0)
    assert n_iter == 1  # one rho=0 evaluation
    assert best <= exact + 1e-6, (
        f'LD bound {best} > exact {exact} (violates lower-bound)')
    assert np.isfinite(best)


def test_ld_dual_bound_monotone_nondecreasing(tmp_path):
    """best_bound must be non-decreasing across iterations (it is a
    max over the per-iteration q(rho))."""
    gg_ops, bounds_by_relu, x_lo, x_hi, input_name = _prep_tiny_fc(tmp_path)
    query_w = np.array([1.0, -1.0])
    query_bias = 0.0

    # Instrument _aggregate_bound to record the per-iteration q values.
    seen = []
    original = verify_ld._aggregate_bound

    def recorder(results, qb):
        val = original(results, qb)
        seen.append(val)
        return val

    verify_ld._aggregate_bound = recorder
    try:
        s = _ld_settings(ld_num_iterations=10, ld_early_stop=False)
        best, _ = verify_ld._ld_iterate_one_query(
            gg_ops, bounds_by_relu, x_lo, x_hi,
            query_w, query_bias, input_name, s, lambda: 10.0)
    finally:
        verify_ld._aggregate_bound = original

    assert len(seen) >= 2
    running_max = -np.inf
    for v in seen:
        running_max = max(running_max, v)
    assert best == pytest.approx(running_max, abs=1e-9)
    # Strictly: best_bound monotone nondecreasing across the running max.
    assert best >= seen[0] - 1e-9


def test_ld_verifies_tiny_robust_instance(tmp_path):
    """For a verifiable spec on the tiny FC the LD bound after enough
    iterations should become > 0 (spec is provably safe)."""
    gg_ops, bounds_by_relu, x_lo, x_hi, input_name = _prep_tiny_fc(tmp_path)
    # Trivially-verifiable spec: Y[0] + 1e6 > 0 (always safe). The LD
    # bound at rho=0 should already be >> 0 because the spec subproblem
    # is dominated by the constant bias.
    query_w = np.array([1.0, 0.0])
    query_bias = 1e6
    s = _ld_settings(ld_num_iterations=5)
    best, _ = verify_ld._ld_iterate_one_query(
        gg_ops, bounds_by_relu, x_lo, x_hi,
        query_w, query_bias, input_name, s, lambda: 10.0)
    assert best > 0


def test_ld_does_not_falsely_verify(tmp_path):
    """On a spec where the unsafe region is reachable (SAT), LD must
    NOT return a positive best_bound (would be a false verification)."""
    gg_ops, bounds_by_relu, x_lo, x_hi, input_name = _prep_tiny_fc(tmp_path)
    # Always-unsafe spec: query_w=[1], bias=-1e6. The min is very
    # negative, so LD bound must stay well below 0.
    query_w = np.array([1.0, 0.0])
    query_bias = -1e6
    s = _ld_settings(ld_num_iterations=50, ld_early_stop=False)
    best, _ = verify_ld._ld_iterate_one_query(
        gg_ops, bounds_by_relu, x_lo, x_hi,
        query_w, query_bias, input_name, s, lambda: 10.0)
    assert best <= 0, f'LD falsely verified a SAT instance: bound={best}'


def test_ld_iteration_converges_tight_single_hidden_layer(tmp_path):
    """1-hidden-layer FC: LD at rho=0 is already tight because the
    spec subproblem encodes the entire last ReLU + linear op. Verify
    that LD bound matches the exact MILP minimum within tolerance."""
    # Build a 1-hidden-layer FC: input(2) -> FC(2) -> Relu -> FC(1)
    rng = np.random.RandomState(1)
    W1 = rng.randn(2, 2).astype(np.float32) * 0.6
    b1 = np.array([0.1, -0.2], dtype=np.float32)
    W2 = rng.randn(1, 2).astype(np.float32) * 0.6
    b2 = np.array([0.3], dtype=np.float32)
    nodes = [
        helper.make_node('Gemm', ['X', 'W1', 'b1'], ['g1'], transB=1),
        helper.make_node('Relu', ['g1'], ['r1']),
        helper.make_node('Gemm', ['r1', 'W2', 'b2'], ['Y'], transB=1),
    ]
    inits = [_init('W1', W1), _init('b1', b1),
             _init('W2', W2), _init('b2', b2)]
    graph = helper.make_graph(
        nodes, 'tiny_fc2',
        [_input_val('X', [1, 2])],
        [_input_val('Y', [1, 1])],
        inits)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid('', 13)])
    g = _save_and_load(model, tmp_path, 'tiny_fc2.onnx')

    import torch
    from vibecheck.verify_zono_bnb import _forward_zonotope_graph
    gg = g.gpu_graph(torch.device('cpu'), torch.float64)
    gg_ops = _serialize_gg_ops(gg)
    x_lo = np.full(2, -1.0)
    x_hi = np.full(2, 1.0)
    xl = torch.tensor(x_lo, dtype=torch.float64)
    xh = torch.tensor(x_hi, dtype=torch.float64)
    sb, _ = _forward_zonotope_graph(xl, xh, gg, torch.device('cpu'), torch.float64)
    bounds_by_relu = {}
    for li in range(gg['n_relu']):
        lo_t, hi_t = sb[li]
        bounds_by_relu[li] = (
            lo_t.cpu().numpy().astype(np.float64),
            hi_t.cpu().numpy().astype(np.float64))

    query_w = np.array([1.0])
    query_bias = 0.0
    exact = _exact_spec_min_via_milp(
        gg_ops, bounds_by_relu, x_lo, x_hi, gg['input_name'],
        query_w, query_bias)
    s = _ld_settings(ld_num_iterations=50, ld_early_stop=False,
                     ld_initial_step=0.01, ld_final_step=0.001)
    best, _ = verify_ld._ld_iterate_one_query(
        gg_ops, bounds_by_relu, x_lo, x_hi,
        query_w, query_bias, gg['input_name'], s, lambda: 30.0)
    # For a 1-hidden-layer FC, LD equals the full MILP at rho=0 (the
    # spec subproblem IS the full problem with a big-M relu encoding).
    assert best == pytest.approx(exact, abs=1e-4), (
        f'LD bound {best} vs exact {exact}')


def test_ld_compute_lr_constant_schedule():
    """With schedule='constant' or n_iter<=1, _compute_lr returns lr0."""
    s = _ld_settings(ld_step_schedule='constant', ld_initial_step=0.3,
                     ld_final_step=0.05)
    assert verify_ld._compute_lr(0, 100, s) == 0.3
    assert verify_ld._compute_lr(50, 100, s) == 0.3
    # n_iter <= 1 also returns lr0 regardless of schedule
    s2 = _ld_settings(ld_step_schedule='linear_decay',
                     ld_initial_step=0.7, ld_final_step=0.1)
    assert verify_ld._compute_lr(0, 1, s2) == 0.7


def test_ld_stable_off_producer_skipped(tmp_path):
    """When a hidden neuron has u <= 0 in bounds_by_relu, its producer
    subproblem is NOT built (stable-off shortcut in _build_all_subproblems).
    Exercise the skip branch by hand-crafting bounds."""
    gg_ops, bounds_by_relu, x_lo, x_hi, input_name = _prep_tiny_fc(tmp_path)
    # Force layer 0 neuron 0 to be stable-off: u_0 <= 0
    lo_0, hi_0 = bounds_by_relu[0]
    lo_0 = lo_0.copy(); hi_0 = hi_0.copy()
    lo_0[0] = -1.0
    hi_0[0] = -0.1  # stable-off
    bounds_by_relu[0] = (lo_0, hi_0)

    blocks = verify_ld._extract_look_back_1_layers(
        gg_ops, bounds_by_relu, input_name)
    subs, _ = verify_ld._build_all_subproblems(
        blocks, bounds_by_relu, x_lo, x_hi, np.array([1.0, 0.0]))
    try:
        # Producer for layer 0 neuron 0 must be absent
        assert ('hidden', 0, 0) not in subs
        # Other neurons at layer 0 still exist
        assert ('hidden', 0, 1) in subs or ('hidden', 0, 2) in subs
    finally:
        verify_ld._dispose_all(subs)


def test_ld_iteration_print_progress(tmp_path, capsys):
    """Setting print_progress=True with a non-zero log_interval must
    emit a progress line from the iteration loop."""
    gg_ops, bounds_by_relu, x_lo, x_hi, input_name = _prep_tiny_fc(tmp_path)
    query_w = np.array([1.0, 0.0])
    query_bias = 1e6
    s = _ld_settings(ld_num_iterations=0, ld_log_interval=1,
                     print_progress=True)
    verify_ld._ld_iterate_one_query(
        gg_ops, bounds_by_relu, x_lo, x_hi,
        query_w, query_bias, input_name, s, lambda: 10.0)
    captured = capsys.readouterr().out
    assert '[LD] iter=' in captured


def test_verify_ld_queries_handles_already_verified_and_timeout(tmp_path):
    """The top-level driver must (1) skip queries whose spec_lb is
    already positive, (2) break out of inner and outer loops when
    time_left() returns 0."""
    import torch
    from vibecheck.verify_zono_bnb import _forward_zonotope_graph

    model = _build_tiny_sequential_fc()
    g = _save_and_load(model, tmp_path, 'q.onnx')
    gg = g.gpu_graph(torch.device('cpu'), torch.float64)
    gg_ops = _serialize_gg_ops(gg)
    x_lo = np.full(2, -1.0)
    x_hi = np.full(2, 1.0)
    xl = torch.tensor(x_lo, dtype=torch.float64)
    xh = torch.tensor(x_hi, dtype=torch.float64)
    sb, _ = _forward_zonotope_graph(
        xl, xh, gg, torch.device('cpu'), torch.float64)
    bounds_by_relu = {
        li: (sb[li][0].cpu().numpy().astype(np.float64),
             sb[li][1].cpu().numpy().astype(np.float64))
        for li in range(gg['n_relu'])
    }

    # Disjunct 0 has TWO queries so the inner loop iterates > 1. qi=0
    # already verified (skip branch, line 53); remaining queries plus
    # outer/inner time_left checks (lines 48, 51).
    disj_queries = {
        0: [(0, np.array([1.0, 0.0]), 1e6),
            (1, np.array([1.0, 0.0]), 1e6),
            (2, np.array([1.0, 0.0]), 1e6)],
        1: [(3, np.array([1.0, 0.0]), 1e6)],
    }
    still_open_disj = {0, 1}
    queries = [(0, np.array([1.0, 0.0]), 1e6),
               (1, np.array([1.0, 0.0]), 1e6),
               (2, np.array([1.0, 0.0]), 1e6),
               (3, np.array([1.0, 0.0]), 1e6)]

    s = _ld_settings(ld_num_iterations=0)
    # Case A: time_left immediately returns 0 -> outer break (line 48)
    spec_lbs_a = {0: -1.0, 1: -1.0, 2: -1.0, 3: -1.0}
    info_to = verify_ld.verify_ld_queries(
        gg, gg_ops, bounds_by_relu, x_lo, x_hi,
        disj_queries, spec_lbs_a, still_open_disj, queries,
        s, time_left=lambda: 0.0)
    assert info_to['ld_queries_run'] == 0

    # Case B: run qi=1 in disj 0, then time_left drops to 0 -> inner
    # break (line 51). qi=0 is already verified -> skip branch (line 53).
    spec_lbs_b = {0: 1.0, 1: -1.0, 2: -1.0, 3: -1.0}
    state = {'steps': 0}
    def tl_b():
        state['steps'] += 1
        # Calls: outer(di=0), inner(qi=0) -> skip, inner(qi=1) -> run,
        # inner(qi=2) -> break on 0
        return 1.0 if state['steps'] <= 3 else 0.0
    info_b = verify_ld.verify_ld_queries(
        gg, gg_ops, bounds_by_relu, x_lo, x_hi,
        disj_queries, spec_lbs_b, still_open_disj, queries,
        s, time_left=tl_b)
    assert info_b['ld_queries_run'] >= 1


def test_verify_graph_with_ld_enabled_on_small_fc(tmp_path):
    """End-to-end `verify_graph()` call with `ld_enabled=True` on a
    synthetic sequential FC. Verification still succeeds (CROWN
    handles this trivial spec), and the LD gated code path remains
    dormant because the query closes at phase 2."""
    model = _build_tiny_sequential_fc()
    g = _save_and_load(model, tmp_path, 'tiny_ld.onnx')
    spec = _easy_spec(2)
    s = default_settings(device='cpu', total_timeout=20,
                         print_progress=False, ld_enabled=True,
                         ld_num_iterations=0)
    result, details = verify_graph(g, spec, s)
    assert result == 'verified'


def test_ld_disabled_pipeline_unchanged(tmp_path):
    """Regression guard: with `ld_enabled=False` the pipeline result
    and its timing keys must match the pre-LD baseline (phase_ld is
    absent, details['ld'] is absent)."""
    model = _build_tiny_sequential_fc()
    g = _save_and_load(model, tmp_path, 'tiny_baseline.onnx')
    spec = _easy_spec(2)
    s = default_settings(device='cpu', total_timeout=20,
                         print_progress=False, ld_enabled=False)
    result, details = verify_graph(g, spec, s)
    assert result == 'verified'
    assert 'phase_ld' not in details['timing']
    assert 'ld' not in details


def test_ld_adam_step_schedule_runs(tmp_path):
    """LD with `ld_step_schedule='adam'` must run without errors and
    produce a valid lower bound (<= exact MILP)."""
    gg_ops, bounds_by_relu, x_lo, x_hi, input_name = _prep_tiny_fc(tmp_path)
    query_w = np.array([1.0, -1.0])
    query_bias = 0.0
    exact = _exact_spec_min_via_milp(
        gg_ops, bounds_by_relu, x_lo, x_hi, input_name,
        query_w, query_bias)
    s = _ld_settings(
        ld_num_iterations=20, ld_early_stop=False,
        ld_step_schedule='adam', ld_adam_lr=0.05)
    best, _ = verify_ld._ld_iterate_one_query(
        gg_ops, bounds_by_relu, x_lo, x_hi,
        query_w, query_bias, input_name, s, lambda: 10.0)
    assert best <= exact + 1e-5
    assert np.isfinite(best)


def test_ld_runs_on_identity_skip_residual(tmp_path):
    """End-to-end LD iteration on an identity-skip residual FC. The
    aggregated dual bound at rho=0 must be a valid lower bound on the
    exact full-network MILP, and LD must terminate without errors."""
    import torch
    from vibecheck.verify_zono_bnb import _forward_zonotope_graph

    model = _build_identity_skip_fc()
    g = _save_and_load(model, tmp_path, 'res_ld.onnx')
    gg = g.gpu_graph(torch.device('cpu'), torch.float64)
    gg_ops = _serialize_gg_ops(gg)
    x_lo = np.full(2, -1.0)
    x_hi = np.full(2, 1.0)
    xl = torch.tensor(x_lo, dtype=torch.float64)
    xh = torch.tensor(x_hi, dtype=torch.float64)
    sb, _ = _forward_zonotope_graph(
        xl, xh, gg, torch.device('cpu'), torch.float64)
    bounds_by_relu = {
        li: (sb[li][0].cpu().numpy().astype(np.float64),
             sb[li][1].cpu().numpy().astype(np.float64))
        for li in range(gg['n_relu'])
    }
    query_w = np.array([1.0])
    query_bias = 0.0
    exact = _exact_spec_min_via_milp(
        gg_ops, bounds_by_relu, x_lo, x_hi, gg['input_name'],
        query_w, query_bias)
    s = _ld_settings(ld_num_iterations=5, ld_early_stop=False)
    best, n_iter = verify_ld._ld_iterate_one_query(
        gg_ops, bounds_by_relu, x_lo, x_hi,
        query_w, query_bias, gg['input_name'], s, lambda: 30.0)
    assert best <= exact + 1e-5, (
        f'LD merge bound {best} > exact {exact}')
    assert n_iter >= 1


def test_ld_acasxu_1_1_runs(vnncomp_benchmarks):
    """Load a small ACAS Xu network + property and run the full
    `verify_graph()` pipeline with LD enabled. The test just asserts
    termination and that the result is one of the valid strings; it
    does not gate on the specific outcome (LD may or may not help on
    sequential ACAS Xu queries)."""
    from vibecheck.vnnlib_loader import load_vnnlib

    onnx_path = vnncomp_benchmarks / (
        'acasxu_2023/onnx/ACASXU_run2a_1_1_batch_2000.onnx.gz')
    spec_path = vnncomp_benchmarks / 'acasxu_2023/vnnlib/prop_2.vnnlib.gz'
    if not onnx_path.exists() or not spec_path.exists():
        pytest.skip('ACAS Xu benchmark not available')
    g = ComputeGraph.from_onnx(str(onnx_path))
    spec = load_vnnlib(str(spec_path))
    s = default_settings(device='cpu', total_timeout=60,
                         print_progress=False, ld_enabled=True,
                         ld_num_iterations=5)
    result, details = verify_graph(g, spec, s)
    assert result in ('verified', 'unknown', 'sat')
    assert 'timing' in details


def test_ld_disabled_gate_skipped(tmp_path):
    """With ld_enabled=False the new gated phase does not execute and
    no LD telemetry appears in the details dict."""
    model = _build_tiny_sequential_fc()
    g = _save_and_load(model, tmp_path, 'tiny.onnx')
    spec = _easy_spec(2)
    s = default_settings(device='cpu', total_timeout=20,
                         print_progress=False, ld_enabled=False)
    result, details = verify_graph(g, spec, s)
    assert result == 'verified'
    assert 'phase_ld' not in details['timing']
    assert 'ld' not in details
