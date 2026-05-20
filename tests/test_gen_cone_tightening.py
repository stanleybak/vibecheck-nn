"""Tests for the G-cone per-neuron tightening path (tighten_mode='gen_cone').

Covers:
- `_dependency_cone` on a hand-built gen_rows_by_layer fixture.
- Bound-equivalence of `_tighten_layer_gen_cone` vs. the full-LP
  reference in `verify_gen_lp.tighten_bounds`.
"""

import numpy as np
import torch

from vibecheck import verify_gen_lp
from vibecheck.verify_graph import (
    _dependency_cone,
    _build_gen_cone_lp,
    _tighten_layer_gen_cone,
)


def _entry(li, j, e_new_col, row_indices, row_values,
           c_in=0.0, lo=-1.0, hi=1.0):
    return {
        'layer_idx': li, 'neuron_idx': j, 'e_new_col': e_new_col,
        'row_indices': np.asarray(row_indices, dtype=np.int32),
        'row_values': np.asarray(row_values, dtype=np.float64),
        'c_in': c_in, 'lo': lo, 'hi': hi,
    }


def test_dependency_cone_minimal():
    """3-layer fixture: inputs [0..2], L0 unstable {0}, L1 unstable {0}.

    - L0 neuron 0 depends on inputs {0, 1} (cols [0, 1])
      e_new_col = 3 (n_input = 3, first new gen)
    - L1 neuron 0 depends on input 2 and L0 neuron 0
      e_new_col = 4 (row_indices = [2, 3])
    - Target at L2 neuron j=0 has row_indices = [1, 4]
      → walks to L1.0 (cols [2, 3]), then L0.0 (cols [0, 1])
      → cone inputs = {0, 1, 2}, upstream = {(L0, 0), (L1, 0)}.
    """
    n_input = 3
    l0_e0 = _entry(li=0, j=0, e_new_col=3,
                   row_indices=[0, 1], row_values=[0.5, -0.25])
    l1_e0 = _entry(li=1, j=0, e_new_col=4,
                   row_indices=[2, 3], row_values=[1.0, 2.0])
    gen_rows_by_layer = {0: {0: l0_e0}, 1: {0: l1_e0}}
    col_origin = {3: (0, 0), 4: (1, 0)}
    target_row_indices = np.asarray([1, 4], dtype=np.int32)
    target_row_values = np.asarray([3.0, 4.0], dtype=np.float64)

    input_cols, upstream_topo = _dependency_cone(
        target_row_indices, target_row_values,
        gen_rows_by_layer, col_origin, n_input)

    # Inputs reached: 1 (directly), 2 (via L1.0), 0 & 1 (via L0.0 chain).
    assert input_cols == [0, 1, 2]
    assert [(e['layer_idx'], e['neuron_idx']) for e in upstream_topo] \
        == [(0, 0), (1, 0)]


def test_dependency_cone_prunes_unreachable():
    """If target row doesn't touch an upstream neuron, the cone omits it."""
    n_input = 2
    l0_e0 = _entry(li=0, j=0, e_new_col=2,
                   row_indices=[0], row_values=[1.0])
    l0_e1 = _entry(li=0, j=1, e_new_col=3,
                   row_indices=[1], row_values=[1.0])
    gen_rows_by_layer = {0: {0: l0_e0, 1: l0_e1}}
    col_origin = {2: (0, 0), 3: (0, 1)}

    input_cols, upstream_topo = _dependency_cone(
        np.asarray([2], dtype=np.int32),
        np.asarray([1.0], dtype=np.float64),
        gen_rows_by_layer, col_origin, n_input)
    # Only the (0, 0) path — neuron (0, 1) is irrelevant.
    assert input_cols == [0]
    assert [(e['layer_idx'], e['neuron_idx']) for e in upstream_topo] \
        == [(0, 0)]


def _fc_op(name, inputs, W, b):
    return {'name': name, 'type': 'fc', 'inputs': list(inputs),
            'W_np': np.asarray(W, dtype=np.float64),
            'bias_np': np.asarray(b, dtype=np.float64)}


def _relu_op(name, inputs, layer_idx):
    return {'name': name, 'type': 'relu', 'inputs': list(inputs),
            'layer_idx': layer_idx}


def _build_2layer_fc():
    """Tiny 2-layer FC net: input(3) → fc1(4) → relu1 → fc2(3) → relu2 → fc3(2).

    Weights are hand-chosen so that both ReLU layers have a mix of dead,
    stable, and unstable neurons at the zonotope-level bounds.
    """
    rng = np.random.default_rng(0)
    W1 = rng.standard_normal((4, 3)) * 0.5
    b1 = rng.standard_normal(4) * 0.1
    W2 = rng.standard_normal((3, 4)) * 0.5
    b2 = rng.standard_normal(3) * 0.1
    W3 = rng.standard_normal((2, 3)) * 0.5
    b3 = rng.standard_normal(2) * 0.1
    ops = [
        _fc_op('fc1', ['x'], W1, b1),
        _relu_op('r1', ['fc1'], 0),
        _fc_op('fc2', ['r1'], W2, b2),
        _relu_op('r2', ['fc2'], 1),
        _fc_op('fc3', ['r2'], W3, b3),
    ]
    return ops


def _cpu_device():
    # Tests run on CPU to avoid CUDA requirement.
    return 'cpu'


def _initial_bounds_from_forward(gg_ops_ser, x_lo, x_hi, input_name):
    """Compute per-ReLU pre-activation bounds by zonotope forward on CPU.

    This gives us the `initial_bounds` seed that both `tighten_bounds`
    and `_tighten_layer_gen_cone` start from.
    """
    device = _cpu_device()
    dtype = torch.float64
    gpu = torch.device(device)
    n_in = len(x_lo)
    half_w = torch.tensor((x_hi - x_lo) / 2, dtype=dtype, device=gpu)
    half_s = torch.tensor((x_hi + x_lo) / 2, dtype=dtype, device=gpu)
    center = {input_name: half_s}
    G_by_op = {input_name: torch.diag(half_w)}
    bounds = {}
    for op in gg_ops_ser:
        nm = op['name']
        t = op['type']
        if t == 'fc':
            prev_c = center[op['inputs'][0]]
            prev_G = G_by_op[op['inputs'][0]]
            W = torch.tensor(op['W_np'], dtype=dtype, device=gpu)
            b = torch.tensor(op['bias_np'], dtype=dtype, device=gpu)
            center[nm] = W @ prev_c + b
            G_by_op[nm] = W @ prev_G
        elif t == 'relu':
            inp = op['inputs'][0]
            c_in = center[inp]
            G_in = G_by_op[inp]
            abs_sum = torch.abs(G_in).sum(dim=1)
            lo = (c_in - abs_sum).cpu().numpy().astype(np.float64)
            hi = (c_in + abs_sum).cpu().numpy().astype(np.float64)
            li = op['layer_idx']
            bounds[li] = (lo, hi)
            # Apply zonotope relu relaxation (min-area) for subsequent ops.
            dead = hi <= 0
            stab = (lo >= 0) & ~dead
            ust = ~dead & ~stab
            lam = np.where(ust, hi / np.where(ust, hi - lo, 1.0),
                           np.where(dead, 0.0, 1.0))
            mu = np.where(ust, -hi * lo / (2.0 * np.where(ust, hi - lo, 1.0)),
                          0.0)
            lam_t = torch.tensor(lam, dtype=dtype, device=gpu)
            mu_t = torch.tensor(mu, dtype=dtype, device=gpu)
            new_c = lam_t * c_in + mu_t
            new_G = lam_t.unsqueeze(1) * G_in
            ui = np.where(ust)[0]
            if len(ui) > 0:
                n = len(c_in)
                extra = torch.zeros(n, len(ui), dtype=dtype, device=gpu)
                for local, j in enumerate(ui):
                    extra[int(j), local] = float(mu[int(j)])
                new_G = torch.cat([new_G, extra], dim=1)
            center[nm] = new_c
            G_by_op[nm] = new_G
    return bounds


def test_gen_cone_matches_tighten_bounds_2layer_fc():
    """Bound-equivalence vs verify_gen_lp.tighten_bounds on a 2-layer FC."""
    gg_ops_ser = _build_2layer_fc()
    x_lo = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
    x_hi = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    input_name = 'x'

    initial_bounds = _initial_bounds_from_forward(
        gg_ops_ser, x_lo, x_hi, input_name)

    # Reference: the full gen-LP tightening pass.
    ref = verify_gen_lp.tighten_bounds(
        gg_ops_ser, x_lo, x_hi, initial_bounds, input_name,
        sample_timeout=10.0, n_threads=1,
        device=_cpu_device(), dtype=torch.float64)

    # Our cone path: iterate layers in order, mirroring tighten_bounds'
    # use of the already-tightened bounds as the seed for subsequent
    # layers' LP rows.
    cone_bounds = {li: (lo.copy(), hi.copy())
                   for li, (lo, hi) in initial_bounds.items()}
    for li in sorted(initial_bounds.keys()):
        lo0, hi0 = cone_bounds[li]
        unstable = np.where((lo0 < 0) & (hi0 > 0))[0]
        if len(unstable) == 0:
            continue
        new_lo, new_hi, method, _, _, _ = _tighten_layer_gen_cone(
            gg_ops_ser, x_lo, x_hi, cone_bounds, li, unstable,
            input_name, sample_timeout=10.0, n_cores=1,
            time_left=lambda: 60.0,
            mode='gen_cone',
            device=_cpu_device(), dtype=torch.float64)
        assert method in ('lp', 'skip', 'lp-partial'), method
        cone_bounds[li] = (new_lo, new_hi)

    # Compare per-layer bounds — triangle LPs should yield identical
    # numbers to reasonable numerical tolerance.
    for li in sorted(initial_bounds.keys()):
        ref_lo, ref_hi = ref[li]
        cone_lo, cone_hi = cone_bounds[li]
        np.testing.assert_allclose(cone_lo, ref_lo, atol=1e-6, rtol=1e-9)
        np.testing.assert_allclose(cone_hi, ref_hi, atol=1e-6, rtol=1e-9)


def test_verify_gen_lp_honors_nonmerge_add_bias():
    """Regression: `verify_gen_lp` must apply the bias on non-merge `Add`
    ops. ACAS Xu's serialization produces fc(bias=0) + add(bias=real)
    pairs, and the earlier code dropped that bias — making `forward_point`
    and `precompute_gen_state` produce wrong centers (confirmed by
    MILP-bound mismatch against the direct-from-ONNX sparse builder).
    """
    # Tiny "fc-then-add-bias" net: x(2) → fc(3, no bias) → add(bias) → out
    W = np.array([[1.0, 0.0],
                  [0.0, 1.0],
                  [0.5, 0.5]], dtype=np.float64)
    b_add = np.array([10.0, -20.0, 30.0], dtype=np.float64)
    ops = [
        {'name': 'fc1', 'type': 'fc', 'inputs': ['x'],
         'W_np': W, 'bias_np': np.zeros(3, dtype=np.float64)},
        {'name': 'add1', 'type': 'add', 'inputs': ['fc1'],
         'is_merge': False, 'bias': b_add},
    ]
    x = np.array([1.0, 2.0], dtype=np.float64)
    expected = W @ x + b_add  # [11, -18, 31.5]

    # forward_point must include the add bias.
    y = verify_gen_lp.forward_point(ops, x, 'x', 'add1')
    np.testing.assert_allclose(y, expected, atol=0, rtol=0)

    # precompute_gen_state must carry the bias into obj_c_out. With
    # x_lo == x_hi, the zonotope has zero radii and obj_c_out equals the
    # forward value at x_center.
    state = verify_gen_lp.precompute_gen_state(
        ops, x, x, {}, 'x', 'add1',
        device='cpu', dtype=torch.float64, formulation='dense')
    np.testing.assert_allclose(state['obj_c_out'], expected,
                               atol=1e-12, rtol=0)


def test_gen_cone_lp_matches_weight_walk_lp():
    """Per-layer LP bounds must be identical for gen_cone and weight_walk.

    Both paths encode the same triangle convex hull (just with different
    variable choices), so the LP bound per neuron is a projection of the
    same polytope. We build a tiny 2-layer FC, run both tighteners on the
    same initial bounds, and compare per-neuron lo/hi.
    """
    from vibecheck.verify_graph import (
        _build_sequential_subgraph, _tighten_sequential_with_probe)

    gg_ops_ser = _build_2layer_fc()
    x_lo = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
    x_hi = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    input_name = 'x'
    initial_bounds = _initial_bounds_from_forward(
        gg_ops_ser, x_lo, x_hi, input_name)

    for li in sorted(initial_bounds.keys()):
        lo0, hi0 = initial_bounds[li]
        unstable = np.where((lo0 < 0) & (hi0 > 0))[0]
        if len(unstable) == 0:
            continue

        # Run both tighteners on a fresh copy of the initial bounds
        # (so neither path sees the other's tightening).
        bounds_gc = {k: (lo.copy(), hi.copy())
                     for k, (lo, hi) in initial_bounds.items()}
        gc_lo, gc_hi, _, _, _, _ = _tighten_layer_gen_cone(
            gg_ops_ser, x_lo, x_hi, bounds_gc, li, unstable,
            input_name, sample_timeout=30.0, n_cores=1,
            time_left=lambda: 60.0, mode='gen_cone',
            device=_cpu_device(), dtype=torch.float64)

        bounds_ww = {k: (lo.copy(), hi.copy())
                     for k, (lo, hi) in initial_bounds.items()}
        layers_np_seq, seq_bounds, seq_li = _build_sequential_subgraph(
            gg_ops_ser, li, bounds_ww)
        ww_lo, ww_hi, _, _, _ = _tighten_sequential_with_probe(
            layers_np_seq, x_lo, x_hi, seq_bounds, seq_li, unstable,
            lp_per_worker=True, sample_timeout=30.0, n_cores=1,
            time_left=lambda: 60.0, mode='lp')

        np.testing.assert_allclose(gc_lo, ww_lo, atol=1e-6, rtol=1e-9,
                                    err_msg=f'L{li} LP lower bound')
        np.testing.assert_allclose(gc_hi, ww_hi, atol=1e-6, rtol=1e-9,
                                    err_msg=f'L{li} LP upper bound')


def test_gen_cone_milp_matches_weight_walk_milp():
    """Per-layer MILP bounds must match between gen_cone and weight_walk
    up to Gurobi's feasibility tolerance.

    `_tighten_sequential_with_probe(mode='milp')` by default skips MILP on
    FC layers past L0 as a cost heuristic — we bypass that here by calling
    `_tighten_layer_parallel` directly with use_milp=True.
    """
    from vibecheck.verify_graph import _build_sequential_subgraph
    from vibecheck.verify_milp import _tighten_layer_parallel

    gg_ops_ser = _build_2layer_fc()
    x_lo = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
    x_hi = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    input_name = 'x'
    initial_bounds = _initial_bounds_from_forward(
        gg_ops_ser, x_lo, x_hi, input_name)

    for li in sorted(initial_bounds.keys()):
        lo0, hi0 = initial_bounds[li]
        unstable = np.where((lo0 < 0) & (hi0 > 0))[0]
        if len(unstable) == 0:
            continue

        # gen_cone MILP
        bounds_gc = {k: (lo.copy(), hi.copy())
                     for k, (lo, hi) in initial_bounds.items()}
        gc_lo, gc_hi, _, _, _, _ = _tighten_layer_gen_cone(
            gg_ops_ser, x_lo, x_hi, bounds_gc, li, unstable,
            input_name, sample_timeout=30.0, n_cores=1,
            time_left=lambda: 60.0, mode='gen_cone_milp',
            device=_cpu_device(), dtype=torch.float64, use_milp=True)

        # weight-walk MILP (direct call, bypassing the FC-skip heuristic)
        bounds_ww = {k: (lo.copy(), hi.copy())
                     for k, (lo, hi) in initial_bounds.items()}
        layers_np_seq, seq_bounds, seq_li = _build_sequential_subgraph(
            gg_ops_ser, li, bounds_ww)
        ww_lo, ww_hi, _ = _tighten_layer_parallel(
            layers_np_seq, x_lo, x_hi, seq_bounds, seq_li,
            use_milp=True, timeout=30.0, n_cores=1,
            neuron_subset=unstable)
        # _tighten_layer_parallel returns full-layer arrays; restrict
        # to only the layer's bounds we queried.
        ww_lo = np.maximum(ww_lo, seq_bounds[seq_li][0])
        ww_hi = np.minimum(ww_hi, seq_bounds[seq_li][1])

        # Tolerance wider than LP: Gurobi feasibility tolerance ~1e-6.
        np.testing.assert_allclose(gc_lo, ww_lo, atol=5e-5, rtol=1e-6,
                                    err_msg=f'L{li} MILP lower bound')
        np.testing.assert_allclose(gc_hi, ww_hi, atol=5e-5, rtol=1e-6,
                                    err_msg=f'L{li} MILP upper bound')


def test_build_gen_cone_lp_bigM_tightens_vs_lp():
    """Big-M on a single upstream neuron should tighten the LP bound."""
    import gurobipy as grb
    # One upstream unstable neuron at (L0, 0) with lo=-1, hi=1, c_in=0.
    # Its pre-activation row is e_in[0] (coef 1).
    upstream = [_entry(li=0, j=0, e_new_col=1,
                       row_indices=[0], row_values=[1.0],
                       c_in=0.0, lo=-1.0, hi=1.0)]
    # Target is "the a-var for (L0,0)" as an identity — objective
    # maximizes that var directly (col 1).
    target_c = 0.0
    target_idx = np.asarray([1], dtype=np.int32)
    target_val = np.asarray([1.0], dtype=np.float64)
    input_cols = [0]

    # LP (triangle): over-approx region; max of a is hi = 1. With triangle
    # the max of a is slope*(c-lo) at e_in=1 → a = slope*(0+1)=1*1/2*(1+1)
    # Actually: slope = hi/(hi-lo) = 1/2. Upper triangle: a <= slope*(z-lo)
    # = 0.5*(z+1). At z=1 (e_in=1), a <= 1. At z=-1, a <= 0. So max a = 1.
    # LP and MILP should both get max=1 here — instead check LP ≥ triangle
    # gives a "relaxed" answer at fractional points. Try objective min on
    # a - 0.5*z to test bigM tightens lower bound.
    #
    # Simpler check: set target = a (col 1) maximized. Same result for
    # LP/MILP. Test instead the *minimum* of a - z (= a - e_in):
    target_idx = np.asarray([0, 1], dtype=np.int32)
    target_val = np.asarray([-1.0, 1.0], dtype=np.float64)

    m_lp, env_lp = _build_gen_cone_lp(
        target_c, target_idx, target_val, input_cols, upstream,
        milp_neurons=set(), sense='min')
    m_lp.optimize()
    lp_min = float(m_lp.ObjVal)
    m_lp.dispose(); env_lp.dispose()

    m_milp, env_milp = _build_gen_cone_lp(
        target_c, target_idx, target_val, input_cols, upstream,
        milp_neurons={(0, 0)}, sense='min')
    m_milp.setParam('OutputFlag', 0)
    m_milp.optimize()
    milp_min = float(m_milp.ObjVal)
    m_milp.dispose(); env_milp.dispose()

    # Triangle (LP): slack allows a - e_in below 0 → LP min < MILP min.
    # MILP exactly encodes relu, so a - e_in = max(0, e_in) - e_in ≥
    # min over e_in∈[-1,1] is 0 (attained at e_in ≥ 0; at e_in=-1 it's
    # 0 - (-1) = 1). So MILP min = 0; LP min ≤ 0.
    assert milp_min >= lp_min - 1e-9
    # And MILP min should be exactly 0 (relu is tight).
    assert abs(milp_min) < 1e-6
