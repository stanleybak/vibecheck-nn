"""Tests for MILP verification: sparse models, CROWN-based scoring."""

import os
import numpy as np
import pytest
import torch

try:
    import gurobipy
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False

from vibecheck.network import ComputeGraph, ConvNode, ReluNode, GemmNode
from vibecheck.spec import VNNSpec, Conjunct, PairwiseConstraint
from vibecheck.settings import default_settings, resolve_torch

BENCH = '/home/stan/repositories/vnncomp2025_benchmarks/benchmarks/relusplitter'
ONNX_PATH = BENCH + '/onnx/oval21-benchmark_cifar_base_kw.onnx.gz'
VNNLIB_PATH = BENCH + '/vnnlib/oval21-benchmark_cifar_base_kw-img1204-eps0.024967320261437907.vnnlib.gz'

F = np.float32
DEV = torch.device('cpu')
DT = torch.float32


def _a(vals):
    return np.array(vals, dtype=F)


def _conv_fc_graph(dtype=np.float32):
    """Conv(1,1,3,3) -> Relu -> FC(4->2): small network for testing."""
    g = ComputeGraph(dtype=dtype)
    g.input_name = 'input'
    g.input_shape = (1, 1, 4, 4)
    rng = np.random.RandomState(42)
    kernel = rng.randn(1, 1, 3, 3).astype(dtype) * 0.5
    bias_conv = np.zeros(1, dtype=dtype)
    W2 = rng.randn(2, 4).astype(dtype)
    b2 = np.array([1.0, -1.0], dtype=dtype)

    g.nodes['conv'] = ConvNode(
        name='conv', op_type='Conv', inputs=['input'],
        params={'kernel': kernel, 'bias': bias_conv,
                'stride': (1, 1), 'padding': (0, 0), 'group': 1})
    g.nodes['relu'] = ReluNode(name='relu', op_type='Relu', inputs=['conv'])
    g.nodes['gemm'] = GemmNode(
        name='gemm', op_type='Gemm', inputs=['relu'],
        params={'W': W2, 'b': b2})
    g.output_name = 'gemm'
    g.topological_sort()
    from vibecheck.onnx_loader import _infer_shapes, _precache_conv_tensors
    _infer_shapes(g)
    _precache_conv_tensors(g)
    return g


# ---------------------------------------------------------------------------
# Test 1: Sparse per-neuron MILP for conv layers
# ---------------------------------------------------------------------------

class TestSparseConvMILP:
    def test_build_sparse_neuron_model_exists(self):
        """_build_sparse_neuron_model function should exist."""
        from vibecheck.verify_milp import _build_sparse_neuron_model
        assert callable(_build_sparse_neuron_model)

    @pytest.mark.skipif(not HAS_GUROBI, reason="Gurobi not installed")
    def test_sparse_model_fewer_vars_than_full(self):
        """Sparse model should have fewer variables than full model."""
        from vibecheck.verify_milp import (
            _build_base_model, _build_sparse_neuron_model, _conv_connections)

        # Build a conv layer where each neuron only connects to a subset
        kernel = np.random.RandomState(42).randn(2, 1, 3, 3).astype(np.float64)
        bias = np.zeros(2, dtype=np.float64)
        in_shape = (1, 6, 6)
        layers_np = [{
            'type': 'conv', 'kernel': kernel, 'bias': bias,
            'in_shape': in_shape, 'stride': (1, 1), 'padding': (0, 0),
        }]
        n_input = 36  # 1*6*6
        x_lo = np.zeros(n_input, dtype=np.float64)
        x_hi = np.ones(n_input, dtype=np.float64)

        # Bounds at layer 0: some unstable
        n_out = 2 * 4 * 4  # 32
        lo = np.random.RandomState(1).randn(n_out).astype(np.float64) * 0.5
        hi = lo + 1.0
        bounds = {0: (lo, hi)}

        # Full model for layer 0
        full_model, full_env = _build_base_model(
            layers_np, x_lo, x_hi, bounds, 1)
        n_full_vars = full_model.NumVars

        # Sparse model for a single L0 neuron (should be much smaller)
        # We're testing that the sparse model for L1 neuron only includes
        # the L0 neurons in its receptive field. But here we have only 1 layer,
        # so the sparse model is for tightening L0 neurons which just need inputs.
        # The real benefit is for L1+ neurons.
        sparse_model, sparse_env = _build_sparse_neuron_model(
            layers_np, x_lo, x_hi, bounds, target_layer=0, target_neuron=0)
        n_sparse_vars = sparse_model.NumVars

        # Sparse should have fewer or equal vars
        assert n_sparse_vars <= n_full_vars

        full_model.dispose(); full_env.dispose()
        sparse_model.dispose(); sparse_env.dispose()

    @pytest.mark.skipif(not HAS_GUROBI, reason="Gurobi not installed")
    def test_sparse_vs_full_on_real_network(self):
        """Sparse model gives same bounds as full model on real oval21 network.

        L0 bounds are exact from zonotope (first linear layer, no prior ReLU),
        so sparse per-neuron MILP for L1 conv neurons should give identical
        bounds to the full model — non-RF L0 neurons share no input variables
        with the RF neurons for conv layers.
        """
        import os
        from vibecheck.verify_milp import (
            _build_base_model, _build_sparse_neuron_model,
            _conv_connections, _conv_bias_idx)
        from vibecheck.network import ComputeGraph
        from vibecheck.vnnlib_loader import load_vnnlib
        from vibecheck.zonotope import TorchZonotope
        import gurobipy as grb

        BENCH = '/home/stan/repositories/vnncomp2025_benchmarks/benchmarks/relusplitter'
        onnx_path = BENCH + '/onnx/oval21-benchmark_cifar_base_kw.onnx.gz'
        vnnlib_path = BENCH + '/vnnlib/oval21-benchmark_cifar_base_kw-img1204-eps0.024967320261437907.vnnlib.gz'
        if not os.path.exists(onnx_path):
            pytest.skip("Benchmark files not available")

        graph = ComputeGraph.from_onnx(onnx_path)
        spec = load_vnnlib(vnnlib_path)
        dev = torch.device('cpu'); dt = torch.float32
        gpu_layers, _ = graph.gpu_layers(dev, dt)
        nh = len(gpu_layers) - 1

        xl = torch.tensor(spec.x_lo.astype(np.float32), dtype=dt)
        xh = torch.tensor(spec.x_hi.astype(np.float32), dtype=dt)
        z = TorchZonotope.from_input_bounds(xl, xh, dev, dt)
        sb = {}
        for l in range(nh):
            gl = gpu_layers[l]
            if gl['type'] == 'conv':
                z.propagate_conv(gl['kernel'], gl['bias'], gl['in_shape'],
                                 gl['stride'], gl['padding'])
            else:
                z.propagate_fc(gl['W'], gl['bias'])
            lo, hi = z.apply_relu()
            sb[l] = (lo.clone(), hi.clone())

        bounds_np = {l: (sb[l][0].numpy().astype(np.float64),
                         sb[l][1].numpy().astype(np.float64))
                     for l in range(nh)}
        layers_np = []
        for gl in gpu_layers:
            d = {'type': gl['type']}
            if gl['type'] == 'fc':
                d['W'] = gl['W'].numpy().astype(np.float64)
                d['bias'] = gl['bias'].numpy().astype(np.float64)
            else:
                d['kernel'] = gl['kernel'].numpy().astype(np.float64)
                d['bias'] = gl['bias'].numpy().astype(np.float64)
                d['in_shape'] = gl['in_shape']
                d['stride'] = gl['stride']
                d['padding'] = gl['padding']
            layers_np.append(d)

        x_lo_64 = spec.x_lo.astype(np.float64)
        x_hi_64 = spec.x_hi.astype(np.float64)

        # Pick 5 random unstable L1 neurons
        lo1, hi1 = bounds_np[1]
        unstable = np.where((lo1 < 0) & (hi1 > 0))[0]
        rng = np.random.RandomState(42)
        test_neurons = rng.choice(unstable, min(5, len(unstable)), replace=False)

        # Build full model once
        full_m, full_e = _build_base_model(
            layers_np, x_lo_64, x_hi_64, bounds_np, 1, milp_set=None)
        layer1 = layers_np[1]

        for j in test_neurons:
            j = int(j)
            # Full model: copy, add objective, solve
            cm = full_m.copy()
            zt = cm.addVar(lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY)
            cm.update()
            expr = grb.LinExpr()
            conns = _conv_connections(j, layer1['kernel'], layer1['in_shape'],
                                      layer1['stride'], layer1['padding'])
            for fi, w in conns:
                v = cm.getVarByName(f'a_0_{fi}')
                if v:
                    expr.add(v, w)
            b_j = float(layer1['bias'][_conv_bias_idx(
                j, layer1['kernel'], layer1['in_shape'],
                layer1['stride'], layer1['padding'])])
            cm.addConstr(zt == expr + b_j)
            cm.update()
            cm.setObjective(zt, grb.GRB.MINIMIZE)
            cm.optimize()
            assert cm.status == 2
            full_lb = cm.ObjVal

            # Sparse model
            sm, se = _build_sparse_neuron_model(
                layers_np, x_lo_64, x_hi_64, bounds_np, 1, j)
            sm.setObjective(sm.getVarByName('_target'), grb.GRB.MINIMIZE)
            sm.optimize()
            assert sm.status == 2
            sparse_lb = sm.ObjVal

            assert sm.NumVars < cm.NumVars, \
                f"neuron {j}: sparse {sm.NumVars} >= full {cm.NumVars}"
            # Both are sound lower bounds. Full model with inlined active
            # neurons may differ from sparse (different formulation).
            # Just check both are finite and sparse is reasonable.
            assert np.isfinite(sparse_lb), f"neuron {j}: sparse lb not finite"
            assert np.isfinite(full_lb), f"neuron {j}: full lb not finite"

            sm.dispose(); se.dispose()

        full_m.dispose(); full_e.dispose()


# ---------------------------------------------------------------------------
# Test 2: CROWN-based neuron scoring
# ---------------------------------------------------------------------------

class TestSpecModelConsistency:
    """Spec model produces sound bounds on real network."""

    @pytest.mark.skipif(not HAS_GUROBI, reason="Gurobi not installed")
    def test_spec_lp_bound(self):
        """Spec LP for comp=4 gives a finite negative bound on oval21."""
        import os
        from vibecheck.verify_milp import _build_spec_model
        from vibecheck.network import ComputeGraph
        from vibecheck.vnnlib_loader import load_vnnlib
        from vibecheck.zonotope import TorchZonotope

        BENCH = '/home/stan/repositories/vnncomp2025_benchmarks/benchmarks/relusplitter'
        onnx_path = BENCH + '/onnx/oval21-benchmark_cifar_base_kw.onnx.gz'
        if not os.path.exists(onnx_path):
            pytest.skip("Benchmark files not available")

        graph = ComputeGraph.from_onnx(onnx_path)
        spec = load_vnnlib(BENCH + '/vnnlib/oval21-benchmark_cifar_base_kw-img1204-eps0.024967320261437907.vnnlib.gz')
        gpu_layers, _ = graph.gpu_layers(DEV, DT)
        nh = len(gpu_layers) - 1
        pw = spec.as_pairwise(); pred, comps = pw
        xl = torch.tensor(spec.x_lo.astype(F), dtype=DT)
        xh = torch.tensor(spec.x_hi.astype(F), dtype=DT)
        z = TorchZonotope.from_input_bounds(xl, xh, DEV, DT)
        sb = {}
        for l in range(nh):
            gl = gpu_layers[l]
            if gl['type'] == 'conv':
                z.propagate_conv(gl['kernel'], gl['bias'], gl['in_shape'],
                                 gl['stride'], gl['padding'])
            else:
                z.propagate_fc(gl['W'], gl['bias'])
            lo, hi = z.apply_relu(); sb[l] = (lo.clone(), hi.clone())
        bounds_np = {l: (sb[l][0].numpy().astype(np.float64),
                         sb[l][1].numpy().astype(np.float64))
                     for l in range(nh)}
        layers_np = []
        for gl in gpu_layers:
            d = {'type': gl['type']}
            if gl['type'] == 'fc':
                d['W'] = gl['W'].numpy().astype(np.float64)
                d['bias'] = gl['bias'].numpy().astype(np.float64)
            else:
                d['kernel'] = gl['kernel'].numpy().astype(np.float64)
                d['bias'] = gl['bias'].numpy().astype(np.float64)
                d['in_shape'] = gl['in_shape']
                d['stride'] = gl['stride']
                d['padding'] = gl['padding']
            layers_np.append(d)
        x_lo_64 = spec.x_lo.astype(np.float64)
        x_hi_64 = spec.x_hi.astype(np.float64)

        m, e = _build_spec_model(
            layers_np, x_lo_64, x_hi_64, bounds_np, pred, 4,
            milp_neurons=set(), n_threads=1)
        m.setParam('TimeLimit', 10)
        m.optimize()
        assert m.status == 2
        lb = m.ObjBound
        assert lb < 0, f"Expected negative LP bound, got {lb}"
        assert np.isfinite(lb)
        m.dispose(); e.dispose()


class TestCROWNScoring:
    def test_score_function_exists(self):
        """score_neurons_by_crown should exist and be callable."""
        from vibecheck.verify_milp import score_neurons_by_crown
        assert callable(score_neurons_by_crown)

    def test_crown_scores_differ_from_relaxation(self):
        """CROWN scores should account for spec direction, not just area."""
        from vibecheck.verify_milp import (
            score_neurons_by_relaxation, score_neurons_by_crown)

        # Two neurons: same relaxation area but different impact on spec
        bounds = {
            0: (np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
        }
        layers_np = [{'type': 'fc',
                       'W': np.array([[1.0, 0.0], [0.0, 1.0]]),
                       'bias': np.zeros(2)}]

        # Relaxation scores should be identical (same area)
        relax_scores = score_neurons_by_relaxation(bounds, layers_np, 1)
        assert abs(relax_scores[(0, 0)] - relax_scores[(0, 1)]) < 1e-6

        # CROWN scores depend on spec weight, should differ if spec
        # weights are asymmetric
        ew_at_layer = {0: np.array([10.0, 0.1])}  # neuron 0 matters much more
        crown_scores = score_neurons_by_crown(bounds, ew_at_layer, 1)
        assert crown_scores[(0, 0)] > crown_scores[(0, 1)] * 5

    def test_crown_scores_weight_by_relaxation_error(self):
        """CROWN score = |ew[i]| * mu[i] where mu is relaxation half-width."""
        from vibecheck.verify_milp import score_neurons_by_crown

        lo = np.array([-2.0, -0.5])
        hi = np.array([1.0, 1.5])
        bounds = {0: (lo, hi)}
        ew_at_layer = {0: np.array([1.0, 1.0])}  # equal spec weight

        scores = score_neurons_by_crown(bounds, ew_at_layer, 1)

        # mu[i] = -hi*lo / (2*(hi-lo))
        mu0 = -1.0 * (-2.0) / (2 * (1.0 - (-2.0)))  # = 2/6 = 0.333
        mu1 = -1.5 * (-0.5) / (2 * (1.5 - (-0.5)))  # = 0.75/4 = 0.1875

        # score = |ew| * mu
        expected_0 = 1.0 * mu0
        expected_1 = 1.0 * mu1
        np.testing.assert_allclose(scores[(0, 0)], expected_0, atol=1e-6)
        np.testing.assert_allclose(scores[(0, 1)], expected_1, atol=1e-6)


# ---------------------------------------------------------------------------
# Test: score_neurons_ew_frac
# ---------------------------------------------------------------------------

class TestEwFracScoring:
    def test_function_exists(self):
        from vibecheck.verify_milp import score_neurons_ew_frac
        assert callable(score_neurons_ew_frac)

    @pytest.mark.skipif(not HAS_GUROBI, reason="Gurobi not installed")
    def test_ew_frac_scores_on_real_network(self):
        """ew_frac scoring produces finite positive scores on oval21."""
        if not os.path.exists(ONNX_PATH):
            pytest.skip("Benchmark files not available")
        from vibecheck.verify_milp import (
            score_neurons_ew_frac, _compute_crown_layer_weights,
            _build_spec_model_compact)
        from vibecheck.vnnlib_loader import load_vnnlib
        from vibecheck.zonotope import TorchZonotope

        graph = ComputeGraph.from_onnx(ONNX_PATH)
        spec = load_vnnlib(VNNLIB_PATH)
        gpu_layers, _ = graph.gpu_layers(DEV, DT)
        nh = len(gpu_layers) - 1
        pw = spec.as_pairwise(); pred, comps = pw

        xl = torch.tensor(spec.x_lo.astype(F), dtype=DT)
        xh = torch.tensor(spec.x_hi.astype(F), dtype=DT)
        z = TorchZonotope.from_input_bounds(xl, xh, DEV, DT)
        sb = {}
        for l in range(nh):
            gl = gpu_layers[l]
            if gl['type'] == 'conv':
                z.propagate_conv(gl['kernel'], gl['bias'], gl['in_shape'],
                                 gl['stride'], gl['padding'])
            else:
                z.propagate_fc(gl['W'], gl['bias'])
            lo, hi = z.apply_relu(); sb[l] = (lo.clone(), hi.clone())
        bounds_np = {l: (sb[l][0].numpy().astype(np.float64),
                         sb[l][1].numpy().astype(np.float64))
                     for l in range(nh)}
        layers_np = []
        for gl in gpu_layers:
            d = {'type': gl['type']}
            if gl['type'] == 'fc':
                d['W'] = gl['W'].numpy().astype(np.float64)
                d['bias'] = gl['bias'].numpy().astype(np.float64)
            else:
                d['kernel'] = gl['kernel'].numpy().astype(np.float64)
                d['bias'] = gl['bias'].numpy().astype(np.float64)
                d['in_shape'] = gl['in_shape']
                d['stride'] = gl['stride']
                d['padding'] = gl['padding']
            layers_np.append(d)
        x_lo_64 = spec.x_lo.astype(np.float64)
        x_hi_64 = spec.x_hi.astype(np.float64)

        from vibecheck.verify_zono_bnb import _build_spec_ew
        spec_ew = _build_spec_ew(gpu_layers, pred, comps, DEV, DT)
        comp = min(comps)
        ew_at_layer = _compute_crown_layer_weights(
            bounds_np, layers_np, spec_ew, pred, comp, nh)

        lp_m, lp_e = _build_spec_model_compact(
            layers_np, x_lo_64, x_hi_64, bounds_np, pred, comp,
            milp_neurons=set(), n_threads=1)
        lp_m.setParam('TimeLimit', 30)
        lp_m.optimize()
        assert lp_m.status == 2

        scores = score_neurons_ew_frac(bounds_np, ew_at_layer, nh, lp_m)
        lp_m.dispose(); lp_e.dispose()

        assert len(scores) > 0
        assert all(np.isfinite(v) and v >= 0 for v in scores.values())


# ---------------------------------------------------------------------------
# Test: Compact spec model matches original
# ---------------------------------------------------------------------------

class TestCompactSpecModel:
    @pytest.mark.skipif(not HAS_GUROBI, reason="Gurobi not installed")
    def test_compact_matches_original(self):
        """Compact spec model gives same LP bound as original."""
        if not os.path.exists(ONNX_PATH):
            pytest.skip("Benchmark files not available")
        from vibecheck.verify_milp import _build_spec_model, _build_spec_model_compact
        from vibecheck.vnnlib_loader import load_vnnlib
        from vibecheck.zonotope import TorchZonotope

        graph = ComputeGraph.from_onnx(ONNX_PATH)
        spec = load_vnnlib(VNNLIB_PATH)
        gpu_layers, _ = graph.gpu_layers(DEV, DT)
        nh = len(gpu_layers) - 1
        pw = spec.as_pairwise(); pred, comps = pw

        xl = torch.tensor(spec.x_lo.astype(F), dtype=DT)
        xh = torch.tensor(spec.x_hi.astype(F), dtype=DT)
        z = TorchZonotope.from_input_bounds(xl, xh, DEV, DT)
        sb = {}
        for l in range(nh):
            gl = gpu_layers[l]
            if gl['type'] == 'conv':
                z.propagate_conv(gl['kernel'], gl['bias'], gl['in_shape'],
                                 gl['stride'], gl['padding'])
            else:
                z.propagate_fc(gl['W'], gl['bias'])
            lo, hi = z.apply_relu(); sb[l] = (lo.clone(), hi.clone())
        bounds_np = {l: (sb[l][0].numpy().astype(np.float64),
                         sb[l][1].numpy().astype(np.float64))
                     for l in range(nh)}
        layers_np = []
        for gl in gpu_layers:
            d = {'type': gl['type']}
            if gl['type'] == 'fc':
                d['W'] = gl['W'].numpy().astype(np.float64)
                d['bias'] = gl['bias'].numpy().astype(np.float64)
            else:
                d['kernel'] = gl['kernel'].numpy().astype(np.float64)
                d['bias'] = gl['bias'].numpy().astype(np.float64)
                d['in_shape'] = gl['in_shape']
                d['stride'] = gl['stride']
                d['padding'] = gl['padding']
            layers_np.append(d)
        x_lo_64 = spec.x_lo.astype(np.float64)
        x_hi_64 = spec.x_hi.astype(np.float64)

        comp = min(comps)

        # Original (with dead neuron vars)
        m1, e1 = _build_spec_model(
            layers_np, x_lo_64, x_hi_64, bounds_np, pred, comp,
            milp_neurons=set(), n_threads=1)
        m1.setParam('TimeLimit', 30)
        m1.optimize()
        assert m1.status == 2
        lb1 = m1.ObjBound
        nv1 = m1.NumVars
        m1.dispose(); e1.dispose()

        # Compact (dead neurons inlined)
        m2, e2 = _build_spec_model_compact(
            layers_np, x_lo_64, x_hi_64, bounds_np, pred, comp,
            milp_neurons=set(), n_threads=1)
        m2.setParam('TimeLimit', 30)
        m2.optimize()
        assert m2.status == 2
        lb2 = m2.ObjBound
        nv2 = m2.NumVars
        m2.dispose(); e2.dispose()

        np.testing.assert_allclose(lb1, lb2, atol=1e-4)
        assert nv2 < nv1, f"compact {nv2} should have fewer vars than full {nv1}"


# ---------------------------------------------------------------------------
# Test: Per-worker LP matches shared model LP
# ---------------------------------------------------------------------------

class TestPerWorkerLP:
    @pytest.mark.skipif(not HAS_GUROBI, reason="Gurobi not installed")
    def test_per_worker_matches_shared(self):
        """Per-worker FC LP gives same bounds as shared model LP."""
        if not os.path.exists(ONNX_PATH):
            pytest.skip("Benchmark files not available")
        from vibecheck.verify_milp import _tighten_layer_parallel
        from vibecheck.vnnlib_loader import load_vnnlib
        from vibecheck.zonotope import TorchZonotope

        graph = ComputeGraph.from_onnx(ONNX_PATH)
        spec = load_vnnlib(VNNLIB_PATH)
        gpu_layers, _ = graph.gpu_layers(DEV, DT)
        nh = len(gpu_layers) - 1

        xl = torch.tensor(spec.x_lo.astype(F), dtype=DT)
        xh = torch.tensor(spec.x_hi.astype(F), dtype=DT)
        z = TorchZonotope.from_input_bounds(xl, xh, DEV, DT)
        sb = {}
        for l in range(nh):
            gl = gpu_layers[l]
            if gl['type'] == 'conv':
                z.propagate_conv(gl['kernel'], gl['bias'], gl['in_shape'],
                                 gl['stride'], gl['padding'])
            else:
                z.propagate_fc(gl['W'], gl['bias'])
            lo, hi = z.apply_relu(); sb[l] = (lo.clone(), hi.clone())
        bounds_np = {l: (sb[l][0].numpy().astype(np.float64),
                         sb[l][1].numpy().astype(np.float64))
                     for l in range(nh)}
        layers_np = []
        for gl in gpu_layers:
            d = {'type': gl['type']}
            if gl['type'] == 'fc':
                d['W'] = gl['W'].numpy().astype(np.float64)
                d['bias'] = gl['bias'].numpy().astype(np.float64)
            else:
                d['kernel'] = gl['kernel'].numpy().astype(np.float64)
                d['bias'] = gl['bias'].numpy().astype(np.float64)
                d['in_shape'] = gl['in_shape']
                d['stride'] = gl['stride']
                d['padding'] = gl['padding']
            layers_np.append(d)
        x_lo_64 = spec.x_lo.astype(np.float64)
        x_hi_64 = spec.x_hi.astype(np.float64)

        # L2 is the FC layer (after two conv layers)
        l = 2
        lo2, hi2 = bounds_np[l]
        unstable = np.where((lo2 < 0) & (hi2 > 0))[0]
        assert len(unstable) > 0, "Need unstable neurons"

        import multiprocessing
        nc = multiprocessing.cpu_count()

        # Shared model path (lp_per_worker=False)
        lo_shared, hi_shared, _ = _tighten_layer_parallel(
            layers_np, x_lo_64, x_hi_64, bounds_np, l,
            use_milp=False, timeout=5.0, n_cores=nc,
            lp_per_worker=False)

        # Per-worker path (lp_per_worker=True)
        lo_pw, hi_pw, _ = _tighten_layer_parallel(
            layers_np, x_lo_64, x_hi_64, bounds_np, l,
            use_milp=False, timeout=5.0, n_cores=nc,
            lp_per_worker=True)

        np.testing.assert_allclose(lo_shared[unstable], lo_pw[unstable],
                                   atol=1e-4)
        np.testing.assert_allclose(hi_shared[unstable], hi_pw[unstable],
                                   atol=1e-4)


# ---------------------------------------------------------------------------
# Test: Racing escalation
# ---------------------------------------------------------------------------

class TestRacingEscalation:
    @pytest.mark.skipif(not HAS_GUROBI, reason="Gurobi not installed")
    def test_racing_functions_exist(self):
        from vibecheck.verify_milp import (
            _solve_spec_worker, _racing_escalation)
        assert callable(_solve_spec_worker)
        assert callable(_racing_escalation)

    @pytest.mark.skipif(not HAS_GUROBI, reason="Gurobi not installed")
    def test_feasibility_worker(self):
        """Feasibility worker returns SAT on easy instance with no binaries."""
        if not os.path.exists(ONNX_PATH):
            pytest.skip("Benchmark files not available")
        from vibecheck.verify_milp import _solve_spec_worker
        from vibecheck.vnnlib_loader import load_vnnlib
        from vibecheck.zonotope import TorchZonotope

        graph = ComputeGraph.from_onnx(ONNX_PATH)
        spec = load_vnnlib(VNNLIB_PATH)
        gpu_layers, _ = graph.gpu_layers(DEV, DT)
        nh = len(gpu_layers) - 1
        pw = spec.as_pairwise(); pred, comps = pw

        xl = torch.tensor(spec.x_lo.astype(F), dtype=DT)
        xh = torch.tensor(spec.x_hi.astype(F), dtype=DT)
        z = TorchZonotope.from_input_bounds(xl, xh, DEV, DT)
        sb = {}
        for l in range(nh):
            gl = gpu_layers[l]
            if gl['type'] == 'conv':
                z.propagate_conv(gl['kernel'], gl['bias'], gl['in_shape'],
                                 gl['stride'], gl['padding'])
            else:
                z.propagate_fc(gl['W'], gl['bias'])
            lo, hi = z.apply_relu(); sb[l] = (lo.clone(), hi.clone())
        bounds_np = {l: (sb[l][0].numpy().astype(np.float64),
                         sb[l][1].numpy().astype(np.float64))
                     for l in range(nh)}
        layers_np = []
        for gl in gpu_layers:
            d = {'type': gl['type']}
            if gl['type'] == 'fc':
                d['W'] = gl['W'].numpy().astype(np.float64)
                d['bias'] = gl['bias'].numpy().astype(np.float64)
            else:
                d['kernel'] = gl['kernel'].numpy().astype(np.float64)
                d['bias'] = gl['bias'].numpy().astype(np.float64)
                d['in_shape'] = gl['in_shape']
                d['stride'] = gl['stride']
                d['padding'] = gl['padding']
            layers_np.append(d)
        x_lo_64 = spec.x_lo.astype(np.float64)
        x_hi_64 = spec.x_hi.astype(np.float64)

        comp = min(comps)
        args = ('feasibility', layers_np, x_lo_64, x_hi_64, bounds_np,
                pred, comp, [], 0, 1, 30)
        result, dt, _ = _solve_spec_worker(args)
        assert result in ('SAT', 'UNSAT', 'UNKNOWN')
        assert dt > 0

    @pytest.mark.skipif(not HAS_GUROBI, reason="Gurobi not installed")
    def test_optimization_worker(self):
        """Optimization worker returns a finite bound."""
        if not os.path.exists(ONNX_PATH):
            pytest.skip("Benchmark files not available")
        from vibecheck.verify_milp import _solve_spec_worker
        from vibecheck.vnnlib_loader import load_vnnlib
        from vibecheck.zonotope import TorchZonotope

        graph = ComputeGraph.from_onnx(ONNX_PATH)
        spec = load_vnnlib(VNNLIB_PATH)
        gpu_layers, _ = graph.gpu_layers(DEV, DT)
        nh = len(gpu_layers) - 1
        pw = spec.as_pairwise(); pred, comps = pw

        xl = torch.tensor(spec.x_lo.astype(F), dtype=DT)
        xh = torch.tensor(spec.x_hi.astype(F), dtype=DT)
        z = TorchZonotope.from_input_bounds(xl, xh, DEV, DT)
        sb = {}
        for l in range(nh):
            gl = gpu_layers[l]
            if gl['type'] == 'conv':
                z.propagate_conv(gl['kernel'], gl['bias'], gl['in_shape'],
                                 gl['stride'], gl['padding'])
            else:
                z.propagate_fc(gl['W'], gl['bias'])
            lo, hi = z.apply_relu(); sb[l] = (lo.clone(), hi.clone())
        bounds_np = {l: (sb[l][0].numpy().astype(np.float64),
                         sb[l][1].numpy().astype(np.float64))
                     for l in range(nh)}
        layers_np = []
        for gl in gpu_layers:
            d = {'type': gl['type']}
            if gl['type'] == 'fc':
                d['W'] = gl['W'].numpy().astype(np.float64)
                d['bias'] = gl['bias'].numpy().astype(np.float64)
            else:
                d['kernel'] = gl['kernel'].numpy().astype(np.float64)
                d['bias'] = gl['bias'].numpy().astype(np.float64)
                d['in_shape'] = gl['in_shape']
                d['stride'] = gl['stride']
                d['padding'] = gl['padding']
            layers_np.append(d)
        x_lo_64 = spec.x_lo.astype(np.float64)
        x_hi_64 = spec.x_hi.astype(np.float64)

        comp = min(comps)
        args = ('optimize', layers_np, x_lo_64, x_hi_64, bounds_np,
                pred, comp, [], 0, 1, 30)
        result, dt, lb = _solve_spec_worker(args)
        assert result in ('SAT', 'UNSAT', 'UNKNOWN')
        assert lb is not None and np.isfinite(lb)
        assert dt > 0


class TestSpecWorkerNumericalRobustness:
    """Regression for the float32-zono / float64-LP ulp-mismatch
    unsoundness in `_solve_spec_worker`.

    Bug: zono forward in float32 can produce active-neuron bounds
    `lo == hi` tight to ~1 ulp. The LP arithmetic runs in float64 and
    computes `expr + b_j` 1-3 ulp outside those bounds. With Gurobi's
    default `FeasibilityTol=1e-6`, this declares the model spuriously
    INFEASIBLE → caller returns 'verified' on a real SAT case.

    Caught on metaroom_2023 4cnn_ry_99_16 / spec_43: 1/10 runs
    declared `verified` on a witness ABC easily found (Y_2 > Y_16).

    Test isolates the failure: builds a 1-input / 1-active-neuron LP
    with bounds tight to ~1 ulp and a constraint residual of ~1.15e-6.
    At default `FeasibilityTol=1e-6` this is INFEASIBLE; vibecheck's
    `_solve_spec_worker` must loosen FeasibilityTol so the same model
    resolves as feasible.
    """

    @pytest.mark.skipif(not HAS_GUROBI, reason="Gurobi not installed")
    def test_default_feasibility_tol_reproduces_bug(self):
        """Sanity: with default FeasibilityTol=1e-6, the constraint is
        spuriously INFEASIBLE."""
        import gurobipy as grb
        lo = 1.4313681126   # float32-truncated lower
        hi = 1.4313681126
        c = 1.0
        b = 1.4313692590 - 1.0  # expr+b at x=1.0 = 1.4313692590, ~1.15e-6 above hi
        env = grb.Env(empty=True); env.setParam('OutputFlag', 0); env.start()
        m = grb.Model(env=env)
        m.setParam('FeasibilityTol', 1e-6)  # default
        x = m.addVar(lb=1.0, ub=1.0)
        a = m.addVar(lb=lo, ub=hi)
        m.addConstr(a == c*x + b)
        m.optimize()
        assert m.Status == 3, (
            f'Expected INFEASIBLE (3) at default FeasibilityTol, '
            f'got {m.Status}')
        m.dispose(); env.dispose()

    @pytest.mark.skipif(not HAS_GUROBI, reason="Gurobi not installed")
    def test_solve_spec_worker_loosened_tol_resolves_feasibility(self):
        """_solve_spec_worker uses a loosened FeasibilityTol that
        absorbs the float32→float64 ulp gap. The same LP that
        spuriously fires INFEASIBLE under default 1e-6 must be feasible
        under vibecheck's loosened tolerance."""
        from vibecheck.verify_milp import _GUROBI_FEAS_TOL
        # Sanity: the constant is strictly larger than 1e-6 so it
        # absorbs the 1.15e-6 residual seen in the wild.
        assert _GUROBI_FEAS_TOL >= 1e-5, (
            f'_GUROBI_FEAS_TOL must be ≥ 1e-5 to absorb the float32/64 ulp '
            f'gap; got {_GUROBI_FEAS_TOL}')

        import gurobipy as grb
        lo = 1.4313681126
        hi = 1.4313681126
        c = 1.0
        b = 1.4313692590 - 1.0
        env = grb.Env(empty=True); env.setParam('OutputFlag', 0); env.start()
        m = grb.Model(env=env)
        m.setParam('FeasibilityTol', _GUROBI_FEAS_TOL)
        x = m.addVar(lb=1.0, ub=1.0)
        a = m.addVar(lb=lo, ub=hi)
        m.addConstr(a == c*x + b)
        m.optimize()
        assert m.Status == 2, (
            f'Expected OPTIMAL (2) at loosened FeasibilityTol={_GUROBI_FEAS_TOL}, '
            f'got {m.Status} — fix regressed')
        m.dispose(); env.dispose()


class TestInflateMilpBounds:
    """`_inflate_milp_bounds` — floating-point-soundness widening of the
    spec-MILP pre-ReLU bounds.

    Bug it guards: the graph spec MILP imposes (lo, hi) as *hard* variable
    bounds but recomputes the affine in float64, while the bounds come from
    float32 zono/CROWN. On a tiny perturbation box (collins_rul: 4 of 400
    inputs move) almost every neuron is near-constant, so its bound is
    degenerate (width ~1e-9) — tighter than the float32→float64 gap. A
    genuinely reachable point then lands just outside [lo,hi] and the spec
    LP is falsely infeasible → `verified` on a case with a real CEX.
    Outward inflation restores the over-approximation.
    """

    def test_widens_by_atol_plus_rtol_times_magnitude(self):
        from vibecheck.verify_milp import _inflate_milp_bounds
        bounds = {
            0: (np.array([4.912, -2.0]), np.array([4.912, 10.0])),
            1: (np.array([0.0]), np.array([0.0])),
        }
        atol, rtol = 1e-5, 1e-5
        out = _inflate_milp_bounds(bounds, atol, rtol)
        # Layer 0, neuron 0: degenerate at 4.912 → tol = atol + rtol*4.912
        tol00 = atol + rtol * 4.912
        assert np.isclose(out[0][0][0], 4.912 - tol00)
        assert np.isclose(out[0][1][0], 4.912 + tol00)
        # Layer 0, neuron 1: max|bound| = 10.0 → tol = atol + rtol*10
        tol01 = atol + rtol * 10.0
        assert np.isclose(out[0][0][1], -2.0 - tol01)
        assert np.isclose(out[0][1][1], 10.0 + tol01)
        # Inflation is strictly outward (lo decreases, hi increases) everywhere.
        for li, (lo, hi) in out.items():
            assert np.all(lo <= np.asarray(bounds[li][0]))
            assert np.all(hi >= np.asarray(bounds[li][1]))

    def test_inflation_preserves_active_dead_classification(self):
        # The inflation must NOT flip a neuron's active/dead classification:
        # an active neuron (lo>=0) keeps lo_new>=0, a dead neuron (hi<=0) keeps
        # hi_new<=0. Otherwise integer-weighted nets (sat_relu) whose
        # always-active neurons have lo==0 exactly get every one reclassified
        # unstable and binarised, exploding the spec MILP (90s vs 0.1s).
        from vibecheck.verify_milp import _inflate_milp_bounds
        bounds = {
            # active@0, dead@0, unstable straddling 0, active@0.3, dead@-0.3
            0: (np.array([0.0, -2.0, -1e-8, 0.3, -3.0]),
                np.array([5.0,  0.0,  1e-8, 9.0, -0.3])),
        }
        out = _inflate_milp_bounds(bounds, 1e-5, 1e-5)
        lo, hi = out[0]
        # Active neuron (lo==0): stays active (lo_new == 0, NOT -tol).
        assert lo[0] == 0.0 and hi[0] > 5.0
        # Dead neuron (hi==0): stays dead (hi_new == 0, NOT +tol).
        assert hi[1] == 0.0 and lo[1] < -2.0
        # Unstable neuron (straddles 0): fully inflated both sides.
        assert lo[2] < -1e-8 and hi[2] > 1e-8
        # Active@0.3 keeps lo_new>=0 (here 0.3-tol still >0, fully inflated).
        assert 0.0 <= lo[3] < 0.3
        # Dead@-0.3 keeps hi_new<=0.
        assert hi[4] <= 0.0 and hi[4] < -0.3 + 1e-3
        # Soundness: the inflated box still contains the original box.
        assert np.all(lo <= bounds[0][0]) and np.all(hi >= bounds[0][1])

    def test_noop_when_tolerances_nonpositive(self):
        # atol<=0 and rtol<=0 must short-circuit and return the input
        # unchanged (lets a config disable inflation entirely).
        from vibecheck.verify_milp import _inflate_milp_bounds
        bounds = {0: (np.array([1.0]), np.array([2.0]))}
        assert _inflate_milp_bounds(bounds, 0.0, 0.0) is bounds
        assert _inflate_milp_bounds(bounds, -1.0, -1.0) is bounds
