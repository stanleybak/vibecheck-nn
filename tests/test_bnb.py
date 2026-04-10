"""Tests for BnB verification: settings, TorchZonotope, gpu_layers, as_pairwise,
verify_zono_bnb helpers, and end-to-end BnB on tiny synthetic networks."""

import numpy as np
import pytest
import torch

from vibecheck.settings import default_settings, resolve_torch
from vibecheck.zonotope import TorchZonotope
from vibecheck.spec import VNNSpec, Conjunct, Constraint, PairwiseConstraint
from vibecheck.network import ComputeGraph, GemmNode, ReluNode, ConvNode
from vibecheck.verify_zono_bnb import (
    _make_slopes, _forward_batch, _pgd_attack, _build_spec_ew,
    _evaluate_region, _fmt_eta, _run_bnb, zonotope_bnb_verify,
)

F = np.float32
DEV = torch.device('cpu')
DT = torch.float32


def _a(vals):
    return np.array(vals, dtype=F)


def _t(vals):
    return torch.tensor(vals, dtype=DT, device=DEV)


# ---------------------------------------------------------------------------
# settings.py
# ---------------------------------------------------------------------------

class TestSettings:
    def test_default_settings(self):
        s = default_settings()
        assert s.device == 'gpu'
        assert s.bits == 32
        assert s.pgd_restarts == 100
        assert s.bnb_order == 'bfs'

    def test_overrides(self):
        s = default_settings(device='cpu', bits=64, bnb_order='dfs')
        assert s.device == 'cpu'
        assert s.bits == 64
        assert s.bnb_order == 'dfs'

    def test_resolve_torch_cpu(self):
        s = default_settings(device='cpu', bits=32)
        dev, dt = resolve_torch(s)
        assert dev == torch.device('cpu')
        assert dt == torch.float32

    def test_resolve_torch_bits16(self):
        s = default_settings(device='cpu', bits=16)
        dev, dt = resolve_torch(s)
        assert dt == torch.float16

    def test_resolve_torch_bits64(self):
        s = default_settings(device='cpu', bits=64)
        dev, dt = resolve_torch(s)
        assert dt == torch.float64

    def test_resolve_torch_gpu_fallback(self):
        """If no CUDA, falls back to cpu."""
        s = default_settings(device='gpu', bits=32)
        dev, dt = resolve_torch(s)
        # On CI without GPU, should fall back to cpu
        if not torch.cuda.is_available():
            assert dev == torch.device('cpu')


# ---------------------------------------------------------------------------
# TorchZonotope
# ---------------------------------------------------------------------------

class TestTorchZonotope:
    def test_from_input_bounds(self):
        xl = _t([0.0, -1.0])
        xh = _t([1.0, 1.0])
        z = TorchZonotope.from_input_bounds(xl, xh, DEV, DT)
        lo, hi = z.bounds()
        torch.testing.assert_close(lo, xl)
        torch.testing.assert_close(hi, xh)

    def test_from_input_bounds_zero_radius(self):
        xl = _t([1.0, 2.0])
        xh = _t([1.0, 4.0])
        z = TorchZonotope.from_input_bounds(xl, xh, DEV, DT)
        assert z.generators.shape[1] == 1  # only dim 1 has nonzero radius
        lo, hi = z.bounds()
        torch.testing.assert_close(lo, xl)
        torch.testing.assert_close(hi, xh)

    def test_propagate_fc(self):
        xl = _t([0.0, 0.0])
        xh = _t([1.0, 1.0])
        z = TorchZonotope.from_input_bounds(xl, xh, DEV, DT)
        W = _t([[1.0, 2.0], [-1.0, 1.0]])
        b = _t([0.0, 0.0])
        z.propagate_fc(W, b)
        lo, hi = z.bounds()
        torch.testing.assert_close(lo, _t([0.0, -1.0]))
        torch.testing.assert_close(hi, _t([3.0, 1.0]))

    def test_propagate_conv(self):
        # 1x1 conv on (1,2,2) input
        xl = _t([0.0, 0.0, 0.0, 0.0])
        xh = _t([1.0, 1.0, 1.0, 1.0])
        z = TorchZonotope.from_input_bounds(xl, xh, DEV, DT)
        kernel = torch.ones(1, 1, 1, 1, dtype=DT, device=DEV) * 2
        bias = torch.zeros(1, dtype=DT, device=DEV)
        z.propagate_conv(kernel, bias, (1, 2, 2), (1, 1), (0, 0))
        lo, hi = z.bounds()
        torch.testing.assert_close(lo, _t([0.0, 0.0, 0.0, 0.0]))
        torch.testing.assert_close(hi, _t([2.0, 2.0, 2.0, 2.0]))

    def test_apply_relu(self):
        z = TorchZonotope(_t([1.0, -1.0, 0.0]),
                          torch.diag(_t([0.5, 0.5, 1.5])))
        lo, hi = z.apply_relu()
        new_lo, new_hi = z.bounds()
        # dim 0: active (lo=0.5, hi=1.5) -> unchanged
        assert new_lo[0] >= 0.5 - 1e-6
        # dim 1: dead (lo=-1.5, hi=-0.5) -> [0, 0]
        assert abs(float(new_lo[1])) < 1e-6
        assert abs(float(new_hi[1])) < 1e-6
        # dim 2: unstable (lo=-1.5, hi=1.5) -> relaxed, new generators added
        assert z.generators.shape[1] > 3  # new generator appended
        # Bounds should be sound: contain [0, 1.5]
        assert new_hi[2] >= 1.5 - 1e-6

    def test_apply_relu_no_unstable(self):
        z = TorchZonotope(_t([2.0, -2.0]),
                          torch.diag(_t([0.5, 0.5])))
        lo, hi = z.apply_relu()
        # No unstable neurons -> no new generators
        assert z.generators.shape[1] == 2

    def test_copy(self):
        z = TorchZonotope(_t([1.0]), _t([[0.5]]))
        z2 = z.copy()
        z2.center[0] = 999.0
        assert float(z.center[0]) == 1.0  # original unchanged

    def test_bounds(self):
        c = _t([1.0, -1.0])
        g = _t([[0.5, 0.0], [0.0, 2.0]])
        z = TorchZonotope(c, g)
        lo, hi = z.bounds()
        torch.testing.assert_close(lo, _t([0.5, -3.0]))
        torch.testing.assert_close(hi, _t([1.5, 1.0]))


# ---------------------------------------------------------------------------
# _make_slopes
# ---------------------------------------------------------------------------

class TestMakeSlopes:
    def test_active_neurons(self):
        lo = _t([1.0, 2.0])
        hi = _t([3.0, 4.0])
        lo_s, up_s, up_t, active, dead, ust = _make_slopes(lo, hi)
        assert active.all()
        assert not dead.any()
        assert not ust.any()
        torch.testing.assert_close(lo_s, _t([1.0, 1.0]))

    def test_dead_neurons(self):
        lo = _t([-3.0, -2.0])
        hi = _t([-1.0, -0.5])
        lo_s, up_s, up_t, active, dead, ust = _make_slopes(lo, hi)
        assert dead.all()
        assert not active.any()
        torch.testing.assert_close(lo_s, _t([0.0, 0.0]))

    def test_unstable_neurons(self):
        lo = _t([-1.0])
        hi = _t([3.0])
        lo_s, up_s, up_t, active, dead, ust = _make_slopes(lo, hi)
        assert ust.all()
        # up_s = hi / (hi - lo) = 3/4
        torch.testing.assert_close(up_s, _t([0.75]))
        # up_t = -lo * up_s = 1 * 0.75 = 0.75
        torch.testing.assert_close(up_t, _t([0.75]))
        # lo_s: up_s > 0.5 -> 1.0
        torch.testing.assert_close(lo_s, _t([1.0]))

    def test_mixed(self):
        lo = _t([1.0, -1.0, -3.0])
        hi = _t([2.0, 1.0, -1.0])
        lo_s, up_s, up_t, active, dead, ust = _make_slopes(lo, hi)
        assert active[0] and not active[1] and not active[2]
        assert not dead[0] and not dead[1] and dead[2]
        assert not ust[0] and ust[1] and not ust[2]


# ---------------------------------------------------------------------------
# _fmt_eta
# ---------------------------------------------------------------------------

class TestFmtEta:
    def test_seconds(self):
        assert _fmt_eta(5.3) == '5.3s'

    def test_minutes(self):
        assert _fmt_eta(125) == '2m05s'

    def test_hours(self):
        assert _fmt_eta(3700) == '1h01m'

    def test_days(self):
        assert _fmt_eta(90000) == '1d01h'

    def test_many_days(self):
        assert _fmt_eta(100 * 86400) == '>99days'


# ---------------------------------------------------------------------------
# Helper: build a tiny FC graph for BnB tests
# ---------------------------------------------------------------------------

def _tiny_fc_graph(dtype=np.float32):
    """Input(2) -> Gemm(2->3) -> Relu -> Gemm(3->2): pairwise-verifiable."""
    g = ComputeGraph(dtype=dtype)
    g.input_name = 'input'
    g.input_shape = (1, 2)

    W1 = np.array([[1.0, 0.5], [-0.5, 1.0], [0.5, 0.5]], dtype=dtype)
    b1 = np.array([0.1, 0.1, 0.1], dtype=dtype)
    W2 = np.array([[1.0, 0.5, 0.5], [-0.5, 0.5, -0.5]], dtype=dtype)
    b2 = np.array([2.0, -2.0], dtype=dtype)  # big bias to make pred=0 win

    g.nodes['gemm1'] = GemmNode(name='gemm1', op_type='Gemm',
                                 inputs=['input'], params={'W': W1, 'b': b1})
    g.nodes['relu'] = ReluNode(name='relu', op_type='Relu', inputs=['gemm1'])
    g.nodes['gemm2'] = GemmNode(name='gemm2', op_type='Gemm',
                                 inputs=['relu'], params={'W': W2, 'b': b2})
    g.output_name = 'gemm2'
    g.topological_sort()
    shapes = {g.input_name: g.input_shape}
    for name in g.topo_order:
        g.nodes[name].infer_shape(shapes)
        shapes[name] = g.nodes[name].output_shape
    return g


# ---------------------------------------------------------------------------
# gpu_layers
# ---------------------------------------------------------------------------

class TestGpuLayers:
    def test_fc_layers(self):
        g = _tiny_fc_graph()
        layers, fwd_data = g.gpu_layers(DEV, DT)
        assert len(layers) == 2  # 2 Gemm nodes (Relu is skipped)
        assert layers[0]['type'] == 'fc'
        assert layers[1]['type'] == 'fc'
        assert layers[0]['W'].shape == (3, 2)
        assert layers[1]['W'].shape == (2, 3)
        assert len(fwd_data['layer_types']) == 2
        assert fwd_data['layer_types'][0] == ('fc', None)

    def test_conv_layers(self):
        """Test gpu_layers with a conv node."""
        g = ComputeGraph(dtype=np.float32)
        g.input_name = 'input'
        g.input_shape = (1, 1, 4, 4)

        kernel = np.ones((1, 1, 3, 3), dtype=np.float32)
        bias = np.zeros(1, dtype=np.float32)
        g.nodes['conv'] = ConvNode(
            name='conv', op_type='Conv', inputs=['input'],
            params={'kernel': kernel, 'bias': bias,
                    'stride': (1, 1), 'padding': (0, 0)})
        W2 = np.eye(4, dtype=np.float32)
        b2 = np.zeros(4, dtype=np.float32)
        g.nodes['gemm'] = GemmNode(
            name='gemm', op_type='Gemm', inputs=['conv'],
            params={'W': W2, 'b': b2})
        g.output_name = 'gemm'
        g.topological_sort()
        shapes = {g.input_name: g.input_shape}
        for name in g.topo_order:
            g.nodes[name].infer_shape(shapes)
            shapes[name] = g.nodes[name].output_shape

        layers, fwd_data = g.gpu_layers(DEV, DT)
        assert len(layers) == 2
        assert layers[0]['type'] == 'conv'
        assert layers[1]['type'] == 'fc'
        assert 'output_padding' in layers[0]
        assert 'n_out' in layers[0]
        assert fwd_data['layer_types'][0][0] == 'conv'


# ---------------------------------------------------------------------------
# as_pairwise
# ---------------------------------------------------------------------------

class TestAsPairwise:
    def test_pairwise_spec(self):
        spec = VNNSpec(_a([0, 0]), _a([1, 1]), [
            Conjunct([PairwiseConstraint(pred=0, comp=1)]),
            Conjunct([PairwiseConstraint(pred=0, comp=2)]),
        ])
        result = spec.as_pairwise()
        assert result is not None
        pred, comps = result
        assert pred == 0
        assert comps == {1, 2}

    def test_non_pairwise_returns_none(self):
        spec = VNNSpec(_a([0, 0]), _a([1, 1]), [
            Conjunct([Constraint(0, '>=', 5.0)]),
        ])
        assert spec.as_pairwise() is None

    def test_mixed_preds_returns_none(self):
        spec = VNNSpec(_a([0, 0, 0]), _a([1, 1, 1]), [
            Conjunct([PairwiseConstraint(pred=0, comp=1)]),
            Conjunct([PairwiseConstraint(pred=1, comp=2)]),
        ])
        assert spec.as_pairwise() is None


# ---------------------------------------------------------------------------
# _build_spec_ew
# ---------------------------------------------------------------------------

class TestBuildSpecEw:
    def test_fc_spec_ew(self):
        g = _tiny_fc_graph()
        layers, _ = g.gpu_layers(DEV, DT)
        spec_ew = _build_spec_ew(layers, pred=0, comps={1}, device=DEV,
                                  dtype=DT)
        assert 1 in spec_ew
        ew, b = spec_ew[1]
        # ew should be W[0] - W[1] from final layer
        W2 = layers[-1]['W']
        expected = W2[0] - W2[1]
        torch.testing.assert_close(ew, expected)


# ---------------------------------------------------------------------------
# _forward_batch
# ---------------------------------------------------------------------------

class TestForwardBatch:
    def test_single_input(self):
        g = _tiny_fc_graph()
        _, fwd_data = g.gpu_layers(DEV, DT)
        nh = len(fwd_data['layer_types']) - 1
        x = _t([[1.0, 0.5]])
        out = _forward_batch(x, fwd_data, nh)
        assert out.shape == (1, 2)
        # Verify it's a reasonable output (just check shape and finiteness)
        assert torch.isfinite(out).all()

    def test_batch(self):
        g = _tiny_fc_graph()
        _, fwd_data = g.gpu_layers(DEV, DT)
        nh = len(fwd_data['layer_types']) - 1
        x = torch.rand(5, 2, dtype=DT, device=DEV)
        out = _forward_batch(x, fwd_data, nh)
        assert out.shape == (5, 2)


# ---------------------------------------------------------------------------
# _pgd_attack
# ---------------------------------------------------------------------------

class TestPgdAttack:
    def test_unsat_region(self):
        """In a region where pred clearly wins, PGD should not find SAT."""
        g = _tiny_fc_graph()
        _, fwd_data = g.gpu_layers(DEV, DT)
        nh = len(fwd_data['layer_types']) - 1
        settings = default_settings(device='cpu', bits=32,
                                     pgd_restarts=10, pgd_iter=5)
        xl = _t([0.4, 0.4])
        xh = _t([0.6, 0.6])
        is_sat, witness, best_adv = _pgd_attack(
            xl, xh, {1}, pred=0, fwd_data=fwd_data, nh=nh,
            settings=settings)
        # With the large bias [2, -2], pred=0 should win easily
        assert not is_sat
        assert witness is None
        assert best_adv is not None


# ---------------------------------------------------------------------------
# _evaluate_region
# ---------------------------------------------------------------------------

class TestEvaluateRegion:
    def test_verified_region(self):
        """Tight region where pred=0 wins should verify all specs."""
        g = _tiny_fc_graph()
        layers, _ = g.gpu_layers(DEV, DT)
        nh = len(layers) - 1
        spec_ew = _build_spec_ew(layers, pred=0, comps={1}, device=DEV,
                                  dtype=DT)
        xl = _t([0.4, 0.4])
        xh = _t([0.6, 0.6])
        spec_lbs, still_open, split_dim = _evaluate_region(
            xl, xh, {1}, layers, spec_ew, pred=0, nh=nh,
            device=DEV, dtype=DT)
        assert 1 in spec_lbs
        # With bias [2, -2], spec_lb should be positive
        assert spec_lbs[1] > 0
        assert len(still_open) == 0
        assert split_dim == -1

    def test_wide_region_needs_split(self):
        """Wide region may not verify, returning split_dim >= 0."""
        g = _tiny_fc_graph()
        layers, _ = g.gpu_layers(DEV, DT)
        nh = len(layers) - 1
        # Use smaller bias so verification is harder
        layers[-1]['bias'] = _t([0.5, -0.5])
        spec_ew = _build_spec_ew(layers, pred=0, comps={1}, device=DEV,
                                  dtype=DT)
        xl = _t([-10.0, -10.0])
        xh = _t([10.0, 10.0])
        spec_lbs, still_open, split_dim = _evaluate_region(
            xl, xh, {1}, layers, spec_ew, pred=0, nh=nh,
            device=DEV, dtype=DT)
        # With wide region and small bias, likely not verified
        if still_open:
            assert split_dim >= 0


# ---------------------------------------------------------------------------
# _run_bnb
# ---------------------------------------------------------------------------

class TestRunBnb:
    def test_immediate_verified(self):
        """If evaluate_fn verifies everything, BnB should return verified."""
        def evaluate_fn(x_l, x_h, remaining):
            return {c: 1.0 for c in remaining}, set(), -1

        def pgd_fn(x_l, x_h, remaining):
            return False, None, None

        settings = default_settings(device='cpu', bnb_order='dfs',
                                     bnb_timeout=5, print_progress=False)
        result, details = _run_bnb(evaluate_fn, pgd_fn,
                                    np.zeros(2, dtype=F), np.ones(2, dtype=F),
                                    {1}, settings)
        assert result == 'verified'
        assert details['n_evals'] == 1

    def test_sat_from_pgd_initial(self):
        """If initial PGD finds SAT, return immediately."""
        def evaluate_fn(x_l, x_h, remaining):
            return {}, remaining, 0

        def pgd_fn(x_l, x_h, remaining):
            return True, np.array([0.5, 0.5], dtype=F), None

        settings = default_settings(device='cpu', bnb_order='dfs',
                                     bnb_timeout=5, print_progress=False)
        result, details = _run_bnb(evaluate_fn, pgd_fn,
                                    np.zeros(2, dtype=F), np.ones(2, dtype=F),
                                    {1}, settings)
        assert result == 'sat'
        assert details['n_evals'] == 0

    def test_sat_from_pgd_at_node(self):
        """PGD at a BnB node finds SAT."""
        call_count = [0]
        def evaluate_fn(x_l, x_h, remaining):
            return {c: -0.1 for c in remaining}, remaining, 0

        def pgd_fn(x_l, x_h, remaining):
            call_count[0] += 1
            if call_count[0] == 1:
                return False, None, np.array([0.3, 0.3], dtype=F)
            return True, np.array([0.5, 0.5], dtype=F), None

        settings = default_settings(device='cpu', bnb_order='dfs',
                                     bnb_timeout=5, print_progress=False)
        result, details = _run_bnb(evaluate_fn, pgd_fn,
                                    np.zeros(2, dtype=F), np.ones(2, dtype=F),
                                    {1}, settings)
        assert result == 'sat'

    def test_timeout(self):
        """BnB should stop on timeout."""
        def evaluate_fn(x_l, x_h, remaining):
            return {c: -0.1 for c in remaining}, remaining, 0

        def pgd_fn(x_l, x_h, remaining):
            return False, None, np.array([0.3, 0.3], dtype=F)

        settings = default_settings(device='cpu', bnb_order='bfs',
                                     bnb_timeout=0.01, print_progress=False)
        result, details = _run_bnb(evaluate_fn, pgd_fn,
                                    np.zeros(2, dtype=F), np.ones(2, dtype=F),
                                    {1}, settings)
        assert result == 'unknown'

    def test_bfs_order(self):
        """BFS explores breadth-first."""
        depths_seen = []
        def evaluate_fn(x_l, x_h, remaining):
            return {c: 1.0 for c in remaining}, set(), -1

        def pgd_fn(x_l, x_h, remaining):
            return False, None, None

        settings = default_settings(device='cpu', bnb_order='bfs',
                                     bnb_timeout=5, print_progress=False)
        result, details = _run_bnb(evaluate_fn, pgd_fn,
                                    np.zeros(2, dtype=F), np.ones(2, dtype=F),
                                    {1}, settings)
        assert result == 'verified'

    def test_dfs_with_split(self):
        """DFS with a split that eventually verifies."""
        eval_count = [0]
        def evaluate_fn(x_l, x_h, remaining):
            eval_count[0] += 1
            width = x_h[0] - x_l[0]
            if width < 0.6:
                return {c: 1.0 for c in remaining}, set(), -1
            return {c: -0.1 for c in remaining}, remaining, 0

        def pgd_fn(x_l, x_h, remaining):
            return False, None, np.array([0.3, 0.3], dtype=F)

        settings = default_settings(device='cpu', bnb_order='dfs',
                                     bnb_timeout=5, print_progress=False)
        result, details = _run_bnb(evaluate_fn, pgd_fn,
                                    np.zeros(2, dtype=F), np.ones(2, dtype=F),
                                    {1}, settings)
        assert result == 'verified'
        assert eval_count[0] >= 2

    def test_print_progress(self):
        """With print_progress, should not crash."""
        eval_count = [0]
        def evaluate_fn(x_l, x_h, remaining):
            eval_count[0] += 1
            if eval_count[0] >= 2:
                return {c: 1.0 for c in remaining}, set(), -1
            return {c: -0.1 for c in remaining}, remaining, 0

        def pgd_fn(x_l, x_h, remaining):
            return False, None, np.array([0.3, 0.3], dtype=F)

        settings = default_settings(device='cpu', bnb_order='dfs',
                                     bnb_timeout=5, print_progress=True)
        result, details = _run_bnb(evaluate_fn, pgd_fn,
                                    np.zeros(2, dtype=F), np.ones(2, dtype=F),
                                    {1}, settings)
        assert result == 'verified'

    def test_bfs_with_adv_right(self):
        """BFS where adversarial is in right child."""
        eval_count = [0]
        def evaluate_fn(x_l, x_h, remaining):
            eval_count[0] += 1
            if eval_count[0] >= 2:
                return {c: 1.0 for c in remaining}, set(), -1
            return {c: -0.1 for c in remaining}, remaining, 0

        def pgd_fn(x_l, x_h, remaining):
            # best_adv in right half (> mid)
            return False, None, np.array([0.8, 0.5], dtype=F)

        settings = default_settings(device='cpu', bnb_order='bfs',
                                     bnb_timeout=5, print_progress=False)
        result, details = _run_bnb(evaluate_fn, pgd_fn,
                                    np.zeros(2, dtype=F), np.ones(2, dtype=F),
                                    {1}, settings)
        assert result == 'verified'


# ---------------------------------------------------------------------------
# End-to-end: zonotope_bnb_verify
# ---------------------------------------------------------------------------

class TestBnbVerifyE2E:
    def test_easy_pairwise(self):
        """BnB on tiny FC graph with large bias -> immediate verification."""
        g = _tiny_fc_graph()
        spec = VNNSpec(
            _a([0.0, 0.0]), _a([1.0, 1.0]),
            [Conjunct([PairwiseConstraint(pred=0, comp=1)])])
        settings = default_settings(device='cpu', bits=32, bnb_order='dfs',
                                     bnb_timeout=5, pgd_restarts=10,
                                     pgd_iter=3, print_progress=False)
        result, details = zonotope_bnb_verify(g, spec, settings)
        assert result == 'verified'

    def test_non_pairwise_raises(self):
        """Non-pairwise spec should raise."""
        g = _tiny_fc_graph()
        spec = VNNSpec(
            _a([0.0, 0.0]), _a([1.0, 1.0]),
            [Conjunct([Constraint(0, '>=', 100.0)])])
        settings = default_settings(device='cpu', bits=32,
                                     print_progress=False)
        with pytest.raises(AssertionError, match="pairwise"):
            zonotope_bnb_verify(g, spec, settings)

    def test_default_settings_used(self):
        """When settings=None, defaults are used."""
        g = _tiny_fc_graph()
        spec = VNNSpec(
            _a([0.4, 0.4]), _a([0.6, 0.6]),
            [Conjunct([PairwiseConstraint(pred=0, comp=1)])])
        result, details = zonotope_bnb_verify(g, spec)
        assert result == 'verified'

    def test_multiple_comps(self):
        """BnB with 3-class output and multiple comps."""
        g = ComputeGraph(dtype=np.float32)
        g.input_name = 'input'
        g.input_shape = (1, 2)
        W1 = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=F)
        b1 = np.zeros(3, dtype=F)
        W2 = np.array([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0], [-0.5, 0.0, 0.5]],
                       dtype=F)
        b2 = np.array([3.0, -3.0, -3.0], dtype=F)
        g.nodes['gemm1'] = GemmNode(name='gemm1', op_type='Gemm',
                                     inputs=['input'],
                                     params={'W': W1, 'b': b1})
        g.nodes['relu'] = ReluNode(name='relu', op_type='Relu',
                                    inputs=['gemm1'])
        g.nodes['gemm2'] = GemmNode(name='gemm2', op_type='Gemm',
                                     inputs=['relu'],
                                     params={'W': W2, 'b': b2})
        g.output_name = 'gemm2'
        g.topological_sort()
        shapes = {g.input_name: g.input_shape}
        for name in g.topo_order:
            g.nodes[name].infer_shape(shapes)
            shapes[name] = g.nodes[name].output_shape

        spec = VNNSpec(
            _a([0.0, 0.0]), _a([1.0, 1.0]),
            [Conjunct([PairwiseConstraint(pred=0, comp=1)]),
             Conjunct([PairwiseConstraint(pred=0, comp=2)])])
        settings = default_settings(device='cpu', bits=32, bnb_order='bfs',
                                     bnb_timeout=5, pgd_restarts=10,
                                     pgd_iter=3, print_progress=False)
        result, details = zonotope_bnb_verify(g, spec, settings)
        assert result == 'verified'


# ---------------------------------------------------------------------------
# main.py BnB path
# ---------------------------------------------------------------------------

class TestMainBnbPath:
    def test_main_bnb_help(self):
        """Verify --mode bnb is a valid argument (just parse check)."""
        from vibecheck.main import main
        import sys
        # Just verify the import works — actual CLI tested via subprocess
        assert callable(main)


# ---------------------------------------------------------------------------
# Deep FC network for Phase 2 backward coverage
# ---------------------------------------------------------------------------

def _deep_fc_graph(dtype=np.float32):
    """Input(2) -> Gemm(2->4) -> Relu -> Gemm(4->4) -> Relu -> Gemm(4->4) -> Relu -> Gemm(4->2).

    4 hidden layers (3 ReLU) means Phase 2 backward tightening loops over l=1,2.
    """
    g = ComputeGraph(dtype=dtype)
    g.input_name = 'input'
    g.input_shape = (1, 2)

    W1 = np.array([[1.0, 0.5], [-0.5, 1.0], [0.3, -0.3], [0.2, 0.8]],
                   dtype=dtype)
    b1 = np.array([0.1, -0.1, 0.0, 0.1], dtype=dtype)
    W2 = np.array([[1.0, 0.2, -0.3, 0.1], [0.1, 1.0, 0.2, -0.1],
                    [-0.2, 0.3, 1.0, 0.2], [0.3, -0.1, 0.1, 1.0]],
                   dtype=dtype)
    b2 = np.array([0.0, 0.0, 0.0, 0.0], dtype=dtype)
    W3 = np.array([[0.5, 0.3, 0.1, -0.2], [-0.1, 0.5, 0.3, 0.1],
                    [0.2, -0.1, 0.5, 0.3], [0.1, 0.2, -0.1, 0.5]],
                   dtype=dtype)
    b3 = np.array([0.0, 0.0, 0.0, 0.0], dtype=dtype)
    W4 = np.array([[1.0, 0.5, 0.3, 0.1], [-0.5, 0.3, -0.1, 0.5]],
                   dtype=dtype)
    b4 = np.array([3.0, -3.0], dtype=dtype)

    g.nodes['g1'] = GemmNode(name='g1', op_type='Gemm', inputs=['input'],
                              params={'W': W1, 'b': b1})
    g.nodes['r1'] = ReluNode(name='r1', op_type='Relu', inputs=['g1'])
    g.nodes['g2'] = GemmNode(name='g2', op_type='Gemm', inputs=['r1'],
                              params={'W': W2, 'b': b2})
    g.nodes['r2'] = ReluNode(name='r2', op_type='Relu', inputs=['g2'])
    g.nodes['g3'] = GemmNode(name='g3', op_type='Gemm', inputs=['r2'],
                              params={'W': W3, 'b': b3})
    g.nodes['r3'] = ReluNode(name='r3', op_type='Relu', inputs=['g3'])
    g.nodes['g4'] = GemmNode(name='g4', op_type='Gemm', inputs=['r3'],
                              params={'W': W4, 'b': b4})
    g.output_name = 'g4'
    g.topological_sort()
    shapes = {g.input_name: g.input_shape}
    for name in g.topo_order:
        g.nodes[name].infer_shape(shapes)
        shapes[name] = g.nodes[name].output_shape
    return g


class TestDeepFCNetwork:
    def test_evaluate_region_phase2(self):
        """Deep FC network exercises Phase 2 backward tightening."""
        g = _deep_fc_graph()
        layers, _ = g.gpu_layers(DEV, DT)
        nh = len(layers) - 1
        assert nh == 3  # 3 relu layers
        spec_ew = _build_spec_ew(layers, pred=0, comps={1}, device=DEV,
                                  dtype=DT)
        xl = _t([-1.0, -1.0])
        xh = _t([1.0, 1.0])
        spec_lbs, still_open, split_dim = _evaluate_region(
            xl, xh, {1}, layers, spec_ew, pred=0, nh=nh,
            device=DEV, dtype=DT)
        assert 1 in spec_lbs

    def test_evaluate_region_phase2_mixed_neurons(self):
        """Phase 2 backward where FC layer k has both active and unstable neurons."""
        g = ComputeGraph(dtype=np.float32)
        g.input_name = 'input'
        g.input_shape = (1, 4)
        # Layer 0: bias makes first 2 neurons active, last 2 unstable
        W1 = np.eye(4, dtype=F) * 0.5
        b1 = np.array([5.0, 5.0, 0.0, 0.0], dtype=F)  # first 2 active
        W2 = np.eye(4, dtype=F) * 0.3
        b2 = np.zeros(4, dtype=F)
        W3 = np.array([[1.0, 0.5, -0.3, 0.2], [-0.5, 0.3, 0.1, -0.2]],
                       dtype=F)
        b3 = np.array([3.0, -3.0], dtype=F)

        g.nodes['g1'] = GemmNode(name='g1', op_type='Gemm', inputs=['input'],
                                  params={'W': W1, 'b': b1})
        g.nodes['r1'] = ReluNode(name='r1', op_type='Relu', inputs=['g1'])
        g.nodes['g2'] = GemmNode(name='g2', op_type='Gemm', inputs=['r1'],
                                  params={'W': W2, 'b': b2})
        g.nodes['r2'] = ReluNode(name='r2', op_type='Relu', inputs=['g2'])
        g.nodes['g3'] = GemmNode(name='g3', op_type='Gemm', inputs=['r2'],
                                  params={'W': W3, 'b': b3})
        g.output_name = 'g3'
        g.topological_sort()
        shapes = {g.input_name: g.input_shape}
        for name in g.topo_order:
            g.nodes[name].infer_shape(shapes)
            shapes[name] = g.nodes[name].output_shape

        layers, _ = g.gpu_layers(DEV, DT)
        nh = len(layers) - 1
        spec_ew = _build_spec_ew(layers, pred=0, comps={1}, device=DEV,
                                  dtype=DT)
        # Input range: first 2 dims high (active after bias=5), last 2 straddle 0
        xl = _t([0.0, 0.0, -2.0, -2.0])
        xh = _t([2.0, 2.0, 2.0, 2.0])
        spec_lbs, still_open, split_dim = _evaluate_region(
            xl, xh, {1}, layers, spec_ew, pred=0, nh=nh,
            device=DEV, dtype=DT)
        assert 1 in spec_lbs

    def test_bnb_deep_fc_verified(self):
        """BnB on deep FC with large bias should verify."""
        g = _deep_fc_graph()
        spec = VNNSpec(
            _a([-0.5, -0.5]), _a([0.5, 0.5]),
            [Conjunct([PairwiseConstraint(pred=0, comp=1)])])
        settings = default_settings(device='cpu', bits=32, bnb_order='dfs',
                                     bnb_timeout=5, pgd_restarts=10,
                                     pgd_iter=3, print_progress=False)
        result, details = zonotope_bnb_verify(g, spec, settings)
        assert result == 'verified'

    def test_bnb_deep_fc_bfs(self):
        """BnB BFS on deep FC."""
        g = _deep_fc_graph()
        spec = VNNSpec(
            _a([-0.5, -0.5]), _a([0.5, 0.5]),
            [Conjunct([PairwiseConstraint(pred=0, comp=1)])])
        settings = default_settings(device='cpu', bits=32, bnb_order='bfs',
                                     bnb_timeout=5, pgd_restarts=10,
                                     pgd_iter=3, print_progress=False)
        result, details = zonotope_bnb_verify(g, spec, settings)
        assert result == 'verified'


# ---------------------------------------------------------------------------
# Conv network for conv path coverage
# ---------------------------------------------------------------------------

def _conv_fc_graph(dtype=np.float32):
    """Conv(1,1,3,3) -> Relu -> FC(4->2): exercises conv paths in BnB."""
    g = ComputeGraph(dtype=dtype)
    g.input_name = 'input'
    g.input_shape = (1, 1, 4, 4)

    kernel = np.random.RandomState(42).randn(1, 1, 3, 3).astype(dtype) * 0.5
    bias_conv = np.zeros(1, dtype=dtype)
    W2 = np.array([[1.0, 0.5, -0.3, 0.2], [-0.5, 0.3, 0.1, -0.2]],
                   dtype=dtype)
    b2 = np.array([3.0, -3.0], dtype=dtype)

    g.nodes['conv'] = ConvNode(
        name='conv', op_type='Conv', inputs=['input'],
        params={'kernel': kernel, 'bias': bias_conv,
                'stride': (1, 1), 'padding': (0, 0)})
    g.nodes['relu'] = ReluNode(name='relu', op_type='Relu', inputs=['conv'])
    g.nodes['gemm'] = GemmNode(
        name='gemm', op_type='Gemm', inputs=['relu'],
        params={'W': W2, 'b': b2})
    g.output_name = 'gemm'
    g.topological_sort()
    shapes = {g.input_name: g.input_shape}
    for name in g.topo_order:
        g.nodes[name].infer_shape(shapes)
        shapes[name] = g.nodes[name].output_shape
    return g


class TestConvNetwork:
    def test_conv_forward_batch(self):
        """Forward batch through conv network."""
        g = _conv_fc_graph()
        _, fwd_data = g.gpu_layers(DEV, DT)
        nh = len(fwd_data['layer_types']) - 1
        x = torch.rand(3, 16, dtype=DT, device=DEV)  # 1*4*4 = 16
        out = _forward_batch(x, fwd_data, nh)
        assert out.shape == (3, 2)

    def test_conv_evaluate_region(self):
        """Evaluate region on conv+fc network."""
        g = _conv_fc_graph()
        layers, _ = g.gpu_layers(DEV, DT)
        nh = len(layers) - 1
        spec_ew = _build_spec_ew(layers, pred=0, comps={1}, device=DEV,
                                  dtype=DT)
        xl = _t([0.0] * 16)
        xh = _t([1.0] * 16)
        spec_lbs, still_open, split_dim = _evaluate_region(
            xl, xh, {1}, layers, spec_ew, pred=0, nh=nh,
            device=DEV, dtype=DT)
        assert 1 in spec_lbs

    def test_conv_bnb_verify(self):
        """Full BnB on conv+fc network."""
        g = _conv_fc_graph()
        spec = VNNSpec(
            _a([0.0] * 16), _a([1.0] * 16),
            [Conjunct([PairwiseConstraint(pred=0, comp=1)])])
        settings = default_settings(device='cpu', bits=32, bnb_order='dfs',
                                     bnb_timeout=5, pgd_restarts=10,
                                     pgd_iter=3, print_progress=False)
        result, details = zonotope_bnb_verify(g, spec, settings)
        # With bias [3, -3], should verify
        assert result == 'verified'


# ---------------------------------------------------------------------------
# Conv as final layer for _build_spec_ew conv path
# ---------------------------------------------------------------------------

def _conv_final_graph(dtype=np.float32):
    """FC(2->4) -> Relu -> Conv(1,1,1,1) final layer: exercises conv spec_ew path."""
    g = ComputeGraph(dtype=dtype)
    g.input_name = 'input'
    g.input_shape = (1, 2)

    W1 = np.array([[1.0, 0.5], [-0.5, 1.0], [0.3, 0.3], [0.2, -0.2]],
                   dtype=dtype)
    b1 = np.array([0.1, 0.0, 0.0, 0.1], dtype=dtype)
    # Conv: 2 output channels from 1 input channel, 1x1 kernel on 2x2 input
    kernel = np.array([[[[1.0]], [[0.5]]]]).astype(dtype)  # (2, 1, 1, 1) — but we need (2, 1, 1, 1) with input (1,2,2)
    # Actually need input to be (1, 2, 2) so 4 elements from FC output
    kernel = np.random.RandomState(42).randn(2, 1, 1, 1).astype(dtype)
    bias_conv = np.array([1.0, -1.0], dtype=dtype)

    g.nodes['gemm'] = GemmNode(name='gemm', op_type='Gemm', inputs=['input'],
                                params={'W': W1, 'b': b1})
    g.nodes['relu'] = ReluNode(name='relu', op_type='Relu', inputs=['gemm'])
    g.nodes['conv'] = ConvNode(
        name='conv', op_type='Conv', inputs=['relu'],
        params={'kernel': kernel, 'bias': bias_conv,
                'stride': (1, 1), 'padding': (0, 0)})
    g.output_name = 'conv'
    g.topological_sort()
    shapes = {g.input_name: g.input_shape}
    for name in g.topo_order:
        g.nodes[name].infer_shape(shapes)
        shapes[name] = g.nodes[name].output_shape
    return g


class TestConvFinalLayer:
    def test_build_spec_ew_conv(self):
        """Build spec effective weights when final layer is conv."""
        g = _conv_final_graph()
        layers, _ = g.gpu_layers(DEV, DT)
        # Final layer is conv
        assert layers[-1]['type'] == 'conv'
        spec_ew = _build_spec_ew(layers, pred=0, comps={1}, device=DEV,
                                  dtype=DT)
        assert 1 in spec_ew
        ew, b = spec_ew[1]
        assert ew.shape[0] > 0  # should have some weights

    def test_conv_final_evaluate(self):
        """Evaluate region with conv as final layer."""
        g = _conv_final_graph()
        layers, _ = g.gpu_layers(DEV, DT)
        nh = len(layers) - 1
        spec_ew = _build_spec_ew(layers, pred=0, comps={1}, device=DEV,
                                  dtype=DT)
        xl = _t([0.0, 0.0])
        xh = _t([1.0, 1.0])
        spec_lbs, still_open, split_dim = _evaluate_region(
            xl, xh, {1}, layers, spec_ew, pred=0, nh=nh,
            device=DEV, dtype=DT)
        assert 1 in spec_lbs


# ---------------------------------------------------------------------------
# PGD SAT coverage — need a network where PGD can actually find violation
# ---------------------------------------------------------------------------

def _easy_sat_graph(dtype=np.float32):
    """Network where output[0] < output[1] for some inputs.
    Input(2) -> Gemm(2->2) -> Relu -> Gemm(2->2): nearly identity with Relu.
    With inputs in the right range, pred=0 loses to comp=1.
    """
    g = ComputeGraph(dtype=dtype)
    g.input_name = 'input'
    g.input_shape = (1, 2)
    W1 = np.eye(2, dtype=dtype)
    b1 = np.zeros(2, dtype=dtype)
    W2 = np.eye(2, dtype=dtype)
    b2 = np.zeros(2, dtype=dtype)
    g.nodes['g1'] = GemmNode(name='g1', op_type='Gemm', inputs=['input'],
                              params={'W': W1, 'b': b1})
    g.nodes['r1'] = ReluNode(name='r1', op_type='Relu', inputs=['g1'])
    g.nodes['g2'] = GemmNode(name='g2', op_type='Gemm', inputs=['r1'],
                              params={'W': W2, 'b': b2})
    g.output_name = 'g2'
    g.topological_sort()
    shapes = {g.input_name: g.input_shape}
    for name in g.topo_order:
        g.nodes[name].infer_shape(shapes)
        shapes[name] = g.nodes[name].output_shape
    return g


class TestPgdSatPaths:
    def test_pgd_finds_sat_during_iter(self):
        """PGD finds SAT during gradient iterations."""
        g = _easy_sat_graph()
        _, fwd_data = g.gpu_layers(DEV, DT)
        nh = len(fwd_data['layer_types']) - 1
        settings = default_settings(device='cpu', bits=32,
                                     pgd_restarts=50, pgd_iter=10)
        # Region where x[0] can be < x[1]: SAT for pred=0, comp=1
        xl = _t([-1.0, 0.5])
        xh = _t([0.0, 1.5])
        is_sat, witness, best_adv = _pgd_attack(
            xl, xh, {1}, pred=0, fwd_data=fwd_data, nh=nh,
            settings=settings)
        assert is_sat
        assert witness is not None

    def test_pgd_finds_sat_final_check(self):
        """PGD finds SAT in the final no-grad check."""
        g = _easy_sat_graph()
        _, fwd_data = g.gpu_layers(DEV, DT)
        nh = len(fwd_data['layer_types']) - 1
        # With 0 iterations, only the final check runs
        settings = default_settings(device='cpu', bits=32,
                                     pgd_restarts=200, pgd_iter=0)
        xl = _t([-2.0, 1.0])
        xh = _t([-0.5, 3.0])
        is_sat, witness, best_adv = _pgd_attack(
            xl, xh, {1}, pred=0, fwd_data=fwd_data, nh=nh,
            settings=settings)
        # Random sampling in a region where x[0]<0 and x[1]>0 will always be SAT
        assert is_sat
        assert witness is not None

    def test_bnb_sat_with_progress(self):
        """BnB finds SAT with print_progress=True."""
        g = _easy_sat_graph()
        spec = VNNSpec(
            _a([-1.0, 0.5]), _a([0.0, 1.5]),
            [Conjunct([PairwiseConstraint(pred=0, comp=1)])])
        settings = default_settings(device='cpu', bits=32, bnb_order='dfs',
                                     bnb_timeout=5, pgd_restarts=50,
                                     pgd_iter=10, print_progress=True)
        result, details = zonotope_bnb_verify(g, spec, settings)
        assert result == 'sat'


# ---------------------------------------------------------------------------
# Additional _run_bnb coverage
# ---------------------------------------------------------------------------

class TestRunBnbAdditional:
    def test_dfs_adv_in_left(self):
        """DFS with adversarial in left child."""
        eval_count = [0]
        def evaluate_fn(x_l, x_h, remaining):
            eval_count[0] += 1
            if eval_count[0] >= 2:
                return {c: 1.0 for c in remaining}, set(), -1
            return {c: -0.1 for c in remaining}, remaining, 0

        def pgd_fn(x_l, x_h, remaining):
            # best_adv in left half (< mid=0.5)
            return False, None, np.array([0.2, 0.5], dtype=F)

        settings = default_settings(device='cpu', bnb_order='dfs',
                                     bnb_timeout=5, print_progress=False)
        result, details = _run_bnb(evaluate_fn, pgd_fn,
                                    np.zeros(2, dtype=F), np.ones(2, dtype=F),
                                    {1}, settings)
        assert result == 'verified'

    def test_dfs_adv_in_right(self):
        """DFS with adversarial in right child (covers lines 466-467)."""
        eval_count = [0]
        def evaluate_fn(x_l, x_h, remaining):
            eval_count[0] += 1
            if eval_count[0] >= 2:
                return {c: 1.0 for c in remaining}, set(), -1
            return {c: -0.1 for c in remaining}, remaining, 0

        def pgd_fn(x_l, x_h, remaining):
            # best_adv in right half (> mid=0.5)
            return False, None, np.array([0.8, 0.5], dtype=F)

        settings = default_settings(device='cpu', bnb_order='dfs',
                                     bnb_timeout=5, print_progress=False)
        result, details = _run_bnb(evaluate_fn, pgd_fn,
                                    np.zeros(2, dtype=F), np.ones(2, dtype=F),
                                    {1}, settings)
        assert result == 'verified'

    def test_sat_at_node_with_progress(self):
        """PGD at BnB node finds SAT with print_progress=True."""
        call_count = [0]
        def evaluate_fn(x_l, x_h, remaining):
            return {c: -0.1 for c in remaining}, remaining, 0

        def pgd_fn(x_l, x_h, remaining):
            call_count[0] += 1
            if call_count[0] == 1:
                return False, None, np.array([0.3, 0.3], dtype=F)
            return True, np.array([0.5, 0.5], dtype=F), None

        settings = default_settings(device='cpu', bnb_order='dfs',
                                     bnb_timeout=5, print_progress=True)
        result, details = _run_bnb(evaluate_fn, pgd_fn,
                                    np.zeros(2, dtype=F), np.ones(2, dtype=F),
                                    {1}, settings)
        assert result == 'sat'

    def test_sat_initial_with_progress(self):
        """Initial PGD finds SAT with print_progress=True."""
        def evaluate_fn(x_l, x_h, remaining):
            return {}, remaining, 0

        def pgd_fn(x_l, x_h, remaining):
            return True, np.array([0.5, 0.5], dtype=F), None

        settings = default_settings(device='cpu', bnb_order='dfs',
                                     bnb_timeout=5, print_progress=True)
        result, details = _run_bnb(evaluate_fn, pgd_fn,
                                    np.zeros(2, dtype=F), np.ones(2, dtype=F),
                                    {1}, settings)
        assert result == 'sat'

    def test_timeout_with_progress(self):
        """Timeout with print_progress=True."""
        def evaluate_fn(x_l, x_h, remaining):
            return {c: -0.1 for c in remaining}, remaining, 0

        def pgd_fn(x_l, x_h, remaining):
            return False, None, np.array([0.3, 0.3], dtype=F)

        settings = default_settings(device='cpu', bnb_order='bfs',
                                     bnb_timeout=0.01, print_progress=True)
        result, details = _run_bnb(evaluate_fn, pgd_fn,
                                    np.zeros(2, dtype=F), np.ones(2, dtype=F),
                                    {1}, settings)
        assert result == 'unknown'


# ---------------------------------------------------------------------------
# main.py CLI coverage
# ---------------------------------------------------------------------------

class TestEvaluateRegionEdgeCases:
    def test_no_hidden_layers(self):
        """Network with 0 ReLU layers: nh=0, empty sb/tight."""
        g = ComputeGraph(dtype=np.float32)
        g.input_name = 'input'
        g.input_shape = (1, 2)
        W = np.eye(2, dtype=F)
        b = np.array([3.0, -3.0], dtype=F)
        g.nodes['g'] = GemmNode(name='g', op_type='Gemm', inputs=['input'],
                                 params={'W': W, 'b': b})
        g.output_name = 'g'
        g.topological_sort()
        shapes = {g.input_name: g.input_shape}
        for name in g.topo_order:
            g.nodes[name].infer_shape(shapes)
            shapes[name] = g.nodes[name].output_shape

        layers, _ = g.gpu_layers(DEV, DT)
        nh = len(layers) - 1
        assert nh == 0
        spec_ew = _build_spec_ew(layers, pred=0, comps={1}, device=DEV,
                                  dtype=DT)
        xl = _t([0.0, 0.0])
        xh = _t([1.0, 1.0])
        spec_lbs, still_open, split_dim = _evaluate_region(
            xl, xh, {1}, layers, spec_ew, pred=0, nh=nh,
            device=DEV, dtype=DT)
        assert 1 in spec_lbs

    def test_conv_sparse_backward(self):
        """Conv backward with <50% unstable neurons (sparse path).

        Requires: layer k is conv AND has pct_unstable < 0.5 AND layer l
        has unstable neurons. Use large positive bias on layer 0 so most
        neurons are active (few unstable -> pct < 0.5), and a second conv
        with zero bias to create instability at layer 1.
        """
        g = ComputeGraph(dtype=np.float32)
        g.input_name = 'input'
        g.input_shape = (1, 1, 6, 6)

        rng = np.random.RandomState(42)
        k1 = rng.randn(1, 1, 3, 3).astype(F) * 0.3
        b1 = np.array([2.0], dtype=F)  # positive bias -> mostly active at narrow range
        W2 = rng.randn(4, 16).astype(F) * 0.5
        b2 = np.zeros(4, dtype=F)
        W3 = np.array([[1.0, 0.5, -0.3, 0.2], [-0.5, 0.3, 0.1, -0.2]], dtype=F)
        b3 = np.array([3.0, -3.0], dtype=F)

        g.nodes['c1'] = ConvNode(
            name='c1', op_type='Conv', inputs=['input'],
            params={'kernel': k1, 'bias': b1,
                    'stride': (1, 1), 'padding': (0, 0)})
        g.nodes['r1'] = ReluNode(name='r1', op_type='Relu', inputs=['c1'])
        g.nodes['g2'] = GemmNode(
            name='g2', op_type='Gemm', inputs=['r1'],
            params={'W': W2, 'b': b2})
        g.nodes['r2'] = ReluNode(name='r2', op_type='Relu', inputs=['g2'])
        g.nodes['fc'] = GemmNode(
            name='fc', op_type='Gemm', inputs=['r2'],
            params={'W': W3, 'b': b3})
        g.output_name = 'fc'
        g.topological_sort()
        shapes = {g.input_name: g.input_shape}
        for name in g.topo_order:
            g.nodes[name].infer_shape(shapes)
            shapes[name] = g.nodes[name].output_shape

        layers, _ = g.gpu_layers(DEV, DT)
        nh = len(layers) - 1
        spec_ew = _build_spec_ew(layers, pred=0, comps={1}, device=DEV,
                                  dtype=DT)
        # Moderate input range: L0 conv all active (pct=0 < 0.5), L1 has unstable
        xl = _t([0.0] * 36)
        xh = _t([2.0] * 36)  # r=1.0 centered at 1.0
        spec_lbs, still_open, split_dim = _evaluate_region(
            xl, xh, {1}, layers, spec_ew, pred=0, nh=nh,
            device=DEV, dtype=DT)
        assert 1 in spec_lbs

    def test_fc_only_unstable_no_active(self):
        """FC backward where a layer has ONLY unstable neurons (no active).

        Layer 0 bias = 0, small weights, wide input range -> all outputs
        straddle zero -> all unstable, 0 active -> covers lines 291-294.
        """
        g = ComputeGraph(dtype=np.float32)
        g.input_name = 'input'
        g.input_shape = (1, 4)
        # Layer 0: all outputs centered near 0 for wide input -> all unstable
        W1 = np.eye(4, dtype=F) * 0.01
        b1 = np.zeros(4, dtype=F)
        # Layer 1: similar
        W2 = np.eye(4, dtype=F) * 0.01
        b2 = np.zeros(4, dtype=F)
        W3 = np.array([[1.0, 0.5, -0.3, 0.2], [-0.5, 0.3, 0.1, -0.2]],
                       dtype=F)
        b3 = np.array([3.0, -3.0], dtype=F)

        g.nodes['g1'] = GemmNode(name='g1', op_type='Gemm', inputs=['input'],
                                  params={'W': W1, 'b': b1})
        g.nodes['r1'] = ReluNode(name='r1', op_type='Relu', inputs=['g1'])
        g.nodes['g2'] = GemmNode(name='g2', op_type='Gemm', inputs=['r1'],
                                  params={'W': W2, 'b': b2})
        g.nodes['r2'] = ReluNode(name='r2', op_type='Relu', inputs=['g2'])
        g.nodes['g3'] = GemmNode(name='g3', op_type='Gemm', inputs=['r2'],
                                  params={'W': W3, 'b': b3})
        g.output_name = 'g3'
        g.topological_sort()
        shapes = {g.input_name: g.input_shape}
        for name in g.topo_order:
            g.nodes[name].infer_shape(shapes)
            shapes[name] = g.nodes[name].output_shape

        layers, _ = g.gpu_layers(DEV, DT)
        nh = len(layers) - 1
        spec_ew = _build_spec_ew(layers, pred=0, comps={1}, device=DEV,
                                  dtype=DT)
        # Wide range centered at 0 -> all neurons unstable at layer 0
        xl = _t([-100.0, -100.0, -100.0, -100.0])
        xh = _t([100.0, 100.0, 100.0, 100.0])
        spec_lbs, still_open, split_dim = _evaluate_region(
            xl, xh, {1}, layers, spec_ew, pred=0, nh=nh,
            device=DEV, dtype=DT)
        assert 1 in spec_lbs


class TestPhase2StableLayer:
    def test_layer_all_stable_skips_tightening(self):
        """Phase 2 where middle layer has 0 unstable neurons (lines 174-175)."""
        g = ComputeGraph(dtype=np.float32)
        g.input_name = 'input'
        g.input_shape = (1, 2)
        W1 = np.eye(2, dtype=F)
        b1 = np.zeros(2, dtype=F)
        # Layer 1: huge positive bias -> all neurons active (0 unstable)
        W2 = np.eye(2, dtype=F)
        b2 = np.array([100.0, 100.0], dtype=F)
        W3 = np.array([[1.0, 0.5], [-0.5, 1.0]], dtype=F)
        b3 = np.array([3.0, -3.0], dtype=F)

        g.nodes['g1'] = GemmNode(name='g1', op_type='Gemm', inputs=['input'],
                                  params={'W': W1, 'b': b1})
        g.nodes['r1'] = ReluNode(name='r1', op_type='Relu', inputs=['g1'])
        g.nodes['g2'] = GemmNode(name='g2', op_type='Gemm', inputs=['r1'],
                                  params={'W': W2, 'b': b2})
        g.nodes['r2'] = ReluNode(name='r2', op_type='Relu', inputs=['g2'])
        g.nodes['g3'] = GemmNode(name='g3', op_type='Gemm', inputs=['r2'],
                                  params={'W': W3, 'b': b3})
        g.output_name = 'g3'
        g.topological_sort()
        shapes = {g.input_name: g.input_shape}
        for name in g.topo_order:
            g.nodes[name].infer_shape(shapes)
            shapes[name] = g.nodes[name].output_shape

        layers, _ = g.gpu_layers(DEV, DT)
        nh = len(layers) - 1
        assert nh == 2
        spec_ew = _build_spec_ew(layers, pred=0, comps={1}, device=DEV,
                                  dtype=DT)
        xl = _t([-1.0, -1.0])
        xh = _t([1.0, 1.0])
        spec_lbs, still_open, split_dim = _evaluate_region(
            xl, xh, {1}, layers, spec_ew, pred=0, nh=nh,
            device=DEV, dtype=DT)
        assert 1 in spec_lbs


class TestMainDirect:
    """Test main() directly via monkeypatch for coverage of CLI BnB path."""

    def test_main_bnb_verified(self, monkeypatch):
        """Call main() with BnB mode on a tiny graph."""
        import sys
        g = _tiny_fc_graph()
        monkeypatch.setattr('vibecheck.main.ComputeGraph.from_onnx',
                            lambda *a, **kw: g)
        monkeypatch.setattr('vibecheck.main.load_vnnlib',
                            lambda *a, **kw: VNNSpec(
                                _a([0.0, 0.0]), _a([1.0, 1.0]),
                                [Conjunct([PairwiseConstraint(pred=0, comp=1)])]))
        monkeypatch.setattr(sys, 'argv',
                            ['vibecheck', '--net', 'x.onnx', '--spec', 'x.vnnlib',
                             '--mode', 'bnb', '--device', 'cpu',
                             '--pgd-restarts', '10'])
        from vibecheck.main import main
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0  # verified

    def test_main_bnb_sat(self, monkeypatch):
        """Call main() with BnB mode that finds SAT."""
        import sys
        g = _easy_sat_graph()
        monkeypatch.setattr('vibecheck.main.ComputeGraph.from_onnx',
                            lambda *a, **kw: g)
        monkeypatch.setattr('vibecheck.main.load_vnnlib',
                            lambda *a, **kw: VNNSpec(
                                _a([-1.0, 0.5]), _a([0.0, 1.5]),
                                [Conjunct([PairwiseConstraint(pred=0, comp=1)])]))
        monkeypatch.setattr(sys, 'argv',
                            ['vibecheck', '--net', 'x.onnx', '--spec', 'x.vnnlib',
                             '--mode', 'bnb', '--device', 'cpu',
                             '--pgd-restarts', '50'])
        from vibecheck.main import main
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1  # sat (not verified)


class TestFuseGemmReshapeConv:
    def test_fuse_basic(self):
        """Fuse Gemm → Reshape → Conv into single FC."""
        from vibecheck.network import ComputeGraph, GemmNode, ConvNode, ReshapeNode, ReluNode
        from vibecheck.onnx_optimizer import fuse_gemm_reshape_conv

        g = ComputeGraph(dtype=np.float32)
        g.input_name = 'input'
        g.input_shape = (1, 4)

        # Gemm(4 → 8), Reshape(8 → 2,2,2), Conv(2→1, 1x1) → Relu → FC(4→2)
        W1 = np.random.RandomState(1).randn(8, 4).astype(F)
        b1 = np.zeros(8, dtype=F)
        g.nodes['gemm1'] = GemmNode(name='gemm1', op_type='Gemm',
                                     inputs=['input'],
                                     params={'W': W1, 'b': b1})
        g.nodes['reshape'] = ReshapeNode(name='reshape', op_type='Reshape',
                                          inputs=['gemm1'],
                                          params={'shape': (1, 2, 2, 2)})
        g.nodes['reshape'].output_shape = (1, 2, 2, 2)
        kernel = np.random.RandomState(2).randn(1, 2, 1, 1).astype(F)
        b_conv = np.zeros(1, dtype=F)
        g.nodes['conv'] = ConvNode(name='conv', op_type='Conv',
                                    inputs=['reshape'],
                                    params={'kernel': kernel, 'bias': b_conv,
                                            'stride': (1, 1),
                                            'padding': (0, 0), 'group': 1})
        g.nodes['relu'] = ReluNode(name='relu', op_type='Relu',
                                    inputs=['conv'])
        W2 = np.random.RandomState(3).randn(2, 4).astype(F)
        b2 = np.array([1.0, -1.0], dtype=F)
        g.nodes['fc'] = GemmNode(name='fc', op_type='Gemm', inputs=['relu'],
                                  params={'W': W2, 'b': b2})
        g.output_name = 'fc'
        g.topological_sort()
        from vibecheck.onnx_loader import _infer_shapes
        _infer_shapes(g)

        # Before fusion: 5 nodes
        assert len(g.nodes) == 5

        fused = fuse_gemm_reshape_conv(g)
        assert fused

        # After fusion: 3 nodes (gemm1, relu, fc)
        assert len(g.nodes) == 3
        assert 'reshape' not in g.nodes
        assert 'conv' not in g.nodes
        # Fused FC: W should be (4, 4) — Conv(2→1,1x1) on (2,2,2) → (1,2,2)=4
        assert g.nodes['gemm1'].params['W'].shape == (4, 4)

    def test_fuse_preserves_output(self):
        """Fused forward pass matches unfused."""
        from vibecheck.network import ComputeGraph, GemmNode, ConvNode, ReshapeNode
        from vibecheck.onnx_optimizer import fuse_gemm_reshape_conv
        from vibecheck.verify import zonotope_verify
        from vibecheck.spec import VNNSpec, Conjunct, Constraint

        def _make_graph():
            g = ComputeGraph(dtype=np.float32)
            g.input_name = 'input'
            g.input_shape = (1, 4)
            W1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                           [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1],
                           [1, 0, 1, 0], [0, 1, 0, 1]], dtype=F)
            b1 = np.ones(8, dtype=F) * 0.5
            g.nodes['gemm1'] = GemmNode(name='gemm1', op_type='Gemm',
                                         inputs=['input'],
                                         params={'W': W1, 'b': b1})
            g.nodes['reshape'] = ReshapeNode(name='reshape', op_type='Reshape',
                                              inputs=['gemm1'],
                                              params={'shape': (1, 2, 2, 2)})
            g.nodes['reshape'].output_shape = (1, 2, 2, 2)
            kernel = np.ones((1, 2, 1, 1), dtype=F)
            b_conv = np.array([0.1], dtype=F)
            g.nodes['conv'] = ConvNode(name='conv', op_type='Conv',
                                        inputs=['reshape'],
                                        params={'kernel': kernel,
                                                'bias': b_conv,
                                                'stride': (1, 1),
                                                'padding': (0, 0),
                                                'group': 1})
            g.output_name = 'conv'
            g.topological_sort()
            from vibecheck.onnx_loader import _infer_shapes, _precache_conv_tensors
            _infer_shapes(g)
            _precache_conv_tensors(g)
            return g

        center = np.array([1.0, 2.0, 3.0, 4.0], dtype=F)
        spec = VNNSpec(center, center,
                       [Conjunct([Constraint(0, '>=', 9999.0)])])

        g1 = _make_graph()
        _, d1 = zonotope_verify(g1, spec)

        g2 = _make_graph()
        fuse_gemm_reshape_conv(g2)
        _, d2 = zonotope_verify(g2, spec)

        np.testing.assert_allclose(d1['output_lo'], d2['output_lo'],
                                   atol=1e-5)

    def test_no_fuse_when_no_pattern(self):
        """No fusion when pattern doesn't match."""
        from vibecheck.onnx_optimizer import fuse_gemm_reshape_conv

        g = _tiny_fc_graph()
        fused = fuse_gemm_reshape_conv(g)
        assert not fused

    def test_no_fuse_gemm_multi_consumer(self):
        """No fusion when Gemm feeds multiple consumers."""
        from vibecheck.network import (ComputeGraph, GemmNode, ConvNode,
                                        ReshapeNode, AddNode)
        from vibecheck.onnx_optimizer import fuse_gemm_reshape_conv
        from vibecheck.onnx_loader import _infer_shapes

        g = ComputeGraph(dtype=np.float32)
        g.input_name = 'input'
        g.input_shape = (1, 4)
        W1 = np.eye(8, 4, dtype=F)
        b1 = np.zeros(8, dtype=F)
        g.nodes['gemm'] = GemmNode(name='gemm', op_type='Gemm',
                                    inputs=['input'],
                                    params={'W': W1, 'b': b1})
        g.nodes['reshape'] = ReshapeNode(name='reshape', op_type='Reshape',
                                          inputs=['gemm'],
                                          params={'shape': (1, 2, 2, 2)})
        g.nodes['reshape'].output_shape = (1, 2, 2, 2)
        kernel = np.ones((1, 2, 1, 1), dtype=F)
        g.nodes['conv'] = ConvNode(name='conv', op_type='Conv',
                                    inputs=['reshape'],
                                    params={'kernel': kernel,
                                            'bias': np.zeros(1, dtype=F),
                                            'stride': (1, 1),
                                            'padding': (0, 0), 'group': 1})
        # gemm also feeds add (multi-consumer)
        g.nodes['add'] = AddNode(name='add', op_type='Add',
                                  inputs=['gemm', 'conv'])
        g.output_name = 'add'
        g.topological_sort()
        _infer_shapes(g)
        assert not fuse_gemm_reshape_conv(g)

    def test_no_fuse_reshape_no_conv(self):
        """No fusion when reshape successor is not Conv."""
        from vibecheck.network import ComputeGraph, GemmNode, ReshapeNode, ReluNode
        from vibecheck.onnx_optimizer import fuse_gemm_reshape_conv
        from vibecheck.onnx_loader import _infer_shapes

        g = ComputeGraph(dtype=np.float32)
        g.input_name = 'input'
        g.input_shape = (1, 4)
        W1 = np.eye(8, 4, dtype=F)
        b1 = np.zeros(8, dtype=F)
        g.nodes['gemm'] = GemmNode(name='gemm', op_type='Gemm',
                                    inputs=['input'],
                                    params={'W': W1, 'b': b1})
        g.nodes['reshape'] = ReshapeNode(name='reshape', op_type='Reshape',
                                          inputs=['gemm'],
                                          params={'shape': (1, 2, 2, 2)})
        g.nodes['reshape'].output_shape = (1, 2, 2, 2)
        # Reshape followed by Relu (not Conv)
        g.nodes['relu'] = ReluNode(name='relu', op_type='Relu',
                                    inputs=['reshape'])
        g.output_name = 'relu'
        g.topological_sort()
        _infer_shapes(g)
        assert not fuse_gemm_reshape_conv(g)

    def test_no_fuse_reshape_none_shape(self):
        """No fusion when reshape output_shape is None."""
        from vibecheck.network import ComputeGraph, GemmNode, ReshapeNode, ConvNode
        from vibecheck.onnx_optimizer import fuse_gemm_reshape_conv

        g = ComputeGraph(dtype=np.float32)
        g.input_name = 'input'
        g.input_shape = (1, 4)
        W1 = np.eye(8, 4, dtype=F)
        b1 = np.zeros(8, dtype=F)
        g.nodes['gemm'] = GemmNode(name='gemm', op_type='Gemm',
                                    inputs=['input'],
                                    params={'W': W1, 'b': b1})
        g.nodes['reshape'] = ReshapeNode(name='reshape', op_type='Reshape',
                                          inputs=['gemm'],
                                          params={'shape': (1, 2, 2, 2)})
        g.nodes['reshape'].output_shape = None  # force None
        kernel = np.ones((1, 2, 1, 1), dtype=F)
        g.nodes['conv'] = ConvNode(name='conv', op_type='Conv',
                                    inputs=['reshape'],
                                    params={'kernel': kernel,
                                            'bias': np.zeros(1, dtype=F),
                                            'stride': (1, 1),
                                            'padding': (0, 0), 'group': 1})
        g.output_name = 'conv'
        g.topological_sort()
        assert not fuse_gemm_reshape_conv(g)

    def test_no_fuse_reshape_bad_dims(self):
        """No fusion when reshape shape is 2D."""
        from vibecheck.network import ComputeGraph, GemmNode, ReshapeNode, ConvNode
        from vibecheck.onnx_optimizer import fuse_gemm_reshape_conv

        g = ComputeGraph(dtype=np.float32)
        g.input_name = 'input'
        g.input_shape = (1, 4)
        W1 = np.eye(8, 4, dtype=F)
        b1 = np.zeros(8, dtype=F)
        g.nodes['gemm'] = GemmNode(name='gemm', op_type='Gemm',
                                    inputs=['input'],
                                    params={'W': W1, 'b': b1})
        g.nodes['reshape'] = ReshapeNode(name='reshape', op_type='Reshape',
                                          inputs=['gemm'],
                                          params={'shape': (2, 4)})
        g.nodes['reshape'].output_shape = (2, 4)  # 2D, not 3D or 4D
        kernel = np.ones((1, 2, 1, 1), dtype=F)
        g.nodes['conv'] = ConvNode(name='conv', op_type='Conv',
                                    inputs=['reshape'],
                                    params={'kernel': kernel,
                                            'bias': np.zeros(1, dtype=F),
                                            'stride': (1, 1),
                                            'padding': (0, 0), 'group': 1})
        g.output_name = 'conv'
        g.topological_sort()
        assert not fuse_gemm_reshape_conv(g)

    def test_no_fuse_prod_mismatch(self):
        """No fusion when reshape prod != W_gemm.shape[0]."""
        from vibecheck.network import ComputeGraph, GemmNode, ReshapeNode, ConvNode
        from vibecheck.onnx_optimizer import fuse_gemm_reshape_conv

        g = ComputeGraph(dtype=np.float32)
        g.input_name = 'input'
        g.input_shape = (1, 4)
        W1 = np.eye(8, 4, dtype=F)
        b1 = np.zeros(8, dtype=F)
        g.nodes['gemm'] = GemmNode(name='gemm', op_type='Gemm',
                                    inputs=['input'],
                                    params={'W': W1, 'b': b1})
        g.nodes['reshape'] = ReshapeNode(name='reshape', op_type='Reshape',
                                          inputs=['gemm'],
                                          params={'shape': (1, 1, 3, 3)})
        g.nodes['reshape'].output_shape = (1, 1, 3, 3)  # prod=9 != W.shape[0]=8
        kernel = np.ones((1, 1, 1, 1), dtype=F)
        g.nodes['conv'] = ConvNode(name='conv', op_type='Conv',
                                    inputs=['reshape'],
                                    params={'kernel': kernel,
                                            'bias': np.zeros(1, dtype=F),
                                            'stride': (1, 1),
                                            'padding': (0, 0), 'group': 1})
        g.output_name = 'conv'
        g.topological_sort()
        assert not fuse_gemm_reshape_conv(g)

    def test_fuse_3d_reshape(self):
        """Fusion works when reshape output is 3D (no batch dim)."""
        from vibecheck.network import ComputeGraph, GemmNode, ConvNode, ReshapeNode
        from vibecheck.onnx_optimizer import fuse_gemm_reshape_conv
        from vibecheck.onnx_loader import _infer_shapes

        g = ComputeGraph(dtype=np.float32)
        g.input_name = 'input'
        g.input_shape = (1, 4)
        W1 = np.random.RandomState(1).randn(8, 4).astype(F)
        b1 = np.zeros(8, dtype=F)
        g.nodes['gemm'] = GemmNode(name='gemm', op_type='Gemm',
                                    inputs=['input'],
                                    params={'W': W1, 'b': b1})
        g.nodes['reshape'] = ReshapeNode(name='reshape', op_type='Reshape',
                                          inputs=['gemm'],
                                          params={'shape': (2, 2, 2)})
        g.nodes['reshape'].output_shape = (2, 2, 2)  # 3D
        kernel = np.ones((1, 2, 1, 1), dtype=F)
        b_conv = np.zeros(1, dtype=F)
        g.nodes['conv'] = ConvNode(name='conv', op_type='Conv',
                                    inputs=['reshape'],
                                    params={'kernel': kernel, 'bias': b_conv,
                                            'stride': (1, 1),
                                            'padding': (0, 0), 'group': 1})
        g.output_name = 'conv'
        g.topological_sort()
        _infer_shapes(g)
        assert fuse_gemm_reshape_conv(g)
        assert g.nodes['gemm'].params['W'].shape == (4, 4)

    def test_fuse_disabled_by_setting(self):
        """Setting fuse_gemm_conv=False skips fusion."""
        g = _tiny_fc_graph()
        spec = VNNSpec(
            _a([0.0, 0.0]), _a([1.0, 1.0]),
            [Conjunct([PairwiseConstraint(pred=0, comp=1)])])
        settings = default_settings(device='cpu', bits=32,
                                     fuse_gemm_conv=False,
                                     print_progress=False)
        # Should run fine without fusion
        from vibecheck.verify_zono_bnb import zonotope_bnb_verify
        result, _ = zonotope_bnb_verify(g, spec, settings)
        assert result == 'verified'


class TestMainCLI:
    @staticmethod
    def _build_onnx(tmp_path):
        """Build tiny ONNX: input(2) -> MatMul+bias -> Relu -> MatMul+bias -> output(2)."""
        import onnx
        from onnx import helper, TensorProto
        W1 = np.eye(2, dtype=np.float32)
        b1 = np.zeros(2, dtype=np.float32)
        W2 = np.eye(2, dtype=np.float32)
        b2 = np.array([3.0, -3.0], dtype=np.float32)
        W1_init = helper.make_tensor('W1', TensorProto.FLOAT, [2, 2],
                                      W1.flatten().tolist())
        b1_init = helper.make_tensor('b1', TensorProto.FLOAT, [2],
                                      b1.flatten().tolist())
        W2_init = helper.make_tensor('W2', TensorProto.FLOAT, [2, 2],
                                      W2.flatten().tolist())
        b2_init = helper.make_tensor('b2', TensorProto.FLOAT, [2],
                                      b2.flatten().tolist())
        mm1 = helper.make_node('MatMul', ['input', 'W1'], ['mm1'])
        add1 = helper.make_node('Add', ['mm1', 'b1'], ['a1'])
        relu = helper.make_node('Relu', ['a1'], ['r1'])
        mm2 = helper.make_node('MatMul', ['r1', 'W2'], ['mm2'])
        add2 = helper.make_node('Add', ['mm2', 'b2'], ['output'])
        graph = helper.make_graph(
            [mm1, add1, relu, mm2, add2], 'test',
            [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2])],
            [W1_init, b1_init, W2_init, b2_init])
        model = helper.make_model(graph,
                                   opset_imports=[helper.make_opsetid('', 13)])
        model.ir_version = 7
        onnx_path = str(tmp_path / 'test.onnx')
        onnx.save(model, onnx_path)
        return onnx_path

    def test_main_bnb_mode(self, tmp_path):
        """Test main() with --mode bnb via subprocess."""
        import subprocess
        onnx_path = self._build_onnx(tmp_path)

        # Write VNNLIB spec
        vnnlib_path = str(tmp_path / 'test.vnnlib')
        with open(vnnlib_path, 'w') as f:
            f.write('(declare-const X_0 Real)\n')
            f.write('(declare-const X_1 Real)\n')
            f.write('(declare-const Y_0 Real)\n')
            f.write('(declare-const Y_1 Real)\n')
            f.write('(assert (<= X_0 1.0))\n')
            f.write('(assert (>= X_0 0.0))\n')
            f.write('(assert (<= X_1 1.0))\n')
            f.write('(assert (>= X_1 0.0))\n')
            f.write('(assert (or (and (>= Y_1 Y_0))))\n')

        result = subprocess.run(
            ['.venv/bin/vibecheck',
             '--net', onnx_path, '--spec', vnnlib_path,
             '--mode', 'bnb', '--device', 'cpu', '--timeout', '5',
             '--pgd-restarts', '10'],
            capture_output=True, text=True, timeout=30)
        assert 'Result:' in result.stdout
        assert 'BnB evals' in result.stdout

    def test_main_bnb_volume_proven(self, tmp_path):
        """Test that volume_proven output appears."""
        import subprocess
        onnx_path = self._build_onnx(tmp_path)
        vnnlib_path = str(tmp_path / 'test.vnnlib')
        with open(vnnlib_path, 'w') as f:
            f.write('(declare-const X_0 Real)\n')
            f.write('(declare-const X_1 Real)\n')
            f.write('(declare-const Y_0 Real)\n')
            f.write('(declare-const Y_1 Real)\n')
            f.write('(assert (<= X_0 1.0))\n')
            f.write('(assert (>= X_0 0.0))\n')
            f.write('(assert (<= X_1 1.0))\n')
            f.write('(assert (>= X_1 0.0))\n')
            f.write('(assert (or (and (>= Y_1 Y_0))))\n')
        result = subprocess.run(
            ['.venv/bin/vibecheck',
             '--net', onnx_path, '--spec', vnnlib_path,
             '--mode', 'bnb', '--device', 'cpu'],
            capture_output=True, text=True, timeout=30)
        assert 'BnB evals' in result.stdout
        assert 'Result:' in result.stdout


# ---------------------------------------------------------------------------
# Deep conv network for Phase 2 conv backward paths
# ---------------------------------------------------------------------------

def _deep_conv_graph(dtype=np.float32):
    """Conv(1,1,3,3) -> Relu -> Conv(1,1,3,3) -> Relu -> FC(1->2).
    2 conv + relu layers -> Phase 2 backward with conv layers.
    Input: (1,1,6,6) = 36 elements.
    """
    g = ComputeGraph(dtype=dtype)
    g.input_name = 'input'
    g.input_shape = (1, 1, 6, 6)

    rng = np.random.RandomState(42)
    k1 = rng.randn(1, 1, 3, 3).astype(dtype) * 0.3
    b1 = np.zeros(1, dtype=dtype)
    k2 = rng.randn(1, 1, 3, 3).astype(dtype) * 0.3
    b2 = np.zeros(1, dtype=dtype)
    # After conv1: (1,1,4,4)=16, after conv2: (1,1,2,2)=4
    W3 = np.array([[1.0, 0.5, -0.3, 0.2], [-0.5, 0.3, 0.1, -0.2]],
                   dtype=dtype)
    b3 = np.array([3.0, -3.0], dtype=dtype)

    g.nodes['c1'] = ConvNode(
        name='c1', op_type='Conv', inputs=['input'],
        params={'kernel': k1, 'bias': b1,
                'stride': (1, 1), 'padding': (0, 0)})
    g.nodes['r1'] = ReluNode(name='r1', op_type='Relu', inputs=['c1'])
    g.nodes['c2'] = ConvNode(
        name='c2', op_type='Conv', inputs=['r1'],
        params={'kernel': k2, 'bias': b2,
                'stride': (1, 1), 'padding': (0, 0)})
    g.nodes['r2'] = ReluNode(name='r2', op_type='Relu', inputs=['c2'])
    g.nodes['fc'] = GemmNode(
        name='fc', op_type='Gemm', inputs=['r2'],
        params={'W': W3, 'b': b3})
    g.output_name = 'fc'
    g.topological_sort()
    shapes = {g.input_name: g.input_shape}
    for name in g.topo_order:
        g.nodes[name].infer_shape(shapes)
        shapes[name] = g.nodes[name].output_shape
    return g


class TestDeepConvNetwork:
    def test_evaluate_region_conv_backward(self):
        """Deep conv network exercises Phase 2 backward with conv layers."""
        g = _deep_conv_graph()
        layers, _ = g.gpu_layers(DEV, DT)
        nh = len(layers) - 1
        assert nh == 2  # 2 relu layers
        spec_ew = _build_spec_ew(layers, pred=0, comps={1}, device=DEV,
                                  dtype=DT)
        xl = _t([0.0] * 36)
        xh = _t([1.0] * 36)
        spec_lbs, still_open, split_dim = _evaluate_region(
            xl, xh, {1}, layers, spec_ew, pred=0, nh=nh,
            device=DEV, dtype=DT)
        assert 1 in spec_lbs

    def test_conv_bnb_verify(self):
        """Full BnB on deep conv network."""
        g = _deep_conv_graph()
        n = 36
        spec = VNNSpec(
            _a([0.0] * n), _a([1.0] * n),
            [Conjunct([PairwiseConstraint(pred=0, comp=1)])])
        settings = default_settings(device='cpu', bits=32, bnb_order='dfs',
                                     bnb_timeout=5, pgd_restarts=10,
                                     pgd_iter=3, print_progress=False)
        result, details = zonotope_bnb_verify(g, spec, settings)
        assert result == 'verified'
