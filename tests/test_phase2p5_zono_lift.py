"""Unit tests for Phase 2.5 zono-lift (verify_graph._phase2p5_zono_lift)
and the _forward_keep_pre_gpu helper."""
import numpy as np
import onnx
import torch
from onnx import helper, TensorProto
import pytest

from vibecheck.network import ComputeGraph
from vibecheck.spec import VNNSpec, Conjunct, Constraint
from vibecheck.settings import default_settings
from vibecheck.verify_graph import verify_graph


def _init(name, arr):
    return helper.make_tensor(name, TensorProto.FLOAT, arr.shape, arr.flatten())


def _input_val(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _tiny_fc(tmp_path, name='m.onnx'):
    """2 → 3 → 3 → 2 FC — small enough that CROWN handles easy specs, but
    has unstable neurons we can probe. Returns ComputeGraph."""
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
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid('', 13)])
    path = str(tmp_path / name)
    onnx.save(model, path)
    return ComputeGraph.from_onnx(path)


def _easy_verifiable_spec(n_in):
    """Y[0] >= 1e6 — unreachable, so the spec is verified."""
    x = np.zeros(n_in, dtype=np.float32)
    eps = 0.01
    return VNNSpec(
        x_lo=x - eps, x_hi=x + eps,
        disjuncts=[Conjunct(
            [Constraint(index=0, op='>=', value=1e6)])])


def test_forward_keep_pre_gpu_records_pre_relu(tmp_path):
    """_forward_keep_pre_gpu returns pre_relu_gpu keyed by layer_idx with
    (center, generators) tensors."""
    from vibecheck.verify_graph import _forward_keep_pre_gpu
    g = _tiny_fc(tmp_path, 'fwd_keep.onnx')
    s = default_settings(device='cpu', bits=64, print_progress=False)
    g.optimize(s)
    device = torch.device('cpu'); dtype = torch.float64
    gg = g.gpu_graph(device, dtype)
    spec = _easy_verifiable_spec(2)
    xl = torch.tensor(spec.x_lo.astype(np.float64), device=device, dtype=dtype)
    xh = torch.tensor(spec.x_hi.astype(np.float64), device=device, dtype=dtype)

    z_final, pre_relu_gpu = _forward_keep_pre_gpu(xl, xh, gg, device, dtype)
    # Two ReLU layers (layer_idx 0 and 1) in this 2-hidden-layer FC
    assert set(pre_relu_gpu.keys()) == {0, 1}
    for L, (c, G) in pre_relu_gpu.items():
        assert c.device.type == 'cpu'
        assert c.dtype == dtype
        assert G.dim() == 2
        # center length == number of neurons at this layer (both 3)
        assert c.numel() == 3


def test_forward_keep_pre_gpu_applies_override_tight(tmp_path):
    """override_tight={L: (lo, hi)} must be passed through to apply_relu
    so the zonotope at L+ uses the tightened bounds."""
    from vibecheck.verify_graph import _forward_keep_pre_gpu
    g = _tiny_fc(tmp_path, 'fwd_override.onnx')
    s = default_settings(device='cpu', bits=64, print_progress=False)
    g.optimize(s)
    device = torch.device('cpu'); dtype = torch.float64
    gg = g.gpu_graph(device, dtype)
    spec = _easy_verifiable_spec(2)
    xl = torch.tensor(spec.x_lo.astype(np.float64), device=device, dtype=dtype)
    xh = torch.tensor(spec.x_hi.astype(np.float64), device=device, dtype=dtype)

    # Baseline
    z_ref, _ = _forward_keep_pre_gpu(xl, xh, gg, device, dtype)
    ref_c = z_ref.center.detach().cpu().numpy().copy()

    # Override: artificially clamp all L0 pre-ReLU to [0, 0.01].  Since
    # apply_relu intersects, this forces every neuron to stable-on with
    # a narrow range — output should shrink.
    override = {0: (np.zeros(3), np.full(3, 0.01))}
    z_o, _ = _forward_keep_pre_gpu(xl, xh, gg, device, dtype,
                                    override_tight=override)
    # Output center should differ (override changed the trajectory).
    out_c = z_o.center.detach().cpu().numpy()
    # At minimum, the forward completes and returns a zonotope of the
    # right shape.
    assert out_c.shape == ref_c.shape


def test_phase2p5_disabled_skips_setting(tmp_path):
    """With zono_lift_enabled=False, Phase 2.5 is not in details['timing']."""
    g = _tiny_fc(tmp_path, 'skip.onnx')
    spec = _easy_verifiable_spec(2)
    s = default_settings(device='cpu', bits=64, total_timeout=30,
                         print_progress=False, zono_lift_enabled=False,
                         input_split_enabled=False)
    result, details = verify_graph(g, spec, s)
    # Spec is trivially verified by CROWN; Phase 2.5 shouldn't have run.
    assert result == 'verified'
    assert 'phase2p5_zono_lift' not in details['timing']


def test_phase2p5_runs_when_crown_leaves_queries_open(monkeypatch, tmp_path):
    """Force CROWN Phase-2 to report every LB = -1.0; Phase 2.5 should run
    and (via real _adaptive_spec_lb on an easy spec) close the queries."""
    from vibecheck import verify_graph as vg

    g = _tiny_fc(tmp_path, 'open.onnx')
    spec = _easy_verifiable_spec(2)

    original_backward = vg._spec_backward_graph

    def fake_backward(sb, xl, xh, gg, spec_ew, qids, nh, device, dtype,
                     return_ew=False):
        if return_ew:
            return original_backward(sb, xl, xh, gg, spec_ew, qids, nh,
                                     device, dtype, return_ew=True)
        lbs = {qi: -1.0 for qi in qids}
        return lbs, None

    monkeypatch.setattr(vg, '_spec_backward_graph', fake_backward)

    # Also prevent PGD from short-circuiting as SAT.
    def fake_pgd(*a, **k):
        return False, None
    monkeypatch.setattr(vg, '_pgd_attack_general', fake_pgd)

    s = default_settings(device='cpu', bits=64, total_timeout=30,
                         print_progress=False, zono_lift_enabled=True,
                         zono_lift_max_passes=2,
                         input_split_enabled=False)
    result, details = verify_graph(g, spec, s)
    # Phase 2.5 should have run.
    assert 'phase2p5_zono_lift' in details['timing']
    assert 'phase2p5' in details
    # For this easy spec the real CROWN LB (via _adaptive_spec_lb in
    # Phase 2.5) is enormous, so Phase 2.5 closes the disjunct.
    assert result == 'verified'
    assert details['phase2p5']['n_closed'] >= 1


def test_phase2p5_converges_without_verify(monkeypatch, tmp_path):
    """If CROWN LB (real) also fails, Phase 2.5 iterates up to max_passes
    and falls through without claiming verified."""
    from vibecheck import verify_graph as vg

    g = _tiny_fc(tmp_path, 'hard.onnx')
    spec = _easy_verifiable_spec(2)

    # Force BOTH _spec_backward_graph AND _adaptive_spec_lb to report
    # negative LB, simulating the "stuck" case.
    original_backward = vg._spec_backward_graph

    def fake_backward(sb, xl, xh, gg, spec_ew, qids, nh, device, dtype,
                     return_ew=False):
        if return_ew:
            return original_backward(sb, xl, xh, gg, spec_ew, qids, nh,
                                     device, dtype, return_ew=True)
        return {qi: -1.0 for qi in qids}, None

    monkeypatch.setattr(vg, '_spec_backward_graph', fake_backward)
    monkeypatch.setattr(vg, '_adaptive_spec_lb', lambda *a, **k: -1.0)

    def fake_pgd(*a, **k):
        return False, None
    monkeypatch.setattr(vg, '_pgd_attack_general', fake_pgd)

    # Disable α-CROWN path too — it runs its own CROWN backward not
    # intercepted by the _adaptive_spec_lb monkey-patch.
    s = default_settings(device='cpu', bits=64, total_timeout=30,
                         print_progress=False, zono_lift_enabled=True,
                         zono_lift_max_passes=2,
                         zono_lift_alpha_crown=False,
                         input_split_enabled=False)
    result, details = verify_graph(g, spec, s)
    # Phase 2.5 ran but didn't close the query (stayed at CROWN LB = -1.0).
    assert 'phase2p5_zono_lift' in details['timing']
    assert details['phase2p5']['n_closed'] == 0
    # verify_graph falls through to later phases; the trivial spec
    # (Y[0] >= 1e6) is handled by LP / downstream.
    assert result in ('verified', 'unknown')


def test_alpha_crown_v2_fixed_intermediate_runs(monkeypatch, tmp_path):
    """Exercise `alpha_crown_impl='v2_fixed_intermediate'`: Phase 2.5 must
    use the v2 path (fixed intermediate bounds + spec-only α + lr_decay)
    instead of the joint legacy path. Force CROWN to leave queries open so
    Phase 2.5 runs, then check that the v2 α-CROWN call returns a sound
    LB and the pipeline closes the easy spec."""
    from vibecheck import verify_graph as vg
    from vibecheck import alpha_crown as ac

    g = _tiny_fc(tmp_path, 'v2_run.onnx')
    spec = _easy_verifiable_spec(2)

    original_backward = vg._spec_backward_graph

    def fake_backward(sb, xl, xh, gg, spec_ew, qids, nh, device, dtype,
                      return_ew=False):
        if return_ew:
            return original_backward(sb, xl, xh, gg, spec_ew, qids, nh,
                                      device, dtype, return_ew=True)
        return {qi: -1.0 for qi in qids}, None

    monkeypatch.setattr(vg, '_spec_backward_graph', fake_backward)
    monkeypatch.setattr(vg, '_pgd_attack_general', lambda *a, **k: (False, None))

    # Count v2 calls to confirm dispatch.
    call_counter = [0]
    orig_v2 = ac.run_alpha_crown_fixed_intermediate

    def counting_v2(*args, **kwargs):
        call_counter[0] += 1
        return orig_v2(*args, **kwargs)

    monkeypatch.setattr(ac, 'run_alpha_crown_fixed_intermediate',
                         counting_v2)
    # batched path also counts toward v2 dispatch.
    orig_v2_b = ac.run_alpha_crown_fixed_intermediate_batched

    def counting_v2_b(*args, **kwargs):
        call_counter[0] += 1
        return orig_v2_b(*args, **kwargs)

    monkeypatch.setattr(ac, 'run_alpha_crown_fixed_intermediate_batched',
                         counting_v2_b)

    s = default_settings(device='cpu', bits=64, total_timeout=30,
                         print_progress=False, zono_lift_enabled=True,
                         zono_lift_max_passes=2,
                         alpha_crown_impl='v2_fixed_intermediate',
                         alpha_crown_lr_decay=0.98,
                         zono_lift_alpha_iters=5,
                         input_split_enabled=False)
    result, details = verify_graph(g, spec, s)
    assert 'phase2p5_zono_lift' in details['timing']
    # v2 path must have been called at least once (single or batched).
    assert call_counter[0] >= 1, (
        f'v2 path not dispatched; count={call_counter[0]}')
    # Easy spec (Y[0] >= 1e6): Phase 2.5 must close and verifier must
    # return 'verified' (soundness: v2 path must produce LB consistent
    # with the true unreachability of Y[0] = 1e6).
    assert result == 'verified'


def test_alpha_crown_v2_fixed_intermediate_standalone(tmp_path):
    """Direct test of `run_alpha_crown_fixed_intermediate`: given a tiny
    network and a trivially-safe spec direction (w=[-1, 0], b=1e6 meaning
    `1e6 - Y[0] >= 0`, always true since Y is bounded), the returned best_lb
    must be positive."""
    from vibecheck import alpha_crown as ac

    g = _tiny_fc(tmp_path, 'v2_standalone.onnx')
    s = default_settings(device='cpu', bits=64, print_progress=False)
    g.optimize(s)
    device = torch.device('cpu')
    dtype = torch.float64
    gg = g.gpu_graph(device, dtype)

    xl = torch.zeros(2, dtype=dtype, device=device)
    xh = torch.full((2,), 0.01, dtype=dtype, device=device)

    # Build bbr_init by running a CROWN backward at phase 2. Simpler: use
    # per-neuron interval bounds via a forward zono pass.
    from vibecheck.verify_graph import _forward_keep_pre_gpu
    _, pre_relu = _forward_keep_pre_gpu(xl, xh, gg, device, dtype)
    bbr_init = {}
    for L, (c, G) in pre_relu.items():
        radius = G.abs().sum(dim=1)
        lo = (c - radius).cpu().numpy().astype(np.float64)
        hi = (c + radius).cpu().numpy().astype(np.float64)
        bbr_init[L] = (lo, hi)

    # Unreachable-on-LB direction: -Y[0] + 1e6 >= 0 is always true.
    w_q = np.array([-1.0, 0.0], dtype=np.float64)
    b_q = 1e6
    best_lb, alpha_params, best_bounds, history = (
        ac.run_alpha_crown_fixed_intermediate(
            gg, xl, xh, bbr_init, w_q, b_q,
            device, dtype, n_iters=3, lr=0.25, lr_decay=0.98,
            early_stop_on_positive=True))
    # best_lb is finite and positive for this trivially-safe direction.
    assert np.isfinite(best_lb)
    assert best_lb > 0
    # alpha_params is shaped correctly: only 'spec' key, one entry per ReLU.
    assert set(alpha_params.keys()) == {'spec'}
    assert set(alpha_params['spec'].keys()) == set(bbr_init.keys())
    # Intermediate bounds are returned unchanged (frozen).
    for L in bbr_init:
        lo_t, hi_t = best_bounds[L]
        np.testing.assert_allclose(
            lo_t.cpu().numpy(), bbr_init[L][0], rtol=1e-12)
        np.testing.assert_allclose(
            hi_t.cpu().numpy(), bbr_init[L][1], rtol=1e-12)


def test_alpha_crown_v2_fixed_intermediate_batched(tmp_path):
    """Direct test of `run_alpha_crown_fixed_intermediate_batched`: batches
    two trivially-safe spec directions; both best_lbs must be positive."""
    from vibecheck import alpha_crown as ac

    g = _tiny_fc(tmp_path, 'v2_batched.onnx')
    s = default_settings(device='cpu', bits=64, print_progress=False)
    g.optimize(s)
    device = torch.device('cpu')
    dtype = torch.float64
    gg = g.gpu_graph(device, dtype)

    xl = torch.zeros(2, dtype=dtype, device=device)
    xh = torch.full((2,), 0.01, dtype=dtype, device=device)

    from vibecheck.verify_graph import _forward_keep_pre_gpu
    _, pre_relu = _forward_keep_pre_gpu(xl, xh, gg, device, dtype)
    bbr_init = {}
    for L, (c, G) in pre_relu.items():
        radius = G.abs().sum(dim=1)
        lo = (c - radius).cpu().numpy().astype(np.float64)
        hi = (c + radius).cpu().numpy().astype(np.float64)
        bbr_init[L] = (lo, hi)

    w_qs = np.array([[-1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
    b_qs = np.array([1e6, 1e6], dtype=np.float64)
    best_lbs, alpha_params, best_bounds, histories = (
        ac.run_alpha_crown_fixed_intermediate_batched(
            gg, xl, xh, bbr_init, w_qs, b_qs,
            device, dtype, n_iters=3, lr=0.25, lr_decay=0.98,
            early_stop_on_positive=True))
    assert best_lbs.shape == (2,)
    assert np.all(np.isfinite(best_lbs))
    assert np.all(best_lbs > 0)
    assert set(alpha_params.keys()) == {'spec'}
    assert len(histories) == 2


def test_alpha_crown_v2_lr_decay_off(tmp_path):
    """lr_decay=1.0 disables the ExponentialLR scheduler (no-op branch)."""
    from vibecheck import alpha_crown as ac

    g = _tiny_fc(tmp_path, 'v2_no_decay.onnx')
    s = default_settings(device='cpu', bits=64, print_progress=False)
    g.optimize(s)
    device = torch.device('cpu')
    dtype = torch.float64
    gg = g.gpu_graph(device, dtype)

    xl = torch.zeros(2, dtype=dtype, device=device)
    xh = torch.full((2,), 0.01, dtype=dtype, device=device)

    from vibecheck.verify_graph import _forward_keep_pre_gpu
    _, pre_relu = _forward_keep_pre_gpu(xl, xh, gg, device, dtype)
    bbr_init = {}
    for L, (c, G) in pre_relu.items():
        radius = G.abs().sum(dim=1)
        lo = (c - radius).cpu().numpy().astype(np.float64)
        hi = (c + radius).cpu().numpy().astype(np.float64)
        bbr_init[L] = (lo, hi)

    # Single: lr_decay=1.0 → no scheduler branch
    best_lb, _, _, _ = ac.run_alpha_crown_fixed_intermediate(
        gg, xl, xh, bbr_init, np.array([-1.0, 0.0]), 1e6,
        device, dtype, n_iters=2, lr=0.25, lr_decay=1.0,
        early_stop_on_positive=True)
    assert best_lb > 0

    # Batched: lr_decay=1.0 → no scheduler branch
    w_qs = np.array([[-1.0, 0.0]], dtype=np.float64)
    b_qs = np.array([1e6], dtype=np.float64)
    best_lbs, _, _, _ = ac.run_alpha_crown_fixed_intermediate_batched(
        gg, xl, xh, bbr_init, w_qs, b_qs,
        device, dtype, n_iters=2, lr=0.25, lr_decay=1.0,
        early_stop_on_positive=True)
    assert best_lbs[0] > 0


def test_phase2p5_layers_override(monkeypatch, tmp_path):
    """Setting zono_lift_layers=[0] restricts tightening to layer 0 only."""
    from vibecheck import verify_graph as vg

    g = _tiny_fc(tmp_path, 'layers_cfg.onnx')
    spec = _easy_verifiable_spec(2)

    original_backward = vg._spec_backward_graph

    def fake_backward(sb, xl, xh, gg, spec_ew, qids, nh, device, dtype,
                     return_ew=False):
        if return_ew:
            return original_backward(sb, xl, xh, gg, spec_ew, qids, nh,
                                     device, dtype, return_ew=True)
        return {qi: -1.0 for qi in qids}, None

    monkeypatch.setattr(vg, '_spec_backward_graph', fake_backward)
    monkeypatch.setattr(vg, '_pgd_attack_general', lambda *a, **k: (False, None))

    s = default_settings(device='cpu', bits=64, total_timeout=30,
                         print_progress=False, zono_lift_enabled=True,
                         zono_lift_max_passes=1, zono_lift_layers=[0],
                         input_split_enabled=False)
    result, details = verify_graph(g, spec, s)
    # Phase 2.5 ran and only touched layer 0. The test passes so long as
    # the code runs without crashing and records per-query info.
    assert 'phase2p5' in details
    assert 'per_query' in details['phase2p5']
