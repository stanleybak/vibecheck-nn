"""Soundness tests for the backward-CROWN handlers of the merged 1-D PWL lookup
table ('pwl') and Floor ('floor') ops in alpha_crown._crown_backward_matrix.

These ops carry fixed (non-alpha) affine bands in the backward pass, so the
guarantee under test is: the ReLU-alpha-CROWN lower bound produced by
run_alpha_crown_fixed_intermediate over a small input box is SOUND, i.e. it never
exceeds the true minimum of the network output on that box. We certify that by
brute-forcing the true min on a dense grid (a valid necessary condition: the
grid min is an upper bound on the true min, so LB <= grid_min must hold).

Sampling is used ONLY to TEST a closed-form bound (the project-sanctioned use);
the bound itself is symbolic.
"""
import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from vibecheck.network import ComputeGraph
from vibecheck.onnx_optimizer import merge_relu_lookup_table
from vibecheck.settings import default_settings
from vibecheck.verify_zono_bnb import (
    _forward_batch_graph,
    _forward_zonotope_graph,
)
from vibecheck.alpha_crown import (
    run_alpha_crown_fixed_intermediate,
    run_alpha_crown_batched,
    _assert_mul_bilinear_one_point,
)
import pytest


# --------------------------------------------------------------------------- #
# Tiny "sandwich" nets: Gemm -> {pwl | floor} -> Gemm, so the backward chain
# threads fc -> nonlinear -> fc just like the real ml4acopf graphs do.
# --------------------------------------------------------------------------- #
def _gemm_nodes(prefix, x_in, w, b, y_out):
    """x_in[1,a] @ w[a,o] + b[o] -> y_out[1,o] via MatMul+Add."""
    return (
        [helper.make_node('MatMul', [x_in, f'{prefix}_W'], [f'{prefix}_mm']),
         helper.make_node('Add', [f'{prefix}_mm', f'{prefix}_B'], [y_out])],
        [numpy_helper.from_array(np.asarray(w, np.float32), f'{prefix}_W'),
         numpy_helper.from_array(np.asarray(b, np.float32), f'{prefix}_B')],
    )


def _pwl_sandwich_onnx(path, W1, b1, off, w, bias, W2, b2):
    n_in, h = W1.shape
    nodes, inits = [], []
    n1, i1 = _gemm_nodes('g1', 'X', W1, b1, 'h_pre')
    nodes += n1 + [helper.make_node('Relu', ['h_pre'], ['h'])]; inits += i1
    nodes += [
        helper.make_node('Unsqueeze', ['h', 'axes'], ['unsq']),
        helper.make_node('Sub', ['unsq', 'OFF'], ['sub']),
        helper.make_node('Relu', ['sub'], ['rl']),
        helper.make_node('MatMul', ['rl', 'LW'], ['mm']),
        helper.make_node('Add', ['mm', 'LB'], ['p']),
    ]
    inits += [
        numpy_helper.from_array(np.array([-1], np.int64), 'axes'),
        numpy_helper.from_array(np.asarray(off, np.float32), 'OFF'),
        numpy_helper.from_array(np.asarray(w, np.float32), 'LW'),
        numpy_helper.from_array(np.asarray(bias, np.float32), 'LB'),
    ]
    n2, i2 = _gemm_nodes('g2', 'p', W2, b2, 'Y')
    nodes += n2; inits += i2
    o = W2.shape[1]
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, n_in])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, o])
    m = helper.make_model(helper.make_graph(nodes, 'pwl_s', [X], [Y], inits),
                          opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    onnx.save(m, path)


def _floor_sandwich_onnx(path, W1, b1, W2, b2):
    n_in, h = W1.shape
    nodes, inits = [], []
    n1, i1 = _gemm_nodes('g1', 'X', W1, b1, 'h_pre')
    nodes += n1 + [helper.make_node('Relu', ['h_pre'], ['h'])]; inits += i1
    nodes += [helper.make_node('Floor', ['h'], ['p'])]
    n2, i2 = _gemm_nodes('g2', 'p', W2, b2, 'Y')
    nodes += n2; inits += i2
    o = W2.shape[1]
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, n_in])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, o])
    m = helper.make_model(helper.make_graph(nodes, 'floor_s', [X], [Y], inits),
                          opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    onnx.save(m, path)


def _grid_min(gg, lo, hi, w_q, b_q, n=160):
    """Brute-force min of (w_q . Y + b_q) on a dense 2-D grid of the input box."""
    ax = [torch.linspace(float(lo[i]), float(hi[i]), n, dtype=torch.float64)
          for i in range(2)]
    gx, gy = torch.meshgrid(ax[0], ax[1], indexing='ij')
    X = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    Y = _forward_batch_graph(X, gg).double()
    wt = torch.tensor(w_q, dtype=torch.float64)
    return float((Y @ wt + b_q).min())


def _alpha_lb(gg, lo, hi, w_q, b_q, s):
    dev, dt = torch.device('cpu'), torch.float64
    xl = torch.tensor(lo, device=dev, dtype=dt)
    xh = torch.tensor(hi, device=dev, dtype=dt)
    ob = {}
    sb, _zf = _forward_zonotope_graph(xl, xh, gg, dev, dt,
                                      settings=s, op_bounds=ob)
    bbr = {L: (l, h) for L, (l, h) in sb.items()}
    lb, _a, _b, _hist = run_alpha_crown_fixed_intermediate(
        gg, xl, xh, bbr, np.asarray(w_q), float(b_q), dev, dt,
        n_iters=40, lr=0.25)
    return float(lb)


def test_pwl_backward_crown_is_sound(tmp_path):
    rng = np.random.default_rng(0)
    W1 = rng.standard_normal((2, 3)); b1 = rng.standard_normal(3)
    off = np.array([-1.5, 0.0, 0.8, 2.0])
    w = np.array([1.0, -2.0, 0.5, 1.5])           # mixed-sign -> non-monotone
    bias = 0.3
    W2 = rng.standard_normal((3, 1)); b2 = rng.standard_normal(1)
    p = str(tmp_path / 'pwl_s.onnx')
    _pwl_sandwich_onnx(p, W1, b1, off, w, bias, W2, b2)
    g = ComputeGraph.from_onnx(p, dtype=np.float64)
    assert merge_relu_lookup_table(g) == 1
    s = default_settings(merge_relu_lookup_table=True, device='cpu')
    gg = g.gpu_graph(device='cpu', dtype=torch.float64)
    lo, hi = [-1.0, -1.0], [1.0, 1.0]
    w_q, b_q = [1.0], 0.0
    lb = _alpha_lb(gg, lo, hi, w_q, b_q, s)
    gmin = _grid_min(gg, lo, hi, w_q, b_q)
    # sound: LB never above the true min (grid min upper-bounds it).
    assert lb <= gmin + 1e-6, f"UNSOUND pwl backward: LB {lb} > grid min {gmin}"
    # non-vacuous: the bound is finite and within a sane band of the truth.
    assert np.isfinite(lb) and lb > gmin - 50.0


def test_floor_backward_crown_is_sound(tmp_path):
    rng = np.random.default_rng(1)
    # scale up so the Gemm output spans several integer cells (exercises the
    # staircase band, not a trivial within-cell exact case).
    W1 = rng.standard_normal((2, 3)) * 3.0; b1 = rng.standard_normal(3)
    W2 = rng.standard_normal((3, 1)); b2 = rng.standard_normal(1)
    p = str(tmp_path / 'floor_s.onnx')
    _floor_sandwich_onnx(p, W1, b1, W2, b2)
    g = ComputeGraph.from_onnx(p, dtype=np.float64)
    s = default_settings(device='cpu')
    gg = g.gpu_graph(device='cpu', dtype=torch.float64)
    lo, hi = [-1.0, -1.0], [1.0, 1.0]
    w_q, b_q = [1.0], 0.0
    lb = _alpha_lb(gg, lo, hi, w_q, b_q, s)
    gmin = _grid_min(gg, lo, hi, w_q, b_q)
    assert lb <= gmin + 1e-6, \
        f"UNSOUND floor backward: LB {lb} > grid min {gmin}"
    assert np.isfinite(lb) and lb > gmin - 50.0


# --------------------------------------------------------------------------- #
# mul_bilinear: no-slack soundness guard + sound McCormick both-vary backward.
# --------------------------------------------------------------------------- #
def test_assert_mul_bilinear_one_point_guard():
    """The guard raises iff BOTH operands vary; passes when one is a point."""
    a_lo = torch.tensor([-1.0, 0.0, 2.0]); a_hi = torch.tensor([1.0, 0.0, 3.0])
    b_lo = torch.tensor([0.5, 0.5, 0.5]); b_hi = torch.tensor([0.5, 0.5, 0.5])
    # b is a point (lo == hi) -> sound, no raise.
    _assert_mul_bilinear_one_point('ok', a_lo, a_hi, b_lo, b_hi)
    # a is a point -> sound, no raise.
    _assert_mul_bilinear_one_point('ok2', b_lo, b_hi, a_lo, a_hi)
    # both vary -> UNSOUND no-slack path -> must raise loudly.
    with pytest.raises(AssertionError, match="both operands vary"):
        _assert_mul_bilinear_one_point(
            'bad', a_lo, a_hi, torch.tensor([-1.0, -1.0, -1.0]),
            torch.tensor([1.0, 1.0, 1.0]))


def _mul_both_vary_onnx(path, W1, b1, W2, b2, W3, b3):
    """X[1,2] -> (Gemm1->Relu->a, Gemm2->Relu->b) -> Mul(a,b) -> Gemm3 -> Y.
    Both Mul operands are variable -> a genuine both-vary mul_bilinear."""
    nodes, inits = [], []
    n1, i1 = _gemm_nodes('g1', 'X', W1, b1, 'h1')
    n2, i2 = _gemm_nodes('g2', 'X', W2, b2, 'h2')
    nodes += n1 + [helper.make_node('Relu', ['h1'], ['a'])]
    nodes += n2 + [helper.make_node('Relu', ['h2'], ['b'])]
    nodes += [helper.make_node('Mul', ['a', 'b'], ['p'])]
    n3, i3 = _gemm_nodes('g3', 'p', W3, b3, 'Y')
    nodes += n3; inits += i1 + i2 + i3
    o = W3.shape[1]
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, o])
    m = helper.make_model(helper.make_graph(nodes, 'mbv', [X], [Y], inits),
                          opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    onnx.save(m, path)


def test_mul_bilinear_both_vary_mccormick_sound(tmp_path):
    """Both-vary mul_bilinear routes to the McCormick envelope; the backward-
    alpha-CROWN lower bound must be SOUND (<= true grid min)."""
    rng = np.random.default_rng(3)
    W1 = rng.standard_normal((2, 3)); b1 = rng.standard_normal(3)
    W2 = rng.standard_normal((2, 3)); b2 = rng.standard_normal(3)
    W3 = rng.standard_normal((3, 1)); b3 = rng.standard_normal(1)
    p = str(tmp_path / 'mbv.onnx')
    _mul_both_vary_onnx(p, W1, b1, W2, b2, W3, b3)
    g = ComputeGraph.from_onnx(p, dtype=np.float64)
    s = default_settings(device='cpu')
    gg = g.gpu_graph(device='cpu', dtype=torch.float64)
    # confirm the graph actually has a both-vary mul_bilinear op.
    assert any(op['type'] == 'mul_bilinear' for op in gg['ops']), \
        "expected a mul_bilinear op in the both-vary graph"
    lo, hi = [-1.0, -1.0], [1.0, 1.0]
    w_q, b_q = [1.0], 0.0
    lb = _alpha_lb(gg, lo, hi, w_q, b_q, s)
    gmin = _grid_min(gg, lo, hi, w_q, b_q)
    assert lb <= gmin + 1e-6, \
        f"UNSOUND mul_bilinear McCormick backward: LB {lb} > grid min {gmin}"
    assert np.isfinite(lb) and lb > gmin - 50.0


# --------------------------------------------------------------------------- #
# init_alpha warm-start for run_alpha_crown_batched.
# --------------------------------------------------------------------------- #
def _relu_net_onnx(path, Ws, bs):
    """X -> (Gemm -> Relu)* -> Gemm -> Y, a small multi-ReLU net."""
    nodes, inits = [], []
    cur = 'X'
    for i, (W, b) in enumerate(zip(Ws, bs)):
        n, ii = _gemm_nodes(f'g{i}', cur, W, b, f'h{i}')
        nodes += n; inits += ii
        if i < len(Ws) - 1:
            nodes += [helper.make_node('Relu', [f'h{i}'], [f'r{i}'])]
            cur = f'r{i}'
        else:
            nodes[-1].output[0] = 'Y'   # rename last Add output to Y
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, Ws[0].shape[0]])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, Ws[-1].shape[1]])
    m = helper.make_model(helper.make_graph(nodes, 'relunet', [X], [Y], inits),
                          opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    onnx.save(m, path)


def test_init_alpha_warmstart_applied(tmp_path):
    """init_alpha must actually load the provided alpha: warm-starting from a
    CONVERGED alpha for one iter reproduces the converged bound, and beats a
    one-iter COLD (min-area) start. Proves the param is wired and matters."""
    dev, dt = torch.device('cpu'), torch.float64
    rng = np.random.default_rng(7)
    Ws = [rng.standard_normal((3, 6)), rng.standard_normal((6, 6)),
          rng.standard_normal((6, 2))]
    bs = [rng.standard_normal(6), rng.standard_normal(6), rng.standard_normal(2)]
    p = str(tmp_path / 'relunet.onnx')
    _relu_net_onnx(p, Ws, bs)
    g = ComputeGraph.from_onnx(p, dtype=np.float64)
    s = default_settings(device='cpu')
    gg = g.gpu_graph(device='cpu', dtype=dt)
    xl = torch.full((3,), -1.0, device=dev, dtype=dt)
    xh = torch.full((3,), 1.0, device=dev, dtype=dt)
    ob = {}
    sb, _zf = _forward_zonotope_graph(xl, xh, gg, dev, dt,
                                      settings=s, op_bounds=ob)
    bbr = {L: (lo.cpu().numpy(), hi.cpu().numpy()) for L, (lo, hi) in sb.items()}
    sn = sorted(sb.keys())
    un = {L: list(np.where((lo.cpu().numpy() < 0) & (hi.cpu().numpy() > 0))[0])
          for L, (lo, hi) in sb.items()}
    # query: minimise output[0].
    w_qs = np.array([[1.0, 0.0]]); b_qs = np.array([0.0])
    common = dict(intermediate_start_nodes=sn, unstable_indices=un,
                  device=dev, dtype=dt, lr=0.1, sparse_alpha=False)
    # 1) converge cold.
    best_conv, ap, _bb, _h = run_alpha_crown_batched(
        gg, xl, xh, bbr, w_qs, b_qs, n_iters=60, **common)
    ap_np = {S: {L: a.detach().cpu().numpy() for L, a in d.items()}
             for S, d in ap.items()}
    # 2) cold, one iter (min-area init).
    best_cold1, *_ = run_alpha_crown_batched(
        gg, xl, xh, bbr, w_qs, b_qs, n_iters=1, **common)
    # 3) warm, one iter from the converged alpha.
    best_warm1, *_ = run_alpha_crown_batched(
        gg, xl, xh, bbr, w_qs, b_qs, n_iters=1, init_alpha=ap_np, **common)
    bc, bw, bk = best_cold1[0], best_warm1[0], best_conv[0]
    # warm-from-converged reproduces the converged bound up to the final-vs-best
    # alpha drift (run_alpha_crown_batched returns the final-iter alpha, which is
    # a hair past the best-iter alpha that achieved best_lbs; ~3e-4 here).
    assert bw >= bk - 5e-3, f"warm {bw} should ~reproduce converged {bk}"
    # and is strictly better than a cold one-iter start (init_alpha matters).
    assert bw > bc + 1e-6, f"warm {bw} should beat cold-1iter {bc}"


# --------------------------------------------------------------------------- #
# split_beta: beta-CROWN split-constraint injection in _crown_backward_matrix.
# --------------------------------------------------------------------------- #
def test_split_beta_is_sound(tmp_path):
    """beta-CROWN split constraint x_j <= 0 on a relu pre-activation: the
    Adam-optimized (alpha+beta) lower bound must be SOUND on the CONSTRAINED
    subdomain (LB <= true min over {x in box : pre-act_j <= 0}), for any beta>=0."""
    from vibecheck.alpha_crown import _crown_backward_matrix, _make_slopes
    dev, dt = torch.device('cpu'), torch.float64
    rng = np.random.default_rng(5)
    W1 = rng.standard_normal((2, 4)); b1 = rng.standard_normal(4)
    W2 = rng.standard_normal((4, 1)); b2 = rng.standard_normal(1)
    p = str(tmp_path / 'rn.onnx')
    _relu_net_onnx(p, [W1, W2], [b1, b2])
    g = ComputeGraph.from_onnx(p, dtype=np.float64)
    s = default_settings(device='cpu')
    gg = g.gpu_graph(device='cpu', dtype=dt)
    xl = torch.tensor([-1.0, -1.0], dtype=dt); xh = torch.tensor([1.0, 1.0], dtype=dt)
    ob = {}
    sb, _zf = _forward_zonotope_graph(xl, xh, gg, dev, dt, settings=s, op_bounds=ob)
    bbr = {L: (lo, hi) for L, (lo, hi) in sb.items()}
    last = gg['ops'][-1]['name']
    w_q = torch.tensor([[1.0]], dtype=dt); b_q = 0.0
    # pick an unstable relu neuron at the (only) relu layer.
    L = sorted(bbr.keys())[0]
    lo, hi = bbr[L]
    un = ((lo < 0) & (hi > 0)).nonzero().flatten()
    assert un.numel() > 0, "need an unstable relu for the test"
    j = int(un[0])
    # split: pre-act_j <= 0 (sign=+1, point=0).
    beta = torch.zeros(1, dtype=dt, requires_grad=True)
    sb_dict = {L: (torch.tensor([j]), torch.tensor([1.0], dtype=dt),
                   torch.tensor([0.0], dtype=dt), beta)}
    # spec alpha (relu lower slope), Adam over alpha+beta.
    lo_s, _, _, active, dead, unstable = _make_slopes(lo, hi)
    alpha = (active.to(dt) + unstable.to(dt) * lo_s).detach().requires_grad_(True)
    opt = torch.optim.Adam([{'params': [alpha], 'lr': 0.2},
                            {'params': [beta], 'lr': 1.0}])
    best = -1e9
    for _ in range(80):
        opt.zero_grad()
        lb, _ = _crown_backward_matrix(gg, xl, xh, {L: alpha}, bbr, last,
                                       w_q, dev, dt, split_beta=sb_dict)
        (-(lb + b_q).sum()).backward(); opt.step()
        with torch.no_grad():
            alpha.clamp_(0, 1); alpha[active] = 1.0; alpha[dead] = 0.0
            beta.clamp_(min=0)
        best = max(best, float((lb + b_q).min()))
    # brute-force min of Y over the CONSTRAINED subdomain {pre-act_j <= 0}.
    N = 60
    ax = torch.linspace(-1, 1, N, dtype=dt)
    gx, gy = torch.meshgrid(ax, ax, indexing='ij')
    X = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    preact = X @ torch.tensor(W1, dtype=dt) + torch.tensor(b1, dtype=dt)
    mask = preact[:, j] <= 0
    Y = _forward_batch_graph(X[mask], gg).double().reshape(-1)
    cmin = float(Y.min())
    assert best <= cmin + 1e-6, \
        f"UNSOUND split_beta: LB {best} > constrained min {cmin}"
    # and beta actually engaged (non-trivial) or stayed 0 — both fine, but the
    # bound must be finite.
    assert np.isfinite(best)


# --------------------------------------------------------------------------- #
# refine_intermediate_bounds_per_node: per-node α tightens intermediate bounds
# (vs forward-zono) and stays SOUND (bounds contain the true pre-activation range).
# --------------------------------------------------------------------------- #
def test_per_node_refine_tightens_and_sound(tmp_path):
    from vibecheck.alpha_crown import refine_intermediate_bounds_per_node
    dev, dt = torch.device('cpu'), torch.float64
    rng = np.random.default_rng(11)
    # 3-layer ReLU MLP so layer 2's bound depends on per-node α over layers 0,1.
    Ws = [rng.standard_normal((3, 6)), rng.standard_normal((6, 6)),
          rng.standard_normal((6, 6)), rng.standard_normal((6, 2))]
    bs = [rng.standard_normal(6), rng.standard_normal(6),
          rng.standard_normal(6), rng.standard_normal(2)]
    p = str(tmp_path / 'mlp.onnx')
    _relu_net_onnx(p, Ws, bs)
    g = ComputeGraph.from_onnx(p, dtype=np.float64)
    s = default_settings(device='cpu')
    gg = g.gpu_graph(device='cpu', dtype=dt)
    xl = torch.full((3,), -1.0, dtype=dt); xh = torch.full((3,), 1.0, dtype=dt)
    ob = {}
    sb, _zf = _forward_zonotope_graph(xl, xh, gg, dev, dt, settings=s, op_bounds=ob)
    fwd = {L: (lo.clone(), hi.clone()) for L, (lo, hi) in sb.items()}
    fwd_np = {L: (lo.cpu().numpy(), hi.cpu().numpy()) for L, (lo, hi) in fwd.items()}
    ref = refine_intermediate_bounds_per_node(gg, xl, xh, fwd_np, dev, dt,
                                              n_iters=40, lr=0.2)
    relu_in = {op['layer_idx']: op['inputs'][0] for op in gg['ops']
               if op['type'] == 'relu' and op.get('layer_idx') is not None}
    # brute-force true pre-activation range per relu layer.
    N = 40
    ax = torch.linspace(-1, 1, N, dtype=dt)
    gx, gy, gz = torch.meshgrid(ax, ax, ax, indexing='ij')
    X = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1)
    tightened = 0
    for L in sorted(relu_in):
        rlo, rhi = ref[L]
        flo, fhi = fwd[L]
        # (1) tighter-or-equal than forward-zono everywhere.
        assert (rlo >= flo - 1e-6).all() and (rhi <= fhi + 1e-6).all(), \
            f"layer {L}: refined not within forward-zono"
        if float((fhi - flo).sum() - (rhi - rlo).sum()) > 1e-3:
            tightened += 1
        # (2) SOUND: refined bounds contain the true pre-activation range.
        pre = _forward_batch_graph(X, gg, capture=relu_in[L]).double()
        tmin = pre.min(0).values; tmax = pre.max(0).values
        assert (rlo <= tmin + 1e-5).all(), f"layer {L}: refined lo > true min (UNSOUND)"
        assert (rhi >= tmax - 1e-5).all(), f"layer {L}: refined hi < true max (UNSOUND)"
    # at least one deeper layer actually tightened (per-node α did something).
    assert tightened >= 1, "per-node refinement tightened nothing"
