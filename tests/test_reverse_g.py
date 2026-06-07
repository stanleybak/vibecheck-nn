"""Regression tests for reverse-mode α-zono state build (`reverse_g.py`).

`build_state_reverse` reconstructs the same per-query α-zono Phase-8 state as
the forward path (`forward_zono_dir_adaptive` + `state_from_alpha_zono`) by
back-propagating from the unstable + output neurons instead of forward-
propagating every generator column. Conv-net speed win.

Two failure modes are pinned here:

1. **Formulation-tag dispatch (the 2026-06-07 cifar100 resnet_large false-verify).**
   `build_gen_lp_from_state` dispatches the LP/MILP builder on the top-level
   `state['formulation']` string. The reverse state's per-neuron data is in the
   α-zono parametrization (`y = λz + μ(1+e_new)`), but it was tagged
   `formulation='sparse'`, which routes to the GENERIC `y=e_new∈[0,hi]`
   direct-ReLU builder — a coordinate-system mismatch that produces an UNSOUND
   (too-tight) polytope. On a SAT case the MILP min-margin came back positive
   (+0.235 vs the correct −0.146), the high-bin MILP fallback declared the
   query CLOSED, and we false-verified. The dual-ascent BnB reads the state
   fields directly (ignores `formulation`), so it stayed sound — which is why
   the bug only surfaced through the MILP fallback. Test: the reverse state is
   tagged `alpha_zono` AND its gen-LP/MILP optimum equals the forward state's.

2. **Backward math per layer type.** The α-relaxed network is affine in the
   generator coordinates `e`, so `torch.autograd.grad` gives the exact ∂z/∂e.
   We check reverse-g's stored rows against this ground truth for fc, conv,
   strided conv, plain+layer ReLU, and a projection-skip residual block
   (resnet downsampling — the structure that exposed the original conv
   `output_padding` work).
"""
import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import onnx
from onnx import helper, TensorProto, numpy_helper
import gurobipy as grb

from vibecheck.network import ComputeGraph
from vibecheck.settings import default_settings
from vibecheck import verify_gen_lp
from vibecheck import alpha_crown as ac
from vibecheck.verify_graph import _forward_keep_pre_gpu
from vibecheck.verify_zono_bnb import _forward_zonotope_graph
from vibecheck.reverse_g import build_state_reverse
from vibecheck.gurobi_util import optimize_checked

DEV = torch.device('cpu')
DT = torch.float64


# ---------------------------------------------------------------------------
# ONNX builders
# ---------------------------------------------------------------------------

def _save(nodes, inits, in_shape, out_shape, name):
    inp = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [1] + list(in_shape))
    out = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [1] + list(out_shape))
    g = helper.make_graph(nodes, name, [inp], [out], initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 7
    fd, p = tempfile.mkstemp(suffix='.onnx')
    os.close(fd)
    onnx.save(m, p)
    return p


def _fc_relu_fc_relu_fc(seed=11):
    """2 → FC(6) → ReLU → FC(6) → ReLU → FC(3). Two unstable ReLU layers."""
    rng = np.random.RandomState(seed)
    W1 = (rng.randn(6, 2) * 0.8).astype(np.float32)
    b1 = (rng.randn(6) * 0.1).astype(np.float32)
    W2 = (rng.randn(6, 6) * 0.8).astype(np.float32)
    b2 = (rng.randn(6) * 0.1).astype(np.float32)
    W3 = (rng.randn(3, 6) * 0.8).astype(np.float32)
    b3 = (rng.randn(3) * 0.1).astype(np.float32)
    nodes = [
        helper.make_node('Gemm', ['input', 'W1', 'b1'], ['fc1'], transB=1),
        helper.make_node('Relu', ['fc1'], ['r1']),
        helper.make_node('Gemm', ['r1', 'W2', 'b2'], ['fc2'], transB=1),
        helper.make_node('Relu', ['fc2'], ['r2']),
        helper.make_node('Gemm', ['r2', 'W3', 'b3'], ['output'], transB=1),
    ]
    inits = [numpy_helper.from_array(a, n) for n, a in
             [('W1', W1), ('b1', b1), ('W2', W2),
              ('b2', b2), ('W3', W3), ('b3', b3)]]
    return _save(nodes, inits, [2], [3], 'fc')


def _proj_skip_residual(seed=0):
    """Strided projection-skip residual block (resnet downsampling) → Gemm.

    Conv(stride2)→ReLU→Conv ⊕ Conv(stride2, 1x1 skip) →ReLU→Flatten→Gemm.
    Exercises conv `output_padding`, the is_merge Add, and two ReLU layers.
    """
    rng = np.random.default_rng(seed)
    C, H = 4, 8

    def cw(co, ci, k, nm):
        return numpy_helper.from_array(
            (rng.standard_normal((co, ci, k, k)) * 0.2).astype(np.float32), nm)

    w_a = cw(8, C, 3, 'wa')
    w_b = cw(8, 8, 3, 'wb')
    w_s = cw(8, C, 1, 'ws')
    wg = numpy_helper.from_array(
        (rng.standard_normal((3, 128)) * 0.1).astype(np.float32), 'wg')
    nodes = [
        helper.make_node('Conv', ['input', 'wa'], ['za'],
                         strides=[2, 2], pads=[1, 1, 1, 1]),
        helper.make_node('Relu', ['za'], ['ya']),
        helper.make_node('Conv', ['ya', 'wb'], ['zmain'],
                         strides=[1, 1], pads=[1, 1, 1, 1]),
        helper.make_node('Conv', ['input', 'ws'], ['zskip'],
                         strides=[2, 2], pads=[0, 0, 0, 0]),
        helper.make_node('Add', ['zmain', 'zskip'], ['zadd']),
        helper.make_node('Relu', ['zadd'], ['yadd']),
        helper.make_node('Flatten', ['yadd'], ['f']),
        helper.make_node('Gemm', ['f', 'wg'], ['output'], transB=1),
    ]
    return _save(nodes, [w_a, w_b, w_s, wg], [C, H, H], [3], 'projskip')


# ---------------------------------------------------------------------------
# Shared state-build scaffolding
# ---------------------------------------------------------------------------

def _bbr_from_forward(xl, xh, gg):
    _, pre = _forward_keep_pre_gpu(xl, xh, gg, DEV, DT)
    bbr = {}
    for L, (c, G) in pre.items():
        r = G.abs().sum(dim=1)
        bbr[L] = ((c - r).cpu().numpy().astype(np.float64),
                  (c + r).cpu().numpy().astype(np.float64))
    return bbr


def _build_both_states(onnx_path, in_shape, qw, qb, eps=0.25):
    """Forward (state_from_alpha_zono) + reverse (build_state_reverse) state
    for the SAME query, alpha, and bounds. Returns (fwd_state, rev_state, gg)."""
    settings = default_settings(device='cpu', bits=64, print_progress=False)
    graph = ComputeGraph.from_onnx(onnx_path)
    graph.optimize(settings)
    from vibecheck.verify_graph import _serialize_gg_ops, _conv_sparse_matrix
    gg = graph.gpu_graph(DEV, DT)
    gg_ops_ser = _serialize_gg_ops(gg)
    for d in gg_ops_ser:
        if d['type'] == 'conv' and 'W_sp' not in d:
            d['W_sp'] = _conv_sparse_matrix(
                d['kernel_np'], d['in_shape'], d['stride'], d['padding'])

    n_in = int(np.prod(in_shape))
    rng = np.random.default_rng(3)
    c0 = rng.random(n_in)
    x_lo = (c0 - eps).astype(np.float64)
    x_hi = (c0 + eps).astype(np.float64)
    xl = torch.tensor(x_lo, device=DEV, dtype=DT)
    xh = torch.tensor(x_hi, device=DEV, dtype=DT)

    bbr = _bbr_from_forward(xl, xh, gg)
    _, alpha_params, _, _ = ac.run_alpha_crown_fixed_intermediate(
        gg, xl, xh, bbr, qw, float(qb), DEV, DT,
        n_iters=8, lr=0.25, lr_decay=0.98, early_stop_on_positive=False)
    _, ew_at_relu = ac.capture_ew_per_relu(
        gg, xl, xh, alpha_params['spec'], bbr, qw, float(qb), DEV, DT)
    alpha_per_layer = ac.build_dir_adaptive_alpha(
        alpha_params['spec'], ew_at_relu, bbr, DEV, DT)

    unstable_per_layer = {}
    for L, (lo, hi) in bbr.items():
        un = np.where((np.asarray(lo) < 0) & (np.asarray(hi) > 0))[0]
        unstable_per_layer[L] = torch.as_tensor(un, dtype=torch.long, device=DEV)

    z_alpha, pre_relu_gpu = ac.forward_zono_dir_adaptive(
        xl, xh, gg, alpha_per_layer, bbr, DEV, DT,
        settings=settings, unstable_per_layer=unstable_per_layer)
    fwd_state = verify_gen_lp.state_from_alpha_zono(
        z_alpha, pre_relu_gpu, alpha_per_layer, bbr, x_lo, x_hi,
        gg_ops_ser, gg['input_name'], gg_ops_ser[-1]['name'],
        unstable_per_layer=unstable_per_layer)

    rev_state = build_state_reverse(
        gg, xl, xh, bbr, alpha_per_layer, DEV, DT)
    ctx = dict(xl=xl, xh=xh, bbr=bbr, alpha_per_layer=alpha_per_layer)
    return fwd_state, rev_state, gg, ctx


def _solve(state, qw, qb, milp_set):
    m, env, _, _ = verify_gen_lp.build_gen_lp_from_state(
        state, qw, qb, milp_set=milp_set, n_threads=1,
        unsafe_halfspace='none')
    m.setParam('TimeLimit', 30.0)
    optimize_checked(m)
    lb = float(m.ObjBound) if m.Status in (
        grb.GRB.OPTIMAL, grb.GRB.USER_OBJ_LIMIT) else None
    m.dispose()
    env.dispose()
    return lb


# ---------------------------------------------------------------------------
# 1. Formulation-tag dispatch + MILP-optimum equivalence (the false-verify)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('builder,in_shape,binarise_all', [
    (_fc_relu_fc_relu_fc, [2], True),     # small: also exact all-binarised MILP
    (_proj_skip_residual, [4, 8, 8], False),  # conv: LP-optimum only (256 bins → slow)
])
def test_reverse_state_alpha_zono_dispatch_and_optimum(
        builder, in_shape, binarise_all):
    qw = np.array([1.0, -1.0, 0.0], dtype=np.float64)
    qb = 0.05
    onnx_path = builder()
    try:
        fwd_state, rev_state, _, _ = _build_both_states(onnx_path, in_shape, qw, qb)

        # (1) The reverse state MUST be tagged alpha_zono so build_gen_lp_from_state
        # routes it to _build_alpha_zono_lp, not the generic direct-ReLU builder.
        assert rev_state['formulation'] == 'alpha_zono'
        assert fwd_state['formulation'] == 'alpha_zono'

        # (2) Same LP relaxation optimum (milp_set=∅). This ALONE catches the
        # formulation mismatch: the generic 'sparse' builder adds the triangle
        # floor (y≥0, y≥z) to every unstable neuron, while _build_alpha_zono_lp
        # leaves them as the bare parallelogram — so the misrouted state's LP
        # optimum differs from the (correct) forward state's.
        lp_fwd = _solve(fwd_state, qw, qb, milp_set=set())
        lp_rev = _solve(rev_state, qw, qb, milp_set=set())
        assert lp_fwd is not None and lp_rev is not None
        assert lp_rev == pytest.approx(lp_fwd, abs=1e-5)

        # (3) Same MILP optimum with EVERY unstable neuron binarised — the
        # tightest sound bound, and exactly what the high-bin fallback computes
        # (the step that false-verified). The 'sparse'-misrouted reverse state
        # gave a too-tight (unsound) value here.
        if binarise_all:
            all_unstable = {(u['layer_idx'], u['neuron_idx'])
                            for u in rev_state['unstable_list']}
            milp_fwd = _solve(fwd_state, qw, qb, milp_set=all_unstable)
            milp_rev = _solve(rev_state, qw, qb, milp_set=all_unstable)
            assert milp_fwd is not None and milp_rev is not None
            assert milp_rev == pytest.approx(milp_fwd, abs=1e-4)
    finally:
        os.remove(onnx_path)


def test_reverse_batched_alpha_zono_dispatch():
    """The BATCHED reverse build (reverse_batched.build_states_reverse_batched,
    the phase8_reverse_g_batched path) had the identical formulation='sparse'
    bug. With D=1 and the same alpha, state[0] must be tagged 'alpha_zono' and
    match the single-state build_state_reverse's gen-LP optimum."""
    from vibecheck.reverse_batched import build_states_reverse_batched
    qw = np.array([1.0, -1.0, 0.0], dtype=np.float64)
    qb = 0.05
    onnx_path = _proj_skip_residual()
    try:
        _, rev_state, gg, ctx = _build_both_states(onnx_path, [4, 8, 8], qw, qb)
        states = build_states_reverse_batched(
            gg, ctx['xl'], ctx['xh'], ctx['bbr'],
            [ctx['alpha_per_layer']], DEV, DT)
        assert len(states) == 1
        assert states[0]['formulation'] == 'alpha_zono'
        lp_batched = _solve(states[0], qw, qb, milp_set=set())
        lp_single = _solve(rev_state, qw, qb, milp_set=set())
        assert lp_batched is not None and lp_single is not None
        assert lp_batched == pytest.approx(lp_single, abs=1e-5)
    finally:
        os.remove(onnx_path)


# ---------------------------------------------------------------------------
# 2. Per-layer-type backward correctness vs autograd ground truth
# ---------------------------------------------------------------------------

def _autograd_check(onnx_path, in_shape, eps=0.3):
    """Compare reverse-g's stored unstable rows to autograd ∂z/∂e on the
    α-relaxed (affine-in-e) forward. Returns max abs row error."""
    graph = ComputeGraph.from_onnx(onnx_path, dtype=np.float32)
    gg = graph.gpu_graph(DEV, DT)
    ops = gg['ops']
    in_name = gg['input_name']
    n_in = int(np.prod(in_shape))

    rng = np.random.default_rng(0)
    c0 = rng.random(n_in)
    xl = torch.tensor(c0 - eps, device=DEV, dtype=DT)
    xh = torch.tensor(c0 + eps, device=DEV, dtype=DT)
    sb, _ = _forward_zonotope_graph(xl, xh, gg, DEV, DT)
    bbr = {L: (lo.cpu().numpy().astype(np.float64),
               hi.cpu().numpy().astype(np.float64))
           for L, (lo, hi) in sb.items()}

    relu_ops = {op['layer_idx']: op for op in ops
                if op['type'] == 'relu' and 'layer_idx' in op}
    Ls = sorted(relu_ops)

    def relax(lo, hi, a):
        active = lo >= 0
        dead = hi <= 0
        ust = (~active) & (~dead)
        lam = np.zeros_like(lo)
        lam[active] = 1.0
        lam[ust] = a[ust]
        mu = np.zeros_like(lo)
        mu[ust] = np.maximum((1 - a[ust]) * hi[ust] / 2, -a[ust] * lo[ust] / 2)
        return lam, mu, ust

    alpha = {L: 0.5 * np.ones_like(bbr[L][0]) for L in Ls}
    lamL = {}
    muL = {}
    ustL = {}
    for L in Ls:
        lamL[L], muL[L], u = relax(bbr[L][0], bbr[L][1], alpha[L])
        ustL[L] = np.where(u)[0]

    st = build_state_reverse(
        gg, xl, xh, bbr,
        {L: torch.tensor(alpha[L], device=DEV, dtype=DT) for L in Ls}, DEV, DT)
    n_gens = st['n_gens']
    n_input = st['n_input']
    e_new_index = {}
    col = n_input
    for L in Ls:
        for j in ustL[L]:
            e_new_index[(L, int(j))] = col
            col += 1

    radii = torch.tensor(c0 * 0 + eps, device=DEV, dtype=DT)
    input_e = torch.zeros(n_input, device=DEV, dtype=DT, requires_grad=True)
    enew = torch.zeros(max(col - n_input, 1), device=DEV, dtype=DT,
                       requires_grad=True)
    center = torch.tensor(c0, device=DEV, dtype=DT)

    def _t(x):
        return (x.detach().to(DT) if torch.is_tensor(x)
                else torch.as_tensor(np.asarray(x), dtype=DT))

    def fwd():
        vals = {in_name: center + radii * input_e}
        pre = {}
        for op in ops:
            t = op['type']
            nm = op['name']
            inp = op['inputs'][0]
            if t == 'conv':
                x = vals[inp].reshape((1,) + tuple(op['in_shape']))
                k = _t(op['kernel'])
                b = _t(op['bias']) if op['bias'] is not None else None
                vals[nm] = F.conv2d(x, k, bias=b, stride=op['stride'],
                                    padding=op['padding'])[0].reshape(-1)
            elif t == 'fc':
                W = _t(op['W'])
                y = vals[inp] @ W.t()
                b = op['bias']
                vals[nm] = y + _t(b) if b is not None else y
            elif t == 'relu':
                z = vals[inp]
                if 'layer_idx' in op:
                    L = op['layer_idx']
                    pre[L] = z
                    lam = _t(lamL[L])
                    mu = _t(muL[L])
                    y = lam * z + mu
                    for j in ustL[L]:
                        y = y.clone()
                        col_j = e_new_index[(L, int(j))] - n_input
                        y[j] = lam[j] * z[j] + mu[j] * (1.0 + enew[col_j])
                    vals[nm] = y
                else:
                    vals[nm] = torch.relu(z)
            elif t == 'reshape':
                vals[nm] = vals[inp]
            elif t == 'add':
                if op.get('is_merge'):
                    vals[nm] = vals[inp] + vals[op['inputs'][1]]
                else:
                    b = op.get('bias')
                    vals[nm] = vals[inp] + (
                        _t(np.asarray(b).flatten()) if b is not None else 0.0)
            elif t == 'sub':
                b = op.get('bias')
                vals[nm] = vals[inp] - (
                    _t(np.asarray(b).flatten()) if b is not None else 0.0)
            else:
                raise NotImplementedError(t)
        return pre, vals[ops[-1]['name']]

    pre, _ = fwd()
    revrow = {(u['layer_idx'], u['neuron_idx']):
              (np.asarray(u['row_indices']), np.asarray(u['row_values']))
              for u in st['unstable_list']}
    worst = 0.0
    for L in Ls:
        for j in ustL[L][:4]:
            gi, gn = torch.autograd.grad(
                pre[L][int(j)], [input_e, enew],
                retain_graph=True, allow_unused=True)
            gt = np.zeros(n_gens)
            gt[:n_input] = gi.cpu().numpy()
            if gn is not None:
                for (_LL, _jj), cc in e_new_index.items():
                    gt[cc] = gn.cpu().numpy()[cc - n_input]
            ri, rv = revrow[(L, int(j))]
            rev = np.zeros(n_gens)
            rev[ri] = rv
            worst = max(worst, float(np.abs(gt - rev).max()))
    return worst


@pytest.mark.parametrize('builder,in_shape', [
    (_fc_relu_fc_relu_fc, [2]),
    (_proj_skip_residual, [4, 8, 8]),
])
def test_reverse_backward_matches_autograd(builder, in_shape):
    onnx_path = builder()
    try:
        worst = _autograd_check(onnx_path, in_shape)
        assert worst < 1e-9, f'reverse-g row diverges from autograd: {worst:.2e}'
    finally:
        os.remove(onnx_path)
