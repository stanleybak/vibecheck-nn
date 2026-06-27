"""Differentiable backward CROWN for attention graphs (vit_2023).

A compact, autograd-capable re-implementation of the spec backward walk
for the op set the vit gg emits (fc / conv / relu / add / sub / mul /
reshape / slice / reduce_sum / exp / reciprocal / mul_bilinear /
matmul_bilinear). The relaxation planes are parametrized exactly like
alpha,beta-CROWN's alpha:

  - ('relu', L):  lam in [0,1]^n   — lower-plane slope for ew+ entries
  - ('cv', name): s in [0,1]^n     — tangent point m = l + s*(u-l) for the
                                     convex exp / reciprocal lower plane
  - ('mc', name): r in [0,1]^(2,*) — McCormick interpolation, r[0] for the
                                     lower planes (ew+), r[1] for the upper
                                     (ew-); r=1 is the corner the fixed
                                     handlers in verify_zono_bnb use
  - ('beta', name): beta >= 0^(2,n) — Lagrangian multipliers for a
                                     bilinear-input value-split l <= z <= u
                                     on the exp/reciprocal INPUT z (the
                                     BnB's `op_clamps`); row 0 = lo side,
                                     row 1 = hi side. beta-CROWN's beta for
                                     general (non-relu) splits: for any
                                     beta >= 0,
  - ('rbeta', L):  beta >= 0^(2,n)  — same Lagrangian for RELU splits
                                     (`relu_clamps`): clamping a split
                                     neuron's bounds makes its relaxation
                                     exact on the subdomain but keeps
                                     spurious traces from inputs OUTSIDE
                                     it (z>0 routed through y=0); the
                                     beta*z term is what actually enforces
                                     the halfspace — beta-CROWN's core
                                     mechanism, measured 30-100x node
                                     collapse vs clamp-only splits.
                                       min_full [f + b_hi(z-u) - b_lo(z-l)]
                                       <= min_subdomain f,
                                     because both added terms are <= 0 on
                                     the subdomain. Without beta a value
                                     split only tightens the local planes;
                                     the concretization still ranges over
                                     the FULL input box, so deep splits
                                     stagnate at the linearized-network
                                     bound (measured on the relu-free
                                     softmax toy). Only applied when the
                                     caller passes `op_clamps`.

Every parameter value in [0,1] (beta in [0,inf)) yields a SOUND lower bound
(each plane is a valid relaxation for any interpolation), so optimizing
them with Adam and taking the best-of over iterations is sound — the
alpha-CROWN argument.

`attn_crown_lb` evaluates the bound (differentiable); `attn_crown_alpha`
runs the optimization for one query and returns (best_lb, params).
"""
import os
import time
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F


def _apply_beta(acc, ew_in, cl, ch, bt, dtype):
    """Inject the value-split Lagrangian  +b_hi*(z-u) - b_lo*(z-l)
    (b >= 0) into the backward functional at the split node's input.
    cl/ch are ±inf where unclamped; those coordinates contribute 0."""
    cl = cl.to(dtype); ch = ch.to(dtype)
    m_lo = torch.isfinite(cl)
    m_hi = torch.isfinite(ch)
    b_ = bt.clamp(min=0)
    b_lo_ = b_[0] * m_lo.to(dtype)
    b_hi_ = b_[1] * m_hi.to(dtype)
    ew_in = ew_in + (b_hi_ - b_lo_)
    acc = acc + (b_lo_ * torch.where(m_lo, cl, torch.zeros_like(cl))
                 - b_hi_ * torch.where(m_hi, ch,
                                       torch.zeros_like(ch))).sum()
    return acc, ew_in


def _mc_planes(xl_t, xh_t, yl_t, yh_t, r_lo, r_up):
    """Interpolated McCormick planes (auto_LiRPA MulHelper formulas).

    Returns (al, bl, gl, au, bu, gu) with broadcastable shapes:
      lower: x*y >= al*x + bl*y + gl   (for ew+ coefficients)
      upper: x*y <= au*x + bu*y + gu   (for ew- coefficients)
    """
    al = r_lo * yl_t + (1 - r_lo) * yh_t
    bl = r_lo * xl_t + (1 - r_lo) * xh_t
    gl = r_lo * (yh_t * xh_t - yl_t * xl_t) - yh_t * xh_t
    au = r_up * yh_t + (1 - r_up) * yl_t
    bu = r_up * xl_t + (1 - r_up) * xh_t
    gu = r_up * (yl_t * xh_t - yh_t * xl_t) - yl_t * xh_t
    return al, bl, gl, au, bu, gu


def attn_crown_lb(gg, xl, xh, sb, op_bounds, w_q, b_q, params,
                  tight_bounds=None, op_clamps=None, relu_clamps=None):
    """Differentiable lower bound of w_q . y + b_q via backward CROWN.

    sb: {layer_idx: (lo, hi)} pre-relu bound TENSORS (from the plain
        forward — possibly clamped per BnB node).
    op_bounds: {op_name: bounds} recorded by the same forward.
    params: the plane-parameter dict described in the module docstring
        (missing keys fall back to the fixed-corner defaults).
    op_clamps: {op_name: (lo_t, hi_t)} value-split intervals on
        exp/reciprocal INPUTS (±inf where unclamped) — enables the
        ('beta', name) Lagrangian terms. The bound is sound with or
        without them for any beta >= 0; betas without clamps are
        ignored (they would be unsound).
    relu_clamps: {layer_idx: (lo_t, hi_t)} the same for RELU splits
        (finite 0.0 at split neurons) — enables ('rbeta', L).
    """
    ops = gg['ops']
    device = xl.device
    dtype = xl.dtype
    w_t = w_q if torch.is_tensor(w_q) else torch.as_tensor(
        np.asarray(w_q, np.float64), device=device, dtype=dtype)
    w_t = w_t.to(dtype)
    ew_at = {ops[-1]['name']: w_t.clone()}
    acc = torch.zeros((), device=device, dtype=dtype) + float(b_q)

    def _push(inp, ew_b):
        ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_b)) + ew_b

    for op in reversed(ops):
        name = op['name']
        if name not in ew_at:
            continue
        ew = ew_at[name]
        t = op['type']

        if t == 'fc':
            acc = acc + ew @ op['bias'].to(dtype)
            _push(op['inputs'][0], ew @ op['W'].to(dtype))
        elif t == 'conv':
            C_out = op['out_shape'][0]
            acc = acc + (ew.reshape(1, *op['out_shape'])
                         .reshape(C_out, -1).sum(dim=1)
                         @ op['bias'].to(dtype))
            ew_b = F.conv_transpose2d(
                ew.reshape(1, *op['out_shape']), op['kernel'].to(dtype),
                stride=op['stride'], padding=op['padding'],
                output_padding=op['output_padding']).flatten()
            _push(op['inputs'][0], ew_b)
        elif t == 'relu':
            L = op.get('layer_idx')
            lo_t, hi_t = sb[L]
            lo_t = lo_t.to(dtype); hi_t = hi_t.to(dtype)
            if tight_bounds is not None and L in tight_bounds:
                tl, th = tight_bounds[L]
                lo_t = torch.maximum(lo_t, torch.as_tensor(
                    tl, dtype=dtype, device=device))
                hi_t = torch.minimum(hi_t, torch.as_tensor(
                    th, dtype=dtype, device=device))
            ub_r = hi_t.clamp(min=0)
            lb_r = lo_t.clamp(max=0)
            ub_r = torch.maximum(ub_r, lb_r + 1e-12)
            up_s = ub_r / (ub_r - lb_r)
            up_t = -lb_r * up_s
            active = (lo_t >= 0).to(dtype)
            dead = (hi_t <= 0).to(dtype)
            unstable = (1 - active) * (1 - dead)
            lam = params.get(('relu', L))
            lam = (lam.clamp(0, 1) if lam is not None
                   else (up_s > 0.5).to(dtype))
            lo_slope = active + unstable * lam
            up_slope = active + unstable * up_s
            up_off = unstable * up_t
            ep = ew.clamp(min=0); en = ew.clamp(max=0)
            acc = acc + (en * up_off).sum()
            ew_in = ep * lo_slope + en * up_slope
            if relu_clamps is not None and L in relu_clamps:
                bt = params.get(('rbeta', L))
                if bt is not None:
                    cl, ch = relu_clamps[L]
                    acc, ew_in = _apply_beta(acc, ew_in, cl, ch, bt, dtype)
            _push(op['inputs'][0], ew_in)
        elif t in ('exp', 'reciprocal'):
            l_in, u_in = op_bounds[name]
            l_in = l_in.to(dtype); u_in = u_in.to(dtype)
            w_in = (u_in - l_in).clamp(min=1e-12)
            s = params.get(('cv', name))
            s = s.clamp(0, 1) if s is not None else 0.5
            m = l_in + s * w_in
            if t == 'exp':
                f_l = torch.exp(l_in); f_u = torch.exp(u_in)
                k_up = (f_u - f_l) / w_in
                b_up = f_l - k_up * l_in
                k_lo = torch.exp(m)
                b_lo = k_lo * (1 - m)
            else:
                k_up = -1.0 / (l_in * u_in)
                b_up = 1.0 / l_in + 1.0 / u_in
                k_lo = -1.0 / (m * m)
                b_lo = 2.0 / m
            ep = ew.clamp(min=0); en = ew.clamp(max=0)
            acc = acc + (ep * b_lo + en * b_up).sum()
            ew_in = ep * k_lo + en * k_up
            if op_clamps is not None and name in op_clamps:
                bt = params.get(('beta', name))
                if bt is not None:
                    cl, ch = op_clamps[name]
                    acc, ew_in = _apply_beta(acc, ew_in, cl, ch, bt, dtype)
            _push(op['inputs'][0], ew_in)
        elif t == 'matmul_bilinear':
            (xlb, xhb), (ylb, yhb) = op_bounds[name]
            sa = op['in_shapes_nd'][0]
            sb_ = op['in_shapes_nd'][1]
            so = op['out_shape_nd']
            Xl = xlb.to(dtype).reshape(sa).unsqueeze(-1)
            Xh = xhb.to(dtype).reshape(sa).unsqueeze(-1)
            Yl = ylb.to(dtype).reshape(sb_).unsqueeze(-3)
            Yh = yhb.to(dtype).reshape(sb_).unsqueeze(-3)
            r = params.get(('mc', name))
            if r is None:
                r_lo = r_up = torch.ones((), device=device, dtype=dtype)
            else:
                r = r.clamp(0, 1)
                r_lo, r_up = r[0], r[1]
            al, bl, gl, au, bu, gu = _mc_planes(Xl, Xh, Yl, Yh, r_lo, r_up)
            ew_nd = ew.reshape(so)
            ep = ew_nd.clamp(min=0).unsqueeze(-2)
            en = ew_nd.clamp(max=0).unsqueeze(-2)
            acc = acc + (ep * gl + en * gu).sum()
            ew_x = (ep * al + en * au).sum(dim=-1).reshape(-1)
            ew_y = (ep * bl + en * bu).sum(dim=-3).reshape(-1)
            _push(op['inputs'][0], ew_x)
            _push(op['inputs'][1], ew_y)
        elif t == 'mul_bilinear':
            from .alpha_crown import _sum_to_shape
            (xlb, xhb), (ylb, yhb) = op_bounds[name]
            sa = op['in_shapes_nd'][0]
            sb_ = op['in_shapes_nd'][1]
            so = op['out_shape_nd']
            ones_o = torch.ones(so, dtype=dtype, device=device)
            Xl = xlb.to(dtype).reshape(sa) * ones_o
            Xh = xhb.to(dtype).reshape(sa) * ones_o
            Yl = ylb.to(dtype).reshape(sb_) * ones_o
            Yh = yhb.to(dtype).reshape(sb_) * ones_o
            r = params.get(('mc', name))
            if r is None:
                r_lo = r_up = torch.ones((), device=device, dtype=dtype)
            else:
                r = r.clamp(0, 1)
                r_lo, r_up = r[0], r[1]
            al, bl, gl, au, bu, gu = _mc_planes(Xl, Xh, Yl, Yh, r_lo, r_up)
            ew_nd = ew.reshape(so)
            ep = ew_nd.clamp(min=0); en = ew_nd.clamp(max=0)
            acc = acc + (ep * gl + en * gu).sum()
            cx = ep * al + en * au
            cy = ep * bl + en * bu
            _push(op['inputs'][0], _sum_to_shape(cx, (), sa).reshape(-1))
            _push(op['inputs'][1], _sum_to_shape(cy, (), sb_).reshape(-1))
        elif t == 'add':
            if op.get('is_merge'):
                for inp in op['inputs']:
                    _push(inp, ew)
            else:
                bias = op.get('bias')
                if bias is not None:
                    bt = torch.as_tensor(
                        np.asarray(bias, np.float64).ravel(),
                        dtype=dtype, device=device)
                    assert bt.numel() == ew.numel(), (
                        f'add {name!r}: bias size {bt.numel()} != ew '
                        f'{ew.numel()} (emission broadcasts biases)')
                    acc = acc + (ew * bt).sum()
                _push(op['inputs'][0], ew)
        elif t == 'sub':
            bias = op.get('bias')
            if bias is not None:
                bt = torch.as_tensor(
                    np.asarray(bias, np.float64).ravel(),
                    dtype=dtype, device=device)
                acc = acc - (ew * bt).sum()
            _push(op['inputs'][0], ew)
        elif t == 'mul':
            scale = op.get('scale')
            if scale is None:
                raise NotImplementedError(
                    f'attn_crown: mul {name!r} without scale')
            st = torch.as_tensor(np.asarray(scale, np.float64).ravel(),
                                 dtype=dtype, device=device)
            _push(op['inputs'][0], ew * st)
        elif t == 'reshape':
            _push(op['inputs'][0], ew)
        elif t in ('slice', 'gather'):
            idx = torch.as_tensor(op['flat_idx'], dtype=torch.long,
                                  device=device)
            n_in = (int(np.prod(op['in_shapes_nd'][0]))
                    if op.get('in_shapes_nd', [None])[0] is not None
                    else int(idx.max()) + 1)
            ew_b = torch.zeros(n_in, dtype=dtype, device=device)
            ew_b = ew_b.index_add(0, idx, ew)
            _push(op['inputs'][0], ew_b)
        elif t == 'reduce_sum':
            in_sh = op['in_shapes_nd'][0]
            out_sh = op.get('out_shape_nd')
            ew_nd = ew.reshape(out_sh)
            ew_b = ew_nd.expand(
                *in_sh) if ew_nd.shape != tuple(in_sh) else ew_nd
            _push(op['inputs'][0], ew_b.reshape(-1).contiguous())
        else:
            raise NotImplementedError(
                f'attn_crown backward: unsupported op {t!r} at {name!r}')

    input_name = gg['input_name']
    ew_inp = ew_at.get(input_name)
    if ew_inp is None:
        raise NotImplementedError('attn_crown: no ew reached the input')
    lb = acc + ew_inp.clamp(min=0) @ xl.to(dtype) \
        + ew_inp.clamp(max=0) @ xh.to(dtype)
    return lb


def init_params(gg, sb, op_bounds, device, dtype, op_clamps=None,
                relu_clamps=None):
    """Leaf parameter tensors at the fixed-handler defaults."""
    params = {}
    if op_clamps is not None:
        for nm, (cl, ch) in op_clamps.items():
            # zero init: iteration 0 equals the no-beta bound
            params[('beta', nm)] = torch.zeros(
                (2, cl.numel()), device=device, dtype=dtype,
                requires_grad=True)
    if relu_clamps is not None:
        for L, (cl, ch) in relu_clamps.items():
            params[('rbeta', L)] = torch.zeros(
                (2, cl.numel()), device=device, dtype=dtype,
                requires_grad=True)
    for op in gg['ops']:
        t = op['type']; name = op['name']
        if t == 'relu' and op.get('layer_idx') in sb:
            L = op['layer_idx']
            lo_t, hi_t = sb[L]
            ub_r = hi_t.clamp(min=0)
            lb_r = lo_t.clamp(max=0)
            up_s = ub_r / (ub_r - lb_r).clamp(min=1e-12)
            params[('relu', L)] = (up_s > 0.5).to(dtype).clone().to(
                device).requires_grad_(True)
        elif t in ('exp', 'reciprocal') and name in op_bounds:
            n = op_bounds[name][0].numel()
            params[('cv', name)] = torch.full(
                (n,), 0.5, device=device, dtype=dtype, requires_grad=True)
        elif t in ('matmul_bilinear', 'mul_bilinear') and name in op_bounds:
            if t == 'matmul_bilinear':
                sa = op['in_shapes_nd'][0]
                sb_ = op['in_shapes_nd'][1]
                # per-term parameters: (.., n, m, p)
                shape = (*sa, sb_[-1])
            else:
                shape = tuple(op['out_shape_nd'])
            params[('mc', name)] = torch.ones(
                (2, *shape), device=device, dtype=dtype,
                requires_grad=True)
    return params


def attn_crown_alpha(gg, xl, xh, sb, op_bounds, w_q, b_q, *,
                     n_iters=50, lr=0.25, time_left=None,
                     tight_bounds=None, params=None, op_clamps=None,
                     relu_clamps=None):
    """Adam over the plane parameters for ONE query; returns
    (best_lb, params). Iteration 0 equals the fixed-corner backward, so
    the best-of is never worse. Early-exits when the bound goes positive.
    """
    device = xl.device
    dtype = xl.dtype
    if params is None:
        params = init_params(gg, sb, op_bounds, device, dtype,
                             op_clamps=op_clamps, relu_clamps=relu_clamps)
    if not params:
        with torch.no_grad():
            lb = attn_crown_lb(gg, xl, xh, sb, op_bounds, w_q, b_q, {},
                               tight_bounds=tight_bounds,
                               op_clamps=op_clamps,
                               relu_clamps=relu_clamps)
        return float(lb), params
    opt = torch.optim.Adam(list(params.values()), lr=lr)
    best = -float('inf')
    for _ in range(n_iters):
        if time_left is not None and time_left() <= 1.0:
            break
        opt.zero_grad()
        lb = attn_crown_lb(gg, xl, xh, sb, op_bounds, w_q, b_q, params,
                           tight_bounds=tight_bounds, op_clamps=op_clamps,
                           relu_clamps=relu_clamps)
        best = max(best, float(lb.detach()))
        if best > 0:
            break
        (-lb).backward()
        opt.step()
        with torch.no_grad():
            for kk, p in params.items():
                if kk[0] in ('beta', 'rbeta'):
                    p.clamp_(min=0)     # Lagrangian: any beta >= 0 sound
                else:
                    p.clamp_(0, 1)
    return best, params


# ---------------------------------------------------------------------------
# Batched (multi-domain) beta-CROWN — the throughput layer.
#
# alpha,beta-CROWN's vit recipe bounds 32 BaB domains per batched GPU call
# (beta-crown iteration 10) WITHOUT re-propagating intermediates per domain:
# each domain's relaxation uses the ROOT intermediate bounds intersected
# with its own split clamps, and the split constraints enter through beta.
# Measured on ibp_3_3_8_1028: our unbatched per-node bound runs ~1.2
# domains/s while ABC effectively runs ~100/s — the tree sizes are similar,
# the throughput is not. This section mirrors attn_crown_lb with a leading
# batch dimension B over domains; every tensor in sb / op_bounds / params /
# clamps carries B first. Soundness per domain is the same argument as the
# single-domain walk (every plane valid on that domain's clamped ranges,
# beta >= 0 Lagrangian for its split halfspaces).
# ---------------------------------------------------------------------------


def _apply_beta_b(acc, ew_in, cl, ch, bt, dtype):
    """Batched _apply_beta: acc (B,), ew_in (B,n), cl/ch (B,n) ±inf where
    unclamped, bt (B,2,n)."""
    cl = cl.to(dtype); ch = ch.to(dtype)
    m_lo = torch.isfinite(cl)
    m_hi = torch.isfinite(ch)
    b_ = bt.clamp(min=0)
    b_lo_ = b_[:, 0] * m_lo.to(dtype)
    b_hi_ = b_[:, 1] * m_hi.to(dtype)
    ew_in = ew_in + (b_hi_ - b_lo_)
    acc = acc + (b_lo_ * torch.where(m_lo, cl, torch.zeros_like(cl))
                 - b_hi_ * torch.where(m_hi, ch, torch.zeros_like(ch))
                 ).sum(dim=-1)
    return acc, ew_in


# Optional torch.compile of the batched bound walk (the FSB eval_only hot path).
# Gated by VC_COMPILE_WALK (probe) / settings.attn_bab_compile_walk; default OFF.
# The walk is a launch-bound op-dispatch loop (~17% GPU util at batch=192), so
# fusing it should cut FSB wall-time without changing any bound (compile is
# semantics-preserving). dynamic=True avoids recompiles as the FSB batch varies.
_COMPILED_WALK = None


def _compiled_walk():
    global _COMPILED_WALK
    if _COMPILED_WALK is None:
        # The full-run walk is called with varying batch B and varying clamp
        # dict KEY-SETS (which layers/exp-ops are clamped this round) -> Dynamo
        # re-traces per distinct structure and the default recompile_limit(8)
        # exhausts -> it gives up to eager (slower). The number of distinct
        # (clamp-key-set, B) combos is small and bounded, so raise the limit;
        # dynamic=True keeps B from being a recompile axis.
        import torch._dynamo as _dyn
        for _attr in ('recompile_limit', 'cache_size_limit'):
            if hasattr(_dyn.config, _attr):
                setattr(_dyn.config, _attr, 32)
        _COMPILED_WALK = torch.compile(attn_crown_lb_batch, dynamic=True)
    return _COMPILED_WALK


def attn_crown_lb_batch(gg, xl, xh, sb_b, op_bounds_b, w_q, b_q, params_b,
                        op_clamps_b=None, relu_clamps_b=None,
                        start_name=None, ew0=None, return_ew=False,
                        return_input_form=False, return_ew_signed=False):
    """Batched attn_crown_lb over B domains; returns lb (B,).

    sb_b: {L: (lo (B,n), hi (B,n))}; op_bounds_b mirrors op_bounds with a
    leading B on every tensor; params_b values carry B first (('mc', name)
    is (B, 2, *shape), betas are (B, 2, n)); clamps are (B, n) ±inf.
    Same plane formulas as the single-domain walk — only shapes differ.

    start_name/ew0: seed the walk at an INTERMEDIATE op's output with
    custom rows ew0 (B, n_node) instead of the final op with w_q —
    used by intermediate-bound refinement (rows of ±I give per-
    coordinate lower/upper bounds of that node). Ops downstream of
    start_name never receive ew and are skipped.
    """
    ops = gg['ops']
    device = xl.device
    dtype = xl.dtype
    if ew0 is not None:
        assert start_name is not None
        B = ew0.shape[0]
        ew_at = {start_name: ew0.to(dtype)}
    else:
        w_t = w_q if torch.is_tensor(w_q) else torch.as_tensor(
            np.asarray(w_q, np.float64), device=device, dtype=dtype)
        w_t = w_t.to(dtype)
        if w_t.dim() == 2:
            # joint spec matrix: one query per batch row
            B = w_t.shape[0]
            ew_at = {ops[-1]['name']: w_t.clone()}
        else:
            # find B from any sb/op_bounds entry
            B = None
            for _lo, _hi in sb_b.values():
                B = _lo.shape[0]
                break
            if B is None:
                for v in op_bounds_b.values():
                    B = (v[0][0] if isinstance(v[0], tuple)
                         else v[0]).shape[0]
                    break
            assert B is not None, \
                'attn_crown_lb_batch: cannot infer batch size'
            ew_at = {ops[-1]['name']:
                     w_t.unsqueeze(0).expand(B, -1).clone()}
    acc = torch.zeros(B, device=device, dtype=dtype) + float(b_q)
    ew_relu = {} if return_ew else None
    ew_relu_s = {} if return_ew_signed else None

    def _push(inp, ew_b):
        ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_b)) + ew_b

    for op in reversed(ops):
        name = op['name']
        if name not in ew_at:
            continue
        ew = ew_at[name]
        t = op['type']

        if t == 'fc':
            acc = acc + ew @ op['bias'].to(dtype)
            _push(op['inputs'][0], ew @ op['W'].to(dtype))
        elif t == 'conv':
            C_out = op['out_shape'][0]
            acc = acc + (ew.reshape(B, C_out, -1).sum(dim=2)
                         @ op['bias'].to(dtype))
            ew_b = F.conv_transpose2d(
                ew.reshape(B, *op['out_shape']), op['kernel'].to(dtype),
                stride=op['stride'], padding=op['padding'],
                output_padding=op['output_padding']).reshape(B, -1)
            _push(op['inputs'][0], ew_b)
        elif t == 'relu':
            L = op.get('layer_idx')
            if ew_relu is not None:
                # per-domain backward sensitivity at this relu's output
                # (|lA|) — the BaBSR weight, replacing the static root ew_w.
                ew_relu[L] = ew.detach().abs()
            if ew_relu_s is not None:
                # SIGNED ew at this relu's output — needed for the BBPS
                # branch-improvement score (ep/en split + branch re-prop).
                ew_relu_s[L] = ew.detach().clone()
            lo_t, hi_t = sb_b[L]
            lo_t = lo_t.to(dtype); hi_t = hi_t.to(dtype)
            ub_r = hi_t.clamp(min=0)
            lb_r = lo_t.clamp(max=0)
            ub_r = torch.maximum(ub_r, lb_r + 1e-12)
            up_s = ub_r / (ub_r - lb_r)
            up_t = -lb_r * up_s
            active = (lo_t >= 0).to(dtype)
            dead = (hi_t <= 0).to(dtype)
            unstable = (1 - active) * (1 - dead)
            lam = params_b.get(('relu', L))
            lam = (lam.clamp(0, 1) if lam is not None
                   else (up_s > 0.5).to(dtype))
            lo_slope = active + unstable * lam
            up_slope = active + unstable * up_s
            up_off = unstable * up_t
            ep = ew.clamp(min=0); en = ew.clamp(max=0)
            acc = acc + (en * up_off).sum(dim=-1)
            ew_in = ep * lo_slope + en * up_slope
            if relu_clamps_b is not None and L in relu_clamps_b:
                bt = params_b.get(('rbeta', L))
                if bt is not None:
                    cl, ch = relu_clamps_b[L]
                    acc, ew_in = _apply_beta_b(acc, ew_in, cl, ch, bt,
                                               dtype)
            _push(op['inputs'][0], ew_in)
        elif t in ('exp', 'reciprocal'):
            l_in, u_in = op_bounds_b[name]
            l_in = l_in.to(dtype); u_in = u_in.to(dtype)
            w_in = (u_in - l_in).clamp(min=1e-12)
            s = params_b.get(('cv', name))
            s = s.clamp(0, 1) if s is not None else 0.5
            m = l_in + s * w_in
            if t == 'exp':
                f_l = torch.exp(l_in); f_u = torch.exp(u_in)
                k_up = (f_u - f_l) / w_in
                b_up = f_l - k_up * l_in
                k_lo = torch.exp(m)
                b_lo = k_lo * (1 - m)
            else:
                k_up = -1.0 / (l_in * u_in)
                b_up = 1.0 / l_in + 1.0 / u_in
                k_lo = -1.0 / (m * m)
                b_lo = 2.0 / m
            ep = ew.clamp(min=0); en = ew.clamp(max=0)
            acc = acc + (ep * b_lo + en * b_up).sum(dim=-1)
            ew_in = ep * k_lo + en * k_up
            if op_clamps_b is not None and name in op_clamps_b:
                bt = params_b.get(('beta', name))
                if bt is not None:
                    cl, ch = op_clamps_b[name]
                    acc, ew_in = _apply_beta_b(acc, ew_in, cl, ch, bt,
                                               dtype)
            _push(op['inputs'][0], ew_in)
        elif t == 'matmul_bilinear':
            (xlb, xhb), (ylb, yhb) = op_bounds_b[name]
            sa = op['in_shapes_nd'][0]
            sb_ = op['in_shapes_nd'][1]
            so = op['out_shape_nd']
            Xl = xlb.to(dtype).reshape(B, *sa).unsqueeze(-1)
            Xh = xhb.to(dtype).reshape(B, *sa).unsqueeze(-1)
            Yl = ylb.to(dtype).reshape(B, *sb_).unsqueeze(-3)
            Yh = yhb.to(dtype).reshape(B, *sb_).unsqueeze(-3)
            r = params_b.get(('mc', name))
            if r is None:
                r_lo = r_up = torch.ones((), device=device, dtype=dtype)
            else:
                r = r.clamp(0, 1)
                r_lo, r_up = r[:, 0], r[:, 1]
            al, bl, gl, au, bu, gu = _mc_planes(Xl, Xh, Yl, Yh, r_lo, r_up)
            ew_nd = ew.reshape(B, *so)
            ep = ew_nd.clamp(min=0).unsqueeze(-2)
            en = ew_nd.clamp(max=0).unsqueeze(-2)
            acc = acc + (ep * gl + en * gu).reshape(B, -1).sum(dim=-1)
            ew_x = (ep * al + en * au).sum(dim=-1).reshape(B, -1)
            ew_y = (ep * bl + en * bu).sum(dim=-3).reshape(B, -1)
            _push(op['inputs'][0], ew_x)
            _push(op['inputs'][1], ew_y)
        elif t == 'mul_bilinear':
            from .alpha_crown import _sum_to_shape
            (xlb, xhb), (ylb, yhb) = op_bounds_b[name]
            sa = op['in_shapes_nd'][0]
            sb_ = op['in_shapes_nd'][1]
            so = op['out_shape_nd']
            ones_o = torch.ones(so, dtype=dtype, device=device)
            Xl = xlb.to(dtype).reshape(B, *sa) * ones_o
            Xh = xhb.to(dtype).reshape(B, *sa) * ones_o
            Yl = ylb.to(dtype).reshape(B, *sb_) * ones_o
            Yh = yhb.to(dtype).reshape(B, *sb_) * ones_o
            r = params_b.get(('mc', name))
            if r is None:
                r_lo = r_up = torch.ones((), device=device, dtype=dtype)
            else:
                r = r.clamp(0, 1)
                r_lo, r_up = r[:, 0], r[:, 1]
            al, bl, gl, au, bu, gu = _mc_planes(Xl, Xh, Yl, Yh, r_lo, r_up)
            ew_nd = ew.reshape(B, *so)
            ep = ew_nd.clamp(min=0); en = ew_nd.clamp(max=0)
            acc = acc + (ep * gl + en * gu).reshape(B, -1).sum(dim=-1)
            cx = ep * al + en * au
            cy = ep * bl + en * bu
            _push(op['inputs'][0],
                  _sum_to_shape(cx, (B,), sa).reshape(B, -1))
            _push(op['inputs'][1],
                  _sum_to_shape(cy, (B,), sb_).reshape(B, -1))
        elif t == 'add':
            if op.get('is_merge'):
                for inp in op['inputs']:
                    _push(inp, ew)
            else:
                bias = op.get('bias')
                if bias is not None:
                    bt = torch.as_tensor(
                        np.asarray(bias, np.float64).ravel(),
                        dtype=dtype, device=device)
                    assert bt.numel() == ew.shape[-1], (
                        f'add {name!r}: bias size {bt.numel()} != ew '
                        f'{ew.shape[-1]} (emission broadcasts biases)')
                    acc = acc + (ew * bt).sum(dim=-1)
                _push(op['inputs'][0], ew)
        elif t == 'sub':
            bias = op.get('bias')
            if bias is not None:
                bt = torch.as_tensor(
                    np.asarray(bias, np.float64).ravel(),
                    dtype=dtype, device=device)
                acc = acc - (ew * bt).sum(dim=-1)
            _push(op['inputs'][0], ew)
        elif t == 'mul':
            scale = op.get('scale')
            if scale is None:
                raise NotImplementedError(
                    f'attn_crown batch: mul {name!r} without scale')
            st = torch.as_tensor(np.asarray(scale, np.float64).ravel(),
                                 dtype=dtype, device=device)
            _push(op['inputs'][0], ew * st)
        elif t == 'reshape':
            _push(op['inputs'][0], ew)
        elif t in ('slice', 'gather'):
            idx = torch.as_tensor(op['flat_idx'], dtype=torch.long,
                                  device=device)
            n_in = (int(np.prod(op['in_shapes_nd'][0]))
                    if op.get('in_shapes_nd', [None])[0] is not None
                    else int(idx.max()) + 1)
            ew_b = torch.zeros(B, n_in, dtype=dtype, device=device)
            ew_b = ew_b.index_add(1, idx, ew)
            _push(op['inputs'][0], ew_b)
        elif t == 'reduce_sum':
            in_sh = op['in_shapes_nd'][0]
            out_sh = op.get('out_shape_nd')
            ew_nd = ew.reshape(B, *out_sh)
            ew_b = (ew_nd.expand(B, *in_sh)
                    if ew_nd.shape != (B, *in_sh) else ew_nd)
            _push(op['inputs'][0], ew_b.reshape(B, -1).contiguous())
        else:
            raise NotImplementedError(
                f'attn_crown batch backward: unsupported op {t!r} at '
                f'{name!r}')

    input_name = gg['input_name']
    ew_inp = ew_at.get(input_name)
    if ew_inp is None:
        raise NotImplementedError('attn_crown batch: no ew reached input')
    lb = acc + ew_inp.clamp(min=0) @ xl.to(dtype) \
        + ew_inp.clamp(max=0) @ xh.to(dtype)
    if return_input_form:
        # (acc (B,), ew_inp (B, n_in)): the input-space linear lower form
        # lb(x) = acc + ew_inp·x. For an upper bound seed ew0=-eye and negate.
        return lb, acc, ew_inp
    if return_ew_signed:
        return lb, ew_relu_s
    if return_ew:
        return lb, ew_relu
    return lb


def _seed_input_forms(gg, xl, xh, sb, ob, params, start_name, ew0,
                      chunk=512):
    """Input-space linear forms (acc (B,), ew_inp (B, n_in)) for the backward
    walk seeded with rows `ew0` at `start_name` — i.e. each row's CROWN bound is
    `acc + ew_inp·x` over the input box. Batched + chunked, no-grad."""
    device = xl.device; dtype = xl.dtype
    B = ew0.shape[0]
    accs = []; ews = []
    for s0 in range(0, B, chunk):
        rows = ew0[s0:s0 + chunk]; b = rows.shape[0]
        sb_b = {L: (lo.unsqueeze(0).expand(b, -1), hi.unsqueeze(0).expand(b, -1))
                for L, (lo, hi) in sb.items()}
        ob_b = {}
        for k, v in ob.items():
            if isinstance(v[0], tuple):
                ob_b[k] = ((v[0][0].unsqueeze(0).expand(b, -1),
                            v[0][1].unsqueeze(0).expand(b, -1)),
                           (v[1][0].unsqueeze(0).expand(b, -1),
                            v[1][1].unsqueeze(0).expand(b, -1)))
            else:
                ob_b[k] = (v[0].unsqueeze(0).expand(b, -1),
                           v[1].unsqueeze(0).expand(b, -1))
        p_b = {k: v.unsqueeze(0).expand(b, *v.shape)
               for k, v in params.items() if k[0] not in ('beta', 'rbeta')}
        with torch.no_grad():
            _lb, acc, ew = attn_crown_lb_batch(
                gg, xl, xh, sb_b, ob_b, None, 0.0, p_b,
                start_name=start_name, ew0=rows, return_input_form=True)
        accs.append(acc); ews.append(ew)
    return torch.cat(accs), torch.cat(ews)


def bbps_root_scores(gg, xl, xh, sb, ob, params, w_q, b_q, chunk=256):
    """BBPS branching score (ABC's nonlinear-branching heuristic, ported to VC).

    ABC branches vit with NonlinearBranching/BBPS, NOT BaBSR: for each unstable
    relu neuron it estimates the WORST-CHILD improvement in the spec lower bound
    from splitting at 0, via a fast backward re-propagation (reuse the spec's input
    linear form + each neuron's downstream-to-input map; re-concretize per branch).
    This scores EVERY unstable neuron cheaply (unlike VC's FSB, which only evaluates
    the BaBSR-top-k and so misses the spec-critical neurons BaBSR under-ranks).

    Returns {layer_idx: score (n,)} (higher = more valuable split; 0 for stable).
    Branching-only -> soundness-irrelevant. Uses the same linear-superposition
    trick as ABC's A_after-A_before: with fixed earlier-layer relaxations the
    downstream propagation is linear in layer L's coefficients, so replacing one
    neuron's coefficient is a rank-1 update of the input form + a re-concretize."""
    device = xl.device
    dtype = xl.dtype
    ops = gg['ops']
    relu_in = {op['layer_idx']: op['inputs'][0] for op in ops
               if op['type'] == 'relu' and op.get('layer_idx') in sb}
    w_t = (w_q.to(device=device, dtype=dtype) if torch.is_tensor(w_q)
           else torch.as_tensor(np.asarray(w_q, np.float64),
                                device=device, dtype=dtype))
    # spec input form: lb(x) = acc_spec + b_q + ewinp_full·x  (b_q cancels in Δ)
    _acc_spec, ewinp_spec = _seed_input_forms(gg, xl, xh, sb, ob, params,
                                              ops[-1]['name'], w_t.unsqueeze(0))
    ewinp_full = ewinp_spec[0]                              # (n_in,)
    base_conc = (ewinp_full.clamp(min=0) @ xl.to(dtype)
                 + ewinp_full.clamp(max=0) @ xh.to(dtype))
    # signed ew at every relu output (one batched B=1 walk)
    sb_b = {L: (lo.unsqueeze(0), hi.unsqueeze(0)) for L, (lo, hi) in sb.items()}
    ob_b = {}
    for k, v in ob.items():
        if isinstance(v[0], tuple):
            ob_b[k] = ((v[0][0].unsqueeze(0), v[0][1].unsqueeze(0)),
                       (v[1][0].unsqueeze(0), v[1][1].unsqueeze(0)))
        else:
            ob_b[k] = (v[0].unsqueeze(0), v[1].unsqueeze(0))
    p_b = {k: v.unsqueeze(0) for k, v in params.items()
           if k[0] not in ('beta', 'rbeta')}
    with torch.no_grad():
        _lb, ew_s = attn_crown_lb_batch(gg, xl, xh, sb_b, ob_b, w_t, float(b_q),
                                        p_b, return_ew_signed=True)
    out = {}
    for L, (lo, hi) in sb.items():
        if L not in relu_in or L not in ew_s:
            continue
        lo = lo.to(dtype); hi = hi.to(dtype)
        n_L = lo.numel()
        ew_out = ew_s[L][0]                                 # (n_L,) signed
        ub_r = hi.clamp(min=0); lb_r = lo.clamp(max=0)
        ub_r = torch.maximum(ub_r, lb_r + 1e-12)
        up_s = ub_r / (ub_r - lb_r); up_t = -lb_r * up_s
        active = (lo >= 0).to(dtype); dead = (hi <= 0).to(dtype)
        unstable = (1 - active) * (1 - dead)
        lam = params.get(('relu', L))
        lam = (lam.clamp(0, 1) if lam is not None else (up_s > 0.5).to(dtype))
        lo_slope = active + unstable * lam
        up_slope = active + unstable * up_s
        up_off = unstable * up_t
        ep = ew_out.clamp(min=0); en = ew_out.clamp(max=0)
        ewin_cur = ep * lo_slope + en * up_slope            # current ew^in_L
        if not bool((unstable > 0).any()):
            out[L] = torch.zeros(n_L, device=device, dtype=dtype); continue
        # per-neuron downstream-to-input map (seed eye at the pre-activation op)
        eye = torch.eye(n_L, device=device, dtype=dtype)
        acc_seed, ewinp_seed = _seed_input_forms(gg, xl, xh, sb, ob, params,
                                                 relu_in[L], eye, chunk=chunk)
        dbounds = []
        for ewin_new in (ew_out, torch.zeros_like(ew_out)):  # active, inactive
            delta = ewin_new - ewin_cur                      # (n_L,)
            dbound = torch.empty(n_L, device=device, dtype=dtype)
            for s0 in range(0, n_L, chunk):
                sl = slice(s0, min(s0 + chunk, n_L))
                ew_new = (ewinp_full.unsqueeze(0)
                          + delta[sl].unsqueeze(1) * ewinp_seed[sl])  # (c, n_in)
                conc = (ew_new.clamp(min=0) @ xl.to(dtype)
                        + ew_new.clamp(max=0) @ xh.to(dtype))
                dacc = -en[sl] * up_off[sl] + delta[sl] * acc_seed[sl]
                dbound[sl] = dacc + (conc - base_conc)
            dbounds.append(dbound)
        score = torch.minimum(dbounds[0], dbounds[1])        # worst-child gain
        out[L] = torch.where(unstable > 0, score,
                             torch.zeros_like(score))
    return out


def box_halfspace_branch_score(gg, xl, xh, sb, ob, params, w_q, b_q):
    """Deterministic spec-aware branching weight (fix #2). For each unstable
    relu neuron, bound its PRE-activation over `box ∩ {spec-halfspace}` using
    CROWN backward linear forms (no generators / forward-zono) and the
    closed-form `box_halfspace` LP; score = the gap that survives the spec
    constraint (the neurons the spec can't pin down -> the spec-critical splits).
    Returns {layer_idx: score (n,)}. Branching-only -> soundness-irrelevant."""
    import numpy as _np
    from . import box_halfspace as _bh
    device = xl.device; dtype = xl.dtype
    center = ((xl + xh) * 0.5).to(torch.float64).cpu().numpy()
    radius = ((xh - xl) * 0.5).to(torch.float64).cpu().numpy()
    # spec lower form m(x)=acc+a·x over the input -> halfspace a·x <= -acc
    # (region where the spec lower bound is non-positive; superset of unsafe),
    # rescaled to e in [-1,1]: x = center + radius*e.
    w_t = (w_q.to(device=device, dtype=dtype) if torch.is_tensor(w_q)
           else torch.as_tensor(_np.asarray(w_q, _np.float64),
                                device=device, dtype=dtype))
    acc_s, ew_s = _seed_input_forms(gg, xl, xh, sb, ob, params,
                                    gg['ops'][-1]['name'], w_t.unsqueeze(0))
    a_in = ew_s[0].to(torch.float64).cpu().numpy()
    a_spec = a_in * radius
    beta_spec = float(-acc_s[0].item() - float((a_in * center).sum()) - float(b_q))
    # relu layer -> pre-activation op name
    relu_in = {op['layer_idx']: op['inputs'][0] for op in gg['ops']
               if op['type'] == 'relu' and op.get('layer_idx') in sb}
    out = {}
    for L, in_name in relu_in.items():
        lo_t, hi_t = sb[L]
        uns = ((lo_t < 0) & (hi_t > 0))
        n = lo_t.numel()
        score = torch.zeros(n, device=device, dtype=dtype)
        if not bool(uns.any()):
            out[L] = score; continue
        eye = torch.eye(n, device=device, dtype=dtype)
        acc_lo, ew_lo = _seed_input_forms(gg, xl, xh, sb, ob, params, in_name, eye)
        acc_up, ew_up = _seed_input_forms(gg, xl, xh, sb, ob, params, in_name, -eye)
        ew_lo = ew_lo.to(torch.float64).cpu().numpy(); acc_lo = acc_lo.to(torch.float64).cpu().numpy()
        ew_up = ew_up.to(torch.float64).cpu().numpy(); acc_up = acc_up.to(torch.float64).cpu().numpy()
        idx = torch.nonzero(uns).flatten().tolist()
        sc = score.cpu().numpy()
        for j in idx:
            d_lo = ew_lo[j] * radius; c0_lo = acc_lo[j] + float((ew_lo[j] * center).sum())
            d_hi = -ew_up[j] * radius; c0_hi = -acc_up[j] - float((ew_up[j] * center).sum())
            tlo = _bh.lagrangian_min(d_lo, c0_lo, a_spec, beta_spec)
            thi = _bh.lagrangian_max(d_hi, c0_hi, a_spec, beta_spec)
            if _np.isfinite(tlo) and _np.isfinite(thi) and thi > tlo:
                sc[j] = min(-tlo, thi)            # gap surviving the spec halfspace
            else:                                  # infeasible/degenerate -> plain gap
                sc[j] = min(-float(lo_t[j]), float(hi_t[j]))
        if os.environ.get('VC_BH_DEBUG'):
            # rank candidate formulas vs ABC's known-good neuron set for 2157
            _abc = {0: {326, 328, 570, 584, 646, 678, 725, 1005, 1033, 1093,
                        1513, 1558}, 1: {652, 1132, 1386, 1463},
                    2: {983, 998, 1118, 1295}}.get(L, set())
            a_arr = a_spec
            proj = _np.array([abs(float((ew_lo[j] * radius * a_arr).sum()))
                              for j in idx])  # |spec . g_lo| (sensitivity)
            gap = sc[idx]                      # surviving spec-halfspace gap
            base = _np.array([min(-float(lo_t[j]), float(hi_t[j])) for j in idx])
            for nm_, val_ in (('bh_gap', gap), ('spec_proj', proj),
                              ('gap*proj', base * proj), ('bh*proj', gap * proj)):
                order = [idx[k] for k in _np.argsort(-val_)]
                topk = set(order[:max(4, len(_abc))])
                print(f'  [bh-dbg] L{L} {nm_}: top{len(topk)} ∩ ABC = '
                      f'{len(topk & _abc)}/{len(_abc)}', flush=True)
        out[L] = torch.as_tensor(sc, device=device, dtype=dtype).clamp(min=0)
    return out


def attn_beta_bab(gg, xl, xh, sb0, ob0, w_q, b_q, root_params, *,
                  time_left, ew_w=None, batch=16, n_iters=12, lr=0.1,
                  max_domains=200000, print_progress=False,
                  gg_work=None, work_dtype=None, kfsb_k=4, perdom_ew=True,
                  hot_warmup=0, hot_kfsb=16, bh_score=False, perdom_la=False,
                  bbps_score=False, bbps_topk=12):
    """Batched no-reforward beta-CROWN BaB on ONE open query.

    Returns (ok, n_domains, reason). The ABC vit recipe: each domain's
    relaxation = ROOT intermediate bounds intersected with that domain's
    split clamps (no zonotope re-propagation), split constraints
    enforced via beta, plane+beta params Adam-optimized over a BATCH of
    domains in one autograd graph, warm-started from the parent domain.
    Sound per domain: planes valid on the clamped ranges (supersets of
    the true subdomain ranges), beta >= 0 Lagrangian for the split
    halfspaces, best-of over iterations.

    Domain clamps: {(L:int, j): side} relu splits / {(name:str, j):
    (l, u)} exp-input value splits. Split choice: ew-weighted BaBSR
    relu score; exp-input slack fallback when no unstable relu remains.

    gg_work/work_dtype: optional low-precision SEARCH mode (fp32 on
    consumer GPUs is ~30x fp64 throughput). The Adam search runs in
    work_dtype on gg_work, but a domain is only PRUNED after its found
    params re-certify lb > 0 in the caller's full-precision walk on
    `gg` — closure decisions never rest on fp32 arithmetic.
    """
    import heapq
    device = xl.device
    dtype = xl.dtype
    w_t = w_q if torch.is_tensor(w_q) else torch.as_tensor(
        np.asarray(w_q, np.float64), device=device, dtype=dtype)
    w_t = w_t.to(dtype)
    wdt = work_dtype or dtype
    wgg = gg_work if gg_work is not None else gg
    xl_w = xl.to(wdt); xh_w = xh.to(wdt)
    w_w = w_t.to(wdt)
    exp_names = [op['name'] for op in gg['ops'] if op['type'] == 'exp']
    # DIAGNOSTIC: restrict BaB relu candidates to an external (block,neuron) set
    # (e.g. ABC's chosen splits) to isolate branching-scoring vs bounding.
    _force_set = None
    _fp = os.environ.get('VC_FORCE_NEURONS')
    if _fp:
        _force_set = set()
        for _ln in open(_fp):
            _p = _ln.split()
            if len(_p) == 2:
                _force_set.add((int(_p[0]), int(_p[1])))
    sb_root = {L: (lo.detach(), hi.detach()) for L, (lo, hi) in sb0.items()}
    base_params = {k: v.detach() for k, v in (root_params or {}).items()}

    def _mk_oc_single(clamps):
        oc = {}
        for key, val in clamps.items():
            if not isinstance(key[0], str):
                continue
            nm, j = key
            if nm not in oc:
                n = ob0[nm][0].numel()
                oc[nm] = (torch.full((n,), -np.inf, device=device,
                                     dtype=dtype),
                          torch.full((n,), np.inf, device=device,
                                     dtype=dtype))
            oc[nm][0][j] = val[0]
            oc[nm][1][j] = val[1]
        return oc or None

    def _mk_rc_single(clamps):
        rc = {}
        for key, side in clamps.items():
            if isinstance(key[0], str):
                continue
            L, j = key
            if L not in rc:
                n = sb_root[L][0].numel()
                rc[L] = (torch.full((n,), -np.inf, device=device,
                                    dtype=dtype),
                         torch.full((n,), np.inf, device=device,
                                    dtype=dtype))
            if side == 0:
                rc[L][1][j] = 0.0
            else:
                rc[L][0][j] = 0.0
        return rc or None

    def recheck_fp64(clamps, params_i):
        """Full-precision certification of a work-dtype closure."""
        sb_d = dom_sb(clamps)
        ob_d = dict(ob0)
        for nm in exp_names:
            if nm in ob_d:
                ob_d[nm] = dom_exp_bounds(clamps, nm)
        p64 = {k: v.to(device=device, dtype=dtype)
               for k, v in params_i.items()}
        with torch.no_grad():
            lb = attn_crown_lb(gg, xl, xh, sb_d, ob_d, w_t, float(b_q),
                               p64, op_clamps=_mk_oc_single(clamps),
                               relu_clamps=_mk_rc_single(clamps))
        return float(lb)

    def dom_sb(clamps):
        sb_d = {}
        for L, (lo, hi) in sb_root.items():
            lo2, hi2 = lo, hi
            for (k, j), side in clamps.items():
                if k != L:
                    continue
                if lo2 is lo:
                    lo2 = lo.clone(); hi2 = hi.clone()
                if side == 0:
                    hi2[j] = min(float(hi2[j]), 0.0)
                else:
                    lo2[j] = max(float(lo2[j]), 0.0)
            sb_d[L] = (lo2, hi2)
        return sb_d

    def dom_exp_bounds(clamps, nm):
        lo, hi = ob0[nm]
        lo2, hi2 = lo, hi
        for (k, j), val in clamps.items():
            if k != nm:
                continue
            if lo2 is lo:
                lo2 = lo.clone(); hi2 = hi.clone()
            lo2[j] = max(float(lo2[j]), val[0])
            hi2[j] = min(float(hi2[j]), val[1])
            hi2[j] = max(float(hi2[j]), float(lo2[j]))
        return lo2, hi2

    def batch_dom_ew(clamps_list):
        """Per-domain backward sensitivity |lA| at each relu layer for a batch
        of domains, via ONE no-beta backward walk (root alpha, the domain's
        tightened relu/exp bounds). Returns {L: (B, n)} or None. This is the
        BaBSR weight ABC uses (per-domain), replacing the static root ew_w —
        the relaxation planes reflect each domain's clamps via dom_sb, so the
        sensitivity is domain-specific. Soundness-irrelevant (branching only)."""
        B = len(clamps_list)
        if B == 0:
            return None
        sb_b = {}
        for L in sb_root:
            los = []; his = []
            for clamps in clamps_list:
                lo_d, hi_d = dom_sb(clamps)[L]
                los.append(lo_d); his.append(hi_d)
            sb_b[L] = (torch.stack(los).to(wdt), torch.stack(his).to(wdt))
        ob_b = {}
        for nm, v in ob0.items():
            if isinstance(v[0], tuple):
                (xlb, xhb), (ylb, yhb) = v
                ob_b[nm] = (
                    (xlb.unsqueeze(0).expand(B, -1).to(wdt),
                     xhb.unsqueeze(0).expand(B, -1).to(wdt)),
                    (ylb.unsqueeze(0).expand(B, -1).to(wdt),
                     yhb.unsqueeze(0).expand(B, -1).to(wdt)))
            elif nm in exp_names:
                los = []; his = []
                for clamps in clamps_list:
                    lo_d, hi_d = dom_exp_bounds(clamps, nm)
                    los.append(lo_d); his.append(hi_d)
                ob_b[nm] = (torch.stack(los).to(wdt), torch.stack(his).to(wdt))
            else:
                ob_b[nm] = (v[0].unsqueeze(0).expand(B, -1).to(wdt),
                            v[1].unsqueeze(0).expand(B, -1).to(wdt))
        pb = {k: v.unsqueeze(0).expand(B, *v.shape).to(wdt)
              for k, v in base_params.items() if k[0] not in ('beta', 'rbeta')}
        with torch.no_grad():
            _lb, ew_relu = attn_crown_lb_batch(
                wgg, xl_w, xh_w, sb_b, ob_b, w_w, float(b_q), pb,
                return_ew=True)
        return ew_relu

    def pick_split(clamps):
        best = None; best_score = -1.0
        sb_d = dom_sb(clamps)
        for L, (lo, hi) in sb_d.items():
            uns = (lo < 0) & (hi > 0)
            if not bool(uns.any()):
                continue
            score = torch.minimum(-lo, hi) * uns
            if ew_w is not None and L in ew_w \
                    and ew_w[L].numel() == score.numel():
                score = score * (ew_w[L] + 1e-12)
            for (L2, j2) in clamps:
                if L2 == L:
                    score[j2] = -1.0
            j = int(score.argmax())
            s = float(score[j])
            if s > best_score and (L, j) not in clamps:
                best_score = s; best = ('relu', L, int(j))
        if best is not None:
            return best
        for nm in exp_names:
            if nm not in ob0:
                continue
            lo, hi = dom_exp_bounds(clamps, nm)
            w_in = hi - lo
            k = (torch.exp(hi) - torch.exp(lo)) / w_in.clamp(min=1e-12)
            xs = torch.log(k.clamp(min=1e-300))
            xs = torch.minimum(torch.maximum(xs, lo), hi)
            slack = (torch.exp(lo) - k * lo) - (torch.exp(xs) - k * xs)
            slack = torch.where(w_in > 1e-6, slack.clamp(min=0.0),
                                torch.full_like(slack, -1.0))
            j = int(slack.argmax())
            s = float(slack[j])
            if s > best_score:
                best_score = s
                best = ('exp', nm, j, float(lo[j]), float(hi[j]))
        return best

    def pick_candidates(clamps, kk, dom_ew=None, allow=None, full_w=None,
                        dom_la=None):
        """Top-kk relu split candidates by the BaBSR-style heuristic
        (global across layers); exp-input fallback when no unstable
        relu remains. Returns a list of split descriptors (possibly
        length < kk), or [] when nothing is splittable.

        dom_ew: optional {L: (n,)} per-domain backward sensitivity |lA|
        (from batch_dom_ew) — the BaBSR weight ABC uses. Falls back to the
        static root ew_w when not provided.
        full_w: optional {L: (n,)} that REPLACES the gap*weight score entirely
        (the box-halfspace spec-aware score already encodes split value).
        dom_la: optional {L: (n,)} per-domain BETA-INCLUSIVE |lA| (ABC's BaBSR);
        when given, score = |lA| * triangle-intercept(-lo*hi/(hi-lo))."""
        cands = []
        sb_d = dom_sb(clamps)
        for L, (lo, hi) in sb_d.items():
            uns = (lo < 0) & (hi > 0)
            if not bool(uns.any()):
                continue
            if dom_la is not None and L in dom_la \
                    and dom_la[L].numel() == lo.numel():
                # proper BaBSR: |lA| * triangle intercept (-lo*hi/(hi-lo))
                intercept = (-lo * hi) / (hi - lo).clamp(min=1e-12)
                score = dom_la[L].to(device=lo.device, dtype=lo.dtype) \
                    * intercept * uns
            elif full_w is not None and L in full_w \
                    and full_w[L].numel() == lo.numel():
                score = full_w[L].to(lo.dtype).clone() * uns
            else:
                score = torch.minimum(-lo, hi) * uns
                _w = (dom_ew.get(L) if dom_ew is not None else None)
                if _w is None and ew_w is not None and L in ew_w:
                    _w = ew_w[L]
                if _w is not None and _w.numel() == score.numel():
                    score = score * (_w.to(score.dtype) + 1e-12)
            for (L2, j2) in clamps:
                if L2 == L:
                    score[j2] = -1.0
            _allow = _force_set if _force_set is not None else allow
            if _allow is not None:
                mask = torch.zeros_like(score, dtype=torch.bool)
                for (fb, fn) in _allow:
                    if fb == L and fn < score.numel():
                        mask[fn] = True
                score = torch.where(mask, score,
                                    torch.full_like(score, -1.0))
            kk_l = min(kk, score.numel())
            v, idx = torch.topk(score, kk_l)
            for s_, j_ in zip(v.tolist(), idx.tolist()):
                if s_ > 0 and (L, int(j_)) not in clamps:
                    if _allow is not None and (L, int(j_)) not in _allow:
                        continue
                    cands.append((s_, ('relu', L, int(j_))))
        if cands:
            cands.sort(key=lambda t: -t[0])
            return [c for _, c in cands[:kk]]
        sp = pick_split(clamps)
        return [sp] if sp is not None else []

    def child_clamps(clamps, sp, side):
        c2 = dict(clamps)
        if sp[0] == 'relu':
            c2[(sp[1], sp[2])] = side
        else:
            _, nm, j, l_, u_ = sp
            m_ = 0.5 * (l_ + u_)
            c2[(nm, j)] = (l_, m_) if side == 0 else (m_, u_)
        return c2

    def bound_batch(doms, eval_only=False, return_la=False):
        """doms: list of (clamps, warm_params). Returns best lb (B,)
        tensor (detached) and per-domain optimized params (detached);
        eval_only skips the Adam loop (one no-grad eval, params=warm —
        the kfsb cheap candidate scorer)."""
        B = len(doms)
        # assemble batched relaxation state
        sb_b = {}
        for L in sb_root:
            los = []; his = []
            for clamps, _ in doms:
                lo_d, hi_d = dom_sb(clamps)[L]
                los.append(lo_d); his.append(hi_d)
            sb_b[L] = (torch.stack(los), torch.stack(his))
        ob_b = {}
        for nm, v in ob0.items():
            if isinstance(v[0], tuple):
                (xlb, xhb), (ylb, yhb) = v
                ob_b[nm] = (
                    (xlb.unsqueeze(0).expand(B, -1),
                     xhb.unsqueeze(0).expand(B, -1)),
                    (ylb.unsqueeze(0).expand(B, -1),
                     yhb.unsqueeze(0).expand(B, -1)))
            elif nm in exp_names:
                los = []; his = []
                for clamps, _ in doms:
                    lo_d, hi_d = dom_exp_bounds(clamps, nm)
                    los.append(lo_d); his.append(hi_d)
                ob_b[nm] = (torch.stack(los), torch.stack(his))
            else:
                ob_b[nm] = (v[0].unsqueeze(0).expand(B, -1),
                            v[1].unsqueeze(0).expand(B, -1))
        # split-constraint clamps (dense per layer/op present in any dom)
        rc_b = {}; oc_b = {}
        for L in sb_root:
            if any(not isinstance(k[0], str) and k[0] == L
                   for clamps, _ in doms for k in clamps):
                n = sb_root[L][0].numel()
                cl = torch.full((B, n), -np.inf, device=device, dtype=dtype)
                ch = torch.full((B, n), np.inf, device=device, dtype=dtype)
                for i, (clamps, _) in enumerate(doms):
                    for (k, j), side in clamps.items():
                        if isinstance(k, str) or k != L:
                            continue
                        if side == 0:
                            ch[i, j] = 0.0
                        else:
                            cl[i, j] = 0.0
                rc_b[L] = (cl, ch)
        for nm in exp_names:
            if any(isinstance(k[0], str) and k[0] == nm
                   for clamps, _ in doms for k in clamps):
                n = ob0[nm][0].numel()
                cl = torch.full((B, n), -np.inf, device=device, dtype=dtype)
                ch = torch.full((B, n), np.inf, device=device, dtype=dtype)
                for i, (clamps, _) in enumerate(doms):
                    for (k, j), val in clamps.items():
                        if not isinstance(k, str) or k != nm:
                            continue
                        cl[i, j] = val[0]
                        ch[i, j] = val[1]
                oc_b[nm] = (cl, ch)
        if eval_only and _compile_walk:
            # CONSTANT-STRUCTURE for torch.compile: give rc_b/oc_b the SAME keys
            # every call (all relu layers + exp ops, ±inf where unclamped) so the
            # compiled walk sees one fixed graph instead of re-tracing per
            # clamp-key-set. ±inf clamp + the (zero) rbeta/beta that params_b then
            # adds for these keys is a NO-OP (the _apply_beta_b mask is all-False
            # at ±inf), so the bound is identical — only the dict structure is
            # stabilized. Confined to the FSB eval_only path (Adam path unchanged).
            for L in sb_root:
                if L not in rc_b:
                    n = sb_root[L][0].numel()
                    rc_b[L] = (
                        torch.full((B, n), -np.inf, device=device, dtype=dtype),
                        torch.full((B, n), np.inf, device=device, dtype=dtype))
            for nm in exp_names:
                if nm not in oc_b:
                    n = ob0[nm][0].numel()
                    oc_b[nm] = (
                        torch.full((B, n), -np.inf, device=device, dtype=dtype),
                        torch.full((B, n), np.inf, device=device, dtype=dtype))
        # batched params: every key in base_params + betas for clamped
        # layers/ops; warm-start per domain where the parent carried one
        params_b = {}
        all_keys = set(base_params)
        for _, wp in doms:
            if wp is not None:
                all_keys.update(wp)
        for L in rc_b:
            all_keys.add(('rbeta', L))
        for nm in oc_b:
            all_keys.add(('beta', nm))
        for key in all_keys:
            rows = []
            for clamps, wp in doms:
                if wp is not None and key in wp:
                    rows.append(wp[key].to(device=device, dtype=dtype))
                elif key in base_params:
                    rows.append(base_params[key])
                else:
                    kind = key[0]
                    if kind == 'rbeta':
                        n = sb_root[key[1]][0].numel()
                        rows.append(torch.zeros((2, n), device=device,
                                                dtype=dtype))
                    elif kind == 'beta':
                        n = ob0[key[1]][0].numel()
                        rows.append(torch.zeros((2, n), device=device,
                                                dtype=dtype))
                    else:
                        raise NotImplementedError(
                            f'attn_beta_bab: param {key!r} missing from '
                            'both warm-start and root')
            params_b[key] = torch.stack(rows).to(wdt) \
                .requires_grad_(True)
        if wdt != dtype:
            # search in the low-precision work dtype (closure decisions
            # re-certified in full precision by the caller loop)
            sb_b = {L: (lo.to(wdt), hi.to(wdt))
                    for L, (lo, hi) in sb_b.items()}
            ob_b = {k: (((v[0][0].to(wdt), v[0][1].to(wdt)),
                         (v[1][0].to(wdt), v[1][1].to(wdt)))
                        if isinstance(v[0], tuple)
                        else (v[0].to(wdt), v[1].to(wdt)))
                    for k, v in ob_b.items()}
            rc_b = {L: (cl.to(wdt), ch.to(wdt))
                    for L, (cl, ch) in rc_b.items()}
            oc_b = {nm: (cl.to(wdt), ch.to(wdt))
                    for nm, (cl, ch) in oc_b.items()}
        if eval_only:
            if _dump_walk and not _walk_dumped[0] and B >= _dump_walk_minB:
                import pickle
                # wgg holds a torch.jit ScriptFunction (some cached kernel) that
                # plain pickle rejects; the backward CROWN walk only reads DATA
                # fields, so strip every non-data object (callables/ScriptFns).
                def _san(o):
                    if isinstance(o, torch.Tensor):
                        return o.detach()
                    if isinstance(o, (int, float, bool, str, bytes,
                                      np.ndarray)) or o is None:
                        return o
                    if isinstance(o, dict):
                        return {k: _san(v) for k, v in o.items()}
                    if isinstance(o, (list, tuple)):
                        return type(o)(_san(v) for v in o)
                    return None  # callables / ScriptFunctions / etc.
                _payload = dict(
                    wgg=_san(wgg), xl_w=xl_w.detach(), xh_w=xh_w.detach(),
                    sb_b=_san(sb_b), ob_b=_san(ob_b), w_w=_san(w_w),
                    b_q=float(b_q), params_b=_san(params_b),
                    oc_b=_san(oc_b or None), rc_b=_san(rc_b or None), B=B)
                with open(_dump_walk, 'wb') as _df:
                    pickle.dump(_payload, _df)
                _walk_dumped[0] = True
                print(f'[dump-walk] wrote eval_only batch B={B} -> '
                      f'{_dump_walk}', flush=True)
            _walk = (_compiled_walk() if _compile_walk
                     else attn_crown_lb_batch)
            with torch.no_grad():
                lb = _walk(wgg, xl_w, xh_w, sb_b, ob_b,
                           w_w, float(b_q), params_b,
                           op_clamps_b=oc_b or None,
                           relu_clamps_b=rc_b or None)
            return lb, None
        opt = torch.optim.Adam(list(params_b.values()), lr=lr)
        best = torch.full((B,), -np.inf, device=device, dtype=wdt)
        for _ in range(n_iters):
            if time_left() <= 1.0:
                break
            opt.zero_grad()
            lb = attn_crown_lb_batch(wgg, xl_w, xh_w, sb_b, ob_b, w_w,
                                     float(b_q), params_b,
                                     op_clamps_b=oc_b or None,
                                     relu_clamps_b=rc_b or None)
            best = torch.maximum(best, lb.detach())
            if bool((best > 0).all()):
                break
            # closed domains stop contributing to the loss
            loss = -(lb.clamp(max=0.01)).sum()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for kk, p in params_b.items():
                    if kk[0] in ('beta', 'rbeta'):
                        p.clamp_(min=0)
                    else:
                        p.clamp_(0, 1)
        out_params = []
        for i in range(B):
            out_params.append({
                k: v[i].detach().to('cpu', torch.float32)
                for k, v in params_b.items()})
        if return_la:
            # (1b) per-domain beta-INCLUSIVE |lA| at each relu, captured from a
            # final no-grad eval at the optimized {alpha,beta} — ABC's BaBSR
            # weight. One extra walk per round (~1/12 of the Adam cost).
            with torch.no_grad():
                _lb2, ew_relu = attn_crown_lb_batch(
                    wgg, xl_w, xh_w, sb_b, ob_b, w_w, float(b_q), params_b,
                    op_clamps_b=oc_b or None, relu_clamps_b=rc_b or None,
                    return_ew=True)
            la_list = [{L: ew_relu[L][i].detach().to('cpu')
                        for L in ew_relu} for i in range(B)]
            return best, out_params, la_list
        return best, out_params

    heap = []
    cnt = 0
    n_domains = 1
    if not pick_candidates({}, 1):
        return False, 1, 'no_split'
    heapq.heappush(heap, (-0.0, cnt, {}, None, None))
    # HOT-SET warmup: the first `hot_warmup` rounds use a WIDE FSB (hot_kfsb) to
    # DISCOVER the high-value split neurons (FSB ranks by actual child-bound
    # improvement); after warmup, restrict candidates to that discovered set at
    # the cheap kfsb_k (the forced-neuron-set experiment showed kfsb=4 closes
    # fast once the candidate set is right). Combines wide-FSB accuracy (brief)
    # with restricted-set speed. Branching-only -> soundness-irrelevant.
    _hot = set(); _round = 0
    _prof = os.environ.get('VC_BAB_PROF')
    _compile_walk = bool(os.environ.get('VC_COMPILE_WALK'))
    _adapt_kfsb = float(os.environ.get('VC_ADAPT_KFSB', 0.0))
    _adapt_near_kk = int(os.environ.get('VC_ADAPT_NEAR_KK', 4))
    # (A) FSB-walk microbench dump: pickle the first steady-state eval_only
    # batch (B >= VC_DUMP_WALK_MINB) so the walk can be timed in isolation.
    _dump_walk = os.environ.get('VC_DUMP_WALK')
    _dump_walk_minB = int(os.environ.get('VC_DUMP_WALK_MINB', 1000))
    _walk_dumped = [False]
    # (B) scoring-fidelity dump: per round/domain, each FSB candidate's raw
    # score components (lo,hi,|lA|) + FSB ground-truth child-bounds, so any
    # cheap-score formula can be ranked against FSB's pick OFFLINE (jsonl,
    # incremental so a timeout-kill keeps what ran). Default off.
    _dump_fsb = os.environ.get('VC_DUMP_FSB')
    _t_fsb = _t_bnd = _t_ew = 0.0
    # (#2) deterministic spec-aware branching score, computed ONCE at the root
    # via the box-halfspace LP over CROWN backward forms; replaces the gap*ew_w
    # heuristic (which mis-ranks the spec-critical neurons -> kfsb chaos).
    _bh_w = None
    if bh_score:
        _t_bh0 = time.perf_counter()
        _bh_w = box_halfspace_branch_score(gg, xl, xh, sb_root, ob0,
                                           base_params, w_q, b_q)
        print(f'  [bh-score] computed root spec-halfspace scores for '
              f'{len(_bh_w)} relu layers ({time.perf_counter()-_t_bh0:.1f}s)',
              flush=True)
    # BBPS branching (ABC's ACTUAL vit heuristic, ported): root bound-improvement
    # score per neuron, used as the pick_candidates pre-filter weight (full_w).
    # Reproduces ABC's neuron set 18/20 vs BaBSR's 8/20 -> kfsb FSB then refines
    # the RIGHT candidates. Gated by attn_bab_bbps_score / VC_BBPS_BRANCH.
    if bbps_score or os.environ.get('VC_BBPS_BRANCH'):
        _t_bb0 = time.perf_counter()
        _bh_w = bbps_root_scores(gg, xl, xh, sb_root, ob0, base_params,
                                 w_q, b_q)
        # RESTRICT candidates to the top-K BBPS neurons per layer (the discovered
        # critical set) — mirrors the forced-20 experiment that closed in budget;
        # _bh_w still orders within the set. K from VC_BBPS_TOPK / attn_bab_bbps_topk.
        _topk = int(os.environ.get('VC_BBPS_TOPK', 0) or bbps_topk or 12)
        _force_set = set()
        for _L, _sc in _bh_w.items():
            _ni = _sc.numel()
            for _j in torch.topk(_sc, min(_topk, _ni)).indices.tolist():
                if float(_sc[_j]) > 0:
                    _force_set.add((_L, int(_j)))
        print(f'  [bbps-score] computed root BBPS scores for {len(_bh_w)} relu '
              f'layers, restricted to {len(_force_set)} neurons (top-{_topk}/layer) '
              f'({time.perf_counter()-_t_bb0:.1f}s)', flush=True)
    if os.environ.get('VC_BH_DEBUG'):
        # Does ROOT scoring match ABC's dynamic neuron set? Compare proper BaBSR
        # (|lA|*triangle-intercept) vs the default (|lA|*min(-lo,hi)) vs parts.
        _abc = {0: {326, 328, 570, 584, 646, 678, 725, 1005, 1033, 1093, 1513,
                    1558}, 1: {652, 1132, 1386, 1463}, 2: {983, 998, 1118, 1295}}
        for L, (lo, hi) in sb_root.items():
            uns = (lo < 0) & (hi > 0)
            if not bool(uns.any()) or L not in _abc:
                continue
            la = (ew_w[L] if (ew_w is not None and L in ew_w
                              and ew_w[L].numel() == lo.numel())
                  else torch.ones_like(lo))
            intercept = (-lo * hi) / (hi - lo).clamp(min=1e-12)
            mn = torch.minimum(-lo, hi)
            cands = {'babsr(|lA|*intercept)': la * intercept * uns,
                     'default(|lA|*min)': la * mn * uns,
                     '|lA|': la * uns, 'intercept': intercept * uns}
            for nm_, sc in cands.items():
                k = max(4, len(_abc[L]))
                top = set(torch.topk(sc, min(k, sc.numel())).indices.tolist())
                print(f'  [babsr-dbg] L{L} {nm_}: top{len(top)} ∩ ABC = '
                      f'{len(top & _abc[L])}/{len(_abc[L])}', flush=True)
    if os.environ.get('VC_BBPS_DEBUG'):
        # Does the ported BBPS score (ABC's ACTUAL vit heuristic) match ABC's
        # neuron set far better than BaBSR? Compare top-k ∩ ABC forced-20.
        _abc = {0: {326, 328, 570, 584, 646, 678, 725, 1005, 1033, 1093, 1513,
                    1558}, 1: {652, 1132, 1386, 1463}, 2: {983, 998, 1118, 1295}}
        _t_b = time.perf_counter()
        _bbps = bbps_root_scores(gg, xl, xh, sb_root, ob0, base_params, w_q, b_q)
        print(f'  [bbps-dbg] computed BBPS root scores '
              f'({time.perf_counter()-_t_b:.1f}s)', flush=True)
        _tot = _hit = 0
        for L in sorted(_abc):
            if L not in _bbps:
                continue
            sc = _bbps[L]
            k = len(_abc[L])
            top = set(torch.topk(sc, min(k, sc.numel())).indices.tolist())
            _hit += len(top & _abc[L]); _tot += len(_abc[L])
            print(f'  [bbps-dbg] L{L} BBPS: top{len(top)} ∩ ABC = '
                  f'{len(top & _abc[L])}/{len(_abc[L])}  (max={float(sc.max()):.4g})',
                  flush=True)
        print(f'  [bbps-dbg] TOTAL ∩ ABC = {_hit}/{_tot} '
              f'(BaBSR baseline was 8/20)', flush=True)
    while heap:
        if time_left() <= 2.0:
            return False, n_domains, 'time'
        if n_domains >= max_domains:
            return False, n_domains, 'cap'
        _round += 1
        _in_warmup = bool(hot_warmup) and _round <= hot_warmup
        _kk = hot_kfsb if _in_warmup else kfsb_k
        _allow = (_hot if (hot_warmup and not _in_warmup and _hot) else None)
        # pop up to batch//2 worst domains; kfsb-pick each one's split
        popped = []; popped_la = []; popped_lb = []
        while heap and (len(popped) + 1) * 2 <= batch:
            lb_p, _, clamps, wp, la = heapq.heappop(heap)
            popped.append((clamps, wp)); popped_la.append(la)
            popped_lb.append(lb_p)
        cand_lists = []
        # per-domain backward sensitivity for this round's popped domains
        # (one batched no-beta walk); the BaBSR weight (ABC-style), replacing
        # the static root ew_w. Gated by attn_bab_perdom_ew (default on).
        _te = time.perf_counter()
        _dom_ews = (batch_dom_ew([c for c, _ in popped])
                    if perdom_ew else None)
        _t_ew += time.perf_counter() - _te
        for _di, (clamps, wp) in enumerate(popped):
            _dew = ({L: _dom_ews[L][_di] for L in _dom_ews}
                    if _dom_ews is not None else None)
            # (1b) per-domain beta-inclusive |lA| stored when this domain was
            # bounded (root domain has None -> falls back to ew_w).
            _dla = popped_la[_di] if perdom_la else None
            # ADAPTIVE kfsb: wide FSB only for domains still FAR from closing
            # (lb < -thr); near-closing domains get a cheap kfsb (one more split
            # usually closes them). Cuts the FSB cost in the convergence tail,
            # where most domains live, without touching the far-domain branching
            # quality that the wide FSB is load-bearing for. Branching-only ->
            # soundness-irrelevant. Gated by VC_ADAPT_KFSB=<thr> (default off).
            _kk_i = _kk
            if _adapt_kfsb and not _in_warmup and popped_lb[_di] >= -_adapt_kfsb:
                _kk_i = min(_kk, _adapt_near_kk)
            cands = pick_candidates(clamps, _kk_i, _dew, _allow, _bh_w, _dla)
            if not cands:
                return False, n_domains, 'exhausted'
            cand_lists.append(cands)
        if _kk > 1 and any(len(c) > 1 for c in cand_lists):
            # FSB: one cheap batched eval of every candidate's two
            # children (parent params, no new betas — ranks the plane /
            # clamp effect; the chosen split still gets the full beta
            # treatment below), pick max(min(child0, child1)).
            rows = []
            for (clamps, wp), cands in zip(popped, cand_lists):
                for sp in cands:
                    rows.append((child_clamps(clamps, sp, 0), wp))
                    rows.append((child_clamps(clamps, sp, 1), wp))
            _tf = time.perf_counter()
            lbs_c, _ = bound_batch(rows, eval_only=True)
            _t_fsb += time.perf_counter() - _tf
            chosen = []
            r = 0
            _fsb_recs = [] if _dump_fsb else None
            for _di2, ((clamps, wp), cands) in enumerate(zip(popped,
                                                             cand_lists)):
                best_sp = None; best_m = -np.inf
                _crec = [] if _dump_fsb else None
                _sb_d = dom_sb(clamps) if _dump_fsb else None
                _dew2 = ({L: _dom_ews[L][_di2] for L in _dom_ews}
                         if (_dump_fsb and _dom_ews is not None) else None)
                _dla2 = popped_la[_di2] if (_dump_fsb and perdom_la) else None
                for sp in cands:
                    c0 = float(lbs_c[r]); c1 = float(lbs_c[r + 1])
                    m = min(c0, c1)
                    r += 2
                    if m > best_m:
                        best_m = m; best_sp = sp
                    if _dump_fsb and sp[0] == 'relu':
                        _L, _j = sp[1], sp[2]
                        _lo, _hi = _sb_d[_L]
                        _ew = (float(_dew2[_L][_j])
                               if (_dew2 is not None and _L in _dew2) else None)
                        _la = (float(_dla2[_L][_j])
                               if (_dla2 is not None and _L in _dla2
                                   and _dla2[_L].numel() > _j) else None)
                        # [L, j, lo, hi, |lA|_nobeta, |lA|_beta, fsb_c0, fsb_c1]
                        _crec.append([int(_L), int(_j), float(_lo[_j]),
                                      float(_hi[_j]), _ew, _la, c0, c1])
                chosen.append(best_sp)
                if _dump_fsb:
                    _chosen_idx = next(
                        (k for k, sp in enumerate(cands) if sp is best_sp), -1)
                    _fsb_recs.append({'lb': float(popped_lb[_di2]),
                                      'cands': _crec,
                                      'chosen': _chosen_idx})
            if _dump_fsb:
                import json
                with open(_dump_fsb, 'a') as _ff:
                    _ff.write(json.dumps({'round': _round,
                                          'doms': _fsb_recs}) + '\n')
        else:
            chosen = [c[0] for c in cand_lists]
        if _in_warmup:
            for sp in chosen:
                if sp is not None and sp[0] == 'relu':
                    _hot.add((sp[1], sp[2]))
        if os.environ.get('VC_DUMP_SPLITS'):
            with open(os.environ['VC_DUMP_SPLITS'], 'a') as _f:
                _f.write(repr(chosen) + '\n')
        children = []
        for (clamps, wp), sp in zip(popped, chosen):
            for side in (0, 1):
                children.append((child_clamps(clamps, sp, side), wp))
        _tb = time.perf_counter()
        if perdom_la:
            lbs, out_params, la_list = bound_batch(children, return_la=True)
        else:
            lbs, out_params = bound_batch(children); la_list = None
        _t_bnd += time.perf_counter() - _tb
        if _prof and _round % 20 == 0:
            print(f'    [bab-prof] round {_round} dom {n_domains}: '
                  f'fsb {_t_fsb:.1f}s bound {_t_bnd:.1f}s ew {_t_ew:.1f}s',
                  flush=True)
        n_domains += len(children)
        for i, (c2, _) in enumerate(children):
            lb_i = float(lbs[i])
            if lb_i > 0 and (wdt != dtype or wgg is not gg):
                # low-precision closure: certify in full precision
                lb_i = recheck_fp64(c2, out_params[i])
            if lb_i > 0:
                continue
            cnt += 1
            heapq.heappush(heap, (lb_i, cnt, c2, out_params[i],
                                  la_list[i] if la_list is not None else None))
        if print_progress and n_domains % 256 < batch:
            worst = heap[0][0] if heap else 0.0
            print(f'    [beta-bab] {n_domains} domains, frontier '
                  f'{len(heap)}, worst lb {worst:+.4f}', flush=True)
    return True, n_domains, 'closed'


def attn_refine_op_bounds(gg, xl, xh, sb, ob, *, params=None, chunk=512,
                          time_left=None, passes=1):
    """CROWN intermediate-bound refinement (ABC's backward intermediate
    bounds, the `forward_before_compute_bounds`-quality pass).

    For every producer node whose bounds feed a recorded relaxation
    (exp/reciprocal inputs, bilinear input sides, pre-relu rows),
    recompute per-coordinate bounds with the batched backward walk
    seeded at that node with ±I rows, and intersect into `ob` / `sb`
    IN PLACE. Topological order, so later nodes' refinements reuse
    earlier ones; `passes` > 1 re-runs the sweep (upstream tightening
    feeds back into downstream planes).

    Sound: each refined value is a backward-CROWN bound on the node's
    TRUE values (same plane validity invariant as the spec walk),
    intersected with the existing forward enclosure — both enclose the
    true set. The zonotope forward's ranges for bilinear-remainder
    boxes are NOT touched (those live in the forward, not in ob).

    Returns the number of (node, coordinate) bounds that tightened.
    """
    device = xl.device
    dtype = xl.dtype
    params = params or {}
    ops = gg['ops']
    op_by_name = {op['name']: op for op in ops}
    input_name = gg['input_name']

    def node_bounds(name, n):
        """(lo, hi) of op `name`'s output via ±I backward, chunked."""
        los = []; his = []
        eye = torch.eye(n, device=device, dtype=dtype)
        for s0 in range(0, n, chunk):
            rows = eye[s0:s0 + chunk]
            B = rows.shape[0]
            sb_b = {L: (lo.unsqueeze(0).expand(B, -1),
                        hi.unsqueeze(0).expand(B, -1))
                    for L, (lo, hi) in sb.items()}
            ob_b = {}
            for k, v in ob.items():
                if isinstance(v[0], tuple):
                    ob_b[k] = ((v[0][0].unsqueeze(0).expand(B, -1),
                                v[0][1].unsqueeze(0).expand(B, -1)),
                               (v[1][0].unsqueeze(0).expand(B, -1),
                                v[1][1].unsqueeze(0).expand(B, -1)))
                else:
                    ob_b[k] = (v[0].unsqueeze(0).expand(B, -1),
                               v[1].unsqueeze(0).expand(B, -1))
            params_b = {k: v.detach().unsqueeze(0).expand(B, *v.shape)
                        for k, v in params.items()
                        if k[0] not in ('beta', 'rbeta')}
            with torch.no_grad():
                lo_r = attn_crown_lb_batch(
                    gg, xl, xh, sb_b, ob_b, None, 0.0, params_b,
                    start_name=name, ew0=rows)
                hi_r = -attn_crown_lb_batch(
                    gg, xl, xh, sb_b, ob_b, None, 0.0, params_b,
                    start_name=name, ew0=-rows)
            los.append(lo_r); his.append(hi_r)
        return torch.cat(los), torch.cat(his)

    n_tight = 0
    for _ in range(passes):
        cache = {}

        def refined(name, n):
            nonlocal n_tight
            if name == input_name or name not in op_by_name:
                return None
            if name in cache:
                return cache[name]
            if time_left is not None and time_left() <= 2.0:
                return None
            lo_r, hi_r = node_bounds(name, n)
            cache[name] = (lo_r, hi_r)
            return cache[name]

        def tighten(pair, name):
            nonlocal n_tight
            lo, hi = pair
            r = refined(name, lo.numel())
            if r is None:
                return pair
            lo_r, hi_r = r
            lo2 = torch.maximum(lo, lo_r)
            hi2 = torch.minimum(hi, hi_r)
            hi2 = torch.maximum(hi2, lo2)
            n_tight += int(((lo2 > lo + 1e-12) | (hi2 < hi - 1e-12))
                           .sum())
            return (lo2, hi2)

        for op in ops:
            t = op['type']; name = op['name']
            if t in ('exp', 'reciprocal') and name in ob:
                ob[name] = tighten(ob[name], op['inputs'][0])
            elif t in ('mul_bilinear', 'matmul_bilinear') and name in ob:
                (pa, pb) = ob[name]
                pa = tighten(pa, op['inputs'][0])
                pb = tighten(pb, op['inputs'][1])
                ob[name] = (pa, pb)
            elif t == 'relu' and op.get('layer_idx') in sb:
                L = op['layer_idx']
                sb[L] = tighten(sb[L], op['inputs'][0])
    return n_tight


def _cuda_free_bytes(device):
    """Bytes this process can still allocate on `device`: driver-free
    memory plus our caching allocator's reserved-but-unallocated pool
    (cached blocks are reusable by new allocations)."""
    free_b, _total = torch.cuda.mem_get_info(device)
    return int(free_b + torch.cuda.memory_reserved(device)
               - torch.cuda.memory_allocated(device))


def _mem_row_cap(free_bytes, bytes_per_row, *, safety=0.6,
                 min_rows=256):
    """Differentiable-row budget that fits in `free_bytes`.

    `bytes_per_row` is the measured live autograd-graph cost of one
    ±I walk row (`_probe_row_bytes`). `safety` reserves headroom for
    everything the probe does not hold live: backward-pass gradients,
    the per-row alpha parameters with their Adam moments, the Q-row
    spec walk and allocator fragmentation. Returns None when the
    reader saw no allocation growth (no cap derivable). `min_rows`
    keeps a degenerate measurement from zeroing the budget (the trim
    below always keeps at least one target regardless)."""
    if bytes_per_row <= 0:
        return None
    return max(min_rows, int(free_bytes * safety / bytes_per_row))


def _probe_row_bytes(gg, xl, xh, sb_w, ob_w, base_p, prod, n, device,
                     wdt, used_bytes, n_probe=64):
    """Measured live-memory bytes per differentiable ±I walk row.

    Runs the first min(n_probe, n) rows of `prod`'s lower+upper
    backward walks with grad-requiring params — the exact state
    attn_alpha_joint keeps alive for EVERY kept row until backward —
    and returns the growth of `used_bytes()` per row while both
    graphs are live. `prod` should be the LATEST target: its walk is
    the longest, so the estimate is conservative for earlier targets;
    per-walk constant overhead amortized over the probe rows only
    rounds the estimate up."""
    B = min(n_probe, n)
    rows = torch.eye(n, device=device, dtype=wdt)[:B]
    sb_b = {L: (lo.unsqueeze(0).expand(B, -1),
                hi.unsqueeze(0).expand(B, -1))
            for L, (lo, hi) in sb_w.items()}
    ob_b = {}
    for k, v in ob_w.items():
        if isinstance(v[0], tuple):
            ob_b[k] = ((v[0][0].unsqueeze(0).expand(B, -1),
                        v[0][1].unsqueeze(0).expand(B, -1)),
                       (v[1][0].unsqueeze(0).expand(B, -1),
                        v[1][1].unsqueeze(0).expand(B, -1)))
        else:
            ob_b[k] = (v[0].unsqueeze(0).expand(B, -1),
                       v[1].unsqueeze(0).expand(B, -1))
    p_leaf = {k: v.detach().clone().requires_grad_(True)
              for k, v in base_p.items()
              if k[0] not in ('beta', 'rbeta')}
    p_b = {k: v.unsqueeze(0).expand(B, *v.shape)
           for k, v in p_leaf.items()}
    m0 = used_bytes()
    with torch.enable_grad():
        lo_r = attn_crown_lb_batch(gg, xl, xh, sb_b, ob_b, None, 0.0,
                                   p_b, start_name=prod, ew0=rows)
        hi_r = -attn_crown_lb_batch(gg, xl, xh, sb_b, ob_b, None, 0.0,
                                    p_b, start_name=prod, ew0=-rows)
        m1 = used_bytes()
    del lo_r, hi_r
    return (m1 - m0) / B


def attn_alpha_joint(gg, xl, xh, sb, ob, W_q, b_vec, *,
                     n_iters=50, lr=0.25, lr_decay=0.98,
                     time_left=None, params=None,
                     refine_chunk=1024, gg_work=None, work_dtype=None,
                     max_rows=4096, per_start_alpha=False,
                     per_row_alpha=True, per_row_rows=1024,
                     freeze_tol=0.0, freeze_patience=2,
                     refresh_every=1, freeze_refresh=8,
                     sign_jump_passes=5, sign_jump_beta=0.6,
                     mem_fns=None, sparse_rows=0):
    """JOINT alpha over all open queries with DIFFERENTIABLE
    intermediate bounds — the alpha,beta-CROWN mechanism that closes
    vit pgd instances in ~2 gradient steps (their `sparse_interm:
    false` + per-start-node alphas; measured on pgd_2_3_16_2043:
    intermediate widths shrink 30-38% in 2 iterations and the worst
    spec margin moves -1.31 -> +0.035).

    W_q (Q, n_out), b_vec (Q,): all open queries at once (ABC's joint
    C matrix). Each iteration: (1) re-derive the bounds of every
    relaxation-feeding intermediate node via the BATCHED backward walk
    (±I rows) with the CURRENT plane params, intersected with the
    frozen forward enclosure — inside the autograd graph, so the spec
    loss backprops into the planes of the intermediate walks; (2)
    evaluate all Q spec rows against the dynamic bounds; (3) Adam on
    loss = -(sum of unverified margins).

    Soundness: every iterate's bounds are valid CROWN bounds for ANY
    params in range (same plane-validity argument), intersected with
    the frozen enclosure — so the per-iteration best-of per query is
    sound. When `work_dtype` (e.g. fp32) is given, the SEARCH runs in
    it on gg_work and the returned best bounds come from a final
    full-precision re-evaluation of the best params on `gg` —
    certification never rests on low precision.

    freeze_tol/freeze_patience: adaptive target freezing — a target
    whose re-derived widths change < freeze_tol (relative, per
    coordinate) for freeze_patience consecutive iterations stops being
    re-derived; its last raw walk bounds are reused as constants
    (detached). Sound: each cached bound is a valid CROWN enclosure
    (any params in range), and the final certification pass below
    re-derives EVERY target in full precision regardless of freezing.
    Measured (pgd nets): most of the ~14 targets stop moving by
    iteration ~3 while the walks dominate the iteration cost — this is
    what buys the ~50 iterations the pgd family needs inside the wall.
    refresh_every=K (>1): targets WITHOUT per-row alphas that are not
    yet frozen are only re-derived every K-th iteration (off-iterations
    reuse the last raw bounds — same soundness argument).
    freeze_refresh=K (>0): frozen targets are re-derived every K-th
    iteration anyway — the cache is updated (and the target unfrozen if
    its widths actually moved past freeze_tol). Bounds staleness is
    therefore <= K iterations, which keeps the search close to the
    full re-derivation trajectory AND bounds the drift between the
    frozen search state and the final full re-derivation cert pass
    (measured without it on pgd_7209: freeze-at-iter-3 quality capped
    the joint worst at -0.18 vs -0.117 with full re-derivation).

    mem_fns: optional (free_bytes_fn, used_bytes_fn) pair backing the
    memory-adaptive differentiable-row cap (see the probe below);
    defaults to the torch.cuda readers on a CUDA device, disabled on
    CPU. Injectable so the cap logic is unit-testable host-side.

    Returns (best (Q,) fp64 ndarray, params, ob_cert, sb_cert) where
    ob_cert/sb_cert are the full-precision certified dynamic bounds
    (valid enclosures; caller may intersect them into shared state).
    """
    device = xl.device
    dtype = xl.dtype
    wdt = work_dtype or dtype
    wgg = gg_work if gg_work is not None else gg
    xl_w = xl.to(wdt); xh_w = xh.to(wdt)
    W_t = W_q if torch.is_tensor(W_q) else torch.as_tensor(
        np.asarray(W_q, np.float64), device=device, dtype=dtype)
    b_t = torch.as_tensor(np.asarray(b_vec, np.float64), device=device,
                          dtype=dtype)
    Q = W_t.shape[0]
    if params is None:
        params = init_params(gg, sb, ob, device, dtype)
    # PER-START-NODE alphas (the ABC default; their vit hook shares
    # alphas only to save memory on >=6-matmul nets): the optimal plane
    # for bounding intermediate node X differs from the spec-optimal
    # plane — sharing one set forces a compromise (measured: shared-set
    # joint alpha stalls at -0.15 on pgd_7209 where ABC reaches -0.015
    # from a worse start). Each start (the spec walk + every refine
    # target) gets an independent copy; all train jointly.
    base_p = {k: v.detach().to(wdt) for k, v in params.items()}
    params = None    # rebuilt per start below

    # relaxation-feeding producers, topological order (the refine set)
    op_by_name = {op['name']: op for op in gg['ops']}
    targets = []   # (producer_name, kind, key, side)
    seen = set()
    for op in gg['ops']:
        t = op['type']; name = op['name']
        if t in ('exp', 'reciprocal') and name in ob:
            tgt = (op['inputs'][0], 'ob1', name, None)
        elif t in ('mul_bilinear', 'matmul_bilinear') and name in ob:
            for side in (0, 1):
                tgt = (op['inputs'][side], 'ob2', name, side)
                if tgt[0] in op_by_name and tgt[0] not in seen:
                    targets.append(tgt); seen.add(tgt[0])
            continue
        elif t == 'relu' and op.get('layer_idx') in sb:
            tgt = (op['inputs'][0], 'sb', op['layer_idx'], None)
        else:
            continue
        if tgt[0] in op_by_name and tgt[0] not in seen:
            targets.append(tgt); seen.add(tgt[0])
    # producer -> all (kind, key, side) consumers needing its bounds
    consumers = {}
    for op in gg['ops']:
        t = op['type']; name = op['name']
        if t in ('exp', 'reciprocal') and name in ob:
            consumers.setdefault(op['inputs'][0], []).append(
                ('ob1', name, None))
        elif t in ('mul_bilinear', 'matmul_bilinear') and name in ob:
            for side in (0, 1):
                consumers.setdefault(op['inputs'][side], []).append(
                    ('ob2', name, side))
        elif t == 'relu' and op.get('layer_idx') in sb:
            consumers.setdefault(op['inputs'][0], []).append(
                ('sb', op['layer_idx'], None))

    sb_frozen = {L: (lo.detach().to(wdt), hi.detach().to(wdt))
                 for L, (lo, hi) in sb.items()}
    ob_frozen = {}
    for k, v in ob.items():
        if isinstance(v[0], tuple):
            ob_frozen[k] = ((v[0][0].detach().to(wdt),
                             v[0][1].detach().to(wdt)),
                            (v[1][0].detach().to(wdt),
                             v[1][1].detach().to(wdt)))
        else:
            ob_frozen[k] = (v[0].detach().to(wdt),
                            v[1].detach().to(wdt))

    # Memory cap: every differentiable ±I row keeps its walk's autograd
    # graph alive until backward — the 3-block ibp net's full target set
    # (~12k rows) OOMs a 22 GB GPU. Trim the EARLIEST targets first:
    # measured on the ABC trace, late-block intermediate bounds carry
    # the gradient value (block-1/QKV bounds barely move), and trimmed
    # targets keep their frozen (still sound) enclosures.
    def _t_width(tgt):
        for kind, key, side in consumers[tgt[0]]:
            if kind == 'ob1':
                return ob[key][0].numel()
            if kind == 'ob2':
                return ob[key][side][0].numel()
            return sb[key][0].numel()
        return 0
    total_rows = sum(_t_width(t) for t in targets)
    # MEMORY-ADAPTIVE row cap (A10G incident, ibp_3_3_8 vnnlib #253):
    # the fixed 4096-row cap kept 3366 rows whose live walk graphs
    # peaked at 21.5/22.06 GB — zero headroom; the one case with TWO
    # open queries (slightly larger baseline + spec walk) OOMed 34 MiB
    # short at iteration 1 while the 56 single-query siblings survived
    # at the knife edge. Measure the real per-row autograd cost of
    # THIS net with a small probe walk, then keep only the rows whose
    # projected peak fits in the currently-free memory with margin.
    # `mem_fns=(free_bytes_fn, used_bytes_fn)` is injectable for
    # CPU-side tests; defaults to the CUDA readers on a CUDA device
    # and stays off on CPU (host RAM is not the constraint here).
    if mem_fns is None and device.type == 'cuda':
        mem_fns = (partial(_cuda_free_bytes, device),
                   partial(torch.cuda.memory_allocated, device))
    if mem_fns is not None and targets and not sparse_rows:
        bpr = _probe_row_bytes(wgg, xl_w, xh_w, sb_frozen, ob_frozen,
                               base_p, targets[-1][0],
                               _t_width(targets[-1]), device, wdt,
                               mem_fns[1])
        cap = _mem_row_cap(mem_fns[0](), bpr)
        if cap is not None and (not max_rows or cap < max_rows):
            print(f'  [joint-alpha] memory cap: {bpr / 1e6:.2f} MB/row '
                  f'measured -> {cap} differentiable rows affordable',
                  flush=True)
            max_rows = cap
    if max_rows and total_rows > max_rows and not sparse_rows:
        kept = []
        acc_rows = 0
        for t in reversed(targets):
            w_t_ = _t_width(t)
            if acc_rows + w_t_ > max_rows and kept:
                break
            kept.append(t); acc_rows += w_t_
        print(f'  [joint-alpha] target cap: {len(kept)}/{len(targets)} '
              f'latest nodes ({acc_rows}/{total_rows} rows) kept '
              f'differentiable', flush=True)
        targets = list(reversed(kept))

    # PER-ROW alphas for the intermediate ±I walks (the localized ABC
    # mechanism: transplanting their per-coordinate boxes into OUR spec
    # walk reproduced their bound exactly; the gap is that a shared
    # alpha set averages gradients across all rows of an intermediate
    # and sacrifices the spec-critical coordinates). Index 0 = lower
    # (+I) walk, 1 = upper (-I) walk — decoupled like ABC's 4-slice.
    prow = {}
    if per_row_alpha and not sparse_rows:
        # per-row autograd is the memory driver (each row keeps its
        # walk graph + its own param slices): only the LATEST targets
        # up to per_row_rows get per-row alphas (the localization put
        # ~80% of the gap in the last pre-relu + last attn@V pair);
        # earlier targets fall back to the shared set.
        # net-adaptive: small nets (pgd ~2.9k rows) take per-row alphas
        # on EVERY target (the configuration that closes 8836); the cap
        # only bites where full per-row OOMs (ibp ~20k rows)
        _pr_budget = (total_rows if total_rows <= 3500
                      else int(per_row_rows))
        _pr_set = set()
        for tgt in reversed(targets):
            w_t_ = _t_width(tgt)
            if w_t_ <= _pr_budget:
                _pr_set.add(tgt[0]); _pr_budget -= w_t_
        for tgt in targets:
            if tgt[0] not in _pr_set:
                continue
            prod = tgt[0]
            n_r = None
            for kind, key, side in consumers[prod]:
                if kind == 'ob1':
                    n_r = ob[key][0].numel()
                elif kind == 'ob2':
                    n_r = ob[key][side][0].numel()
                else:
                    n_r = sb[key][0].numel()
                break
            prow[prod] = {
                k: v.detach().to(wdt).unsqueeze(0).unsqueeze(0)
                    .expand(2, n_r, *v.shape).clone()
                    .requires_grad_(True)
                for k, v in base_p.items()
                if k[0] not in ('beta', 'rbeta')}

    pstart = {'__spec__': {k: v.clone().requires_grad_(True)
                           for k, v in base_p.items()}}
    for tgt in targets:
        # per-start sets are ABC's default but cost ~(n_targets+1)x per
        # iteration; at fixed wall the shared set converges further
        # (measured on pgd_7209: -0.179 shared vs -0.203 per-start)
        pstart[tgt[0]] = ({k: v.clone().requires_grad_(True)
                           for k, v in base_p.items()}
                          if per_start_alpha else pstart['__spec__'])

    def dynamic_bounds(p, use_gg, use_xl, use_xh, grad, prow_use=None,
                       reuse=None, raw_out=None):
        """One sweep of differentiable intermediate re-derivation.
        Returns (sb_dyn, ob_dyn) for the given params.

        reuse: {prod: (lo_r, hi_r)} raw walk bounds (detached) used in
        place of the walk for frozen / off-refresh targets (sound:
        each is a valid CROWN enclosure from an earlier iteration).
        raw_out: dict collecting the detached raw walk bounds of every
        target actually re-derived this sweep."""
        sb_dyn = dict(sb_frozen) if use_gg is wgg else {
            L: (lo.detach(), hi.detach()) for L, (lo, hi) in sb.items()}
        ob_dyn = dict(ob_frozen) if use_gg is wgg else {
            k: v for k, v in ob.items()}
        for prod, _, _, _ in targets:
            if reuse is not None and prod in reuse:
                lo_r, hi_r = reuse[prod]
                for kind, key, side in consumers[prod]:
                    if kind == 'ob1':
                        lo0, hi0 = ob_dyn[key]
                    elif kind == 'ob2':
                        lo0, hi0 = ob_dyn[key][side]
                    else:
                        lo0, hi0 = sb_dyn[key]
                    lo2 = torch.maximum(lo0, lo_r)
                    hi2 = torch.minimum(hi0, hi_r)
                    hi2 = torch.maximum(hi2, lo2)
                    if kind == 'ob1':
                        ob_dyn[key] = (lo2, hi2)
                    elif kind == 'ob2':
                        pa, pb = ob_dyn[key]
                        ob_dyn[key] = ((lo2, hi2), pb) if side == 0 \
                            else (pa, (lo2, hi2))
                    else:
                        sb_dyn[key] = (lo2, hi2)
                continue
            n = None
            for kind, key, side in consumers[prod]:
                if kind == 'ob1':
                    n = ob_dyn[key][0].numel()
                elif kind == 'ob2':
                    n = ob_dyn[key][side][0].numel()
                else:
                    n = sb_dyn[key][0].numel()
                break
            _d0 = next(iter(p.values()), None)
            if isinstance(_d0, dict):
                _d0 = next(iter(_d0.values()), None)
            eye = torch.eye(n, device=device,
                            dtype=_d0.dtype if _d0 is not None else wdt)
            # SPARSE intermediate refinement: differentiably re-derive only the
            # K WIDEST rows of this node (the loosest coords, most to gain) and
            # keep the frozen forward enclosure for the rest. Sound: un-walked
            # rows retain a valid CROWN enclosure; walked rows are intersected
            # as before. Lets EVERY target be refined within memory (the full
            # +-I walk of all nodes OOMs -> only ~2/27 nodes fit otherwise).
            # Node-level (pstart) alphas only (prow is empty when sparse_rows>0).
            sel = None
            if sparse_rows and n > sparse_rows:
                # base = this node's CURRENT dynamic enclosure (right precision
                # for the search vs fp64 cert pass, and a valid CROWN bound);
                # the un-walked rows keep it (intersection no-op).
                _kc, _kk, _ks = consumers[prod][0]
                if _kc == 'ob1':
                    _fl, _fh = ob_dyn[_kk]
                elif _kc == 'ob2':
                    _fl, _fh = ob_dyn[_kk][_ks]
                else:
                    _fl, _fh = sb_dyn[_kk]
                sel = torch.sort(torch.topk(
                    (_fh - _fl).detach(), int(sparse_rows)).indices).values
                walk_eye = eye[sel]
                base_lo, base_hi = _fl.detach(), _fh.detach()
            else:
                walk_eye = eye
            los = []; his = []
            for s0 in range(0, walk_eye.shape[0], refine_chunk):
                rows = walk_eye[s0:s0 + refine_chunk]
                B = rows.shape[0]
                sb_b = {L: (lo.unsqueeze(0).expand(B, -1),
                            hi.unsqueeze(0).expand(B, -1))
                        for L, (lo, hi) in sb_dyn.items()}
                ob_b = {}
                for k2, v2 in ob_dyn.items():
                    if isinstance(v2[0], tuple):
                        ob_b[k2] = (
                            (v2[0][0].unsqueeze(0).expand(B, -1),
                             v2[0][1].unsqueeze(0).expand(B, -1)),
                            (v2[1][0].unsqueeze(0).expand(B, -1),
                             v2[1][1].unsqueeze(0).expand(B, -1)))
                    else:
                        ob_b[k2] = (v2[0].unsqueeze(0).expand(B, -1),
                                    v2[1].unsqueeze(0).expand(B, -1))
                p_src = p[prod] if isinstance(
                    next(iter(p.values()), None), dict) else p
                pr = prow_use.get(prod) if prow_use else None
                if pr is not None:
                    p_lo = {k2: v2[0, s0:s0 + B] for k2, v2 in pr.items()}
                    p_hi = {k2: v2[1, s0:s0 + B] for k2, v2 in pr.items()}
                else:
                    p_b = {k2: v2.unsqueeze(0).expand(B, *v2.shape)
                           for k2, v2 in p_src.items()
                           if k2[0] not in ('beta', 'rbeta')}
                    p_lo = p_hi = p_b
                ctx = torch.enable_grad() if grad else torch.no_grad()
                with ctx:
                    lo_r = attn_crown_lb_batch(
                        use_gg, use_xl, use_xh, sb_b, ob_b, None, 0.0,
                        p_lo, start_name=prod, ew0=rows)
                    hi_r = -attn_crown_lb_batch(
                        use_gg, use_xl, use_xh, sb_b, ob_b, None, 0.0,
                        p_hi, start_name=prod, ew0=-rows)
                los.append(lo_r); his.append(hi_r)
            lo_w = torch.cat(los); hi_w = torch.cat(his)
            if sel is not None:
                # scatter the K walked rows into the frozen full-length bound;
                # un-walked rows keep the (detached) frozen enclosure -> their
                # intersection below is a no-op (no spurious tightening).
                lo_r = base_lo.clone().index_copy(0, sel, lo_w)
                hi_r = base_hi.clone().index_copy(0, sel, hi_w)
            else:
                lo_r, hi_r = lo_w, hi_w
            if raw_out is not None:
                raw_out[prod] = (lo_r.detach(), hi_r.detach())
            for kind, key, side in consumers[prod]:
                if kind == 'ob1':
                    lo0, hi0 = ob_dyn[key]
                elif kind == 'ob2':
                    lo0, hi0 = ob_dyn[key][side]
                else:
                    lo0, hi0 = sb_dyn[key]
                # Straight-through intersection: VALUE = intersected
                # (tightest sound box, recip domain protected), GRADIENT
                # = raw walk bounds. torch.maximum alone routes the
                # subgradient to the winning arg, so coordinates where
                # the frozen box wins get ZERO gradient into their
                # alphas permanently — measured as a hard plateau
                # (-0.187 at 400 iters on pgd_7209) while ABC, which
                # replaces instead of intersecting, decays to -0.015.
                if lo_r.requires_grad:
                    lo2 = lo_r + (torch.maximum(lo0, lo_r)
                                  - lo_r).detach()
                    hi2 = hi_r + (torch.minimum(hi0, hi_r)
                                  - hi_r).detach()
                    hi2 = hi2 + (torch.maximum(
                        hi2.detach(), lo2.detach()) - hi2.detach())
                else:
                    lo2 = torch.maximum(lo0, lo_r)
                    hi2 = torch.minimum(hi0, hi_r)
                    hi2 = torch.maximum(hi2, lo2)
                if kind == 'ob1':
                    ob_dyn[key] = (lo2, hi2)
                elif kind == 'ob2':
                    pa, pb = ob_dyn[key]
                    ob_dyn[key] = ((lo2, hi2), pb) if side == 0 \
                        else (pa, (lo2, hi2))
                else:
                    sb_dyn[key] = (lo2, hi2)
        return sb_dyn, ob_dyn

    W_w = W_t.to(wdt)
    all_leaves = list({id(v): v for d in pstart.values()
                       for v in d.values()}.values())
    all_leaves += [v for d in prow.values() for v in d.values()]
    opt = torch.optim.Adam(all_leaves, lr=lr)
    # ABC's alpha schedule: aggressive start + exponential decay
    # (lr_alpha 0.5, decay ~0.98). Ablation-measured: their pgd-family
    # closes NEED ~50 alpha iterations (iteration=2 flips unsat ->
    # timeout); the schedule is what makes 50 iterations converge.
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=lr_decay)
    best = torch.full((Q,), -np.inf, device=device, dtype=wdt)
    best_params = {sk: {k: v.detach().clone() for k, v in d.items()}
                   for sk, d in pstart.items()}
    best_prow = {sk: {k: v.detach().clone() for k, v in d.items()}
                 for sk, d in prow.items()}
    best_worst = -np.inf
    _t_joint0 = time.perf_counter()
    # adaptive target freezing state (see docstring)
    frozen = {}      # prod -> (lo_r, hi_r) detached raw walk bounds
    streak = {}      # prod -> consecutive small-width-change count
    prev_w = {}      # prod -> last raw width tensor
    last_raw = {}    # prod -> last raw walk bounds (refresh reuse)
    n_done = 0
    for it in range(n_iters):
        if time_left is not None and time_left() <= 3.0:
            break
        refresh_now = (freeze_refresh > 0 and it > 0
                       and it % freeze_refresh == 0)
        reuse = {} if refresh_now else dict(frozen)
        if refresh_every > 1 and it % refresh_every != 0:
            for tgt in targets:
                prod = tgt[0]
                if (prod not in reuse and prod not in prow
                        and prod in last_raw):
                    reuse[prod] = last_raw[prod]
        raw = {}
        opt.zero_grad()
        sb_dyn, ob_dyn = dynamic_bounds(pstart, wgg, xl_w, xh_w, True,
                                        prow_use=prow, reuse=reuse,
                                        raw_out=raw)
        n_done = it + 1
        with torch.no_grad():
            for prod, (lo_r, hi_r) in raw.items():
                last_raw[prod] = (lo_r, hi_r)
                w_r = hi_r - lo_r
                wp = prev_w.get(prod)
                if wp is not None:
                    rel = float(((w_r - wp).abs()
                                 / wp.abs().clamp(min=1e-8)).max())
                    if rel < freeze_tol:
                        streak[prod] = streak.get(prod, 0) + 1
                        if streak[prod] >= freeze_patience:
                            frozen[prod] = (lo_r, hi_r)
                    else:
                        streak[prod] = 0
                        # widths actually moved: unfreeze (refresh
                        # sweeps re-derive frozen targets so a stale
                        # cache can't cap the search trajectory)
                        frozen.pop(prod, None)
                prev_w[prod] = w_r
        lb = attn_crown_lb_batch(wgg, xl_w, xh_w,
                                 {L: (lo.unsqueeze(0).expand(Q, -1),
                                      hi.unsqueeze(0).expand(Q, -1))
                                  for L, (lo, hi) in sb_dyn.items()},
                                 {k: (((v[0][0].unsqueeze(0).expand(Q, -1),
                                        v[0][1].unsqueeze(0).expand(Q, -1)),
                                       (v[1][0].unsqueeze(0).expand(Q, -1),
                                        v[1][1].unsqueeze(0).expand(Q, -1)))
                                      if isinstance(v[0], tuple)
                                      else (v[0].unsqueeze(0).expand(Q, -1),
                                            v[1].unsqueeze(0).expand(Q, -1)))
                                  for k, v in ob_dyn.items()},
                                 W_w, 0.0, {k: v.unsqueeze(0).expand(
                                     Q, *v.shape)
                                     for k, v in pstart['__spec__'].items()
                                     if k[0] not in ('beta', 'rbeta')})
        lb = lb + b_t.to(wdt)
        best = torch.maximum(best, lb.detach())
        worst = float(lb.detach().min())
        if worst > best_worst:
            best_worst = worst
            best_params = {sk: {k: v.detach().clone()
                                for k, v in d.items()}
                           for sk, d in pstart.items()}
            best_prow = {sk: {k: v.detach().clone()
                              for k, v in d.items()}
                         for sk, d in prow.items()}
        if bool((best > 0).all()):
            break
        loss = -(lb.clamp(max=0.01)).sum()
        loss.backward()
        if it < sign_jump_passes:
            # SIGN-JUMP warm start (vertex-seeking): g(alpha) is
            # piecewise-linear in the relu/McCormick interpolation
            # params, so each coordinate's optimum sits at 0 or 1 —
            # jump toward the endpoint the gradient's SIGN points at,
            # damped because coordinates interact (full jumps
            # oscillate). 4-6 damped passes buy ~95% of Adam's
            # 20-pass value; Adam then polishes (and owns the
            # interior-optimum params: cv tangents, betas).
            with torch.no_grad():
                _seen_sj = set()
                for d in list(pstart.values()) + list(prow.values()):
                    if id(d) in _seen_sj:
                        continue    # shared-alpha aliases: jump once
                    _seen_sj.add(id(d))
                    for kk, pp in d.items():
                        if kk[0] not in ('relu', 'mc'):
                            continue
                        if pp.grad is None:
                            continue
                        tgt_v = (pp.grad < 0).to(pp.dtype)
                        pp.mul_(1 - sign_jump_beta).add_(
                            sign_jump_beta * tgt_v)
                        pp.grad.zero_()
        else:
            opt.step()
        sched.step()
        with torch.no_grad():
            for d in pstart.values():
                for kk, pp in d.items():
                    if kk[0] in ('beta', 'rbeta'):
                        pp.clamp_(min=0)
                    else:
                        pp.clamp_(0, 1)
            for d in prow.values():
                for pp in d.values():
                    pp.clamp_(0, 1)
    print(f'  [joint-alpha] {n_done} iters '
          f'({time.perf_counter() - _t_joint0:.1f}s), '
          f'{len(frozen)}/{len(targets)} targets frozen, '
          f'search worst {best_worst:+.4f}', flush=True)
    # full-precision certification of the best params
    p64 = {sk: {k: v.detach().to(device=device, dtype=dtype)
                for k, v in d.items()}
           for sk, d in best_params.items()}
    prow64 = {sk: {k: v.detach().to(device=device, dtype=dtype)
                   for k, v in d.items()}
              for sk, d in best_prow.items()}
    with torch.no_grad():
        sb_c, ob_c = dynamic_bounds(p64, gg, xl, xh, False,
                                    prow_use=prow64)
        lb64 = attn_crown_lb_batch(
            gg, xl, xh,
            {L: (lo.unsqueeze(0).expand(Q, -1),
                 hi.unsqueeze(0).expand(Q, -1))
             for L, (lo, hi) in sb_c.items()},
            {k: (((v[0][0].unsqueeze(0).expand(Q, -1),
                   v[0][1].unsqueeze(0).expand(Q, -1)),
                  (v[1][0].unsqueeze(0).expand(Q, -1),
                   v[1][1].unsqueeze(0).expand(Q, -1)))
                 if isinstance(v[0], tuple)
                 else (v[0].unsqueeze(0).expand(Q, -1),
                       v[1].unsqueeze(0).expand(Q, -1)))
             for k, v in ob_c.items()},
            W_t, 0.0, {k: v.unsqueeze(0).expand(Q, *v.shape)
                       for k, v in p64['__spec__'].items()
                       if k[0] not in ('beta', 'rbeta')})
        lb64 = lb64 + b_t
    # callers warm-start single-walk alphas from the SPEC set
    return (lb64.detach().cpu().numpy(),
            {k: v.detach() for k, v in p64['__spec__'].items()},
            ob_c, sb_c)
