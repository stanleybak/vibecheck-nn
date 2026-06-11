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

Every parameter value in [0,1] yields a SOUND lower bound (each plane is a
valid relaxation for any interpolation), so optimizing them with Adam and
taking the best-of over iterations is sound — the alpha-CROWN argument.

`attn_crown_lb` evaluates the bound (differentiable); `attn_crown_alpha`
runs the optimization for one query and returns (best_lb, params).
"""
import numpy as np
import torch
import torch.nn.functional as F


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
                  tight_bounds=None):
    """Differentiable lower bound of w_q . y + b_q via backward CROWN.

    sb: {layer_idx: (lo, hi)} pre-relu bound TENSORS (from the plain
        forward — possibly clamped per BnB node).
    op_bounds: {op_name: bounds} recorded by the same forward.
    params: the plane-parameter dict described in the module docstring
        (missing keys fall back to the fixed-corner defaults).
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
            _push(op['inputs'][0], ep * lo_slope + en * up_slope)
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
            _push(op['inputs'][0], ep * k_lo + en * k_up)
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


def init_params(gg, sb, op_bounds, device, dtype):
    """Leaf parameter tensors at the fixed-handler defaults."""
    params = {}
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
                     tight_bounds=None, params=None):
    """Adam over the plane parameters for ONE query; returns
    (best_lb, params). Iteration 0 equals the fixed-corner backward, so
    the best-of is never worse. Early-exits when the bound goes positive.
    """
    device = xl.device
    dtype = xl.dtype
    if params is None:
        params = init_params(gg, sb, op_bounds, device, dtype)
    if not params:
        with torch.no_grad():
            lb = attn_crown_lb(gg, xl, xh, sb, op_bounds, w_q, b_q, {},
                               tight_bounds=tight_bounds)
        return float(lb), params
    opt = torch.optim.Adam(list(params.values()), lr=lr)
    best = -float('inf')
    for _ in range(n_iters):
        if time_left is not None and time_left() <= 1.0:
            break
        opt.zero_grad()
        lb = attn_crown_lb(gg, xl, xh, sb, op_bounds, w_q, b_q, params,
                           tight_bounds=tight_bounds)
        best = max(best, float(lb))
        if best > 0:
            break
        (-lb).backward()
        opt.step()
        with torch.no_grad():
            for p in params.values():
                p.clamp_(0, 1)
    return best, params
