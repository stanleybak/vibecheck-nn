"""Saturating u8xs8 quantized GEMM/Conv — the NON-VNNI MLAS path.

Root cause of smart_turn_multimodal_2026's platform-dependence (measured): onnxruntime's
fused u8xs8 quantized matmul/conv (QLinearMatMul/QLinearConv/QGemm) computes a DIFFERENT
function on CPUs WITHOUT AVX-VNNI than on CPUs with it. With VNNI (e.g. Intel) the kernel
uses VPDPBUSD -> exact int32 accumulation. Without VNNI (e.g. AMD Zen2) MLAS emulates it with
VPMADDUBSW, which sums adjacent u8xs8 product PAIRS into int16 WITH SATURATION (clamp to
+-32767). Large same-sign products (a deep transformer's activations) saturate on non-VNNI
only -> different int32 -> a different requantized code -> the final logit can flip
(measured: one smart_turn witness gives 0.918 on Intel, 0.500 on AMD; both at ORT 1.26.0).

This module provides:
  * numpy ground-truth twins (sat_matmul / sat_conv) — validated BIT-EXACT against real AWS
    non-VNNI ORT (matmul 8/8, conv 20/20 random cases). These reproduce the scorer's output
    on any machine, no hardware needed.
  * torch differentiable surrogate ops (SatMatmul / sat_conv_torch) — forward hard-saturates
    (matches the twins), backward uses a SOFT saturation gradient so PGD keeps a signal in
    the saturated band (the search lever for finding non-VNNI-valid counterexamples).

Conventions confirmed against ORT:
  * matmul pairs ADJACENT elements of the K (reduction) axis; conv uses K-order (kh,kw,cin)
    i.e. channels-INNERMOST ('hwc'), and pads with x_zero_point (so padded == real 0).
  * zero-point corrections and per-channel requant scales are applied EXACTLY (int32/float)
    AFTER the saturating integer accumulation, never inside it.

See detect_quant_oracle()/resolve_saturation() in surrogate_pgd.py for picking exact-vs-
saturating automatically from the local ORT.
"""
import numpy as np

INT16_MIN, INT16_MAX = -32768, 32767


# --------------------------------------------------------------- numpy ground truth (oracle)

def _sat_pairsum_np(prod_even, prod_odd):
    """Adjacent-pair int16-saturating sum of two product streams -> the per-pair int16 term."""
    return np.clip(prod_even + prod_odd, INT16_MIN, INT16_MAX)


def sat_matmul(a_u8, b_s8, a_zp, b_zp, a_s, b_s, y_s, y_zp):
    """Non-VNNI QLinearMatMul twin. a_u8:[M,K] uint8 codes, b_s8:[K,N] int8 codes; returns
    uint8 output [M,N]. Bit-exact vs real non-VNNI ORT."""
    a = a_u8.astype(np.int64)
    b = b_s8.astype(np.int64)
    M, K = a.shape
    N = b.shape[1]
    ae, be = (a[:, :-1], b[:-1, :]) if K % 2 else (a, b)   # even part for the paired sat-dot
    Ke = ae.shape[1]
    ps = _sat_pairsum_np(np.einsum('mp,pn->mpn', ae[:, 0:Ke:2], be[0:Ke:2, :]),
                         np.einsum('mp,pn->mpn', ae[:, 1:Ke:2], be[1:Ke:2, :]))
    raw = ps.sum(1)
    if K % 2:
        raw = raw + np.outer(a[:, -1], b[-1, :])          # odd tail: single product, no pair-sat
    # zero-point corrections use the FULL K (originals), applied exactly after saturation
    acc = (raw - int(a_zp) * b.sum(0)[None, :] - int(b_zp) * a.sum(1)[:, None]
           + K * int(a_zp) * int(b_zp))
    y = np.rint(acc.astype(np.float64) * (float(a_s) * float(b_s) / float(y_s))) + int(y_zp)
    return np.clip(y, 0, 255).astype(np.uint8)


def _im2col(x, KH, KW, sh, sw, ph, pw, pad_val):
    """x:[1,Cin,Hi,Wi] -> cols:[Ho*Wo, KH*KW*Cin] in K-order (kh,kw,cin) ('hwc'); Ho, Wo.
    Padding uses pad_val (QLinearConv pads with x_zero_point)."""
    _, C, Hi, Wi = x.shape
    Ho = (Hi + 2 * ph - KH) // sh + 1
    Wo = (Wi + 2 * pw - KW) // sw + 1
    xp = np.pad(x[0], ((0, 0), (ph, ph), (pw, pw)), constant_values=pad_val)
    cols = np.empty((C, KH, KW, Ho, Wo), x.dtype)
    for kh in range(KH):
        for kw in range(KW):
            cols[:, kh, kw] = xp[:, kh:kh + sh * Ho:sh, kw:kw + sw * Wo:sw]
    return np.transpose(cols, (3, 4, 1, 2, 0)).reshape(Ho * Wo, -1), Ho, Wo


def sat_conv(x_u8, w_s8, x_zp, w_zp, x_s, w_s, y_s, y_zp, bias=None, stride=1, pad=0):
    """Non-VNNI QLinearConv twin. x_u8:[1,Cin,H,W] uint8, w_s8:[Cout,Cin,KH,KW] int8; per-
    channel w_s allowed; bias int32 [Cout]. Returns uint8 [1,Cout,Ho,Wo]. Bit-exact vs ORT."""
    Cout, Cin, KH, KW = w_s8.shape
    A, Ho, Wo = _im2col(x_u8, KH, KW, stride, stride, pad, pad, int(x_zp))
    B = np.transpose(w_s8, (2, 3, 1, 0)).reshape(KH * KW * Cin, Cout)   # K-order (kh,kw,cin)
    K = A.shape[1]
    a = A.astype(np.int64)
    b = B.astype(np.int64)
    tail = None
    if K % 2:
        tail = np.outer(a[:, -1], b[-1, :])
        a, b = a[:, :-1], b[:-1, :]
    ps = _sat_pairsum_np(np.einsum('mp,pn->mpn', a[:, 0::2], b[0::2, :]),
                         np.einsum('mp,pn->mpn', a[:, 1::2], b[1::2, :]))
    raw = ps.sum(1)
    if tail is not None:
        raw = raw + tail
    acc = (raw - int(x_zp) * B.astype(np.int64).sum(0)[None, :]
           - int(w_zp) * A.astype(np.int64).sum(1)[:, None] + K * int(x_zp) * int(w_zp))
    if bias is not None:
        acc = acc + bias.astype(np.int64)[None, :]
    ws = np.asarray(w_s, np.float64).reshape(1, -1)
    y = np.rint(acc.astype(np.float64) * (float(x_s) * ws / float(y_s))) + int(y_zp)
    return np.clip(y, 0, 255).astype(np.uint8).reshape(Ho, Wo, Cout).transpose(2, 0, 1)[None]


# ----------------------------------------------------- torch differentiable surrogate ops

def _torch():
    import torch
    return torch


class _SatClamp:
    """forward: hard clamp to [lo,hi] (matches the oracle). backward: SOFT bump grad
    1/(1+(d/T)^2) where d is distance past the boundary -> full grad in-band, decaying (not
    zero) in saturation, so PGD keeps a usable signal where the real output is flat."""

    @staticmethod
    def make(torch):
        class _F(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, lo, hi, T):
                ctx.save_for_backward(x)
                ctx.lo, ctx.hi, ctx.T = lo, hi, T
                return x.clamp(lo, hi)

            @staticmethod
            def backward(ctx, g):
                (x,) = ctx.saved_tensors
                over = torch.relu(x - ctx.hi) + torch.relu(ctx.lo - x)
                return g * (1.0 / (1.0 + (over / ctx.T) ** 2)), None, None, None
        return _F


def _ste_round(torch):
    class _F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return torch.round(x)

        @staticmethod
        def backward(ctx, g):
            return g
    return _F


def _sat_dot_core(a, b, hard, T):
    """The pair-saturating reduction. Materializes a [M, K/2, N] pair tensor — the memory
    bottleneck — which is why _sat_dot_torch optionally gradient-CHECKPOINTS this (recompute in
    backward instead of storing it, so the GPU backward fits)."""
    torch = _torch()
    K = a.shape[1]
    tail = None
    if K % 2:
        tail = a[:, -1:] * b[-1:, :]
        a, b, K = a[:, :-1], b[:-1, :], K - 1
    ps = (torch.einsum('mp,pn->mpn', a[:, 0:K:2], b[0:K:2, :])
          + torch.einsum('mp,pn->mpn', a[:, 1:K:2], b[1:K:2, :]))
    ps = ps.clamp(INT16_MIN, INT16_MAX) if hard else \
        _SatClamp.make(torch).apply(ps, float(INT16_MIN), float(INT16_MAX), T)
    out = ps.sum(1)
    return out + tail if tail is not None else out


def _sat_dot_torch(a, b, hard, T=4096.0, ckpt=False):
    """a:[M,K] b:[K,N] integer-code tensors -> [M,N] non-VNNI saturating dot. ckpt=True
    gradient-checkpoints the [M,K/2,N] pair materialization (recompute in backward) so the
    GPU backward holds ~one pair tensor instead of all of them — the difference between OOM and
    fitting in ~4GB. Only worth it on GPU (it ~doubles compute); CPU leaves it off."""
    torch = _torch()
    if ckpt and a.requires_grad:
        from torch.utils.checkpoint import checkpoint
        return checkpoint(_sat_dot_core, a, b, hard, T, use_reentrant=False)
    return _sat_dot_core(a, b, hard, T)


def sat_matmul_torch(a_code, b_code, a_zp, b_zp, a_s, b_s, y_s, y_zp, hard=True, out_clamp=True):
    """Saturating matmul on integer-code tensors. hard=True forward matches sat_matmul;
    hard=False returns the continuous dequantized logit with the soft-saturation gradient."""
    K = a_code.shape[1]
    a = a_code.double() if hard else a_code
    b = b_code.double() if hard else b_code
    raw = _sat_dot_torch(a, b, hard)
    acc = raw - a_zp * b.sum(0, keepdim=True) - b_zp * a.sum(1, keepdim=True) + K * a_zp * b_zp
    if hard:
        torch = _torch()
        y = torch.round(acc * (a_s * b_s / y_s)) + y_zp
        return y.clamp(0, 255) if out_clamp else acc * (a_s * b_s / y_s) + y_zp
    return acc * (a_s * b_s / y_s) + y_zp


def sat_conv_torch(x_code, w_code, x_zp, w_zp, x_s, w_s, y_s, y_zp,
                   stride=1, pad=0, bias=None, hard=True, out_clamp=True):
    """Saturating conv via F.unfold (im2col) + saturating GEMM, K-order (kh,kw,cin), pad with
    x_zp. x_code:[1,Cin,H,W], w_code:[Cout,Cin,KH,KW] (integer codes as float). Returns
    [1,Cout,Ho,Wo] (hard+out_clamp -> uint8-valued; else continuous logit proxy)."""
    torch = _torch()
    import torch.nn.functional as F
    Cout, Cin, KH, KW = w_code.shape
    xp = F.pad(x_code, (pad, pad, pad, pad), value=float(x_zp))
    cols = F.unfold(xp, (KH, KW), stride=stride)            # [1, Cin*KH*KW, L]
    L = cols.shape[-1]
    A = cols[0].reshape(Cin, KH, KW, L).permute(1, 2, 0, 3).reshape(KH * KW * Cin, L).T
    B = w_code.permute(2, 3, 1, 0).reshape(KH * KW * Cin, Cout)
    raw = _sat_dot_torch(A.double() if hard else A, B.double() if hard else B, hard)
    acc = raw - x_zp * B.sum(0, keepdim=True) - w_zp * A.sum(1, keepdim=True) \
        + (KH * KW * Cin) * x_zp * w_zp
    if bias is not None:
        acc = acc + bias.reshape(1, -1)
    ws = torch.as_tensor(np.asarray(w_s, np.float64)).reshape(1, -1)
    Ho = (xp.shape[2] - KH) // stride + 1
    Wo = (xp.shape[3] - KW) // stride + 1
    if hard:
        y = torch.round(acc * (x_s * ws / y_s)) + y_zp
        y = y.clamp(0, 255) if out_clamp else acc * (x_s * ws / y_s) + y_zp
    else:
        y = acc * (x_s * ws / y_s) + y_zp
    return y.reshape(Ho, Wo, Cout).permute(2, 0, 1).unsqueeze(0)


# ---------------------------------------------- graft into an onnx2torch surrogate (gradient)

def extract_matmul_quant_params(onnx_path):
    """For each MatMul in the QDQ onnx whose inputs come from DequantizeLinear, return
    {node_name: dict(act_scale, act_zp, w_code[K,N], w_scale, w_zp, weight_input_idx)} for the
    act×weight matmuls (one DQ input is a weight constant). act×act matmuls are skipped (v1)."""
    import onnx
    from onnx import numpy_helper
    from .surrogate_pgd import _load_onnx_model
    g = _load_onnx_model(onnx_path).graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}

    def dq(inp):
        nd = prod.get(inp)
        if nd is None or nd.op_type != 'DequantizeLinear':
            return None
        z = init.get(nd.input[2]) if len(nd.input) > 2 else np.array(0)
        return dict(src=nd.input[0], scale=init.get(nd.input[1]), zp=z,
                    is_const=nd.input[0] in init, code=init.get(nd.input[0]))

    out = {}
    for n in g.node:
        if n.op_type != 'MatMul':
            continue
        a, b = dq(n.input[0]), dq(n.input[1])
        if not (a and b):
            continue
        if b['is_const']:
            act, w, widx = a, b, 1
        elif a['is_const']:
            act, w, widx = b, a, 0
        else:
            continue                                   # act×act: v1 leaves float
        out[n.name] = dict(act_scale=float(np.ravel(act['scale']).mean()),
                           act_zp=int(np.ravel(act['zp']).reshape(-1)[0]),
                           w_code=w['code'], w_scale=np.asarray(w['scale'], np.float64).ravel(),
                           w_zp=int(np.ravel(w['zp']).reshape(-1)[0]), weight_input_idx=widx)
    return out


class SaturatingMatMul:
    """Wraps an onnx2torch OnnxMatMul: quantizes the activation input (STE) and does the
    non-VNNI pair-saturating u8×s8 matmul with soft-saturation gradient, returning the
    dequantized result (same units as the float matmul it replaces). Forward needn't match
    ORT bit-for-bit — it only steers PGD; AWS ORT is the ground-truth validator."""

    def __init__(self, params):
        import torch
        p = params
        self.a_s, self.a_z, self.w_z = p['act_scale'], p['act_zp'], p['w_zp']
        wc = p['w_code']
        # float32 throughout (the surrogate runs in float32; this only steers PGD, ORT validates)
        self.w = torch.as_tensor(wc.astype(np.float32))             # [K, N] codes
        self.w_s = torch.as_tensor(p['w_scale'].astype(np.float32)).reshape(1, -1)  # [1, N]
        self.widx = p['weight_input_idx']
        self._ste = _ste_round(torch)

    def __call__(self, *args):
        torch = _torch()
        act = args[1 - self.widx]                                   # the activation input
        shp = act.shape
        x = act.reshape(-1, shp[-1])                                # [M, K]
        code = (self._ste.apply(x / self.a_s) + self.a_z).clamp(0, 255)
        w = self.w.to(code.device)
        K = code.shape[1]
        raw = _sat_dot_torch(code, w, hard=False, ckpt=code.is_cuda)
        acc = raw - self.a_z * w.sum(0, keepdim=True) - self.w_z * code.sum(1, keepdim=True) \
            + K * self.a_z * self.w_z
        y = acc * (self.a_s * self.w_s.to(code.device))            # dequant (continuous logit)
        return y.reshape(*shp[:-1], y.shape[-1])


def graft_saturating_matmuls(converted_model, onnx_path, log=print):
    """In-place replace act×weight OnnxMatMul submodules of an onnx2torch-converted surrogate
    with SaturatingMatMul (non-VNNI saturating, soft-grad). Returns the count grafted.
    onnx2torch sanitizes node names `/encoder/layers.0/.../MatMul` -> module name
    `encoder/layers/0/.../MatMul` (strip leading '/', '.'->'/'); we key params the same way."""
    params = extract_matmul_quant_params(onnx_path)
    psani = {k.lstrip('/').replace('.', '/'): k for k in params}
    n = 0
    for name, mod in list(converted_model.named_modules()):
        if 'MatMul' not in type(mod).__name__ or name not in psani:
            continue
        setattr(converted_model, name, _AsModule(SaturatingMatMul(params[psani[name]])))
        n += 1
    log(f'[surrogate] grafted saturating matmul into {n}/{len(params)} act×weight MatMuls')
    return n


def _AsModule(fn):
    import torch

    class _M(torch.nn.Module):
        def forward(self, *a):
            return fn(*a)
    return _M()


def graft_ste_round(converted_model, log=print):
    """Replace OnnxRound submodules with straight-through round so gradients reach the input.
    The fakequant surrogate's 119 activation Round ops otherwise zero the PGD gradient."""
    import torch
    ste = _ste_round(torch)

    class _STERound(torch.nn.Module):
        def forward(self, x):
            return ste.apply(x)
    n = 0
    for name, mod in list(converted_model.named_modules()):
        if type(mod).__name__ == 'OnnxRound':
            setattr(converted_model, name, _STERound())
            n += 1
    log(f'[surrogate] STE round grafted onto {n} OnnxRound ops (differentiable fakequant base)')
    return n


def graft_ste_clip(converted_model, log=print):
    """Wrap OnnxClip submodules with straight-through clip: forward = the original clamp
    (faithful), backward = identity, so gradients pass the uint8-range clip that otherwise
    zeroes them where activations saturate at the quantization bound."""
    import torch

    class _STEClip(torch.nn.Module):
        def __init__(self, orig):
            super().__init__()
            self.orig = orig

        def forward(self, *a):
            x = a[0]
            y = self.orig(*a)                 # faithful clamp (keeps exact bounds)
            return x + (y - x).detach()       # forward=y, backward passes through to x
    n = 0
    for name, mod in list(converted_model.named_modules()):
        if type(mod).__name__ == 'OnnxClip':
            setattr(converted_model, name, _STEClip(mod))
            n += 1
    log(f'[surrogate] STE clip grafted onto {n} OnnxClip ops')
    return n


def make_fakequant_differentiable(converted_model, log=print):
    """STE through Round + Clip so PGD gradients reach the input while the forward stays the
    faithful fakequant. Returns (n_round, n_clip)."""
    return graft_ste_round(converted_model, log), graft_ste_clip(converted_model, log)


def extract_conv_quant_params(onnx_path):
    """Per Conv node fed by DequantizeLinear inputs: act scale/zp, weight codes/scale/zp,
    bias int32, strides, pads. Returns {node_name: dict}. Handles 1D/2D/3D (group=1)."""
    from onnx import numpy_helper
    from .surrogate_pgd import _load_onnx_model
    g = _load_onnx_model(onnx_path).graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}

    def dq(inp):
        nd = prod.get(inp)
        if nd is None or nd.op_type != 'DequantizeLinear':
            return None
        z = init.get(nd.input[2]) if len(nd.input) > 2 else np.array(0)
        return dict(scale=init.get(nd.input[1]), zp=z, is_const=nd.input[0] in init,
                    code=init.get(nd.input[0]))

    out = {}
    for n in g.node:
        if n.op_type != 'Conv':
            continue
        a, w = dq(n.input[0]), dq(n.input[1])
        if not (a and w and w['is_const'] and not a['is_const']):
            continue
        attr = {at.name: at for at in n.attribute}
        strides = list(attr['strides'].ints) if 'strides' in attr else None
        pads = list(attr['pads'].ints) if 'pads' in attr else None
        bias = None
        if len(n.input) > 2:
            bdq = dq(n.input[2])
            bias = (bdq['code'] if bdq else init.get(n.input[2]))
        out[n.name] = dict(act_scale=float(np.ravel(a['scale']).mean()),
                           act_zp=int(np.ravel(a['zp']).reshape(-1)[0]),
                           w_code=w['code'], w_scale=np.asarray(w['scale'], np.float64).ravel(),
                           w_zp=int(np.ravel(w['zp']).reshape(-1)[0]),
                           bias=None if bias is None else np.asarray(bias).ravel(),
                           strides=strides, pads=pads)
    return out


def _im2col_nd(x, ksize, strides, pads, pad_val):
    """x:[1,Cin,*spatial] -> A:[L, prod(k)*Cin] in K-order (k...,cin innermost), and out spatial."""
    torch = _torch()
    import torch.nn.functional as F
    nd = len(ksize)
    padpairs = []
    for p in reversed(pads[:nd] if len(pads) == nd else pads[:nd]):
        padpairs += [p, p]                                   # F.pad wants reversed dim order
    xp = F.pad(x, padpairs, value=float(pad_val))
    win = xp
    for d in range(nd):
        win = win.unfold(2 + d, ksize[d], strides[d])        # -> ...,outdim, ...,k
    # win: [1, Cin, *out_spatial, *ksize]
    Cin = x.shape[1]
    out_spatial = win.shape[2:2 + nd]
    # permute to [*out_spatial, *ksize, Cin]
    perm = [0] + [2 + d for d in range(nd)] + [2 + nd + d for d in range(nd)] + [1]
    win = win.permute(*perm).reshape(int(np.prod(out_spatial)), -1)
    return win, out_spatial


class SaturatingConv:
    """Replaces an onnx2torch ConvNd: ND im2col on the STE-quantized activation + saturating
    u8×s8 GEMM (soft grad) + per-channel dequant + bias. Steers PGD; ORT validates."""

    def __init__(self, params):
        import torch
        p = params
        self.a_s, self.a_z, self.w_z = p['act_scale'], p['act_zp'], p['w_zp']
        wc = p['w_code']                                     # [Cout, Cin, *k]
        self.Cout, self.Cin = wc.shape[0], wc.shape[1]
        self.ksize = list(wc.shape[2:])
        nd = len(self.ksize)
        # weight -> [K, Cout] in K-order (k..., cin innermost)
        wt = np.transpose(wc, [0] + [2 + d for d in range(nd)] + [1]).reshape(self.Cout, -1).T
        self.w = torch.as_tensor(wt.astype(np.float32))
        self.w_s = torch.as_tensor(p['w_scale'].astype(np.float32)).reshape(1, -1)
        self.strides = p['strides'] or [1] * nd
        self.pads = p['pads'] or [0] * (2 * nd)
        self.bias = None if p['bias'] is None else torch.as_tensor(
            (p['bias'] * p['act_scale'] * p['w_scale']).astype(np.float32)).reshape(1, -1)
        self._ste = _ste_round(torch)

    def __call__(self, x):
        torch = _torch()
        code = (self._ste.apply(x / self.a_s) + self.a_z).clamp(0, 255)
        A, out_spatial = _im2col_nd(code, self.ksize, self.strides, self.pads, self.a_z)
        w = self.w.to(code.device)
        K = A.shape[1]
        raw = _sat_dot_torch(A, w, hard=False, ckpt=A.is_cuda)
        acc = raw - self.a_z * w.sum(0, keepdim=True) - self.w_z * A.sum(1, keepdim=True) \
            + K * self.a_z * self.w_z
        y = acc * (self.a_s * self.w_s.to(code.device))
        if self.bias is not None:
            y = y + self.bias.to(code.device)
        y = y.reshape(*out_spatial, self.Cout)
        nd = len(self.ksize)
        return y.permute(nd, *range(nd)).unsqueeze(0)        # [1, Cout, *out_spatial]


def graft_saturating_convs(converted_model, onnx_path, log=print, only_types=None):
    """Replace ConvNd submodules (act×weight) with SaturatingConv. only_types restricts to
    e.g. ('Conv1d',) — the saturating GEMM materializes a [L, K/2, Cout] pair tensor, which is
    huge for the video Conv3d (1.2M-elem backbone) and OOMs; skip those unless you have RAM."""
    params = extract_conv_quant_params(onnx_path)
    psani = {k.lstrip('/').replace('.', '/'): k for k in params}
    n = skipped = 0
    for name, mod in list(converted_model.named_modules()):
        tname = type(mod).__name__
        if not tname.startswith('Conv') or name not in psani:
            continue
        if only_types is not None and tname not in only_types:
            skipped += 1
            continue
        setattr(converted_model, name, _AsModule(SaturatingConv(params[psani[name]])))
        n += 1
    log(f'[surrogate] grafted saturating conv into {n}/{len(params)} Convs'
        + (f' ({skipped} skipped by only_types={only_types})' if skipped else ''))
    return n
