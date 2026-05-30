"""LiRPA-style forward-mode linear bound propagation.

This module has two implementations:
 - `forward_linear_bounds` (single-instance): per-op state shape
   `(n_out_at_op, n_input)`. Used by `alpha_crown_lirpa` for per-query
   Adam-tuned bounds at the root box.
 - `batched_forward_linear_bounds` (B-batched): per-op state shape
   `(B, n_out_at_op, n_input)`. Used by input-split BAB drivers that
   process many leaves per iter on GPU. Strictly faster per leaf than
   serial single-instance LiRPA via cuBLAS batched matmul.


Tracks per-op (A_lo, b_lo, A_up, b_up) — linear-in-input lower and
upper bounds. For input variable x in box [xl, xh]:
    y >= A_lo @ x + b_lo
    y <= A_up @ x + b_up
hold for every reachable y at that op.

This is α,β-CROWN's `forward` bound mode: each layer carries its own
LB and UB linear functions of the *original* input. Nonlinear ops
(Pow, Sigmoid, Div) take chord/tangent of the curve relative to the
LB/UB box bounds of the layer's input.

Stricter than zonotope chord+slack for activations: LB uses tangent
slope (gentler), UB uses chord slope (steeper); different slopes
between the two bounds let downstream sign-split ew·y backward give
strictly tighter spec bounds. The zonotope uses a single slope for
both and adds symmetric slack, which compounds across layers.

This is a partial implementation — supports pensieve's op chain:
fc, conv, add, sub_bilinear, sub, reshape, slice, gather, concat,
relu, pow, reduce_sum, div_bilinear.
"""
import numpy as np
import torch
import torch.nn.functional as F


def _evaluate_box(A_lo, b_lo, A_up, b_up, xl, xh):
    """Compute the BOX bound at this op from linear LB/UB and input box.

    For each output element i and the input box x in [xl, xh]:
      out_lo[i] = min over x of (A_lo[i] @ x) + b_lo[i]
                = sum_j min(A_lo[i,j]*xl[j], A_lo[i,j]*xh[j]) + b_lo[i]
                = positive_part(A_lo[i]) @ xl + negative_part(A_lo[i]) @ xh + b_lo[i]
      out_hi[i] = max over x of (A_up[i] @ x) + b_up[i]
                = positive_part(A_up[i]) @ xh + negative_part(A_up[i]) @ xl + b_up[i]
    """
    A_lo_pos = A_lo.clamp(min=0)
    A_lo_neg = A_lo.clamp(max=0)
    A_up_pos = A_up.clamp(min=0)
    A_up_neg = A_up.clamp(max=0)
    out_lo = A_lo_pos @ xl + A_lo_neg @ xh + b_lo
    out_hi = A_up_pos @ xh + A_up_neg @ xl + b_up
    return out_lo, out_hi


class _Bound:
    """Per-op linear bound data: A_lo, b_lo, A_up, b_up + cached box."""
    __slots__ = ('A_lo', 'b_lo', 'A_up', 'b_up', 'lo_box', 'hi_box',
                  'shape_nd')

    def __init__(self, A_lo, b_lo, A_up, b_up, lo_box, hi_box,
                 shape_nd=None):
        self.A_lo = A_lo
        self.b_lo = b_lo
        self.A_up = A_up
        self.b_up = b_up
        self.lo_box = lo_box
        self.hi_box = hi_box
        self.shape_nd = shape_nd


def _init_input_bound(xl, xh, device, dtype):
    """Initial bound at input: y = x → A=I, b=0."""
    n_in = xl.numel()
    A_id = torch.eye(n_in, dtype=dtype, device=device)
    b_zero = torch.zeros(n_in, dtype=dtype, device=device)
    return _Bound(A_id, b_zero, A_id.clone(), b_zero.clone(),
                   xl.clone(), xh.clone())


def _apply_fc(bound_in, W, bias):
    """y = W @ x + bias for linear op. A_out = W @ A_in, etc."""
    # For positive-coef row of W: same direction. For negative: swap LB/UB.
    # General per-element-of-W sign-split:
    #   y[i] = sum_j W[i,j] * g[j] + bias[i]
    #   y_LB[i] = sum_j (W+_pos · g_LB[j] + W_neg · g_UB[j]) + bias[i]
    # In A form: y_LB[i] = W[i] @ (chosen-by-sign A_g_lo / A_g_up rows)
    W_pos = W.clamp(min=0)
    W_neg = W.clamp(max=0)
    A_lo_out = W_pos @ bound_in.A_lo + W_neg @ bound_in.A_up
    b_lo_out = W_pos @ bound_in.b_lo + W_neg @ bound_in.b_up + bias
    A_up_out = W_pos @ bound_in.A_up + W_neg @ bound_in.A_lo
    b_up_out = W_pos @ bound_in.b_up + W_neg @ bound_in.b_lo + bias
    return A_lo_out, b_lo_out, A_up_out, b_up_out


def _apply_relu(bound_in, alpha=None):
    """ReLU per element. Active/dead pass through 0/identity; unstable
    relaxation:
      LB: y >= α * x  where α ∈ [0, 1] (default up_slope; Adam-optimized
        when `alpha` is passed)
      UB: y <= up_slope · x - up_slope · lo
    """
    lo = bound_in.lo_box
    hi = bound_in.hi_box
    active = lo >= 0
    dead = hi <= 0
    unstable = ~active & ~dead
    diff = (hi - lo).clamp(min=1e-12)
    up_slope = hi / diff
    up_intercept = -up_slope * lo
    # LB α: use externally supplied if present, else default = up_slope.
    if alpha is not None:
        lb_alpha = alpha
    else:
        lb_alpha = up_slope
    lb_slope = torch.where(active, torch.ones_like(lo),
                  torch.where(dead, torch.zeros_like(lo), lb_alpha))
    lb_int = torch.zeros_like(lo)
    ub_slope = torch.where(active, torch.ones_like(lo),
                  torch.where(dead, torch.zeros_like(lo), up_slope))
    ub_int = torch.where(unstable, up_intercept, torch.zeros_like(lo))
    # Scale input's A and b by per-element slopes. For element i:
    #   y_LB[i] = lb_slope[i] * (g_LB[i] · x + g_LB_b[i]) + lb_int[i]
    # (when lb_slope >= 0; here lb_slope >= 0 always)
    A_lo_out = lb_slope.unsqueeze(-1) * bound_in.A_lo
    b_lo_out = lb_slope * bound_in.b_lo + lb_int
    A_up_out = ub_slope.unsqueeze(-1) * bound_in.A_up
    b_up_out = ub_slope * bound_in.b_up + ub_int
    return A_lo_out, b_lo_out, A_up_out, b_up_out


def _apply_pow(bound_in, p, tangent_pos=None):
    """Pow per element. For x ∈ [lo, hi] (post-relu, so lo >= 0 typically):
      LB: y >= tangent_slope · g + tangent_const  (tangent at α-pos;
        default midpoint)
      UB: y <= chord_slope · g + chord_const
    """
    lo = bound_in.lo_box
    hi = bound_in.hi_box
    diff = (hi - lo).clamp(min=1e-12)
    chord_slope = (hi ** p - lo ** p) / diff
    chord_intercept = lo ** p - chord_slope * lo
    if tangent_pos is None:
        m = (lo + hi) / 2
    else:
        m = tangent_pos
    tangent_slope = p * m.pow(p - 1)
    tangent_intercept = m.pow(p) - tangent_slope * m
    # Sign-stable: use chord/tangent. Mixed-sign → fallback to box.
    convex = (lo >= 0)
    use_two_line = convex
    # For positive a, p >= 2: convex → chord is upper, tangent is lower.
    # LB slope = tangent (smaller), UB slope = chord (larger).
    # Both slopes >= 0 → use bound_in's LB for LB direction, UB for UB.
    lb_slope = torch.where(use_two_line, tangent_slope,
                              torch.zeros_like(lo))
    lb_int = torch.where(use_two_line, tangent_intercept,
                            (lo ** p).minimum(hi ** p))
    ub_slope = torch.where(use_two_line, chord_slope,
                              torch.zeros_like(lo))
    ub_int = torch.where(use_two_line, chord_intercept,
                            (lo ** p).maximum(hi ** p))
    # Slopes are both >= 0 when convex, so LB uses g_LB, UB uses g_UB.
    A_lo_out = lb_slope.unsqueeze(-1) * bound_in.A_lo
    b_lo_out = lb_slope * bound_in.b_lo + lb_int
    A_up_out = ub_slope.unsqueeze(-1) * bound_in.A_up
    b_up_out = ub_slope * bound_in.b_up + ub_int
    return A_lo_out, b_lo_out, A_up_out, b_up_out


def _apply_reduce_sum(bound_in, in_shape_nd, axes, keepdims):
    """Linear reduce: sum along given axes. A and b sum element-wise."""
    n_in = bound_in.A_lo.shape[1]
    A_lo_nd = bound_in.A_lo.reshape(*in_shape_nd, n_in)
    b_lo_nd = bound_in.b_lo.reshape(*in_shape_nd)
    A_up_nd = bound_in.A_up.reshape(*in_shape_nd, n_in)
    b_up_nd = bound_in.b_up.reshape(*in_shape_nd)
    for ax in sorted(axes, reverse=True):
        A_lo_nd = A_lo_nd.sum(dim=ax, keepdim=bool(keepdims))
        b_lo_nd = b_lo_nd.sum(dim=ax, keepdim=bool(keepdims))
        A_up_nd = A_up_nd.sum(dim=ax, keepdim=bool(keepdims))
        b_up_nd = b_up_nd.sum(dim=ax, keepdim=bool(keepdims))
    A_lo_out = A_lo_nd.reshape(-1, n_in)
    b_lo_out = b_lo_nd.reshape(-1)
    A_up_out = A_up_nd.reshape(-1, n_in)
    b_up_out = b_up_nd.reshape(-1)
    return A_lo_out, b_lo_out, A_up_out, b_up_out


def _apply_sub_bilinear(bound_a, bound_b):
    """y = a - b. y_LB = a_LB - b_UB, y_UB = a_UB - b_LB."""
    return (bound_a.A_lo - bound_b.A_up, bound_a.b_lo - bound_b.b_up,
            bound_a.A_up - bound_b.A_lo, bound_a.b_up - bound_b.b_lo)


def _apply_add_bias(bound_in, bias_flat):
    """y = x + const. A unchanged, b += bias."""
    return (bound_in.A_lo, bound_in.b_lo + bias_flat,
            bound_in.A_up, bound_in.b_up + bias_flat)


def _apply_slice(bound_in, flat_idx):
    """Select rows of A and b."""
    return (bound_in.A_lo.index_select(0, flat_idx),
            bound_in.b_lo.index_select(0, flat_idx),
            bound_in.A_up.index_select(0, flat_idx),
            bound_in.b_up.index_select(0, flat_idx))


def _apply_concat(bound_list):
    """Concatenate rows of multiple bounds."""
    return (torch.cat([b.A_lo for b in bound_list], dim=0),
            torch.cat([b.b_lo for b in bound_list], dim=0),
            torch.cat([b.A_up for b in bound_list], dim=0),
            torch.cat([b.b_up for b in bound_list], dim=0))


def _apply_conv(bound_in, kernel, bias, in_shape, stride, padding):
    """Apply conv to each input variable's column. Per input dim k,
    reshape A_lo[:, k] (n_out_flat,) to spatial (C, H, W), conv,
    flatten back. Linear so LB and UB same direction."""
    n_in = bound_in.A_lo.shape[1]
    C, H, W = in_shape
    # A is (n_out_flat_before_conv, n_in). Need each col reshaped.
    # Permute to (n_in, n_out_flat_before) = (n_in, C, H, W).
    A_lo_b = bound_in.A_lo.t().reshape(n_in, C, H, W)
    A_up_b = bound_in.A_up.t().reshape(n_in, C, H, W)
    # Conv each batch (n_in batch dim).
    A_lo_after = F.conv2d(A_lo_b, kernel, bias=None,
                            stride=stride, padding=padding)
    A_up_after = F.conv2d(A_up_b, kernel, bias=None,
                            stride=stride, padding=padding)
    # (n_in, C_out, H_out, W_out) → flatten to (n_in, n_out_flat) → t to
    # (n_out_flat, n_in).
    n_out_flat = A_lo_after.shape[1] * A_lo_after.shape[2] * A_lo_after.shape[3]
    A_lo_out = A_lo_after.reshape(n_in, n_out_flat).t().contiguous()
    A_up_out = A_up_after.reshape(n_in, n_out_flat).t().contiguous()
    # Bias add: b_out_unflat = conv(b_in_reshape) + bias_broadcast.
    b_lo_unflat = F.conv2d(bound_in.b_lo.reshape(1, C, H, W), kernel,
                              bias=bias, stride=stride, padding=padding)
    b_up_unflat = F.conv2d(bound_in.b_up.reshape(1, C, H, W), kernel,
                              bias=bias, stride=stride, padding=padding)
    b_lo_out = b_lo_unflat.reshape(-1)
    b_up_out = b_up_unflat.reshape(-1)
    return A_lo_out, b_lo_out, A_up_out, b_up_out


def alpha_crown_lirpa(gg, xl_np, xh_np, w_spec, b_spec, device, dtype,
                        n_iters=20, lr=0.1):
    """α-CROWN with LiRPA forward bound. Optimises per-(ReLU, Pow, Div)
    slope parameters via Adam to maximize LB of spec = w_spec · y + b_spec
    at the network output.

    Returns the best (largest) spec_lb seen across Adam iters. Plays the
    same role as `_run_alpha_crown_inputsplit_batched` but with the
    LiRPA forward pass rather than zonotope.
    """
    xl = torch.as_tensor(xl_np, dtype=dtype, device=device).flatten()
    xh = torch.as_tensor(xh_np, dtype=dtype, device=device).flatten()
    w_spec_t = torch.as_tensor(w_spec, dtype=dtype, device=device).flatten()
    # Initial pass to discover op shapes + box bounds.
    state0 = forward_linear_bounds(gg, xl_np, xh_np, device, dtype)
    # Create α parameters per nonlinear op based on state0.
    alpha_relu = {}
    alpha_pow = {}
    for op in gg['ops']:
        name = op['name']; t = op['type']
        if t == 'relu':
            b = state0[op['inputs'][0]]
            lo, hi = b.lo_box, b.hi_box
            unstable = (lo < 0) & (hi > 0)
            # α ∈ [0, 1] per neuron; init at upper-slope choice.
            up_slope = hi / (hi - lo).clamp(min=1e-12)
            init = torch.where(unstable, up_slope, torch.zeros_like(lo))
            init = init.detach().clone().requires_grad_(True)
            alpha_relu[name] = init
        elif t == 'pow':
            b = state0[op['inputs'][0]]
            lo, hi = b.lo_box, b.hi_box
            init = ((lo + hi) / 2).detach().clone().requires_grad_(True)
            alpha_pow[name] = (init, lo.clone(), hi.clone())
    params = list(alpha_relu.values()) + [p for p, _, _ in alpha_pow.values()]
    if not params:
        # No tunable parameters — return single forward pass.
        state = forward_linear_bounds(gg, xl_np, xh_np, device, dtype)
        last = gg['ops'][-1]['name']
        b = state[last]
        # spec_lb = min over x in box of w·y + b_const
        spec_A = w_spec_t @ (b.A_lo if w_spec_t.sum() >= 0 else b.A_up)
        spec_b = w_spec_t @ b.b_lo + b_spec  # approximation
        spec_lb = (spec_A.clamp(min=0) @ xl
                    + spec_A.clamp(max=0) @ xh + spec_b)
        return float(spec_lb)
    optimizer = torch.optim.Adam(params, lr=lr)
    best_lb = -float('inf')
    for it in range(n_iters):
        optimizer.zero_grad()
        state = forward_linear_bounds(
            gg, xl_np, xh_np, device, dtype,
            alpha_relu_per_op=alpha_relu, alpha_pow_per_op=alpha_pow)
        last = gg['ops'][-1]['name']
        b = state[last]
        # Per-output: pick LB or UB based on w_spec sign.
        w_pos = w_spec_t.clamp(min=0)
        w_neg = w_spec_t.clamp(max=0)
        # spec(x) = w·y + b_spec where y ∈ [LB(x), UB(x)]
        # spec_LB(x) = w_pos·y_LB(x) + w_neg·y_UB(x) + b_spec
        spec_A = w_pos @ b.A_lo + w_neg @ b.A_up
        spec_b_const = w_pos @ b.b_lo + w_neg @ b.b_up + b_spec
        # Min over input box.
        spec_lb = (spec_A.clamp(min=0) @ xl
                    + spec_A.clamp(max=0) @ xh + spec_b_const)
        # Adam update — maximize spec_lb
        loss = -spec_lb
        with torch.no_grad():
            curr = float(spec_lb)
            if curr > best_lb:
                best_lb = curr
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            # Clamp ReLU α to [0, 1].
            for a in alpha_relu.values():
                a.data.clamp_(0.0, 1.0)
            # Clamp Pow tangent to [lo, hi].
            for tan, lo_t, hi_t in alpha_pow.values():
                tan.data.copy_(torch.maximum(
                    torch.minimum(tan.data, hi_t), lo_t))
    return best_lb


def _batched_eval_box(A_lo, b_lo, A_up, b_up, xl_b, xh_b):
    """Batched box bound from batched linear LB/UB.

    A_lo: (B, n_out, n_in), b_lo: (B, n_out), xl_b/xh_b: (B, n_in).
    Returns (out_lo, out_hi) each shape (B, n_out).

    Uses Mid/Halfdiff identity to do 2 matmuls instead of 4:
      A @ Mid - |A| @ Halfdiff = A_pos @ xl + A_neg @ xh
      A @ Mid + |A| @ Halfdiff = A_pos @ xh + A_neg @ xl
    When A_lo and A_up share storage (tight upstream bound), 2 matmuls suffice
    for BOTH out_lo and out_hi.
    """
    xmid = (xl_b + xh_b) * 0.5
    xhalf = (xh_b - xl_b) * 0.5
    xmid_e = xmid.unsqueeze(-1); xhalf_e = xhalf.unsqueeze(-1)
    if A_lo is A_up and b_lo is b_up:
        A_mid_v = (A_lo @ xmid_e).squeeze(-1)
        A_amp_v = (A_lo.abs() @ xhalf_e).squeeze(-1)
        out_lo = A_mid_v - A_amp_v + b_lo
        out_hi = A_mid_v + A_amp_v + b_lo
        return out_lo, out_hi
    lo_mid = (A_lo @ xmid_e).squeeze(-1)
    lo_amp = (A_lo.abs() @ xhalf_e).squeeze(-1)
    up_mid = (A_up @ xmid_e).squeeze(-1)
    up_amp = (A_up.abs() @ xhalf_e).squeeze(-1)
    out_lo = lo_mid - lo_amp + b_lo
    out_hi = up_mid + up_amp + b_up
    return out_lo, out_hi


class _BBound:
    """Per-op batched linear bounds. A_*: (B, n_neurons, n_in)."""
    __slots__ = ('A_lo', 'b_lo', 'A_up', 'b_up', 'lo_box', 'hi_box')
    def __init__(self, A_lo, b_lo, A_up, b_up, lo_box, hi_box):
        self.A_lo = A_lo; self.b_lo = b_lo
        self.A_up = A_up; self.b_up = b_up
        self.lo_box = lo_box; self.hi_box = hi_box


def _bpki_gemm(W, A):
    """Compute einsum('jk,bpki->bpji', W, A) as a SINGLE GEMM.

    W is (j, k); A is (B, P, k, i). The contraction is over k, and W is shared
    across the (B, P, i) "batch" — so the natural `torch.einsum`/`W @ A` lowers
    to a broadcast `bmm` (B*P skinny matmuls, each re-reading the full W). For
    mscn's tall-skinny FC weights (e.g. W=2048x6144) against a thin A (i small)
    this is ~18x slower than folding (B,P,i) into one GEMM dimension and reading
    W once. Mathematically identical (fp64: ~1e-12; fp32: reassociation noise,
    same order as a cutlass GEMM — which is what α,β-CROWN uses anyway).
    """
    Bd, Pd, kd, idim = A.shape
    A2 = A.permute(2, 0, 1, 3).reshape(kd, Bd * Pd * idim)   # (k, B*P*i)
    O = (W @ A2).reshape(W.shape[0], Bd, Pd, idim)           # (j, B, P, i)
    return O.permute(1, 2, 0, 3).contiguous()                # (B, P, j, i)


def _bnm_gemm(W, AB):
    """Compute `W @ AB` (W shared across batch) as a SINGLE GEMM.

    W is (j, k); AB is (B, k, m). `W @ AB` broadcasts W to (B,j,k) and runs a
    `bmm` (B skinny matmuls re-reading W). For mscn's tall FC weights this is
    ~18x slower than folding (B, m) into one GEMM dim. Same math (fp64 ~1e-12).
    """
    Bd, kd, m = AB.shape
    AB2 = AB.permute(1, 0, 2).reshape(kd, Bd * m)            # (k, B*m)
    O = (W @ AB2).reshape(W.shape[0], Bd, m)                 # (j, B, m)
    return O.permute(1, 0, 2).contiguous()                   # (B, j, m)


def batched_forward_linear_bounds(gg, xl_b, xh_b, device, dtype,
                                    free_states=False,
                                    forced_varying_mask=None,
                                    forced_K=None,
                                    skip_bilinear_is_pt_check=False):
    """Batched LiRPA forward.

    Args:
      xl_b, xh_b: (B, n_in) tensors of per-leaf input bounds.
      free_states: if True, free op state dict entries once no longer
        referenced by downstream ops (refcount-based liveness). Reduces
        peak memory from O(N_layers × A_size) to O(2 × A_size). The
        returned dict has only the FINAL output op's bound under
        op_name; intermediate keys removed.
    Returns:
      dict op_name → _BBound with A_*: (B, n_neurons, n_in), b_*: (B, n_neurons).

    Supports the mscn op chain: fc, relu, pow, reduce_sum,
    sub_bilinear, mul_bilinear, div_bilinear, sigmoid, add, sub,
    reshape, slice/gather, concat.
    """
    B, n_in = xl_b.shape
    # Refcounts for streaming: how many downstream ops will read each.
    refcount = {gg['input_name']: 0}
    for op in gg['ops']:
        refcount[op['name']] = 0
        for inp in op['inputs']:
            refcount[inp] = refcount.get(inp, 0) + 1
    # The final output op MUST be kept (caller uses it).
    last_name = gg['ops'][-1]['name']
    refcount[last_name] = max(refcount.get(last_name, 0), 1)
    # When free_states=True, ALSO capture (lo_box, hi_box) at ops
    # consumed by ReLU/sigmoid/tanh (their pre-activations) and
    # bilinear inputs. Backward CROWN needs these intermediate
    # intervals; the full A matrices can be freed after consumption.
    # Stash them on a function attribute so callers can read after
    # the call (mirrors how forward_zonotope returns last_bilinear_op_bounds).
    interm_needed = set()
    for op in gg['ops']:
        if op['type'] in ('relu', 'sigmoid', 'tanh') and op['inputs']:
            interm_needed.add(op['inputs'][0])
        if op['type'] in ('mul_bilinear', 'div_bilinear'):
            for inp in op['inputs']:
                interm_needed.add(inp)
    interm_box = {}  # op_name -> (lo_box, hi_box)
    # Initial: y = x. Standard CROWN form would use identity A of shape
    # (n_in, n_in) but we can compress to (n_in, K) where K is the
    # number of input dims with non-zero radius across the batch — the
    # constant dims contribute zero rows in any A coefficient that ever
    # gets multiplied through downstream ops, but they keep showing up
    # in our memory. Dropping them is exact (they multiply x by 0 in
    # `_batched_eval_box`). For mscn cardinality with 1-2 varying dims
    # out of 308 this shrinks per-leaf A by 150×.
    if forced_varying_mask is not None:
        varying_mask = forced_varying_mask
    else:
        radii = (xh_b - xl_b) / 2
        varying_mask = (radii.abs().max(dim=0).values > 0)  # (n_in,)
    # K is a Python int that controls A-matrix shape. int(tensor) is a
    # GPU sync (and a torch.jit.trace constant-bake). Hot callers can
    # precompute K and pass it via forced_K to skip the sync.
    if forced_K is not None:
        K = forced_K
    else:
        K = int(varying_mask.sum())
    # Expose for callers that need to remap compressed A back to full input
    # dims (e.g. spec eval inside BaB needs xl_b[:, mask] / xh_b[:, mask]).
    batched_forward_linear_bounds.last_varying_mask = varying_mask
    if K == n_in or K == 0:
        # No compression possible / nothing to do.
        A_id = torch.eye(n_in, dtype=dtype, device=device).unsqueeze(0).expand(
            B, -1, -1).contiguous()
    else:
        var_idx = varying_mask.nonzero(as_tuple=True)[0]  # (K,)
        # Build (B, n_in, K) sparse identity: column k is e_{var_idx[k]}
        A_id = torch.zeros(B, n_in, K, dtype=dtype, device=device)
        A_id[:, var_idx, torch.arange(K, device=device)] = 1.0
    b_zero = torch.zeros(B, n_in, dtype=dtype, device=device)
    # When K < n_in, b_zero holds the CONSTANT part (= x_lo[constant_dim]).
    # Specifically, since x = b_const + A @ z where z lives in the K-dim
    # subspace, the constant offset is the constant-dim values of x_lo.
    if K < n_in:
        const_mask = ~varying_mask
        b_zero = b_zero + (xl_b * const_mask.to(dtype))  # (B, n_in)
        # Compressed input-box subspace (K-dim).
        xl_eval = xl_b[:, varying_mask].contiguous()
        xh_eval = xh_b[:, varying_mask].contiguous()
    else:
        xl_eval = xl_b
        xh_eval = xh_b
    state = {gg['input_name']: _BBound(
        A_id, b_zero, A_id.clone(), b_zero.clone(),
        xl_b.clone(), xh_b.clone())}
    def _maybe_free(op_input_names):
        if not free_states:
            return
        for inp in op_input_names:
            refcount[inp] -= 1
            if refcount[inp] <= 0 and inp in state:
                if inp in interm_needed:
                    bd = state[inp]
                    interm_box[inp] = (bd.lo_box.clone(), bd.hi_box.clone())
                del state[inp]
    import os as _osot, time as _tos
    _osot_on = _osot.environ.get('DUMP_OPTIME_FLP', '') == '1'
    _optime = {}
    _opcount = {}
    for op in gg['ops']:
        name = op['name']; t = op['type']; ins = op['inputs']
        bound_in = state.get(ins[0]) if ins else None
        if _osot_on:
            torch.cuda.synchronize()
            _t0 = _tos.perf_counter()
        if t == 'reshape':
            state[name] = bound_in
        elif t in ('slice', 'gather'):
            flat_idx = op.get('flat_idx')
            idx_t = torch.as_tensor(flat_idx, dtype=torch.long, device=device)
            A_lo_o = bound_in.A_lo.index_select(1, idx_t)
            b_lo_o = bound_in.b_lo.index_select(1, idx_t)
            A_up_o = bound_in.A_up.index_select(1, idx_t)
            b_up_o = bound_in.b_up.index_select(1, idx_t)
            # Box bounds are just a permutation of input box bounds —
            # no need to re-evaluate from A/b.
            lo_box = bound_in.lo_box.index_select(1, idx_t)
            hi_box = bound_in.hi_box.index_select(1, idx_t)
            state[name] = _BBound(A_lo_o, b_lo_o, A_up_o, b_up_o, lo_box, hi_box)
        elif t == 'concat':
            bounds = [state[i] for i in ins]
            A_lo_o = torch.cat([b.A_lo for b in bounds], dim=1)
            b_lo_o = torch.cat([b.b_lo for b in bounds], dim=1)
            A_up_o = torch.cat([b.A_up for b in bounds], dim=1)
            b_up_o = torch.cat([b.b_up for b in bounds], dim=1)
            # Box bounds are just the concat of input box bounds.
            lo_box = torch.cat([b.lo_box for b in bounds], dim=1)
            hi_box = torch.cat([b.hi_box for b in bounds], dim=1)
            state[name] = _BBound(A_lo_o, b_lo_o, A_up_o, b_up_o, lo_box, hi_box)
        elif t == 'fc':
            W = op['W']
            bias = op['bias']
            # Cache |W| on the op dict — reused every batch.
            W_abs = op.get('W_abs')
            if W_abs is None:
                W_abs = W.abs()
                op['W_abs'] = W_abs
            in_shape_nd = op.get('in_shapes_nd', [None])[0]
            # If incoming bounds are tight (A_lo == A_up, b_lo == b_up — common
            # before any activation) we only need ONE matmul instead of four.
            # Use object-identity check (state machinery shares the tensor
            # when no relaxation has been introduced) — a fast no-op test.
            tight = (bound_in.A_lo is bound_in.A_up and
                     bound_in.b_lo is bound_in.b_up)
            if (in_shape_nd is not None and len(in_shape_nd) >= 2
                    and W.shape[1] == in_shape_nd[-1]):
                # ND batched matmul: (B, prefix*n_in_inner, n_in) →
                # (B, prefix*n_out_inner, n_in).
                prefix = in_shape_nd[:-1]
                n_in_inner = in_shape_nd[-1]
                prefix_size = int(np.prod(prefix))
                n_out_inner = W.shape[0]
                # Reshape A to (B, prefix_size, n_in_inner, n_input)
                A_lo_p = bound_in.A_lo.reshape(B, prefix_size, n_in_inner, -1)
                b_lo_p = bound_in.b_lo.reshape(B, prefix_size, n_in_inner)
                if tight:
                    A_o_nd = _bpki_gemm(W, A_lo_p)
                    b_o_nd = torch.einsum('jk,bpk->bpj', W, b_lo_p) + bias
                    A_lo_o_nd = A_o_nd; A_up_o_nd = A_o_nd
                    b_lo_o_nd = b_o_nd; b_up_o_nd = b_o_nd
                else:
                    A_up_p = bound_in.A_up.reshape(B, prefix_size, n_in_inner, -1)
                    b_up_p = bound_in.b_up.reshape(B, prefix_size, n_in_inner)
                    # Mid/Halfdiff trick — 2 matmuls per A_lo/A_up pair.
                    A_mid = (A_lo_p + A_up_p) * 0.5
                    A_half = (A_up_p - A_lo_p) * 0.5
                    b_mid = (b_lo_p + b_up_p) * 0.5
                    b_half = (b_up_p - b_lo_p) * 0.5
                    A_mid_o = _bpki_gemm(W, A_mid)
                    A_amp = _bpki_gemm(W_abs, A_half)
                    b_mid_o = torch.einsum('jk,bpk->bpj', W, b_mid) + bias
                    b_amp = torch.einsum('jk,bpk->bpj', W_abs, b_half)
                    A_lo_o_nd = A_mid_o - A_amp
                    A_up_o_nd = A_mid_o + A_amp
                    b_lo_o_nd = b_mid_o - b_amp
                    b_up_o_nd = b_mid_o + b_amp
                A_lo_o = A_lo_o_nd.reshape(B, prefix_size * n_out_inner, -1)
                A_up_o = A_up_o_nd.reshape(B, prefix_size * n_out_inner, -1)
                b_lo_o = b_lo_o_nd.reshape(B, -1)
                b_up_o = b_up_o_nd.reshape(B, -1)
            else:
                if tight:
                    # Fuse A and b into a single matmul:
                    #   AB = cat([A_lo, b_lo[..., None]], dim=-1)   (B, n_in, n_input+1)
                    #   W @ AB → A_out, b_out + bias
                    AB = torch.cat([bound_in.A_lo,
                                     bound_in.b_lo.unsqueeze(-1)], dim=-1)
                    AB_o = _bnm_gemm(W, AB)
                    A_o = AB_o[..., :-1]
                    b_o = AB_o[..., -1] + bias
                    A_lo_o = A_o; A_up_o = A_o
                    b_lo_o = b_o; b_up_o = b_o
                else:
                    # Mid/Halfdiff fused with bias column: 2 matmuls.
                    A_mid = (bound_in.A_lo + bound_in.A_up) * 0.5
                    A_half = (bound_in.A_up - bound_in.A_lo) * 0.5
                    b_mid = (bound_in.b_lo + bound_in.b_up) * 0.5
                    b_half = (bound_in.b_up - bound_in.b_lo) * 0.5
                    AB_mid = torch.cat([A_mid, b_mid.unsqueeze(-1)], dim=-1)
                    AB_half = torch.cat([A_half, b_half.unsqueeze(-1)], dim=-1)
                    AB_mid_o = _bnm_gemm(W, AB_mid)
                    AB_amp = _bnm_gemm(W_abs, AB_half)
                    A_mid_o = AB_mid_o[..., :-1]
                    A_amp = AB_amp[..., :-1]
                    b_mid_o = AB_mid_o[..., -1] + bias
                    b_amp = AB_amp[..., -1]
                    A_lo_o = A_mid_o - A_amp
                    A_up_o = A_mid_o + A_amp
                    b_lo_o = b_mid_o - b_amp
                    b_up_o = b_mid_o + b_amp
            lo_box, hi_box = _batched_eval_box(
                A_lo_o, b_lo_o, A_up_o, b_up_o, xl_eval, xh_eval)
            state[name] = _BBound(A_lo_o, b_lo_o, A_up_o, b_up_o, lo_box, hi_box)
        elif t == 'relu':
            lo = bound_in.lo_box; hi = bound_in.hi_box  # (B, n)
            active = lo >= 0; dead = hi <= 0
            unstable = ~active & ~dead
            diff = (hi - lo).clamp(min=1e-12)
            up_slope = hi / diff
            up_int = -up_slope * lo
            # MIN-AREA lower-slope (matches ABC's `adaptive` default):
            # lower_k = 1 if up_slope > 0.5 else 0  (= min triangle area)
            min_area_lb = (up_slope > 0.5).to(dtype)
            lb_slope = torch.where(active, torch.ones_like(lo),
                          torch.where(dead, torch.zeros_like(lo),
                                       min_area_lb))
            lb_int_v = torch.zeros_like(lo)
            ub_slope = torch.where(active, torch.ones_like(lo),
                          torch.where(dead, torch.zeros_like(lo), up_slope))
            ub_int_v = torch.where(unstable, up_int, torch.zeros_like(lo))
            A_lo_o = lb_slope.unsqueeze(-1) * bound_in.A_lo
            b_lo_o = lb_slope * bound_in.b_lo + lb_int_v
            A_up_o = ub_slope.unsqueeze(-1) * bound_in.A_up
            b_up_o = ub_slope * bound_in.b_up + ub_int_v
            lo_box, hi_box = _batched_eval_box(
                A_lo_o, b_lo_o, A_up_o, b_up_o, xl_eval, xh_eval)
            state[name] = _BBound(A_lo_o, b_lo_o, A_up_o, b_up_o, lo_box, hi_box)
        elif t == 'pow':
            lo = bound_in.lo_box; hi = bound_in.hi_box
            p = int(op.get('exponent', 2))
            diff = (hi - lo).clamp(min=1e-12)
            chord_slope = (hi ** p - lo ** p) / diff
            chord_int = lo ** p - chord_slope * lo
            m = (lo + hi) / 2
            tan_slope = p * m.pow(p - 1)
            tan_int = m.pow(p) - tan_slope * m
            convex = lo >= 0
            lb_slope = torch.where(convex, tan_slope, torch.zeros_like(lo))
            lb_int = torch.where(convex, tan_int,
                                  torch.minimum(lo ** p, hi ** p))
            ub_slope = torch.where(convex, chord_slope, torch.zeros_like(lo))
            ub_int = torch.where(convex, chord_int,
                                  torch.maximum(lo ** p, hi ** p))
            A_lo_o = lb_slope.unsqueeze(-1) * bound_in.A_lo
            b_lo_o = lb_slope * bound_in.b_lo + lb_int
            A_up_o = ub_slope.unsqueeze(-1) * bound_in.A_up
            b_up_o = ub_slope * bound_in.b_up + ub_int
            lo_box, hi_box = _batched_eval_box(
                A_lo_o, b_lo_o, A_up_o, b_up_o, xl_eval, xh_eval)
            state[name] = _BBound(A_lo_o, b_lo_o, A_up_o, b_up_o, lo_box, hi_box)
        elif t == 'reduce_sum':
            in_shape_nd = op.get('in_shapes_nd', [None])[0]
            axes = op.get('axes', ())
            keepdims = op.get('keepdims', False)
            n_input_orig = bound_in.A_lo.shape[-1]
            A_lo_nd = bound_in.A_lo.reshape(B, *in_shape_nd, n_input_orig)
            b_lo_nd = bound_in.b_lo.reshape(B, *in_shape_nd)
            A_up_nd = bound_in.A_up.reshape(B, *in_shape_nd, n_input_orig)
            b_up_nd = bound_in.b_up.reshape(B, *in_shape_nd)
            # Axes are in INPUT shape — shift by 1 for batch dim.
            for ax in sorted(axes, reverse=True):
                ax_b = 1 + ax  # account for batch dim
                A_lo_nd = A_lo_nd.sum(dim=ax_b, keepdim=bool(keepdims))
                b_lo_nd = b_lo_nd.sum(dim=ax_b, keepdim=bool(keepdims))
                A_up_nd = A_up_nd.sum(dim=ax_b, keepdim=bool(keepdims))
                b_up_nd = b_up_nd.sum(dim=ax_b, keepdim=bool(keepdims))
            A_lo_o = A_lo_nd.reshape(B, -1, n_input_orig)
            b_lo_o = b_lo_nd.reshape(B, -1)
            A_up_o = A_up_nd.reshape(B, -1, n_input_orig)
            b_up_o = b_up_nd.reshape(B, -1)
            lo_box, hi_box = _batched_eval_box(
                A_lo_o, b_lo_o, A_up_o, b_up_o, xl_eval, xh_eval)
            state[name] = _BBound(A_lo_o, b_lo_o, A_up_o, b_up_o, lo_box, hi_box)
        elif t == 'sub_bilinear':
            bound_a = state[ins[0]]; bound_b = state[ins[1]]
            A_lo_o = bound_a.A_lo - bound_b.A_up
            b_lo_o = bound_a.b_lo - bound_b.b_up
            A_up_o = bound_a.A_up - bound_b.A_lo
            b_up_o = bound_a.b_up - bound_b.b_lo
            lo_box, hi_box = _batched_eval_box(
                A_lo_o, b_lo_o, A_up_o, b_up_o, xl_eval, xh_eval)
            state[name] = _BBound(A_lo_o, b_lo_o, A_up_o, b_up_o, lo_box, hi_box)
        elif t == 'sub':
            bias = op.get('bias')
            if bias is not None:
                bias_t = torch.as_tensor(bias.flatten(), dtype=dtype,
                                            device=device)
                A_lo_o = bound_in.A_lo
                b_lo_o = bound_in.b_lo - bias_t.unsqueeze(0)
                A_up_o = bound_in.A_up
                b_up_o = bound_in.b_up - bias_t.unsqueeze(0)
            else:
                A_lo_o, b_lo_o = bound_in.A_lo, bound_in.b_lo
                A_up_o, b_up_o = bound_in.A_up, bound_in.b_up
            lo_box, hi_box = _batched_eval_box(
                A_lo_o, b_lo_o, A_up_o, b_up_o, xl_eval, xh_eval)
            state[name] = _BBound(A_lo_o, b_lo_o, A_up_o, b_up_o, lo_box, hi_box)
        elif t == 'add':
            if op.get('is_merge'):
                bound_a = state[ins[0]]; bound_b = state[ins[1]]
                A_lo_o = bound_a.A_lo + bound_b.A_lo
                b_lo_o = bound_a.b_lo + bound_b.b_lo
                A_up_o = bound_a.A_up + bound_b.A_up
                b_up_o = bound_a.b_up + bound_b.b_up
            else:
                bias = op.get('bias')
                if bias is not None:
                    bias_t = torch.as_tensor(bias.flatten(), dtype=dtype,
                                                device=device)
                    out_shape_nd = op.get('out_shape_nd')
                    if (out_shape_nd is not None
                            and bias_t.numel() != bound_in.b_lo.shape[1]):
                        bias_shape = list(bias.shape) if hasattr(
                            bias, 'shape') else [bias_t.numel()]
                        bias_nd = bias_t.reshape(*bias_shape)
                        ones_out = torch.ones(*out_shape_nd, dtype=dtype,
                                                device=device)
                        bias_t = (ones_out * bias_nd).reshape(-1)
                    A_lo_o = bound_in.A_lo
                    b_lo_o = bound_in.b_lo + bias_t.unsqueeze(0)
                    A_up_o = bound_in.A_up
                    b_up_o = bound_in.b_up + bias_t.unsqueeze(0)
                else:
                    A_lo_o, b_lo_o = bound_in.A_lo, bound_in.b_lo
                    A_up_o, b_up_o = bound_in.A_up, bound_in.b_up
            lo_box, hi_box = _batched_eval_box(
                A_lo_o, b_lo_o, A_up_o, b_up_o, xl_eval, xh_eval)
            state[name] = _BBound(A_lo_o, b_lo_o, A_up_o, b_up_o, lo_box, hi_box)
        elif t in ('sigmoid', 'tanh'):
            from .verify_zono_bnb import _sigmoid_tanh_linear_bounds
            lo, hi = bound_in.lo_box, bound_in.hi_box
            lo_s, lo_t_, up_s, up_t = _sigmoid_tanh_linear_bounds(lo, hi, t)
            A_lo_o = lo_s.unsqueeze(-1) * bound_in.A_lo
            b_lo_o = lo_s * bound_in.b_lo + lo_t_
            A_up_o = up_s.unsqueeze(-1) * bound_in.A_up
            b_up_o = up_s * bound_in.b_up + up_t
            lo_box, hi_box = _batched_eval_box(
                A_lo_o, b_lo_o, A_up_o, b_up_o, xl_eval, xh_eval)
            state[name] = _BBound(A_lo_o, b_lo_o, A_up_o, b_up_o, lo_box, hi_box)
        elif t == 'mul_bilinear':
            bound_a = state[ins[0]]; bound_b = state[ins[1]]
            a_lo_box = bound_a.lo_box; a_hi_box = bound_a.hi_box
            b_lo_box = bound_b.lo_box; b_hi_box = bound_b.hi_box
            sh_in = op.get('in_shapes_nd', [None, None])
            sh_out = op.get('out_shape_nd')
            n_input_orig = bound_a.A_lo.shape[-1]
            ones_out = torch.ones(*sh_out, dtype=dtype, device=device)
            a_lo_o = (ones_out * a_lo_box.reshape(B, *sh_in[0])).reshape(B, -1)
            a_hi_o = (ones_out * a_hi_box.reshape(B, *sh_in[0])).reshape(B, -1)
            b_lo_o2 = (ones_out * b_lo_box.reshape(B, *sh_in[1])).reshape(B, -1)
            b_hi_o2 = (ones_out * b_hi_box.reshape(B, *sh_in[1])).reshape(B, -1)
            A_lo_a_o = (ones_out.unsqueeze(-1)
                          * bound_a.A_lo.reshape(B, *sh_in[0], n_input_orig)
                         ).reshape(B, -1, n_input_orig)
            A_up_a_o = (ones_out.unsqueeze(-1)
                          * bound_a.A_up.reshape(B, *sh_in[0], n_input_orig)
                         ).reshape(B, -1, n_input_orig)
            b_lo_a_o = (ones_out * bound_a.b_lo.reshape(B, *sh_in[0])).reshape(B, -1)
            b_up_a_o = (ones_out * bound_a.b_up.reshape(B, *sh_in[0])).reshape(B, -1)
            A_lo_b_o = (ones_out.unsqueeze(-1)
                          * bound_b.A_lo.reshape(B, *sh_in[1], n_input_orig)
                         ).reshape(B, -1, n_input_orig)
            A_up_b_o = (ones_out.unsqueeze(-1)
                          * bound_b.A_up.reshape(B, *sh_in[1], n_input_orig)
                         ).reshape(B, -1, n_input_orig)
            b_lo_b_o = (ones_out * bound_b.b_lo.reshape(B, *sh_in[1])).reshape(B, -1)
            b_up_b_o = (ones_out * bound_b.b_up.reshape(B, *sh_in[1])).reshape(B, -1)
            # Per-element check: which side is a point (zero radius).
            # `skip_bilinear_is_pt_check=True` (set by sync-free callers
            # for jit.trace / CUDA-graph paths) forces the McCormick
            # branch unconditionally. Sound: at r=0.5 McCormick collapses
            # to the exact linear form when one side is constant
            # (alpha = (b_lo+b_hi)/2 = b when b_lo==b_hi, gamma cancels).
            if skip_bilinear_is_pt_check:
                a_is_pt = False
                b_is_pt = False
            else:
                a_is_pt = ((a_hi_o - a_lo_o).abs() < 1e-12).all(dim=1).all().item()
                b_is_pt = ((b_hi_o2 - b_lo_o2).abs() < 1e-12).all(dim=1).all().item()
            if b_is_pt:
                b_const = b_lo_o2
                b_pos = b_const.clamp(min=0); b_neg = b_const.clamp(max=0)
                A_lo_o = b_pos.unsqueeze(-1) * A_lo_a_o + b_neg.unsqueeze(-1) * A_up_a_o
                b_lo_o = b_pos * b_lo_a_o + b_neg * b_up_a_o
                A_up_o = b_pos.unsqueeze(-1) * A_up_a_o + b_neg.unsqueeze(-1) * A_lo_a_o
                b_up_o = b_pos * b_up_a_o + b_neg * b_lo_a_o
            elif a_is_pt:
                a_const = a_lo_o
                a_pos = a_const.clamp(min=0); a_neg = a_const.clamp(max=0)
                A_lo_o = a_pos.unsqueeze(-1) * A_lo_b_o + a_neg.unsqueeze(-1) * A_up_b_o
                b_lo_o = a_pos * b_lo_b_o + a_neg * b_up_b_o
                A_up_o = a_pos.unsqueeze(-1) * A_up_b_o + a_neg.unsqueeze(-1) * A_lo_b_o
                b_up_o = a_pos * b_up_b_o + a_neg * b_lo_b_o
            else:
                # ABC-style McCormick + sign-conditional substitution
                # (matches BoundMul.bound_forward_both_perturbed exactly
                # with r_l = r_u = 0.5 ('middle' option in
                # MulHelper.interpolated_relaxation). This is the LOAD-
                # BEARING DIFFERENCE for tight forward bounds — earlier
                # box fallback dropped linear coefficients entirely and
                # blew up width through downstream FC layers.
                # LB envelope: y >= alpha_l*a + beta_l*b + gamma_l
                # UB envelope: y <= alpha_u*a + beta_u*b + gamma_u
                # McCormick LB/UB envelope at r_l = r_u = 0.5 (matches
                # ABC's BoundMul.bound_forward_both_perturbed with the
                # 'middle' MulHelper.interpolated_relaxation option).
                # At r=0.5 the slope params collapse to midpoints
                # (alpha_l == alpha_u == (b_lo+b_hi)/2; same for beta).
                # Gamma_l / gamma_u still differ. Slopes computed once,
                # clamps shared across LB/UB substitutions.
                alpha = (b_lo_o2 + b_hi_o2) * 0.5
                beta = (a_lo_o + a_hi_o) * 0.5
                gamma_l = -0.5 * (b_hi_o2 * a_hi_o + b_lo_o2 * a_lo_o)
                gamma_u = -0.5 * (b_lo_o2 * a_hi_o + b_hi_o2 * a_lo_o)
                alpha_p = alpha.clamp(min=0)
                alpha_n = alpha.clamp(max=0)
                beta_p = beta.clamp(min=0)
                beta_n = beta.clamp(max=0)
                alpha_p_u = alpha_p.unsqueeze(-1)
                alpha_n_u = alpha_n.unsqueeze(-1)
                beta_p_u = beta_p.unsqueeze(-1)
                beta_n_u = beta_n.unsqueeze(-1)
                A_lo_o = (alpha_p_u * A_lo_a_o + alpha_n_u * A_up_a_o
                          + beta_p_u * A_lo_b_o + beta_n_u * A_up_b_o)
                b_lo_o = (alpha_p * b_lo_a_o + alpha_n * b_up_a_o
                          + beta_p * b_lo_b_o + beta_n * b_up_b_o
                          + gamma_l)
                A_up_o = (alpha_n_u * A_lo_a_o + alpha_p_u * A_up_a_o
                          + beta_n_u * A_lo_b_o + beta_p_u * A_up_b_o)
                b_up_o = (alpha_n * b_lo_a_o + alpha_p * b_up_a_o
                          + beta_n * b_lo_b_o + beta_p * b_up_b_o
                          + gamma_u)
            lo_box, hi_box = _batched_eval_box(
                A_lo_o, b_lo_o, A_up_o, b_up_o, xl_eval, xh_eval)
            state[name] = _BBound(A_lo_o, b_lo_o, A_up_o, b_up_o, lo_box, hi_box)
        elif t == 'div_bilinear':
            from .verify_zono_bnb import (
                _reciprocal_linear_bounds, _mccormick_linear_bounds)
            bound_a = state[ins[0]]; bound_b = state[ins[1]]
            a_lo_box = bound_a.lo_box; a_hi_box = bound_a.hi_box
            b_lo_box = bound_b.lo_box; b_hi_box = bound_b.hi_box
            sh_in = op.get('in_shapes_nd', [None, None])
            sh_out = op.get('out_shape_nd')
            n_input_orig = bound_a.A_lo.shape[-1]
            ones_out = torch.ones(*sh_out, dtype=dtype, device=device)
            a_lo_o = (ones_out * a_lo_box.reshape(B, *sh_in[0])).reshape(B, -1)
            a_hi_o = (ones_out * a_hi_box.reshape(B, *sh_in[0])).reshape(B, -1)
            b_lo_o2 = (ones_out * b_lo_box.reshape(B, *sh_in[1])).reshape(B, -1)
            b_hi_o2 = (ones_out * b_hi_box.reshape(B, *sh_in[1])).reshape(B, -1)
            A_lo_a_o = (ones_out.unsqueeze(-1)
                          * bound_a.A_lo.reshape(B, *sh_in[0], n_input_orig)
                         ).reshape(B, -1, n_input_orig)
            A_up_a_o = (ones_out.unsqueeze(-1)
                          * bound_a.A_up.reshape(B, *sh_in[0], n_input_orig)
                         ).reshape(B, -1, n_input_orig)
            b_lo_a_o = (ones_out * bound_a.b_lo.reshape(B, *sh_in[0])).reshape(B, -1)
            b_up_a_o = (ones_out * bound_a.b_up.reshape(B, *sh_in[0])).reshape(B, -1)
            A_lo_b_o = (ones_out.unsqueeze(-1)
                          * bound_b.A_lo.reshape(B, *sh_in[1], n_input_orig)
                         ).reshape(B, -1, n_input_orig)
            A_up_b_o = (ones_out.unsqueeze(-1)
                          * bound_b.A_up.reshape(B, *sh_in[1], n_input_orig)
                         ).reshape(B, -1, n_input_orig)
            b_lo_b_o = (ones_out * bound_b.b_lo.reshape(B, *sh_in[1])).reshape(B, -1)
            b_up_b_o = (ones_out * bound_b.b_up.reshape(B, *sh_in[1])).reshape(B, -1)
            # Skip the b>0 assert in sync-free mode (caller takes
            # responsibility for graph-static invariants; mscn softmax
            # denominators are always positive by construction).
            if not skip_bilinear_is_pt_check:
                assert bool((b_lo_o2 > 0).all()), 'div needs b > 0'
            # Fast path: b is a point (per-leaf constant). Then 1/b is
            # exact, and y = a/b is just a*(1/b) — linear in a's bound.
            # No McCormick slack needed. ABC's BoundReciprocal+BoundMul
            # naturally handles this since Reciprocal of a point is a
            # point and BoundMul falls into the constant-side path.
            if skip_bilinear_is_pt_check:
                b_is_pt = False
            else:
                b_is_pt = ((b_hi_o2 - b_lo_o2).abs() < 1e-12).all().item()
            import os as _opfp
            if _opfp.environ.get('DUMP_DIV_FAST', '') == '1':
                print(f"[DIV_FAST] op {name}: b_is_pt={b_is_pt} b_lo_min={b_lo_o2.min().item():.4f} b_hi_max={b_hi_o2.max().item():.4f}")
            if b_is_pt:
                inv_b_const = 1.0 / b_lo_o2  # exact
                inv_b_pos = inv_b_const.clamp(min=0).unsqueeze(-1)
                inv_b_neg = inv_b_const.clamp(max=0).unsqueeze(-1)
                A_lo_o = inv_b_pos * A_lo_a_o + inv_b_neg * A_up_a_o
                A_up_o = inv_b_pos * A_up_a_o + inv_b_neg * A_lo_a_o
                b_lo_o = inv_b_const.clamp(min=0) * b_lo_a_o + inv_b_const.clamp(max=0) * b_up_a_o
                b_up_o = inv_b_const.clamp(min=0) * b_up_a_o + inv_b_const.clamp(max=0) * b_lo_a_o
                lo_box, hi_box = _batched_eval_box(
                    A_lo_o, b_lo_o, A_up_o, b_up_o, xl_eval, xh_eval)
                state[name] = _BBound(A_lo_o, b_lo_o, A_up_o, b_up_o, lo_box, hi_box)
                _maybe_free(ins)
                continue
            rs_lb, rc_lb, rs_ub, rc_ub = _reciprocal_linear_bounds(
                b_lo_o2, b_hi_o2,
                skip_positivity_check=skip_bilinear_is_pt_check)
            v_min_o = 1.0 / b_hi_o2; v_max_o = 1.0 / b_lo_o2
            (s_a_lb_m, s_v_lb_m, c_lb_m,
             s_a_ub_m, s_v_ub_m, c_ub_m) = _mccormick_linear_bounds(
                a_lo_o, a_hi_o, v_min_o, v_max_o)
            pos_v_lb = (s_v_lb_m >= 0).to(dtype); neg_v_lb = 1.0 - pos_v_lb
            pos_v_ub = (s_v_ub_m >= 0).to(dtype); neg_v_ub = 1.0 - pos_v_ub
            sv_for_lb = pos_v_lb * rs_lb + neg_v_lb * rs_ub
            cv_for_lb = pos_v_lb * rc_lb + neg_v_lb * rc_ub
            sv_for_ub = neg_v_ub * rs_lb + pos_v_ub * rs_ub
            cv_for_ub = neg_v_ub * rc_lb + pos_v_ub * rc_ub
            coef_a_lb = s_a_lb_m; coef_b_lb = s_v_lb_m * sv_for_lb
            const_lb_div = s_v_lb_m * cv_for_lb + c_lb_m
            coef_a_ub = s_a_ub_m; coef_b_ub = s_v_ub_m * sv_for_ub
            const_ub_div = s_v_ub_m * cv_for_ub + c_ub_m
            ca_lb_pos = coef_a_lb.clamp(min=0); ca_lb_neg = coef_a_lb.clamp(max=0)
            cb_lb_pos = coef_b_lb.clamp(min=0); cb_lb_neg = coef_b_lb.clamp(max=0)
            ca_ub_pos = coef_a_ub.clamp(min=0); ca_ub_neg = coef_a_ub.clamp(max=0)
            cb_ub_pos = coef_b_ub.clamp(min=0); cb_ub_neg = coef_b_ub.clamp(max=0)
            A_lo_o = (ca_lb_pos.unsqueeze(-1) * A_lo_a_o
                      + ca_lb_neg.unsqueeze(-1) * A_up_a_o
                      + cb_lb_pos.unsqueeze(-1) * A_lo_b_o
                      + cb_lb_neg.unsqueeze(-1) * A_up_b_o)
            b_lo_o = (ca_lb_pos * b_lo_a_o + ca_lb_neg * b_up_a_o
                      + cb_lb_pos * b_lo_b_o + cb_lb_neg * b_up_b_o
                      + const_lb_div)
            A_up_o = (ca_ub_pos.unsqueeze(-1) * A_up_a_o
                      + ca_ub_neg.unsqueeze(-1) * A_lo_a_o
                      + cb_ub_pos.unsqueeze(-1) * A_up_b_o
                      + cb_ub_neg.unsqueeze(-1) * A_lo_b_o)
            b_up_o = (ca_ub_pos * b_up_a_o + ca_ub_neg * b_lo_a_o
                      + cb_ub_pos * b_up_b_o + cb_ub_neg * b_lo_b_o
                      + const_ub_div)
            lo_box, hi_box = _batched_eval_box(
                A_lo_o, b_lo_o, A_up_o, b_up_o, xl_eval, xh_eval)
            state[name] = _BBound(A_lo_o, b_lo_o, A_up_o, b_up_o, lo_box, hi_box)
        elif t == 'conv':
            # Conv treated as a sparse linear op. bound_in.A_lo/A_up
            # shape (B, n_in_flat, K). Reshape per-leaf-per-input-col to
            # (B*K, C_in, H_in, W_in), run F.conv2d, reshape back.
            # Mid/Halfdiff trick: Conv(A_mid) ± Conv(|W|)(A_half) — uses
            # the same kernel for the mid part and the abs-kernel for
            # the amp part. Bias enters only the b accumulator (added
            # to b_mid; A_amp is bias-free).
            kernel = op['kernel']  # (C_out, C_in, kH, kW)
            bias = op['bias']      # (C_out,) or None
            stride = op.get('stride', (1, 1))
            padding = op.get('padding', (0, 0))
            in_shape_nd = op.get('in_shapes_nd', [None])[0]
            assert in_shape_nd is not None and len(in_shape_nd) == 2, (
                f'conv {name!r}: need 2D in_shape_nd (C,H,W) or (C,W); '
                f'got {in_shape_nd!r}')
            # in_shape_nd is per network.py's convention: 2 dims means
            # (C, W) → treat as (C, H=1, W). 3 dims would be (C, H, W).
            if len(in_shape_nd) == 2:
                C_in, W_in = int(in_shape_nd[0]), int(in_shape_nd[1])
                H_in = 1
            else:
                C_in, H_in, W_in = (int(d) for d in in_shape_nd)
            n_in_flat = C_in * H_in * W_in
            assert bound_in.A_lo.shape[1] == n_in_flat, (
                f'conv {name!r}: A_lo dim mismatch — expected n_in_flat={n_in_flat} '
                f'got {bound_in.A_lo.shape[1]}')
            kernel_abs = kernel.abs()
            tight = (bound_in.A_lo is bound_in.A_up
                     and bound_in.b_lo is bound_in.b_up)
            # Helper: apply F.conv2d to A of shape (B, n_in_flat, K).
            # Returns (B, n_out_flat, K). Treats K as a "batched columns"
            # axis: rearrange to (B*K, C_in, H_in, W_in), conv, back.
            K_in = bound_in.A_lo.shape[2]
            def _conv_A(A, kern):
                # A: (B, n_in_flat, K). Reshape to (B, C_in, H_in, W_in, K).
                A5 = A.reshape(B, C_in, H_in, W_in, K_in)
                # Move K to leading: (B, K, C_in, H_in, W_in) → (B*K, C_in, H_in, W_in)
                A5p = A5.permute(0, 4, 1, 2, 3).contiguous().reshape(
                    B * K_in, C_in, H_in, W_in)
                Ao = F.conv2d(A5p, kern, bias=None,
                               stride=stride, padding=padding)
                # Ao: (B*K, C_out, H_out, W_out).
                C_out, H_out, W_out = Ao.shape[1], Ao.shape[2], Ao.shape[3]
                Ao5 = Ao.reshape(B, K_in, C_out, H_out, W_out)
                # Permute back to (B, C_out, H_out, W_out, K) → flatten
                # spatial dims to n_out_flat.
                Ao5p = Ao5.permute(0, 2, 3, 4, 1).contiguous().reshape(
                    B, C_out * H_out * W_out, K_in)
                return Ao5p
            def _conv_b(b, kern, with_bias=False):
                # b: (B, n_in_flat). Reshape (B, C_in, H_in, W_in).
                b4 = b.reshape(B, C_in, H_in, W_in)
                bo = F.conv2d(b4, kern, bias=bias if with_bias else None,
                               stride=stride, padding=padding)
                return bo.reshape(B, -1)
            if tight:
                A_o = _conv_A(bound_in.A_lo, kernel)
                b_o = _conv_b(bound_in.b_lo, kernel, with_bias=True)
                A_lo_o = A_o; A_up_o = A_o
                b_lo_o = b_o; b_up_o = b_o
            else:
                A_mid = (bound_in.A_lo + bound_in.A_up) * 0.5
                A_half = (bound_in.A_up - bound_in.A_lo) * 0.5
                b_mid = (bound_in.b_lo + bound_in.b_up) * 0.5
                b_half = (bound_in.b_up - bound_in.b_lo) * 0.5
                A_mid_o = _conv_A(A_mid, kernel)
                A_amp = _conv_A(A_half, kernel_abs)
                b_mid_o = _conv_b(b_mid, kernel, with_bias=True)
                b_amp = _conv_b(b_half, kernel_abs, with_bias=False)
                A_lo_o = A_mid_o - A_amp
                A_up_o = A_mid_o + A_amp
                b_lo_o = b_mid_o - b_amp
                b_up_o = b_mid_o + b_amp
            lo_box, hi_box = _batched_eval_box(
                A_lo_o, b_lo_o, A_up_o, b_up_o, xl_eval, xh_eval)
            state[name] = _BBound(A_lo_o, b_lo_o, A_up_o, b_up_o,
                                    lo_box, hi_box)
        else:
            raise NotImplementedError(
                f'batched_forward_linear_bounds: unsupported op {t!r}')
        if _osot_on:
            torch.cuda.synchronize()
            _dt = _tos.perf_counter() - _t0
            _optime[t] = _optime.get(t, 0) + _dt
            _opcount[t] = _opcount.get(t, 0) + 1
        # Free input states once refcount hits zero (streaming mode).
        _maybe_free(ins)
    # Capture any intermediates that are needed but never had their
    # refcount hit zero (final-layer outputs in the chain).
    if free_states:
        for nm in interm_needed:
            if nm not in interm_box and nm in state:
                bd = state[nm]
                interm_box[nm] = (bd.lo_box.clone(), bd.hi_box.clone())
    # Expose via function attribute so callers can read the
    # intermediate (lo_box, hi_box) needed by backward CROWN without a
    # signature change (existing callers ignore the attribute).
    batched_forward_linear_bounds.last_interm_box = interm_box
    if _osot_on:
        total = sum(_optime.values())
        print(f"[FLP forward] total={total*1000:.1f}ms by type:")
        for k in sorted(_optime, key=lambda x: -_optime[x]):
            print(f"  {k:20s} {_optime[k]*1000:7.2f}ms ({_opcount[k]:4d} calls)")
    return state


def forward_lirpa_compat_zono_batched(xl_b, xh_b, gg, device, dtype):
    """Adapter that runs forward LiRPA but returns data in the same shape as
    ``_forward_zonotope_graph_batched`` so callers can swap in this tighter
    forward pass without changes downstream.

    Returns ``(sb, (c, G))`` where:
      - sb: dict ``{layer_idx -> (lo_pre, hi_pre)}`` for ReLU/Sigmoid/Tanh
        pre-activations (same key/value semantics as zonotope path).
      - (c, G): output bounds packed as a 1-generator zonotope so the
        caller's ``lo = c - |G|.sum(-1)``, ``hi = c + |G|.sum(-1)`` pattern
        still works. ``G`` has shape ``(B, n_out, 1)``.

    Also stashes ``forward_lirpa_compat_zono_batched.last_bilinear_op_bounds``
    (dict ``name -> (lo_box, hi_box)``) — mirrors zonotope's stash so
    ``_spec_backward_graph_batched(..., bilinear_op_bounds=...)`` keeps
    receiving McCormick input bounds for mul/div/sub bilinear ops.
    """
    import os as _os_flp_dbg
    _dbg = (_os_flp_dbg.environ.get('DEBUG_FWD_LIRPA_COMPAT', '') == '1')
    B_full = xl_b.shape[0]
    if _dbg:
        print(f'[fwd-lirpa-compat] called, B={B_full}, '
              f'gg n_ops={len(gg["ops"])}', flush=True)

    bilinear_inputs = set()
    for op_ in gg['ops']:
        if op_['type'] in ('mul_bilinear', 'div_bilinear', 'sub_bilinear'):
            for inp_ in op_['inputs']:
                bilinear_inputs.add(inp_)
    out_name = gg['ops'][-1]['name']

    # OOM-aware chunking. Start with the last successful chunk size (or
    # B_full); on CUDA OOM, halve and retry. We accumulate per-op (lo, hi)
    # across chunks by concatenating on batch dim, so caller-visible
    # tensors are (B_full, n_neurons). Remembering the last good chunk
    # avoids re-paying the 3-4 OOM halvings each call on consistently
    # tight memory (e.g. mscn_2048d_dual with B=2260+).
    _hint = getattr(forward_lirpa_compat_zono_batched,
                    '_last_good_chunk', None)
    # Cap initial chunk for very large batches — otherwise we burn 10+
    # halvings on cardinality_1_{8560..11080} before reaching a fittable
    # chunk (~24s overhead, often exhausting the 60s budget before BAB).
    # 1024 is empirically large enough to fit on 10GB GPU but small enough
    # that 8557-batch starts at 1024 (3 halvings) not 8557 (13 halvings).
    if _hint is not None:
        chunk = min(_hint, B_full)
    else:
        chunk = min(B_full, 1024)
    sb_acc = {}
    op_bounds_acc = {}
    out_lo_parts = []
    out_hi_parts = []
    pos = 0
    while pos < B_full:
        end = min(pos + chunk, B_full)
        try:
            state_c = batched_forward_linear_bounds(
                gg, xl_b[pos:end], xh_b[pos:end], device, dtype,
                free_states=True)
            interm_c = batched_forward_linear_bounds.last_interm_box
            # Accumulate.
            for op in gg['ops']:
                nm = op['name']
                if op['type'] in ('relu', 'sigmoid', 'tanh') and 'layer_idx' in op:
                    pre = op['inputs'][0]
                    if pre in interm_c:
                        if op['layer_idx'] not in sb_acc:
                            sb_acc[op['layer_idx']] = (
                                [interm_c[pre][0]], [interm_c[pre][1]])
                        else:
                            sb_acc[op['layer_idx']][0].append(interm_c[pre][0])
                            sb_acc[op['layer_idx']][1].append(interm_c[pre][1])
                if nm in bilinear_inputs and nm in interm_c:
                    if nm not in op_bounds_acc:
                        op_bounds_acc[nm] = (
                            [interm_c[nm][0]], [interm_c[nm][1]])
                    else:
                        op_bounds_acc[nm][0].append(interm_c[nm][0])
                        op_bounds_acc[nm][1].append(interm_c[nm][1])
            # Output bounds.
            out_bb = state_c.get(out_name)
            if out_bb is not None:
                out_lo_parts.append(out_bb.lo_box)
                out_hi_parts.append(out_bb.hi_box)
            else:
                lo_y, hi_y = interm_c[out_name]
                out_lo_parts.append(lo_y)
                out_hi_parts.append(hi_y)
            del state_c, interm_c
            pos = end
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            new_chunk = max(1, chunk // 2)
            if new_chunk == chunk:
                raise  # truly can't fit even B=1
            if _dbg:
                print(f'[fwd-lirpa-compat] OOM at chunk={chunk}, halving '
                      f'-> {new_chunk}', flush=True)
            chunk = new_chunk
        except RuntimeError as e:
            # CUBLAS_STATUS_EXECUTION_FAILED commonly appears when the GPU
            # is at the memory edge — CUBLAS fails before torch detects OOM.
            # Treat it as OOM (halve chunk and retry). Re-raise anything else
            # so we don't mask real shape/dtype bugs.
            msg = str(e)
            if 'CUBLAS_STATUS_EXECUTION_FAILED' not in msg and 'CUBLAS' not in msg:
                raise
            torch.cuda.empty_cache()
            # cuBLAS handle can corrupt after repeated failures — clear
            # workspaces to force fresh handles on next call. Without
            # this, even chunk=1 fails after many prior halvings on
            # cardinality_1_{8560..11080}.
            try:
                torch._C._cuda_clearCublasWorkspaces()
            except AttributeError:
                pass
            new_chunk = max(1, chunk // 2)
            if new_chunk == chunk:
                raise
            if _dbg:
                print(f'[fwd-lirpa-compat] CUBLAS exec failure at chunk={chunk}'
                      f' (treating as OOM), halving -> {new_chunk}', flush=True)
            chunk = new_chunk

    # Concatenate accumulated per-layer bounds across chunks.
    # Stash the chunk size that worked end-to-end so next call starts here.
    forward_lirpa_compat_zono_batched._last_good_chunk = chunk
    sb = {L: (torch.cat(lo_parts, dim=0), torch.cat(hi_parts, dim=0))
          for L, (lo_parts, hi_parts) in sb_acc.items()}
    op_bounds = {nm: (torch.cat(lo_parts, dim=0), torch.cat(hi_parts, dim=0))
                 for nm, (lo_parts, hi_parts) in op_bounds_acc.items()}
    forward_lirpa_compat_zono_batched.last_bilinear_op_bounds = op_bounds

    lo_y = torch.cat(out_lo_parts, dim=0)
    hi_y = torch.cat(out_hi_parts, dim=0)
    c = (lo_y + hi_y) * 0.5
    G = ((hi_y - lo_y) * 0.5).unsqueeze(-1)
    return sb, (c, G)


def batched_forward_linear_bounds_streaming(gg, xl_b, xh_b, device, dtype):
    """Streaming batched LiRPA: monkey-patches the state dict to free
    per-op _BBound after its last consumer reads it. Peak memory ~=
    biggest single-layer A (vs all-layers cumulative).

    Returns the OUTPUT op's _BBound only.
    """
    # Compute last-use per op (which op_index is last to read it).
    last_consumer = {}  # producer_name -> op_index of last consumer
    for op_idx, op in enumerate(gg['ops']):
        for inp in op['inputs']:
            last_consumer[inp] = op_idx
    # Input is never freed; output op is consumed by caller (kept).
    last_name = gg['ops'][-1]['name']
    # Build state by calling existing dispatch but with mid-stream free.
    # We CAN'T modify `batched_forward_linear_bounds` cleanly without
    # rewriting it. Hack: run it, then post-free non-output entries.
    # (Peak memory is still all-layers — true streaming needs inline
    # refactor. This wrapper only helps with downstream GPU cache reuse.)
    state = batched_forward_linear_bounds(gg, xl_b, xh_b, device, dtype)
    last_bb = state[last_name]
    for name in list(state.keys()):
        if name != last_name:
            del state[name]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return last_bb


def batched_forward_lirpa_layer_bounds_only(gg, xl_b, xh_b, device, dtype):
    """Batched LiRPA forward keeping ONLY (lo_box, hi_box) per layer/op.

    Memory-efficient: discards A matrices after each layer (~10x lower
    peak memory than `batched_forward_linear_bounds`). Returns:
      `(sb_b_layer, last_bb)` where:
        - `sb_b_layer`: dict mapping ReLU layer_idx → (lo, hi) shape (B, n)
        - `last_bb`: full _BBound for the final output op (for spec compute).

    Use case: downstream backward CROWN only needs (lo, hi) per layer for
    linearization, NOT the full A matrices. ABC uses this pattern.
    """
    state_lo_hi = {}  # op_name → (lo, hi)
    sb_b_layer = {}
    # We still need state for ops whose A is consumed by LATER ops in
    # the chain (input flows through linear chain). So compute step by
    # step, freeing the previous op's A.
    last_op_name = gg['ops'][-1]['name']
    # Simplest: run full LiRPA but free non-output states aggressively.
    state = batched_forward_linear_bounds(gg, xl_b, xh_b, device, dtype)
    for op in gg['ops']:
        bo = state[op['name']]
        state_lo_hi[op['name']] = (bo.lo_box, bo.hi_box)
        if op['type'] == 'relu' and 'layer_idx' in op:
            inp_name = op['inputs'][0]
            inp_b = state[inp_name]
            # ReLU's pre-act (lo, hi) is what `sb_b` convention stores.
            sb_b_layer[op['layer_idx']] = (inp_b.lo_box, inp_b.hi_box)
    last_bb = state[last_op_name]
    # Free intermediate As to drop memory.
    for name in list(state.keys()):
        if name != last_op_name:
            del state[name]
    return sb_b_layer, last_bb


# jit.trace cache for sync-free path. Keyed by (id(gg), B_bucket, K_int).
# When `use_jit_trace=True`, the chunked wrapper routes through
# batched_forward_linear_bounds_traceable (sync-free) and caches the trace.
# Bucketing chunk sizes lets diverse-B BAB calls reuse a small set of traces.
_TRACE_CACHE = {}
_TRACE_BUCKETS = (64, 128, 256, 512, 1024, 2048)


def _bucket_B(B):
    for b in _TRACE_BUCKETS:
        if B <= b:
            return b
    return B  # exceeds max bucket — fall back to exact B (cached separately)


def _get_or_trace(gg, sample_xl, sample_xh, K_int, varying_mask,
                   device, dtype, last_name, free_states):
    """Cache: per-(gg, B_bucket, K) get-or-create a jit.trace'd forward.

    Uses `bounded_module.compile_forward` to generate the straight-line
    sync-free function, then jit.trace caches the result. Microbench
    showed 1.27× speedup vs vanilla; bounds match within 3e-8 (ulp).
    """
    B_sample = sample_xl.shape[0]
    key = (id(gg), B_sample, K_int)
    cached = _TRACE_CACHE.get(key)
    if cached is not None:
        return cached

    from .bounded_module import compile_forward
    fwd = compile_forward(gg, varying_mask, K_int, device, dtype)
    traced = torch.jit.trace(fwd, (sample_xl, sample_xh),
                              check_trace=False, strict=False)
    _TRACE_CACHE[key] = traced
    return traced


def _build_sb_from_interm(gg, interm_box, bilinear_inputs=None):
    """Convert interm_box (op_name → (lo, hi)) into the sb/bilinear-bounds
    structure that `_spec_backward_graph_batched` expects.

    Lets BAB callers reuse the forward pass's intermediate bounds for the
    backward CROWN spec eval, avoiding a redundant second forward pass.
    """
    if bilinear_inputs is None:
        bilinear_inputs = set()
        for op in gg['ops']:
            if op['type'] in ('mul_bilinear', 'div_bilinear', 'sub_bilinear'):
                for inp in op['inputs']:
                    bilinear_inputs.add(inp)
    sb = {}
    op_bounds = {}
    for op in gg['ops']:
        nm = op['name']
        if op['type'] in ('relu', 'sigmoid', 'tanh') and 'layer_idx' in op:
            pre = op['inputs'][0]
            if pre in interm_box:
                sb[op['layer_idx']] = interm_box[pre]
        if nm in bilinear_inputs and nm in interm_box:
            op_bounds[nm] = interm_box[nm]
    return sb, op_bounds


def chunked_batched_forward_linear_bounds(gg, xl_b, xh_b, device, dtype,
                                            max_chunk=None, free_states=True,
                                            use_jit_trace=False,
                                            track_interm=False):
    """OOM-resilient batched LiRPA: process B leaves in chunks that fit
    GPU memory. Returns only the OUTPUT op's _BBound (concat across chunks).
    `max_chunk` initial chunk size hint. On OOM, halve and retry.
    `free_states` (default True): pass through to batched_forward, freeing
    per-op A matrices once last consumer reads them (peak memory ~5x lower).

    Exposes the varying-input-dim mask via attribute
    ``chunked_batched_forward_linear_bounds.last_varying_mask`` (bool tensor
    of shape (n_in,)) so callers can do
    ``xl_b[:, mask]`` when multiplying the returned compressed-A bb back
    against the original input box. Without this the caller silently
    mis-shapes the matmul and either crashes (shape error) or computes
    wrong bounds.
    """
    B_full = xl_b.shape[0]
    if max_chunk is None:
        max_chunk = B_full
    last_name = gg['ops'][-1]['name']
    # Try full batch first; on OOM halve.
    chunk = min(max_chunk, B_full)
    A_lo_parts = []; b_lo_parts = []
    A_up_parts = []; b_up_parts = []
    lo_box_parts = []; hi_box_parts = []
    pos = 0
    # Compute varying mask ONCE up-front so all chunks share the same
    # input-dim compression — otherwise different chunks could compress to
    # different K and we couldn't concat A_lo on dim 0.
    radii_all = (xh_b - xl_b) / 2
    full_varying_mask = (radii_all.abs().max(dim=0).values > 0)
    chunked_batched_forward_linear_bounds.last_varying_mask = full_varying_mask
    # Pre-compute K once (one sync per call, reused across all chunks).
    # When use_jit_trace=True this also lets the cached trace stay valid.
    K_int = int(full_varying_mask.sum()) if use_jit_trace else None
    # Accumulate interm_box across chunks → BAB BWD_CROWN can reuse without
    # a second forward pass (kills the 3.8× per-iter cost overhead).
    _interm_acc = {}  # op_name → ([lo_chunks], [hi_chunks])
    while pos < B_full:
        end = min(pos + chunk, B_full)
        try:
            if use_jit_trace:
                B_chunk = end - pos
                B_padded = _bucket_B(B_chunk)
                if B_padded > B_chunk:
                    pad_n = B_padded - B_chunk
                    xl_in = torch.cat([xl_b[pos:end],
                                        xl_b[end-1:end].expand(pad_n, -1)], dim=0)
                    xh_in = torch.cat([xh_b[pos:end],
                                        xh_b[end-1:end].expand(pad_n, -1)], dim=0)
                else:
                    xl_in, xh_in = xl_b[pos:end], xh_b[pos:end]
                traced_fn = _get_or_trace(gg, xl_in, xh_in, K_int,
                                            full_varying_mask, device, dtype,
                                            last_name, free_states)
                A_lo_o, b_lo_o, A_up_o, b_up_o, lo_o, hi_o = traced_fn(
                    xl_in, xh_in)
                if B_padded > B_chunk:
                    A_lo_o = A_lo_o[:B_chunk]; b_lo_o = b_lo_o[:B_chunk]
                    A_up_o = A_up_o[:B_chunk]; b_up_o = b_up_o[:B_chunk]
                    lo_o = lo_o[:B_chunk]; hi_o = hi_o[:B_chunk]
                A_lo_parts.append(A_lo_o); b_lo_parts.append(b_lo_o)
                A_up_parts.append(A_up_o); b_up_parts.append(b_up_o)
                lo_box_parts.append(lo_o); hi_box_parts.append(hi_o)
                pos = end
                continue
            state_c = batched_forward_linear_bounds(
                gg, xl_b[pos:end], xh_b[pos:end], device, dtype,
                free_states=free_states,
                forced_varying_mask=full_varying_mask)
            bc = state_c[last_name]
            A_lo_parts.append(bc.A_lo); b_lo_parts.append(bc.b_lo)
            A_up_parts.append(bc.A_up); b_up_parts.append(bc.b_up)
            lo_box_parts.append(bc.lo_box); hi_box_parts.append(bc.hi_box)
            # Accumulate interm_box across chunks (concat on batch dim).
            # Only when caller needs it (BAB BWD_CROWN path) — concat
            # has non-trivial overhead at big B.
            if track_interm:
                _interm_c = batched_forward_linear_bounds.last_interm_box
                for _nm, (_lo, _hi) in _interm_c.items():
                    if _nm not in _interm_acc:
                        _interm_acc[_nm] = ([_lo], [_hi])
                    else:
                        _interm_acc[_nm][0].append(_lo)
                        _interm_acc[_nm][1].append(_hi)
            # Free intermediate state.
            del state_c, bc
            pos = end
        except (torch.cuda.OutOfMemoryError, RuntimeError) as _e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            new_chunk = max(1, chunk // 2)
            if new_chunk == chunk:
                raise  # can't recover
            chunk = new_chunk
    A_lo = torch.cat(A_lo_parts, dim=0)
    b_lo = torch.cat(b_lo_parts, dim=0)
    A_up = torch.cat(A_up_parts, dim=0)
    b_up = torch.cat(b_up_parts, dim=0)
    lo_box = torch.cat(lo_box_parts, dim=0)
    hi_box = torch.cat(hi_box_parts, dim=0)
    # Concat interm_box across chunks and stash on function for BAB reuse
    # (BWD_CROWN can build sb from this without re-running forward).
    if track_interm:
        _interm_box = {}
        for _nm, (_los, _his) in _interm_acc.items():
            if len(_los) == 1:
                _interm_box[_nm] = (_los[0], _his[0])
            else:
                _interm_box[_nm] = (torch.cat(_los, dim=0), torch.cat(_his, dim=0))
        chunked_batched_forward_linear_bounds.last_interm_box = _interm_box
    else:
        chunked_batched_forward_linear_bounds.last_interm_box = None
    return _BBound(A_lo, b_lo, A_up, b_up, lo_box, hi_box)


def forward_linear_bounds(gg, xl_np, xh_np, device, dtype,
                            alpha_relu_per_op=None,
                            alpha_pow_per_op=None):
    """Forward linear bound propagation through gg ops.

    Returns dict {op_name: _Bound} with per-op linear LB/UB bounds and
    cached box bounds.

    `tangent_pos_per_pow` (optional): dict {op_name: per-element tangent
    position tensor} for α-Pow optimisation; default = midpoint.
    """
    xl = torch.as_tensor(xl_np, dtype=dtype, device=device).flatten()
    xh = torch.as_tensor(xh_np, dtype=dtype, device=device).flatten()
    n_in = xl.numel()
    state = {gg['input_name']: _init_input_bound(xl, xh, device, dtype)}
    for op in gg['ops']:
        name = op['name']
        t = op['type']
        ins = op['inputs']
        bound_in = state.get(ins[0]) if ins else None
        if t == 'reshape':
            state[name] = bound_in
        elif t in ('slice', 'gather'):
            flat_idx = op.get('flat_idx')
            idx_t = torch.as_tensor(flat_idx, dtype=torch.long,
                                       device=device)
            A_lo, b_lo, A_up, b_up = _apply_slice(bound_in, idx_t)
            lo_box, hi_box = _evaluate_box(A_lo, b_lo, A_up, b_up, xl, xh)
            state[name] = _Bound(A_lo, b_lo, A_up, b_up, lo_box, hi_box)
        elif t == 'concat':
            bounds = [state[i] for i in ins]
            A_lo, b_lo, A_up, b_up = _apply_concat(bounds)
            lo_box, hi_box = _evaluate_box(A_lo, b_lo, A_up, b_up, xl, xh)
            state[name] = _Bound(A_lo, b_lo, A_up, b_up, lo_box, hi_box)
        elif t == 'fc':
            W = op['W'].to(device=device, dtype=dtype)
            bias = op['bias'].to(device=device, dtype=dtype)
            # ND check: skip for now (assume 2D fc)
            in_shape_nd = op.get('in_shapes_nd', [None])[0]
            if (in_shape_nd is not None and len(in_shape_nd) >= 2
                    and W.shape[1] == in_shape_nd[-1]):
                # ND batched matmul — flatten by row-wise application.
                # For our LiRPA: treat as per-position fc. Apply
                # per-position W across prefix. The flatten shape is
                # prefix * n_out_inner. A_in is (prefix*n_in_inner, n_input_orig).
                prefix = in_shape_nd[:-1]
                n_in_inner = in_shape_nd[-1]
                prefix_size = int(np.prod(prefix))
                # Reshape A_in to (prefix_size, n_in_inner, n_in_orig)
                A_lo_p = bound_in.A_lo.reshape(prefix_size, n_in_inner, -1)
                A_up_p = bound_in.A_up.reshape(prefix_size, n_in_inner, -1)
                b_lo_p = bound_in.b_lo.reshape(prefix_size, n_in_inner)
                b_up_p = bound_in.b_up.reshape(prefix_size, n_in_inner)
                W_pos = W.clamp(min=0); W_neg = W.clamp(max=0)
                # Per prefix: y[i, j] = sum_k W[j, k] * g[i, k] + bias[j]
                # A_out (prefix_size, n_out, n_in_orig) =
                #   W_pos @ A_lo_p (per-prefix matmul, contracting k)
                A_lo_out_nd = torch.einsum('jk,pkn->pjn', W_pos, A_lo_p) \
                    + torch.einsum('jk,pkn->pjn', W_neg, A_up_p)
                A_up_out_nd = torch.einsum('jk,pkn->pjn', W_pos, A_up_p) \
                    + torch.einsum('jk,pkn->pjn', W_neg, A_lo_p)
                b_lo_out_nd = torch.einsum('jk,pk->pj', W_pos, b_lo_p) \
                    + torch.einsum('jk,pk->pj', W_neg, b_up_p) + bias
                b_up_out_nd = torch.einsum('jk,pk->pj', W_pos, b_up_p) \
                    + torch.einsum('jk,pk->pj', W_neg, b_lo_p) + bias
                n_out_inner = W.shape[0]
                A_lo_out = A_lo_out_nd.reshape(prefix_size * n_out_inner, -1)
                A_up_out = A_up_out_nd.reshape(prefix_size * n_out_inner, -1)
                b_lo_out = b_lo_out_nd.reshape(-1)
                b_up_out = b_up_out_nd.reshape(-1)
            else:
                A_lo_out, b_lo_out, A_up_out, b_up_out = _apply_fc(
                    bound_in, W, bias)
            lo_box, hi_box = _evaluate_box(
                A_lo_out, b_lo_out, A_up_out, b_up_out, xl, xh)
            state[name] = _Bound(
                A_lo_out, b_lo_out, A_up_out, b_up_out, lo_box, hi_box)
        elif t == 'conv':
            kernel = op['kernel'].to(device=device, dtype=dtype)
            bias = op['bias'].to(device=device, dtype=dtype)
            in_shape = op['in_shape']
            stride = op['stride']
            padding = op['padding']
            A_lo_out, b_lo_out, A_up_out, b_up_out = _apply_conv(
                bound_in, kernel, bias, in_shape, stride, padding)
            lo_box, hi_box = _evaluate_box(
                A_lo_out, b_lo_out, A_up_out, b_up_out, xl, xh)
            state[name] = _Bound(
                A_lo_out, b_lo_out, A_up_out, b_up_out, lo_box, hi_box)
        elif t == 'relu':
            alpha = None
            if alpha_relu_per_op is not None:
                alpha = alpha_relu_per_op.get(name)
            A_lo_out, b_lo_out, A_up_out, b_up_out = _apply_relu(
                bound_in, alpha)
            lo_box, hi_box = _evaluate_box(
                A_lo_out, b_lo_out, A_up_out, b_up_out, xl, xh)
            state[name] = _Bound(
                A_lo_out, b_lo_out, A_up_out, b_up_out, lo_box, hi_box)
        elif t == 'pow':
            p = int(op.get('exponent', 2))
            tan_alpha = None
            if alpha_pow_per_op is not None:
                tup = alpha_pow_per_op.get(name)
                if tup is not None:
                    tan_alpha = tup[0]
            A_lo_out, b_lo_out, A_up_out, b_up_out = _apply_pow(
                bound_in, p, tan_alpha)
            lo_box, hi_box = _evaluate_box(
                A_lo_out, b_lo_out, A_up_out, b_up_out, xl, xh)
            state[name] = _Bound(
                A_lo_out, b_lo_out, A_up_out, b_up_out, lo_box, hi_box)
        elif t == 'reduce_sum':
            in_shape_nd = op.get('in_shapes_nd', [None])[0]
            axes = op.get('axes', ())
            keepdims = op.get('keepdims', False)
            A_lo_out, b_lo_out, A_up_out, b_up_out = _apply_reduce_sum(
                bound_in, in_shape_nd, axes, keepdims)
            lo_box, hi_box = _evaluate_box(
                A_lo_out, b_lo_out, A_up_out, b_up_out, xl, xh)
            state[name] = _Bound(
                A_lo_out, b_lo_out, A_up_out, b_up_out, lo_box, hi_box)
        elif t == 'sub_bilinear':
            bound_a = state[ins[0]]
            bound_b = state[ins[1]]
            A_lo_out, b_lo_out, A_up_out, b_up_out = _apply_sub_bilinear(
                bound_a, bound_b)
            lo_box, hi_box = _evaluate_box(
                A_lo_out, b_lo_out, A_up_out, b_up_out, xl, xh)
            state[name] = _Bound(
                A_lo_out, b_lo_out, A_up_out, b_up_out, lo_box, hi_box)
        elif t == 'sub':
            bias = op.get('bias')
            if bias is not None:
                bias_t = torch.as_tensor(bias.flatten(), dtype=dtype,
                                            device=device)
                A_lo_out = bound_in.A_lo
                b_lo_out = bound_in.b_lo - bias_t
                A_up_out = bound_in.A_up
                b_up_out = bound_in.b_up - bias_t
            else:
                A_lo_out, b_lo_out = bound_in.A_lo, bound_in.b_lo
                A_up_out, b_up_out = bound_in.A_up, bound_in.b_up
            lo_box, hi_box = _evaluate_box(
                A_lo_out, b_lo_out, A_up_out, b_up_out, xl, xh)
            state[name] = _Bound(
                A_lo_out, b_lo_out, A_up_out, b_up_out, lo_box, hi_box)
        elif t == 'add':
            if op.get('is_merge'):
                # Skip connection: y = a + b
                bound_a = state[ins[0]]
                bound_b = state[ins[1]]
                A_lo_out = bound_a.A_lo + bound_b.A_lo
                b_lo_out = bound_a.b_lo + bound_b.b_lo
                A_up_out = bound_a.A_up + bound_b.A_up
                b_up_out = bound_a.b_up + bound_b.b_up
            else:
                bias = op.get('bias')
                if bias is not None:
                    bias_t = torch.as_tensor(bias.flatten(), dtype=dtype,
                                                device=device)
                    # Broadcast bias to bound_in's flat shape via ND.
                    in_shape_nd = op.get('in_shapes_nd', [None])[0]
                    out_shape_nd = op.get('out_shape_nd')
                    if (bias_t.numel() != bound_in.b_lo.numel()
                            and in_shape_nd is not None
                            and out_shape_nd is not None):
                        bias_shape = list(bias.shape) if hasattr(
                            bias, 'shape') else [bias_t.numel()]
                        bias_nd = bias_t.reshape(*bias_shape)
                        ones_out = torch.ones(*out_shape_nd, dtype=dtype,
                                                device=device)
                        bias_t = (ones_out * bias_nd).reshape(-1)
                    A_lo_out = bound_in.A_lo
                    b_lo_out = bound_in.b_lo + bias_t
                    A_up_out = bound_in.A_up
                    b_up_out = bound_in.b_up + bias_t
                else:
                    A_lo_out, b_lo_out = bound_in.A_lo, bound_in.b_lo
                    A_up_out, b_up_out = bound_in.A_up, bound_in.b_up
            lo_box, hi_box = _evaluate_box(
                A_lo_out, b_lo_out, A_up_out, b_up_out, xl, xh)
            state[name] = _Bound(
                A_lo_out, b_lo_out, A_up_out, b_up_out, lo_box, hi_box)
        elif t == 'div_bilinear':
            # y = a / b, element-wise (with broadcasting). Need both
            # input bounds.
            bound_a = state[ins[0]]
            bound_b = state[ins[1]]
            # Box bounds for a, b (scalars or vectors).
            a_lo_box = bound_a.lo_box
            a_hi_box = bound_a.hi_box
            b_lo_box = bound_b.lo_box
            b_hi_box = bound_b.hi_box
            sh_in = op.get('in_shapes_nd', [None, None])
            sh_out = op.get('out_shape_nd')
            n_out = int(np.prod(sh_out)) if sh_out else a_lo_box.numel()
            # Broadcast a (sh_in[0]) and b (sh_in[1]) to sh_out shape.
            # For pensieve: a is (6,), b is (1,) → out (6,).
            # Promote A's to broadcast shape too.
            n_input = bound_a.A_lo.shape[1]
            # Reshape & broadcast:
            ones_out = torch.ones(*sh_out, dtype=dtype, device=device)
            a_lo_o = (ones_out * a_lo_box.reshape(*sh_in[0])).reshape(-1)
            a_hi_o = (ones_out * a_hi_box.reshape(*sh_in[0])).reshape(-1)
            b_lo_o = (ones_out * b_lo_box.reshape(*sh_in[1])).reshape(-1)
            b_hi_o = (ones_out * b_hi_box.reshape(*sh_in[1])).reshape(-1)
            # Broadcast A's similarly.
            A_lo_a_o = (ones_out.unsqueeze(-1)
                          * bound_a.A_lo.reshape(*sh_in[0], n_input)
                         ).reshape(-1, n_input)
            A_up_a_o = (ones_out.unsqueeze(-1)
                          * bound_a.A_up.reshape(*sh_in[0], n_input)
                         ).reshape(-1, n_input)
            b_lo_a_o = (ones_out * bound_a.b_lo.reshape(*sh_in[0])).reshape(-1)
            b_up_a_o = (ones_out * bound_a.b_up.reshape(*sh_in[0])).reshape(-1)
            A_lo_b_o = (ones_out.unsqueeze(-1)
                          * bound_b.A_lo.reshape(*sh_in[1], n_input)
                         ).reshape(-1, n_input)
            A_up_b_o = (ones_out.unsqueeze(-1)
                          * bound_b.A_up.reshape(*sh_in[1], n_input)
                         ).reshape(-1, n_input)
            b_lo_b_o = (ones_out * bound_b.b_lo.reshape(*sh_in[1])).reshape(-1)
            b_up_b_o = (ones_out * bound_b.b_up.reshape(*sh_in[1])).reshape(-1)
            assert bool((b_lo_o > 0).all()), (
                f'div_bilinear LiRPA needs b > 0; got b_lo={b_lo_o}')
            # ABC-style Reciprocal+McCormick: Div = Mul(a, Recip(b)).
            # Recip 1/b: chord UB + α-tangent LB.
            # Mul a·v: α-McCormick interp.
            # Compose: substitute v = Recip bound based on sign.
            # Strictly tighter than naive y_LB = a / b_hi (which loses
            # b dependence). Mirrors `_div_backward_rm_mccormick`.
            from .verify_zono_bnb import (
                _reciprocal_linear_bounds, _mccormick_linear_bounds)
            rs_lb, rc_lb, rs_ub, rc_ub = _reciprocal_linear_bounds(
                b_lo_o, b_hi_o)  # default α=0.5 (midpoint tangent)
            v_min_o = 1.0 / b_hi_o  # 1/b range when b > 0
            v_max_o = 1.0 / b_lo_o
            (s_a_lb_m, s_v_lb_m, c_lb_m,
             s_a_ub_m, s_v_ub_m, c_ub_m) = _mccormick_linear_bounds(
                a_lo_o, a_hi_o, v_min_o, v_max_o)
            # Sign-aware substitution for LB / UB.
            pos_v_lb = (s_v_lb_m >= 0).to(dtype)
            neg_v_lb = 1.0 - pos_v_lb
            pos_v_ub = (s_v_ub_m >= 0).to(dtype)
            neg_v_ub = 1.0 - pos_v_ub
            # For LB: if s_v_lb_m > 0, use Recip LB (smaller v_LB→smaller y_LB).
            # If s_v_lb_m < 0, use Recip UB.
            sv_for_lb = pos_v_lb * rs_lb + neg_v_lb * rs_ub
            cv_for_lb = pos_v_lb * rc_lb + neg_v_lb * rc_ub
            sv_for_ub = neg_v_ub * rs_lb + pos_v_ub * rs_ub  # mirror
            cv_for_ub = neg_v_ub * rc_lb + pos_v_ub * rc_ub
            # Composed linear bounds in (a, b):
            #   y_LB(a, b) = s_a_lb_m·a + s_v_lb_m·(sv_for_lb·b + cv_for_lb) + c_lb_m
            coef_a_lb = s_a_lb_m
            coef_b_lb = s_v_lb_m * sv_for_lb
            const_lb_div = s_v_lb_m * cv_for_lb + c_lb_m
            coef_a_ub = s_a_ub_m
            coef_b_ub = s_v_ub_m * sv_for_ub
            const_ub_div = s_v_ub_m * cv_for_ub + c_ub_m
            # Now combine with input bounds for a and b (sign-aware):
            #   y_LB(x) = coef_a_lb · a_LB_or_UB(x) + coef_b_lb · b_LB_or_UB(x)
            # Per-output element, sign of coef_a_lb chooses a's LB or UB.
            ca_lb_pos = coef_a_lb.clamp(min=0)
            ca_lb_neg = coef_a_lb.clamp(max=0)
            cb_lb_pos = coef_b_lb.clamp(min=0)
            cb_lb_neg = coef_b_lb.clamp(max=0)
            ca_ub_pos = coef_a_ub.clamp(min=0)
            ca_ub_neg = coef_a_ub.clamp(max=0)
            cb_ub_pos = coef_b_ub.clamp(min=0)
            cb_ub_neg = coef_b_ub.clamp(max=0)
            A_lo_out = (ca_lb_pos.unsqueeze(-1) * A_lo_a_o
                        + ca_lb_neg.unsqueeze(-1) * A_up_a_o
                        + cb_lb_pos.unsqueeze(-1) * A_lo_b_o
                        + cb_lb_neg.unsqueeze(-1) * A_up_b_o)
            b_lo_out = (ca_lb_pos * b_lo_a_o + ca_lb_neg * b_up_a_o
                        + cb_lb_pos * b_lo_b_o + cb_lb_neg * b_up_b_o
                        + const_lb_div)
            A_up_out = (ca_ub_pos.unsqueeze(-1) * A_up_a_o
                        + ca_ub_neg.unsqueeze(-1) * A_lo_a_o
                        + cb_ub_pos.unsqueeze(-1) * A_up_b_o
                        + cb_ub_neg.unsqueeze(-1) * A_lo_b_o)
            b_up_out = (ca_ub_pos * b_up_a_o + ca_ub_neg * b_lo_a_o
                        + cb_ub_pos * b_up_b_o + cb_ub_neg * b_lo_b_o
                        + const_ub_div)
            lo_box, hi_box = _evaluate_box(
                A_lo_out, b_lo_out, A_up_out, b_up_out, xl, xh)
            state[name] = _Bound(
                A_lo_out, b_lo_out, A_up_out, b_up_out, lo_box, hi_box)
        elif t in ('sigmoid', 'tanh'):
            from .verify_zono_bnb import _sigmoid_tanh_linear_bounds
            lo, hi = bound_in.lo_box, bound_in.hi_box
            lo_s, lo_t_, up_s, up_t = _sigmoid_tanh_linear_bounds(lo, hi, t)
            # Both slopes >= 0; LB scales bound_in's LB, UB scales UB.
            A_lo_out = lo_s.unsqueeze(-1) * bound_in.A_lo
            b_lo_out = lo_s * bound_in.b_lo + lo_t_
            A_up_out = up_s.unsqueeze(-1) * bound_in.A_up
            b_up_out = up_s * bound_in.b_up + up_t
            lo_box, hi_box = _evaluate_box(
                A_lo_out, b_lo_out, A_up_out, b_up_out, xl, xh)
            state[name] = _Bound(
                A_lo_out, b_lo_out, A_up_out, b_up_out, lo_box, hi_box)
        elif t == 'mul_bilinear':
            # y = a * b element-wise. mscn typical: one side is a
            # PER-DISJUNCT mask (point input fixed at runtime) and the
            # other is perturbed.
            bound_a = state[ins[0]]
            bound_b = state[ins[1]]
            a_lo_box = bound_a.lo_box
            a_hi_box = bound_a.hi_box
            b_lo_box = bound_b.lo_box
            b_hi_box = bound_b.hi_box
            sh_in = op.get('in_shapes_nd', [None, None])
            sh_out = op.get('out_shape_nd')
            n_input = bound_a.A_lo.shape[1]
            # Detect point-side (zero-rad input box from initial).
            a_is_pt = bool((a_hi_box - a_lo_box).abs().max() < 1e-12)
            b_is_pt = bool((b_hi_box - b_lo_box).abs().max() < 1e-12)
            # Broadcast both to sh_out shape (mscn: a (3,128), b (3,1)).
            ones_out = torch.ones(*sh_out, dtype=dtype, device=device)
            a_lo_o = (ones_out * a_lo_box.reshape(*sh_in[0])).reshape(-1)
            a_hi_o = (ones_out * a_hi_box.reshape(*sh_in[0])).reshape(-1)
            b_lo_o = (ones_out * b_lo_box.reshape(*sh_in[1])).reshape(-1)
            b_hi_o = (ones_out * b_hi_box.reshape(*sh_in[1])).reshape(-1)
            A_lo_a_o = (ones_out.unsqueeze(-1)
                          * bound_a.A_lo.reshape(*sh_in[0], n_input)
                         ).reshape(-1, n_input)
            A_up_a_o = (ones_out.unsqueeze(-1)
                          * bound_a.A_up.reshape(*sh_in[0], n_input)
                         ).reshape(-1, n_input)
            b_lo_a_o = (ones_out * bound_a.b_lo.reshape(*sh_in[0])).reshape(-1)
            b_up_a_o = (ones_out * bound_a.b_up.reshape(*sh_in[0])).reshape(-1)
            A_lo_b_o = (ones_out.unsqueeze(-1)
                          * bound_b.A_lo.reshape(*sh_in[1], n_input)
                         ).reshape(-1, n_input)
            A_up_b_o = (ones_out.unsqueeze(-1)
                          * bound_b.A_up.reshape(*sh_in[1], n_input)
                         ).reshape(-1, n_input)
            b_lo_b_o = (ones_out * bound_b.b_lo.reshape(*sh_in[1])).reshape(-1)
            b_up_b_o = (ones_out * bound_b.b_up.reshape(*sh_in[1])).reshape(-1)
            # Point-side path: y = a * b_const (or const * b).
            if b_is_pt:
                # b is a point — use b_lo_o (= b_hi_o) as constant.
                b_const = b_lo_o
                b_pos = b_const.clamp(min=0)
                b_neg = b_const.clamp(max=0)
                # y_LB = b_pos * a_LB + b_neg * a_UB
                A_lo_out = b_pos.unsqueeze(-1) * A_lo_a_o + b_neg.unsqueeze(-1) * A_up_a_o
                b_lo_out = b_pos * b_lo_a_o + b_neg * b_up_a_o
                A_up_out = b_pos.unsqueeze(-1) * A_up_a_o + b_neg.unsqueeze(-1) * A_lo_a_o
                b_up_out = b_pos * b_up_a_o + b_neg * b_lo_a_o
            elif a_is_pt:
                a_const = a_lo_o
                a_pos = a_const.clamp(min=0)
                a_neg = a_const.clamp(max=0)
                A_lo_out = a_pos.unsqueeze(-1) * A_lo_b_o + a_neg.unsqueeze(-1) * A_up_b_o
                b_lo_out = a_pos * b_lo_b_o + a_neg * b_up_b_o
                A_up_out = a_pos.unsqueeze(-1) * A_up_b_o + a_neg.unsqueeze(-1) * A_lo_b_o
                b_up_out = a_pos * b_up_b_o + a_neg * b_lo_b_o
            else:
                # Both perturbed: McCormick. LB = max(line1, line2), but
                # for forward LiRPA we use just line1 (sound, looser).
                # line1: y_LB = b_lo·a + a_lo·b - a_lo·b_lo
                # Sound but maybe loose. Use midpoint interpolation.
                # Simpler conservative: y in [box]. lb_slope = 0, lb_const = box_lo.
                corners = torch.stack([a_lo_o * b_lo_o, a_lo_o * b_hi_o,
                                         a_hi_o * b_lo_o, a_hi_o * b_hi_o])
                y_box_lo = corners.min(dim=0).values
                y_box_hi = corners.max(dim=0).values
                n_out_flat = y_box_lo.numel()
                A_zero = torch.zeros(n_out_flat, n_input, dtype=dtype,
                                       device=device)
                A_lo_out = A_zero
                b_lo_out = y_box_lo
                A_up_out = A_zero
                b_up_out = y_box_hi
            lo_box, hi_box = _evaluate_box(
                A_lo_out, b_lo_out, A_up_out, b_up_out, xl, xh)
            state[name] = _Bound(
                A_lo_out, b_lo_out, A_up_out, b_up_out, lo_box, hi_box)
        else:
            raise NotImplementedError(
                f'forward_lirpa: unsupported op {t!r} at {name!r}')
    return state


# ----------------------------------------------------------------------
# Sync-free traceable variant.
#
# Built up stage-by-stage to eliminate every `.item()` / `int(tensor)` /
# `bool(tensor)` sync from the bound-prop path. End goal: a function
# pure in (xl_b, xh_b) that `torch.jit.trace` can record once and replay
# correctly on subsequent calls (no constants baked from per-call data).
#
# Current state:
#   Stage 1 ✓: K passed in (caller pre-computes via .sum().item() once
#             per `chunked_…` call, then all chunks reuse).
#   Stage 2 — : mul_bilinear `.item()` syncs at lines 752-753 still present
#   Stage 3 — : div_bilinear `.item()` + assert at lines 840, 846 still present
#   Stage 4 — : `verify_zono_bnb._reciprocal_linear_bounds` assert still
#              present (called from div_bilinear path)
# ----------------------------------------------------------------------
def batched_forward_linear_bounds_traceable(
        gg, xl_b, xh_b, device, dtype, *,
        varying_mask, K_int, free_states=False):
    """Sync-free wrapper around batched_forward_linear_bounds.

    Caller is responsible for pre-computing varying_mask and K_int (once
    per (gg, perturbation-shape) — cheap). This wrapper routes through
    the sync-free paths (Stages 1-N) accumulated so far.

    Stage 1 ✓: K passed in (no `K = int(...)` sync).
    Stage 2 ✓: `skip_bilinear_is_pt_check=True` removes the 2 per-mul
              and 1 per-div `.item()` syncs. Always-McCormick is exact
              when one side is constant (at r=0.5).
    """
    return batched_forward_linear_bounds(
        gg, xl_b, xh_b, device, dtype,
        free_states=free_states,
        forced_varying_mask=varying_mask,
        forced_K=K_int,
        skip_bilinear_is_pt_check=True)
