"""Codegen specialized forward bound prop for a fixed graph.

Walks `gg['ops']` once at compile time and emits a Python function whose
body is straight-line ops on LOCAL VARIABLES (no state dict, no Python
dispatch, no per-op type-check). Python's GC frees intermediate bounds
as soon as they go out of scope — works with jit.trace/torch.compile,
no OOM from leaked state.

PROOF point (mscn 1-op subset): straight-line forward gets 2.54× speedup
under jit.trace with bit-exact bounds. Scaling to full mscn op set here.

Per-op handlers below are pure tensor functions (no `.item()`, no
`int(tensor)`, no `bool(tensor)`). Bilinear ops always use the
McCormick branch (sound — exact at r=0.5 when one side is constant).

Each handler takes a tuple of 6 tensors representing a `_BBound`:
    (A_lo, b_lo, A_up, b_up, lo_box, hi_box)
plus op-specific args. Returns the same 6-tuple for the output.

This module currently targets the mscn_2048d_dual op set: fc, relu,
slice, concat, add, reduce_sum, mul_bilinear, div_bilinear, sigmoid,
sub_bilinear.
"""
import torch
import numpy as np
from typing import Tuple, List, Dict, Optional


# ----------------------------------------------------------------------
# Op handlers (pure tensor, no syncs)
# ----------------------------------------------------------------------

def _box_lo_hi(A_lo, b_lo, A_up, b_up, xl_eval, xh_eval):
    """Concretize linear bound over compressed eval box. Mid/halfdiff trick."""
    xmid = (xl_eval + xh_eval) * 0.5
    xhalf = (xh_eval - xl_eval) * 0.5
    xmid_e = xmid.unsqueeze(-1); xhalf_e = xhalf.unsqueeze(-1)
    if A_lo is A_up and b_lo is b_up:
        Am = (A_lo @ xmid_e).squeeze(-1)
        Aa = (A_lo.abs() @ xhalf_e).squeeze(-1)
        return Am - Aa + b_lo, Am + Aa + b_lo
    lo_m = (A_lo @ xmid_e).squeeze(-1)
    lo_a = (A_lo.abs() @ xhalf_e).squeeze(-1)
    up_m = (A_up @ xmid_e).squeeze(-1)
    up_a = (A_up.abs() @ xhalf_e).squeeze(-1)
    return lo_m - lo_a + b_lo, up_m + up_a + b_up


def fc_op(A_lo, b_lo, A_up, b_up, lo_box, hi_box,
          W, W_abs, bias, xl_eval, xh_eval, in_shape_nd):
    """fc / Gemm / MatMul — y = W @ x + bias.

    Uses the mid/halfdiff trick (2 matmuls instead of 4) for the
    general case. ND-batched if in_shape_nd has prefix dims.
    """
    B = A_lo.shape[0]
    if (in_shape_nd is not None and len(in_shape_nd) >= 2
            and W.shape[1] == in_shape_nd[-1]):
        prefix = in_shape_nd[:-1]
        n_in_inner = in_shape_nd[-1]
        prefix_size = int(np.prod(prefix))
        n_out_inner = W.shape[0]
        # Reshape A to (B, prefix_size, n_in_inner, K)
        A_lo_p = A_lo.reshape(B, prefix_size, n_in_inner, -1)
        b_lo_p = b_lo.reshape(B, prefix_size, n_in_inner)
        if A_lo is A_up and b_lo is b_up:
            A_o_nd = torch.einsum('jk,bpki->bpji', W, A_lo_p)
            b_o_nd = torch.einsum('jk,bpk->bpj', W, b_lo_p) + bias
            A_lo_o = A_o_nd.reshape(B, prefix_size * n_out_inner, -1)
            b_lo_o = b_o_nd.reshape(B, -1)
            A_up_o = A_lo_o; b_up_o = b_lo_o
        else:
            A_up_p = A_up.reshape(B, prefix_size, n_in_inner, -1)
            b_up_p = b_up.reshape(B, prefix_size, n_in_inner)
            A_mid = (A_lo_p + A_up_p) * 0.5
            A_half = (A_up_p - A_lo_p) * 0.5
            b_mid = (b_lo_p + b_up_p) * 0.5
            b_half = (b_up_p - b_lo_p) * 0.5
            A_mid_o = torch.einsum('jk,bpki->bpji', W, A_mid)
            A_amp = torch.einsum('jk,bpki->bpji', W_abs, A_half)
            b_mid_o = torch.einsum('jk,bpk->bpj', W, b_mid) + bias
            b_amp = torch.einsum('jk,bpk->bpj', W_abs, b_half)
            A_lo_o = (A_mid_o - A_amp).reshape(B, prefix_size * n_out_inner, -1)
            A_up_o = (A_mid_o + A_amp).reshape(B, prefix_size * n_out_inner, -1)
            b_lo_o = (b_mid_o - b_amp).reshape(B, -1)
            b_up_o = (b_mid_o + b_amp).reshape(B, -1)
    else:
        # 1D matmul path
        if A_lo is A_up and b_lo is b_up:
            AB = torch.cat([A_lo, b_lo.unsqueeze(-1)], dim=-1)
            AB_o = W @ AB
            A_o = AB_o[..., :-1]; b_o = AB_o[..., -1] + bias
            A_lo_o = A_o; A_up_o = A_o
            b_lo_o = b_o; b_up_o = b_o
        else:
            A_mid = (A_lo + A_up) * 0.5; A_half = (A_up - A_lo) * 0.5
            b_mid = (b_lo + b_up) * 0.5; b_half = (b_up - b_lo) * 0.5
            AB_mid = torch.cat([A_mid, b_mid.unsqueeze(-1)], dim=-1)
            AB_half = torch.cat([A_half, b_half.unsqueeze(-1)], dim=-1)
            AB_mid_o = W @ AB_mid
            AB_amp = W_abs @ AB_half
            A_mid_o = AB_mid_o[..., :-1]; A_amp = AB_amp[..., :-1]
            b_mid_o = AB_mid_o[..., -1] + bias; b_amp = AB_amp[..., -1]
            A_lo_o = A_mid_o - A_amp; A_up_o = A_mid_o + A_amp
            b_lo_o = b_mid_o - b_amp; b_up_o = b_mid_o + b_amp
    lo_box_o, hi_box_o = _box_lo_hi(A_lo_o, b_lo_o, A_up_o, b_up_o, xl_eval, xh_eval)
    return A_lo_o, b_lo_o, A_up_o, b_up_o, lo_box_o, hi_box_o


def relu_op(A_lo, b_lo, A_up, b_up, lo_box, hi_box, xl_eval, xh_eval):
    """ReLU with min-area triangle relaxation.

    For mixed-sign neurons: UB chord through (lo, 0)-(hi, hi), LB tangent
    at slope=hi/(hi-lo) anchored at 0 (gives min-area triangle).
    """
    pre_lo, pre_hi = lo_box, hi_box
    active = pre_lo >= 0
    inactive = pre_hi <= 0
    # Unstable: slope_u = hi/(hi-lo), intercept_ub = -slope_u*lo (for UB chord)
    safe_denom = pre_hi - pre_lo
    safe_denom = torch.where(safe_denom == 0, torch.ones_like(safe_denom), safe_denom)
    slope_u = pre_hi / safe_denom
    intercept_u_ub = -slope_u * pre_lo
    # LB slope (min-area): if -lo > hi, use 0 slope; else 1 slope.
    # Equivalent to slope_lb = (pre_hi > -pre_lo).float()
    slope_lb = (pre_hi > -pre_lo).to(A_lo.dtype)
    # Combine
    one = torch.ones_like(slope_u); zero = torch.zeros_like(slope_u)
    slope_ub_eff = torch.where(active, one, torch.where(inactive, zero, slope_u))
    slope_lb_eff = torch.where(active, one, torch.where(inactive, zero, slope_lb))
    int_ub_eff = torch.where(active, zero, torch.where(inactive, zero, intercept_u_ub))
    int_lb_eff = zero  # min-area LB passes through origin

    A_lo_o = A_lo * slope_lb_eff.unsqueeze(-1)
    b_lo_o = b_lo * slope_lb_eff + int_lb_eff
    A_up_o = A_up * slope_ub_eff.unsqueeze(-1)
    b_up_o = b_up * slope_ub_eff + int_ub_eff
    lo_box_o, hi_box_o = _box_lo_hi(A_lo_o, b_lo_o, A_up_o, b_up_o, xl_eval, xh_eval)
    return A_lo_o, b_lo_o, A_up_o, b_up_o, lo_box_o, hi_box_o


def slice_op(A_lo, b_lo, A_up, b_up, lo_box, hi_box, flat_idx_t):
    """Slice — gather along the neuron dim using precomputed flat indices."""
    A_lo_o = A_lo.index_select(1, flat_idx_t)
    A_up_o = A_up.index_select(1, flat_idx_t)
    b_lo_o = b_lo.index_select(1, flat_idx_t)
    b_up_o = b_up.index_select(1, flat_idx_t)
    lo_box_o = lo_box.index_select(1, flat_idx_t)
    hi_box_o = hi_box.index_select(1, flat_idx_t)
    return A_lo_o, b_lo_o, A_up_o, b_up_o, lo_box_o, hi_box_o


def add_op(A_lo_a, b_lo_a, A_up_a, b_up_a, lo_box_a, hi_box_a,
           A_lo_b, b_lo_b, A_up_b, b_up_b, lo_box_b, hi_box_b,
           xl_eval, xh_eval):
    """Elementwise add of two bounds (is_merge=True) — linear."""
    A_lo_o = A_lo_a + A_lo_b
    b_lo_o = b_lo_a + b_lo_b
    A_up_o = A_up_a + A_up_b
    b_up_o = b_up_a + b_up_b
    lo_box_o, hi_box_o = _box_lo_hi(A_lo_o, b_lo_o, A_up_o, b_up_o, xl_eval, xh_eval)
    return A_lo_o, b_lo_o, A_up_o, b_up_o, lo_box_o, hi_box_o


def bias_add_op(A_lo, b_lo, A_up, b_up, lo_box, hi_box, bias_t,
                xl_eval, xh_eval):
    """Add a constant bias tensor (1-input add op, no perturbation)."""
    if bias_t is None:
        return A_lo, b_lo, A_up, b_up, lo_box, hi_box
    bias_b = bias_t.unsqueeze(0)
    b_lo_o = b_lo + bias_b
    b_up_o = b_up + bias_b
    lo_box_o, hi_box_o = _box_lo_hi(A_lo, b_lo_o, A_up, b_up_o, xl_eval, xh_eval)
    return A_lo, b_lo_o, A_up, b_up_o, lo_box_o, hi_box_o


def reduce_sum_op(A_lo, b_lo, A_up, b_up, lo_box, hi_box,
                  in_shape_nd, axes, keepdims, xl_eval, xh_eval):
    """ReduceSum along given axes (relative to the ND shape).

    Linear, so each entry's bound = sum of the input bounds along that axis.
    """
    B = A_lo.shape[0]
    n_in_neurons = A_lo.shape[1]
    K = A_lo.shape[-1]
    # Reshape to (B, *in_shape_nd, K) for A, (B, *in_shape_nd) for b
    A_lo_nd = A_lo.reshape(B, *in_shape_nd, K)
    A_up_nd = A_up.reshape(B, *in_shape_nd, K)
    b_lo_nd = b_lo.reshape(B, *in_shape_nd)
    b_up_nd = b_up.reshape(B, *in_shape_nd)
    # Adjust axes for batch prefix (+1).
    axes_b = [int(a) + 1 for a in axes]
    keep = bool(keepdims)
    A_lo_o_nd = A_lo_nd.sum(dim=axes_b, keepdim=keep)
    A_up_o_nd = A_up_nd.sum(dim=axes_b, keepdim=keep)
    b_lo_o_nd = b_lo_nd.sum(dim=axes_b, keepdim=keep)
    b_up_o_nd = b_up_nd.sum(dim=axes_b, keepdim=keep)
    A_lo_o = A_lo_o_nd.reshape(B, -1, K)
    A_up_o = A_up_o_nd.reshape(B, -1, K)
    b_lo_o = b_lo_o_nd.reshape(B, -1)
    b_up_o = b_up_o_nd.reshape(B, -1)
    lo_box_o, hi_box_o = _box_lo_hi(A_lo_o, b_lo_o, A_up_o, b_up_o, xl_eval, xh_eval)
    return A_lo_o, b_lo_o, A_up_o, b_up_o, lo_box_o, hi_box_o


def concat_op(bounds_list, xl_eval, xh_eval):
    """Concat along the neuron dim (matches vanilla forward_lirpa concat).

    bounds_list: list of (A_lo, b_lo, A_up, b_up, lo_box, hi_box) — one per input.
    """
    A_lo_o = torch.cat([b[0] for b in bounds_list], dim=1)
    b_lo_o = torch.cat([b[1] for b in bounds_list], dim=1)
    A_up_o = torch.cat([b[2] for b in bounds_list], dim=1)
    b_up_o = torch.cat([b[3] for b in bounds_list], dim=1)
    lo_box_o = torch.cat([b[4] for b in bounds_list], dim=1)
    hi_box_o = torch.cat([b[5] for b in bounds_list], dim=1)
    return A_lo_o, b_lo_o, A_up_o, b_up_o, lo_box_o, hi_box_o


def sigmoid_op(A_lo, b_lo, A_up, b_up, lo_box, hi_box, xl_eval, xh_eval):
    """Sigmoid — defer to vanilla _sigmoid_tanh_linear_bounds for the
    LB/UB slope+intercept (exact match with vanilla forward_lirpa)."""
    from .verify_zono_bnb import _sigmoid_tanh_linear_bounds
    lo_s, lo_t, up_s, up_t = _sigmoid_tanh_linear_bounds(lo_box, hi_box, 'sigmoid')
    A_lo_o = lo_s.unsqueeze(-1) * A_lo
    b_lo_o = lo_s * b_lo + lo_t
    A_up_o = up_s.unsqueeze(-1) * A_up
    b_up_o = up_s * b_up + up_t
    lo_box_o, hi_box_o = _box_lo_hi(A_lo_o, b_lo_o, A_up_o, b_up_o, xl_eval, xh_eval)
    return A_lo_o, b_lo_o, A_up_o, b_up_o, lo_box_o, hi_box_o


def sub_bilinear_op(A_lo_a, b_lo_a, A_up_a, b_up_a, lo_box_a, hi_box_a,
                    A_lo_b, b_lo_b, A_up_b, b_up_b, lo_box_b, hi_box_b,
                    xl_eval, xh_eval):
    """Elementwise sub (a - b) — linear."""
    A_lo_o = A_lo_a - A_up_b
    b_lo_o = b_lo_a - b_up_b
    A_up_o = A_up_a - A_lo_b
    b_up_o = b_up_a - b_lo_b
    lo_box_o, hi_box_o = _box_lo_hi(A_lo_o, b_lo_o, A_up_o, b_up_o, xl_eval, xh_eval)
    return A_lo_o, b_lo_o, A_up_o, b_up_o, lo_box_o, hi_box_o


def _mccormick_substitute(a_lo, a_hi, b_lo, b_hi,
                          A_lo_a, A_up_a, b_lo_a, b_up_a,
                          A_lo_b, A_up_b, b_lo_b, b_up_b):
    """McCormick LB/UB envelope at r_l = r_u = 0.5, substituted into the
    linear bounds for a and b. Returns (A_lo_o, b_lo_o, A_up_o, b_up_o).

    Sound: at r=0.5 the slope params collapse to midpoints and the envelope
    becomes exact when one side is constant (a_lo == a_hi or b_lo == b_hi).
    Matches ABC's `BoundMul.bound_forward_both_perturbed` exactly.
    """
    alpha = (b_lo + b_hi) * 0.5
    beta = (a_lo + a_hi) * 0.5
    gamma_l = -0.5 * (b_hi * a_hi + b_lo * a_lo)
    gamma_u = -0.5 * (b_lo * a_hi + b_hi * a_lo)
    alpha_p = alpha.clamp(min=0); alpha_n = alpha.clamp(max=0)
    beta_p = beta.clamp(min=0); beta_n = beta.clamp(max=0)
    alpha_p_u = alpha_p.unsqueeze(-1); alpha_n_u = alpha_n.unsqueeze(-1)
    beta_p_u = beta_p.unsqueeze(-1); beta_n_u = beta_n.unsqueeze(-1)
    A_lo_o = (alpha_p_u * A_lo_a + alpha_n_u * A_up_a
              + beta_p_u * A_lo_b + beta_n_u * A_up_b)
    b_lo_o = (alpha_p * b_lo_a + alpha_n * b_up_a
              + beta_p * b_lo_b + beta_n * b_up_b + gamma_l)
    A_up_o = (alpha_n_u * A_lo_a + alpha_p_u * A_up_a
              + beta_n_u * A_lo_b + beta_p_u * A_up_b)
    b_up_o = (alpha_n * b_lo_a + alpha_p * b_up_a
              + beta_n * b_lo_b + beta_p * b_up_b + gamma_u)
    return A_lo_o, b_lo_o, A_up_o, b_up_o


def mul_bilinear_op(A_lo_a, b_lo_a, A_up_a, b_up_a, lo_box_a, hi_box_a,
                    A_lo_b, b_lo_b, A_up_b, b_up_b, lo_box_b, hi_box_b,
                    sh_a, sh_b, sh_out, xl_eval, xh_eval):
    """Elementwise multiply with broadcasting + McCormick.

    Matches the inline mul_bilinear in batched_forward_linear_bounds with
    `skip_bilinear_is_pt_check=True` (always-McCormick path). Sound at r=0.5.
    """
    B = A_lo_a.shape[0]
    n_input = A_lo_a.shape[-1]
    # Broadcast both to sh_out via ones_out
    ones_out = torch.ones(*sh_out, dtype=A_lo_a.dtype, device=A_lo_a.device)
    a_lo_o = (ones_out * lo_box_a.reshape(B, *sh_a)).reshape(B, -1)
    a_hi_o = (ones_out * hi_box_a.reshape(B, *sh_a)).reshape(B, -1)
    b_lo_o2 = (ones_out * lo_box_b.reshape(B, *sh_b)).reshape(B, -1)
    b_hi_o2 = (ones_out * hi_box_b.reshape(B, *sh_b)).reshape(B, -1)
    A_lo_a_o = (ones_out.unsqueeze(-1)
                  * A_lo_a.reshape(B, *sh_a, n_input)).reshape(B, -1, n_input)
    A_up_a_o = (ones_out.unsqueeze(-1)
                  * A_up_a.reshape(B, *sh_a, n_input)).reshape(B, -1, n_input)
    b_lo_a_o = (ones_out * b_lo_a.reshape(B, *sh_a)).reshape(B, -1)
    b_up_a_o = (ones_out * b_up_a.reshape(B, *sh_a)).reshape(B, -1)
    A_lo_b_o = (ones_out.unsqueeze(-1)
                  * A_lo_b.reshape(B, *sh_b, n_input)).reshape(B, -1, n_input)
    A_up_b_o = (ones_out.unsqueeze(-1)
                  * A_up_b.reshape(B, *sh_b, n_input)).reshape(B, -1, n_input)
    b_lo_b_o = (ones_out * b_lo_b.reshape(B, *sh_b)).reshape(B, -1)
    b_up_b_o = (ones_out * b_up_b.reshape(B, *sh_b)).reshape(B, -1)
    A_lo_o, b_lo_o, A_up_o, b_up_o = _mccormick_substitute(
        a_lo_o, a_hi_o, b_lo_o2, b_hi_o2,
        A_lo_a_o, A_up_a_o, b_lo_a_o, b_up_a_o,
        A_lo_b_o, A_up_b_o, b_lo_b_o, b_up_b_o)
    lo_box_o, hi_box_o = _box_lo_hi(A_lo_o, b_lo_o, A_up_o, b_up_o, xl_eval, xh_eval)
    return A_lo_o, b_lo_o, A_up_o, b_up_o, lo_box_o, hi_box_o


def div_bilinear_op(A_lo_a, b_lo_a, A_up_a, b_up_a, lo_box_a, hi_box_a,
                    A_lo_b, b_lo_b, A_up_b, b_up_b, lo_box_b, hi_box_b,
                    sh_a, sh_b, sh_out, xl_eval, xh_eval):
    """y = a / b — Reciprocal · Mul refactor (matches forward_lirpa).

    Reciprocal of b: chord UB, tangent LB at midpoint.
    Then McCormick(a, 1/b).
    Assumes b > 0 (graph-static for mscn softmax denominators).
    """
    from .verify_zono_bnb import _reciprocal_linear_bounds
    B = A_lo_a.shape[0]
    n_input = A_lo_a.shape[-1]
    ones_out = torch.ones(*sh_out, dtype=A_lo_a.dtype, device=A_lo_a.device)
    # Broadcast a + b
    a_lo_o = (ones_out * lo_box_a.reshape(B, *sh_a)).reshape(B, -1)
    a_hi_o = (ones_out * hi_box_a.reshape(B, *sh_a)).reshape(B, -1)
    b_lo_o2 = (ones_out * lo_box_b.reshape(B, *sh_b)).reshape(B, -1)
    b_hi_o2 = (ones_out * hi_box_b.reshape(B, *sh_b)).reshape(B, -1)
    A_lo_a_o = (ones_out.unsqueeze(-1)
                  * A_lo_a.reshape(B, *sh_a, n_input)).reshape(B, -1, n_input)
    A_up_a_o = (ones_out.unsqueeze(-1)
                  * A_up_a.reshape(B, *sh_a, n_input)).reshape(B, -1, n_input)
    b_lo_a_o = (ones_out * b_lo_a.reshape(B, *sh_a)).reshape(B, -1)
    b_up_a_o = (ones_out * b_up_a.reshape(B, *sh_a)).reshape(B, -1)
    A_lo_b_o = (ones_out.unsqueeze(-1)
                  * A_lo_b.reshape(B, *sh_b, n_input)).reshape(B, -1, n_input)
    A_up_b_o = (ones_out.unsqueeze(-1)
                  * A_up_b.reshape(B, *sh_b, n_input)).reshape(B, -1, n_input)
    b_lo_b_o = (ones_out * b_lo_b.reshape(B, *sh_b)).reshape(B, -1)
    b_up_b_o = (ones_out * b_up_b.reshape(B, *sh_b)).reshape(B, -1)
    # Reciprocal bound of b: returns (slope_lb, const_lb, slope_ub, const_ub)
    rs_lb, rc_lb, rs_ub, rc_ub = _reciprocal_linear_bounds(
        b_lo_o2, b_hi_o2, skip_positivity_check=True)
    # Substitute reciprocal into b's linear form: ib_LB = rs_lb*b_LB + rc_lb
    # (when rs_lb < 0 swap LB/UB on b). Use sign-conditional.
    rs_lb_p = rs_lb.clamp(min=0); rs_lb_n = rs_lb.clamp(max=0)
    rs_ub_p = rs_ub.clamp(min=0); rs_ub_n = rs_ub.clamp(max=0)
    # Linear bounds for 1/b
    A_lo_ib = rs_lb_p.unsqueeze(-1) * A_lo_b_o + rs_lb_n.unsqueeze(-1) * A_up_b_o
    b_lo_ib = rs_lb_p * b_lo_b_o + rs_lb_n * b_up_b_o + rc_lb
    A_up_ib = rs_ub_p.unsqueeze(-1) * A_up_b_o + rs_ub_n.unsqueeze(-1) * A_lo_b_o
    b_up_ib = rs_ub_p * b_up_b_o + rs_ub_n * b_lo_b_o + rc_ub
    # Box bounds for 1/b
    ib_lo = 1.0 / b_hi_o2
    ib_hi = 1.0 / b_lo_o2
    # Now McCormick: y = a * (1/b)
    A_lo_o, b_lo_o, A_up_o, b_up_o = _mccormick_substitute(
        a_lo_o, a_hi_o, ib_lo, ib_hi,
        A_lo_a_o, A_up_a_o, b_lo_a_o, b_up_a_o,
        A_lo_ib, A_up_ib, b_lo_ib, b_up_ib)
    lo_box_o, hi_box_o = _box_lo_hi(A_lo_o, b_lo_o, A_up_o, b_up_o, xl_eval, xh_eval)
    return A_lo_o, b_lo_o, A_up_o, b_up_o, lo_box_o, hi_box_o


def _sanitize(name):
    """Make `name` a valid Python identifier — gg op names like '21' or
    'op/x' won't work as variable names directly."""
    out = []
    for c in str(name):
        if c.isalnum() or c == '_':
            out.append(c)
        else:
            out.append('_')
    s = ''.join(out)
    if s and s[0].isdigit():
        s = 'op_' + s
    return s


def compile_forward(gg, varying_mask, K_int, device, dtype):
    """Walk gg, emit a straight-line forward function.

    Returns a callable: `fwd(xl_b, xh_b) → (A_lo, b_lo, A_up, b_up,
    lo_box, hi_box)` for the OUTPUT op. All intermediate state is local
    — Python GC frees as each op's output goes out of scope.

    The generated function captures op constants (W, bias, etc.) via
    closure. Caller pre-computes varying_mask and K_int (no per-call sync).
    """
    input_name = gg['input_name']
    ops = gg['ops']
    last_name = ops[-1]['name']
    n_in = int(varying_mask.numel())

    # Pre-extract op constants and stage them with sanitized var names.
    consts = {}  # var_name → tensor
    closure = {}
    closure['_fc_op'] = fc_op
    closure['_relu_op'] = relu_op
    closure['_slice_op'] = slice_op
    closure['_add_op'] = add_op
    closure['_bias_add_op'] = bias_add_op
    closure['_reduce_sum_op'] = reduce_sum_op
    closure['_concat_op'] = concat_op
    closure['_sigmoid_op'] = sigmoid_op
    closure['_sub_bilinear_op'] = sub_bilinear_op
    closure['_mul_bilinear_op'] = mul_bilinear_op
    closure['_div_bilinear_op'] = div_bilinear_op
    closure['torch'] = torch
    closure['_n_in'] = n_in
    closure['_K'] = K_int
    closure['_varying_mask'] = varying_mask
    closure['_const_mask'] = ~varying_mask

    # Pre-compute var_idx and A_id template
    var_idx = varying_mask.nonzero(as_tuple=True)[0]
    closure['_var_idx'] = var_idx

    lines = []
    # Initial input state — build A_id (B, n_in, K) + b_zero + xl_eval/xh_eval
    in_var = _sanitize(input_name)
    lines.append(f"B = xl_b.shape[0]")
    lines.append(f"_A_id_proto = torch.zeros(B, _n_in, _K, dtype=xl_b.dtype, device=xl_b.device)")
    lines.append(f"_A_id_proto[:, _var_idx, torch.arange(_K, device=xl_b.device)] = 1.0")
    lines.append(f"_const_mask_f = _const_mask.to(xl_b.dtype)")
    lines.append(f"_b_zero_init = xl_b * _const_mask_f")
    lines.append(f"_xl_eval = xl_b[:, _varying_mask].contiguous()")
    lines.append(f"_xh_eval = xh_b[:, _varying_mask].contiguous()")
    lines.append(f"A_lo_{in_var} = _A_id_proto")
    lines.append(f"A_up_{in_var} = _A_id_proto")
    lines.append(f"b_lo_{in_var} = _b_zero_init")
    lines.append(f"b_up_{in_var} = _b_zero_init")
    lines.append(f"lo_box_{in_var} = xl_b.clone()")
    lines.append(f"hi_box_{in_var} = xh_b.clone()")

    # Per-op codegen
    for op in ops:
        nm = _sanitize(op['name'])
        t = op['type']
        ins = [_sanitize(i) for i in op['inputs']]
        if t == 'fc':
            # Stage const tensors
            W_var = f"_W_{nm}"; W_abs_var = f"_W_abs_{nm}"; bias_var = f"_bias_{nm}"
            W = op['W']
            if isinstance(W, np.ndarray):
                W = torch.from_numpy(W).to(device=device, dtype=dtype)
            elif not isinstance(W, torch.Tensor):
                W = torch.tensor(W, device=device, dtype=dtype)
            else:
                W = W.to(device=device, dtype=dtype)
            bias = op['bias']
            if isinstance(bias, np.ndarray):
                bias = torch.from_numpy(bias).to(device=device, dtype=dtype)
            elif not isinstance(bias, torch.Tensor):
                bias = torch.tensor(bias, device=device, dtype=dtype)
            else:
                bias = bias.to(device=device, dtype=dtype)
            closure[W_var] = W
            closure[W_abs_var] = W.abs()
            closure[bias_var] = bias
            in_shape_nd = op.get('in_shapes_nd', [None])[0]
            in_shape_var = f"_in_shape_{nm}"
            closure[in_shape_var] = in_shape_nd
            in0 = ins[0]
            lines.append(
                f"A_lo_{nm}, b_lo_{nm}, A_up_{nm}, b_up_{nm}, lo_box_{nm}, hi_box_{nm} = "
                f"_fc_op(A_lo_{in0}, b_lo_{in0}, A_up_{in0}, b_up_{in0}, "
                f"lo_box_{in0}, hi_box_{in0}, "
                f"{W_var}, {W_abs_var}, {bias_var}, _xl_eval, _xh_eval, {in_shape_var})")
        elif t == 'relu':
            in0 = ins[0]
            lines.append(
                f"A_lo_{nm}, b_lo_{nm}, A_up_{nm}, b_up_{nm}, lo_box_{nm}, hi_box_{nm} = "
                f"_relu_op(A_lo_{in0}, b_lo_{in0}, A_up_{in0}, b_up_{in0}, "
                f"lo_box_{in0}, hi_box_{in0}, _xl_eval, _xh_eval)")
        elif t == 'slice':
            flat_idx = op.get('flat_idx')
            if flat_idx is None:
                raise NotImplementedError(f"slice op {op['name']!r}: no flat_idx")
            if isinstance(flat_idx, torch.Tensor):
                idx_t = flat_idx.to(device=device, dtype=torch.long)
            else:
                idx_t = torch.as_tensor(flat_idx, dtype=torch.long, device=device)
            idx_var = f"_idx_{nm}"
            closure[idx_var] = idx_t
            in0 = ins[0]
            lines.append(
                f"A_lo_{nm}, b_lo_{nm}, A_up_{nm}, b_up_{nm}, lo_box_{nm}, hi_box_{nm} = "
                f"_slice_op(A_lo_{in0}, b_lo_{in0}, A_up_{in0}, b_up_{in0}, "
                f"lo_box_{in0}, hi_box_{in0}, {idx_var})")
        elif t == 'add':
            if op.get('is_merge'):
                i0, i1 = ins[0], ins[1]
                lines.append(
                    f"A_lo_{nm}, b_lo_{nm}, A_up_{nm}, b_up_{nm}, lo_box_{nm}, hi_box_{nm} = "
                    f"_add_op(A_lo_{i0}, b_lo_{i0}, A_up_{i0}, b_up_{i0}, lo_box_{i0}, hi_box_{i0}, "
                    f"A_lo_{i1}, b_lo_{i1}, A_up_{i1}, b_up_{i1}, lo_box_{i1}, hi_box_{i1}, "
                    f"_xl_eval, _xh_eval)")
            else:
                # bias-add: 1 bound input + a constant bias (with optional broadcasting)
                i0 = ins[0]
                bias = op.get('bias')
                bias_var = f"_bias_add_{nm}"
                if bias is None:
                    closure[bias_var] = None
                else:
                    bias_t = torch.as_tensor(np.asarray(bias).flatten(),
                                              dtype=dtype, device=device)
                    out_shape_nd = op.get('out_shape_nd')
                    # If bias needs broadcasting to out_shape, do it once at compile
                    if (out_shape_nd is not None
                            and bias_t.numel() != int(np.prod(out_shape_nd))):
                        bias_shape = list(bias.shape) if hasattr(bias, 'shape') else [bias_t.numel()]
                        bias_nd = bias_t.reshape(*bias_shape)
                        ones_out = torch.ones(*out_shape_nd, dtype=dtype, device=device)
                        bias_t = (ones_out * bias_nd).reshape(-1)
                    closure[bias_var] = bias_t
                lines.append(
                    f"A_lo_{nm}, b_lo_{nm}, A_up_{nm}, b_up_{nm}, lo_box_{nm}, hi_box_{nm} = "
                    f"_bias_add_op(A_lo_{i0}, b_lo_{i0}, A_up_{i0}, b_up_{i0}, "
                    f"lo_box_{i0}, hi_box_{i0}, {bias_var}, _xl_eval, _xh_eval)")
        elif t == 'reduce_sum':
            in_shape_nd = op.get('in_shapes_nd', [None])[0]
            if in_shape_nd is None:
                in_shape_nd = op.get('in_shape_nd')
            axes = op.get('axes', [])
            keepdims = op.get('keepdims', False)
            in_shape_var = f"_in_shape_{nm}"
            axes_var = f"_axes_{nm}"
            keepdims_var = f"_keepdims_{nm}"
            closure[in_shape_var] = list(in_shape_nd)
            closure[axes_var] = list(axes)
            closure[keepdims_var] = bool(keepdims)
            in0 = ins[0]
            lines.append(
                f"A_lo_{nm}, b_lo_{nm}, A_up_{nm}, b_up_{nm}, lo_box_{nm}, hi_box_{nm} = "
                f"_reduce_sum_op(A_lo_{in0}, b_lo_{in0}, A_up_{in0}, b_up_{in0}, "
                f"lo_box_{in0}, hi_box_{in0}, "
                f"{in_shape_var}, {axes_var}, {keepdims_var}, _xl_eval, _xh_eval)")
        elif t == 'concat':
            bd_list = ", ".join(
                f"(A_lo_{i}, b_lo_{i}, A_up_{i}, b_up_{i}, lo_box_{i}, hi_box_{i})"
                for i in ins)
            lines.append(
                f"A_lo_{nm}, b_lo_{nm}, A_up_{nm}, b_up_{nm}, lo_box_{nm}, hi_box_{nm} = "
                f"_concat_op([{bd_list}], _xl_eval, _xh_eval)")
        elif t == 'sigmoid':
            in0 = ins[0]
            lines.append(
                f"A_lo_{nm}, b_lo_{nm}, A_up_{nm}, b_up_{nm}, lo_box_{nm}, hi_box_{nm} = "
                f"_sigmoid_op(A_lo_{in0}, b_lo_{in0}, A_up_{in0}, b_up_{in0}, "
                f"lo_box_{in0}, hi_box_{in0}, _xl_eval, _xh_eval)")
        elif t == 'sub_bilinear':
            i0, i1 = ins[0], ins[1]
            lines.append(
                f"A_lo_{nm}, b_lo_{nm}, A_up_{nm}, b_up_{nm}, lo_box_{nm}, hi_box_{nm} = "
                f"_sub_bilinear_op(A_lo_{i0}, b_lo_{i0}, A_up_{i0}, b_up_{i0}, lo_box_{i0}, hi_box_{i0}, "
                f"A_lo_{i1}, b_lo_{i1}, A_up_{i1}, b_up_{i1}, lo_box_{i1}, hi_box_{i1}, "
                f"_xl_eval, _xh_eval)")
        elif t == 'mul_bilinear':
            i0, i1 = ins[0], ins[1]
            sh_a = op.get('in_shapes_nd', [None, None])[0]
            sh_b = op.get('in_shapes_nd', [None, None])[1]
            sh_out = op.get('out_shape_nd')
            sh_a_var = f"_sh_a_{nm}"; sh_b_var = f"_sh_b_{nm}"; sh_out_var = f"_sh_out_{nm}"
            closure[sh_a_var] = list(sh_a); closure[sh_b_var] = list(sh_b)
            closure[sh_out_var] = list(sh_out)
            lines.append(
                f"A_lo_{nm}, b_lo_{nm}, A_up_{nm}, b_up_{nm}, lo_box_{nm}, hi_box_{nm} = "
                f"_mul_bilinear_op(A_lo_{i0}, b_lo_{i0}, A_up_{i0}, b_up_{i0}, lo_box_{i0}, hi_box_{i0}, "
                f"A_lo_{i1}, b_lo_{i1}, A_up_{i1}, b_up_{i1}, lo_box_{i1}, hi_box_{i1}, "
                f"{sh_a_var}, {sh_b_var}, {sh_out_var}, _xl_eval, _xh_eval)")
        elif t == 'div_bilinear':
            i0, i1 = ins[0], ins[1]
            sh_a = op.get('in_shapes_nd', [None, None])[0]
            sh_b = op.get('in_shapes_nd', [None, None])[1]
            sh_out = op.get('out_shape_nd')
            sh_a_var = f"_sh_a_{nm}"; sh_b_var = f"_sh_b_{nm}"; sh_out_var = f"_sh_out_{nm}"
            closure[sh_a_var] = list(sh_a); closure[sh_b_var] = list(sh_b)
            closure[sh_out_var] = list(sh_out)
            lines.append(
                f"A_lo_{nm}, b_lo_{nm}, A_up_{nm}, b_up_{nm}, lo_box_{nm}, hi_box_{nm} = "
                f"_div_bilinear_op(A_lo_{i0}, b_lo_{i0}, A_up_{i0}, b_up_{i0}, lo_box_{i0}, hi_box_{i0}, "
                f"A_lo_{i1}, b_lo_{i1}, A_up_{i1}, b_up_{i1}, lo_box_{i1}, hi_box_{i1}, "
                f"{sh_a_var}, {sh_b_var}, {sh_out_var}, _xl_eval, _xh_eval)")
        else:
            raise NotImplementedError(
                f'compile_forward: unsupported op type {t!r} at {op["name"]!r}')

    # Return final op's bound
    out = _sanitize(last_name)
    lines.append(f"return A_lo_{out}, b_lo_{out}, A_up_{out}, b_up_{out}, lo_box_{out}, hi_box_{out}")

    # Assemble the function source
    body = "\n    ".join(lines)
    source = f"def _generated_fwd(xl_b, xh_b):\n    {body}\n"
    code = compile(source, "<vibecheck-bounded-forward>", "exec")
    exec(code, closure)
    fwd = closure['_generated_fwd']
    # Required by torch.jit.trace: function needs a __module__ attribute.
    fwd.__module__ = __name__
    fwd._source = source  # for debugging
    fwd._n_ops = len(ops)
    return fwd
