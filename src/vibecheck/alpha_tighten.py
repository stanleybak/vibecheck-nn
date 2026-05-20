"""GPU-batched α-CROWN layer tightening with shared- or per-target-α.

For each unstable target neuron j at layer L, run CROWN backward from L's
pre-activation through every earlier ReLU layer with a learnable α. Adam-
optimize α to maximize the LB (and separately UB).

Supports general graph ops (conv, fc, relu, add, sub, reshape) by mirroring
`alpha_crown._crown_backward_matrix`'s per-op walk, with per-target slopes
(shape `(n_targets, n_neur_at_layer)`) instead of the shared `(n_neur,)`.

Two modes:
  per_target=True  → one α-vector per (target, prior layer's unstable).
                     Matches the LP triangle relaxation at convergence.
                     Memory: `O(sum_L n_unstable_L · n_targets)` floats.
  per_target=False → one α-vector per layer, shared across all targets.
                     Faster but bound is slightly looser than LP (Adam
                     balances across the sum-loss).
"""
import time
import numpy as np
import torch
import torch.nn.functional as F


def _walk_backward_per_target(
        gg, xl_t, xh_t, alpha_per_layer, bbr_t, start_op_name,
        ew_init, n_targets, device, dtype, mode='lb', per_target=True):
    """Generic CROWN-style backward walk through gg ops with per-target α.

    Args:
        ew_init: shape `(n_targets, n_neur_at_start)` — coefficient of
                 z_target on the start op's output.
        alpha_per_layer: dict {layer_idx: α-tensor}.
                         If per_target=True, α-tensor shape `(n_targets, n_un)`
                         OR `(n_targets, n_neur)` (sparse only-unstable form
                         vs full-layer form, detected by sizes).
                         If per_target=False, α-tensor shape `(n_un,)` or
                         `(n_neur,)`.
        bbr_t: dict {layer_idx: (lo, hi)} torch tensors.
        mode: 'lb' returns LB on ew_init·z_target, 'ub' returns UB.
              For UB, equivalent to LB(-z_target) negated; we just flip
              the ew_init sign internally.
    """
    ops = gg['ops']
    if mode == 'ub':
        ew_init = -ew_init
    start_idx = next(i for i, op in enumerate(ops)
                      if op['name'] == start_op_name)
    ew_at = {start_op_name: ew_init}
    acc = torch.zeros(n_targets, device=device, dtype=dtype)

    for i in range(start_idx, -1, -1):
        op = ops[i]; name = op['name']
        if name not in ew_at: continue
        ew = ew_at[name]; t = op['type']
        if t == 'conv':
            out_shape = op['out_shape']
            kernel = op['kernel'].to(dtype=dtype, device=device)
            bias = op['bias'].to(dtype=dtype, device=device)
            C_out, H_out, W_out = out_shape
            ew_4d = ew.reshape(n_targets, C_out, H_out, W_out)
            acc = acc + (ew_4d.sum(dim=(-1, -2)) * bias).sum(dim=-1)
            ew_back = F.conv_transpose2d(
                ew_4d, kernel, stride=op['stride'], padding=op['padding'],
                output_padding=op['output_padding']).reshape(n_targets, -1)
            inp = op['inputs'][0]
            ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_back)) + ew_back
        elif t == 'fc':
            W = op['W'].to(dtype=dtype, device=device)
            bias = op['bias'].to(dtype=dtype, device=device)
            acc = acc + ew @ bias
            ew_back = ew @ W
            inp = op['inputs'][0]
            ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_back)) + ew_back
        elif t == 'relu':
            if 'layer_idx' in op:
                L = op['layer_idx']
                lo_t, hi_t = bbr_t[L]
                ub_r = torch.clamp(hi_t, min=0)
                lb_r = torch.clamp(lo_t, max=0)
                ub_r = torch.maximum(ub_r, lb_r + 1e-8)
                up_s = ub_r / (ub_r - lb_r)        # chord slope (n_neur,)
                up_t = -lb_r * up_s                # chord offset (n_neur,)
                active = lo_t >= 0
                dead = hi_t <= 0
                unstable = (~active) & (~dead)
                n_neur = lo_t.numel()

                if alpha_per_layer is not None and L in alpha_per_layer:
                    alpha = alpha_per_layer[L]
                    if per_target:
                        # alpha shape: (n_targets, n_un) sparse-only-unstable form
                        un_idx = unstable.nonzero().flatten()
                        # Build full per-target slope (n_targets, n_neur)
                        eff_slope = active.to(dtype).unsqueeze(0).expand(
                            n_targets, n_neur).contiguous()
                        if un_idx.numel() > 0:
                            eff_slope = eff_slope.index_copy(1, un_idx, alpha)
                    else:
                        # Shared mode: alpha shape (n_un,) sparse
                        un_idx = unstable.nonzero().flatten()
                        if un_idx.numel() > 0:
                            alpha_full = active.to(dtype).index_copy(
                                0, un_idx, alpha)
                            eff_slope = alpha_full.unsqueeze(0)  # (1, n_neur)
                        else:
                            eff_slope = active.to(dtype).unsqueeze(0)
                else:
                    # No α: min-area default
                    eff_slope = (active.to(dtype) +
                                  unstable.to(dtype) * (up_s > 0.5).to(dtype))
                    eff_slope = eff_slope.unsqueeze(0)
                ep = ew.clamp(min=0); en = ew.clamp(max=0)
                acc = acc + (en * up_t).sum(dim=-1)
                ew_back = ep * eff_slope + en * up_s
            else:
                ew_back = ew
            inp = op['inputs'][0]
            ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_back)) + ew_back
        elif t == 'add':
            if op.get('is_merge'):
                for inp in op['inputs']:
                    ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew)) + ew
            else:
                bias = op.get('bias')
                if bias is not None:
                    bt = torch.tensor(
                        bias.flatten(), dtype=dtype, device=device)
                    acc = acc + ew @ bt
                inp = op['inputs'][0]
                ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew)) + ew
        elif t == 'sub':
            bias = op.get('bias')
            if bias is not None:
                bt = torch.tensor(bias.flatten(), dtype=dtype, device=device)
                acc = acc - ew @ bt
            inp = op['inputs'][0]
            ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew)) + ew
        elif t == 'reshape':
            inp = op['inputs'][0]
            ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew)) + ew

    input_name = gg['input_name']
    ew_inp = ew_at.get(input_name)
    box_min = (ew_inp.clamp(min=0) * xl_t.unsqueeze(0) +
               ew_inp.clamp(max=0) * xh_t.unsqueeze(0)).sum(dim=-1)
    lb = acc + box_min
    if mode == 'ub':
        return -lb
    return lb


def tighten_layer_alpha_crown(
        gg, xl, xh, bbr, target_layer, *, device, dtype,
        n_iters=50, lr=0.05, target_indices=None,
        per_target=True, return_timing=False):
    """Tighten unstable neurons at `target_layer` via per-target α-CROWN.

    Args:
        gg: gpu_graph dict.
        xl, xh: input box (numpy or tensor).
        bbr: dict {layer_idx: (lo, hi)} numpy arrays.
        target_layer: int — ReLU layer index to tighten.
        n_iters, lr: Adam hyperparameters.
        target_indices: explicit list of neuron indices; defaults to all unstable.
        per_target: True → per-target α (LP-equivalent), False → shared α.

    Returns: (new_lo, new_hi) numpy arrays of size matching bbr[target_layer].
    """
    # Find the op producing target_layer's pre-activation.
    target_pre_op = None
    for op in gg['ops']:
        if op['type'] == 'relu' and op.get('layer_idx') == target_layer:
            target_pre_op = op['inputs'][0]
            break
    if target_pre_op is None:
        raise ValueError(f'no relu op for layer_idx={target_layer}')

    lo_t_np, hi_t_np = bbr[target_layer]
    if target_indices is None:
        target_indices = np.where((lo_t_np < 0) & (hi_t_np > 0))[0]
    target_indices = np.asarray(target_indices, dtype=np.int64)
    n_targets = len(target_indices)
    if n_targets == 0:
        if return_timing:
            return lo_t_np.copy(), hi_t_np.copy(), {'wall': 0.0}
        return lo_t_np.copy(), hi_t_np.copy()

    xl_t = torch.as_tensor(np.asarray(xl).flatten(), device=device, dtype=dtype)
    xh_t = torch.as_tensor(np.asarray(xh).flatten(), device=device, dtype=dtype)

    # bbr tensors for prior layers (and target layer for slope at it isn't
    # needed since walk stops AT target's pre-relu).
    bbr_t = {L: (torch.as_tensor(bbr[L][0], device=device, dtype=dtype),
                  torch.as_tensor(bbr[L][1], device=device, dtype=dtype))
              for L in bbr if L < target_layer}

    # Per-prior-layer unstable indices and α params.
    alpha_lb = {}; alpha_ub = {}
    for L in range(target_layer):
        lo_l = bbr_t[L][0]; hi_l = bbr_t[L][1]
        un = ((lo_l < 0) & (hi_l > 0)).nonzero().flatten()
        if un.numel() == 0:
            continue
        if per_target:
            shape = (n_targets, un.numel())
        else:
            shape = (un.numel(),)
        alpha_lb[L] = torch.full(shape, 0.5, device=device, dtype=dtype,
                                  requires_grad=True)
        alpha_ub[L] = torch.full(shape, 0.5, device=device, dtype=dtype,
                                  requires_grad=True)

    # ew_init: identity rows for target indices, shape (n_targets, n_at_target)
    n_at_target = lo_t_np.size
    target_idx_t = torch.as_tensor(target_indices, device=device,
                                    dtype=torch.long)
    ew_init = torch.zeros(n_targets, n_at_target, device=device, dtype=dtype)
    ew_init[torch.arange(n_targets, device=device), target_idx_t] = 1.0

    t0 = time.perf_counter()
    # Optimize LB
    if alpha_lb:
        opt = torch.optim.Adam(list(alpha_lb.values()), lr=lr)
        best_lbs = torch.full((n_targets,), -float('inf'), device=device,
                                dtype=dtype)
        for it in range(n_iters):
            opt.zero_grad()
            lb = _walk_backward_per_target(
                gg, xl_t, xh_t, alpha_lb, bbr_t, target_pre_op,
                ew_init, n_targets, device, dtype, mode='lb',
                per_target=per_target)
            (-lb.sum()).backward()
            opt.step()
            with torch.no_grad():
                for p in alpha_lb.values():
                    p.clamp_(0, 1)
                best_lbs = torch.maximum(best_lbs, lb.detach())
    else:
        with torch.no_grad():
            best_lbs = _walk_backward_per_target(
                gg, xl_t, xh_t, alpha_lb, bbr_t, target_pre_op,
                ew_init, n_targets, device, dtype, mode='lb',
                per_target=per_target).detach()

    # Optimize UB
    if alpha_ub:
        opt = torch.optim.Adam(list(alpha_ub.values()), lr=lr)
        best_ubs = torch.full((n_targets,), float('inf'), device=device,
                                dtype=dtype)
        for it in range(n_iters):
            opt.zero_grad()
            ub = _walk_backward_per_target(
                gg, xl_t, xh_t, alpha_ub, bbr_t, target_pre_op,
                ew_init, n_targets, device, dtype, mode='ub',
                per_target=per_target)
            (ub.sum()).backward()
            opt.step()
            with torch.no_grad():
                for p in alpha_ub.values():
                    p.clamp_(0, 1)
                best_ubs = torch.minimum(best_ubs, ub.detach())
    else:
        with torch.no_grad():
            best_ubs = _walk_backward_per_target(
                gg, xl_t, xh_t, alpha_ub, bbr_t, target_pre_op,
                ew_init, n_targets, device, dtype, mode='ub',
                per_target=per_target).detach()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    new_lo = lo_t_np.astype(np.float64).copy()
    new_hi = hi_t_np.astype(np.float64).copy()
    lbs_np = best_lbs.cpu().numpy().astype(np.float64)
    ubs_np = best_ubs.cpu().numpy().astype(np.float64)
    new_lo[target_indices] = np.maximum(new_lo[target_indices], lbs_np)
    new_hi[target_indices] = np.minimum(new_hi[target_indices], ubs_np)

    if return_timing:
        return new_lo, new_hi, {'wall': dt, 'n_targets': n_targets,
                                 'n_iters': n_iters, 'per_target': per_target}
    return new_lo, new_hi
