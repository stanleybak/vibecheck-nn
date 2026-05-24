"""α-CROWN optimization + direction-adaptive forward zonotope reconstruction.

Pipeline:
  1. `run_alpha_crown(...)` — per-query α-CROWN with per-(L, start_node) α
     and joint intermediate bound recomputation each Adam iteration.
  2. `capture_ew_per_relu(...)` — after α-CROWN converges, do one more
     CROWN backward pass with the optimal α's to record the accumulated
     backward weight `ew` at each unstable ReLU.
  3. `build_dir_adaptive_alpha(...)` — per unstable neuron k at layer L:
        λ_k = α_k        if ep_k > 0  (lower triangle line α·z is tight)
        λ_k = up_s_k     else         (upper triangle line up_s·z + up_t is tight)
     Stable-on: λ = 1. Dead: λ = 0.
  4. `forward_zono_dir_adaptive(...)` — forward zonotope with per-neuron
     (λ, μ, shift). For λ = α: μ = max((1-α)·hi/2, -α·lo/2), shift = μ.
     For λ = up_s: μ = (1-up_s)·hi/2 = -up_s·lo/2, shift = μ (min-area case).

The forward zonotope's spec LB equals α-CROWN's backward LB to machine
precision (the direction-adaptive reconstruction picks the tight triangle
edge per neuron for the spec direction). The same zonotope is usable for
Phase 2.5 halfspace LP tightening of per-neuron pre-ReLU bounds.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .verify_zono_bnb import (
    _make_slopes, _find_shared_gens_count, _sigmoid_tanh_linear_bounds)
from .zonotope import TorchZonotope


def _mul_scale_to_tensor(op, dtype, device):
    scale_t = op.get('scale')
    if isinstance(scale_t, np.ndarray):
        scale_t = torch.from_numpy(scale_t).to(device=device, dtype=dtype)
    elif not isinstance(scale_t, torch.Tensor):
        scale_t = torch.tensor(scale_t, dtype=dtype, device=device)
    else:
        scale_t = scale_t.to(device=device, dtype=dtype)
    return scale_t.flatten()


def _mul_scale_broadcast(op, dtype, device, n_flat):
    """Resolve op['scale'] to a (n_flat,) tensor with per-channel broadcast."""
    sflat = _mul_scale_to_tensor(op, dtype, device)
    if sflat.numel() == 1 or sflat.numel() == n_flat:
        return sflat
    in_shape = op.get('in_shapes_nd', [None])[0]
    if in_shape is None or len(in_shape) != 3:
        raise ValueError(
            f'mul: scale {sflat.shape} incompatible with n={n_flat}; '
            f'no spatial shape')
    C, H, W = in_shape
    assert sflat.numel() == C
    return sflat.view(1, C, 1, 1).expand(1, C, H, W).reshape(-1)


def _mul_scale_backward(op, ew, dtype, device):
    """CROWN backward through y = scale * x: ew_back = ew * scale."""
    n = ew.shape[-1]
    sflat = _mul_scale_broadcast(op, dtype, device, n)
    return ew * sflat


def _mul_scale_zono(op, center, generators, dtype, device):
    """Forward zono through y = scale * x: scale center and gens."""
    n = center.numel()
    sflat = _mul_scale_broadcast(op, dtype, device, n)
    if sflat.numel() == 1:
        return center * sflat, generators * sflat
    return center * sflat, generators * sflat.unsqueeze(-1)


def _bias_dot_ew(ew, bias_np, dtype, device):
    """Compute the bias contribution to `acc` for an Add backward.

    Returns the tensor to ADD to `acc`. Handles scalar / per-channel
    broadcast against ew's last dim (the layer width):
      - scalar bias (numel=1) → scalar * ew.sum(dim=-1)
      - matching-size bias    → ew @ bias_vec
    """
    bt_full = torch.as_tensor(np.asarray(bias_np).flatten(),
                                 dtype=dtype, device=device)
    n = ew.shape[-1]
    if bt_full.numel() == n:
        return ew @ bt_full
    if bt_full.numel() == 1:
        return ew.sum(dim=-1) * bt_full
    raise ValueError(
        f'_bias_dot_ew: bias size {bt_full.numel()} incompatible with '
        f'ew last dim {n}')


# ---------------------------------------------------------------------------
# CROWN backward helpers (batched, gradient-capable)
# ---------------------------------------------------------------------------

def _crown_backward_matrix(gg, xl, xh, alpha_at_layer, bbr_tensors,
                            start_op_name, ew_init, device, dtype,
                            unstable_at_layer=None):
    """Batched CROWN backward from `start_op_name`'s output back to the
    network input. Returns `(lb_per_batch, ew_at_input)`.

    At each ReLU with `layer_idx == L`, uses `alpha_at_layer[L]` as the
    lower slope (falls back to min-area if absent). `bbr_tensors[L]`
    provides (lo, hi) as differentiable tensors for slope computation.

    Sparse-α support: if `unstable_at_layer is not None` and `L` is in it,
    `alpha_at_layer[L]` is interpreted as a (n_unstable,) sparse vector.
    A full-size slope vector is built by scatter-ing into a 1-tensor at
    the unstable indices (active gets 1.0, dead/stable-on/off get 0/1
    respectively). Matches AB-CROWN's `sparse_features_alpha` (relu.py:64).
    """
    ops = gg['ops']
    start_idx = next(i for i, op in enumerate(ops)
                     if op['name'] == start_op_name)
    ew_at = {start_op_name: ew_init}
    B = ew_init.shape[0]
    acc = torch.zeros(B, dtype=dtype, device=device)

    for i in range(start_idx, -1, -1):
        op = ops[i]; name = op['name']
        if name not in ew_at: continue
        ew = ew_at[name]; t = op['type']
        if t == 'conv':
            out_shape = op['out_shape']
            kernel = op['kernel'].to(dtype=dtype, device=device)
            bias = op['bias'].to(dtype=dtype, device=device)
            C_out, H_out, W_out = out_shape
            ew_4d = ew.reshape(B, C_out, H_out, W_out)
            acc = acc + (ew_4d.sum(dim=(-1, -2)) * bias).sum(dim=-1)
            ew_back = F.conv_transpose2d(
                ew_4d, kernel, stride=op['stride'], padding=op['padding'],
                output_padding=op['output_padding']).reshape(B, -1)
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
                lo_t, hi_t = bbr_tensors[L]
                ub_r = torch.clamp(hi_t, min=0)
                lb_r = torch.clamp(lo_t, max=0)
                ub_r = torch.maximum(ub_r, lb_r + 1e-8)
                up_s = ub_r / (ub_r - lb_r)
                up_t = -lb_r * up_s
                active = lo_t >= 0
                dead = hi_t <= 0
                unstable = (~active) & (~dead)
                # Build eff_slope as a single (n_neurons,) tensor with
                # autograd connected to alpha. Sparse-α: densify on-the-fly
                # via scatter so the autograd graph mirrors dense exactly:
                # alpha_full = zeros; alpha_full[un_idx] = alpha; eff = active +
                # alpha_full. The alpha->alpha_full edge is a single sparse
                # op, ~n_un elements; downstream graph (multiply by ep, etc.)
                # is identical in shape to dense. Adam state is still the
                # n_un-sized leaf alpha, so optimizer cost remains sparse.
                if alpha_at_layer is not None and L in alpha_at_layer:
                    alpha = alpha_at_layer[L]
                    if (unstable_at_layer is not None
                            and L in unstable_at_layer):
                        un_idx = unstable_at_layer[L]
                        # densify: alpha_full has zeros at non-unstable.
                        # Use index_copy on a fresh active.to(dtype) base —
                        # equivalent to dense eff_slope but autograd only
                        # connects through the n_un elements written.
                        alpha_full = active.to(dtype).index_copy(
                            0, un_idx, alpha)
                        eff_slope = alpha_full
                    else:
                        eff_slope = active.to(dtype) + unstable.to(dtype) * alpha
                else:
                    eff_slope = active.to(dtype) + (
                        unstable.to(dtype) * (up_s > 0.5).to(dtype))
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
                    acc = acc + _bias_dot_ew(ew, bias, dtype, device)
                inp = op['inputs'][0]
                ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew)) + ew
        elif t == 'sub':
            bias = op.get('bias')
            if bias is not None:
                acc = acc - _bias_dot_ew(ew, bias, dtype, device)
            inp = op['inputs'][0]
            ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew)) + ew
        elif t == 'reshape':
            inp = op['inputs'][0]
            ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew)) + ew

        elif t == 'slice':
            flat_idx = op.get('flat_idx')
            in_shape_nd = op.get('in_shapes_nd', [None])[0]
            n_in = int(np.prod(in_shape_nd)) if in_shape_nd is not None else None
            if flat_idx is None or n_in is None:
                raise ValueError("slice backward missing flat_idx/in_shape")
            idx_t = torch.as_tensor(flat_idx, dtype=torch.long, device=device)
            ew_back = torch.zeros(ew.shape[0], n_in, dtype=ew.dtype, device=device)
            ew_back.index_copy_(-1, idx_t, ew)
            inp = op['inputs'][0]
            ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_back)) + ew_back

        elif t == 'concat':
            # Split ew along last dim per input's flat size.
            in_shapes = op.get('in_shapes_nd', [])
            offset = 0
            for inp, in_shape_nd in zip(op['inputs'], in_shapes):
                n_in = int(np.prod(in_shape_nd))
                ew_i = ew[..., offset:offset + n_in]
                ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_i)) + ew_i
                offset += n_in

        elif t == 'mul':
            ew_back = _mul_scale_backward(op, ew, dtype, device)
            inp = op['inputs'][0]
            ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_back)) + ew_back

        elif t in ('sigmoid', 'tanh'):
            L = op.get('layer_idx')
            lo_pre, hi_pre = bbr_tensors[L][0], bbr_tensors[L][1]
            lo_s, lo_t, up_s, up_t = _sigmoid_tanh_linear_bounds(
                lo_pre, hi_pre, t)
            ep = ew.clamp(min=0); en = ew.clamp(max=0)
            acc = acc + (ep * lo_t).sum(dim=-1) + (en * up_t).sum(dim=-1)
            ew_back = ep * lo_s + en * up_s
            inp = op['inputs'][0]
            ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_back)) + ew_back

        else:
            raise NotImplementedError(
                f'_crown_backward_matrix: unsupported op {t!r} (name={name!r}). '
                'Silent skip would drop ew and produce unsound bounds.')

    input_name = gg['input_name']
    ew_inp = ew_at.get(input_name)
    xl_t = xl.to(dtype=dtype, device=device)
    xh_t = xh.to(dtype=dtype, device=device)
    lb = acc + ew_inp.clamp(min=0) @ xl_t + ew_inp.clamp(max=0) @ xh_t
    return lb, ew_inp


def _find_op_producing_relu_input(gg, L):
    for op in gg['ops']:
        if op['type'] == 'relu' and op.get('layer_idx') == L:
            return op['inputs'][0]
    return None


# ---------------------------------------------------------------------------
# α-CROWN optimization (per query, per-(L, start_node) α, joint intermediate)
# ---------------------------------------------------------------------------

def run_alpha_crown(gg, xl, xh, bbr_init, w_q, b_q,
                     intermediate_start_nodes, unstable_indices,
                     device, dtype, n_iters=20, lr=0.25, lr_decay=1.0,
                     max_iters=None, early_stop_eps_spec=None,
                     early_stop_eps_bounds=None, early_stop_patience=3,
                     early_stop_on_positive=False):
    """Run α-CROWN for a single query direction (w_q, b_q).

    Parameters per-(start_node, L) with start_node ∈ {intermediate_start_nodes, 'spec'}:
    for each pair (S, L < S), a trainable α_{S,L} tensor of shape (n_neurons_at_L,).
    Each backward pass uses α_params[S] (the α's associated with that start_node).

    Returns:
      best_lb (float): best spec LB seen.
      alpha_params (dict): {S: {L: tensor}} — trained α tensors.
      best_bounds (dict): {L: (lo_tensor, hi_tensor)} — iter-best intermediate
                           bounds (element-wise max/min across iters).
      history (list[float]): spec LB per iteration.

    If `early_stop_on_positive` is True, breaks as soon as `best_lb > 0`
    (after the current iter's α update has been applied to the best-bound
    tracker). Matches α,β-CROWN's `stop_criterion_final` — once the spec
    is provably safe, further Adam steps only waste wall time.
    """
    all_relu_layers = sorted(bbr_init.keys())
    start_nodes = list(intermediate_start_nodes) + ['spec']

    # Initialize α per (S, L < S) at min-area's lo_s (0 or 1).
    alpha_params = {}
    slopes_init_cache = {}
    for L in all_relu_layers:
        lo_t = torch.as_tensor(bbr_init[L][0], dtype=dtype, device=device)
        hi_t = torch.as_tensor(bbr_init[L][1], dtype=dtype, device=device)
        lo_s, up_s, up_t, active, dead, unstable = _make_slopes(lo_t, hi_t)
        slopes_init_cache[L] = (lo_s, up_s, active, dead, unstable)

    for S in start_nodes:
        alpha_params[S] = {}
        S_val = 10**9 if S == 'spec' else S
        for L in all_relu_layers:
            if L >= S_val: continue
            lo_s, up_s, active, dead, unstable = slopes_init_cache[L]
            lo_t = torch.as_tensor(bbr_init[L][0], dtype=dtype, device=device)
            alpha = torch.zeros_like(lo_t)
            alpha = alpha + active.to(dtype) * 1.0
            alpha = alpha + unstable.to(dtype) * lo_s
            alpha = alpha.clone().detach().requires_grad_(True)
            alpha_params[S][L] = alpha

    all_tensors = [alpha_params[S][L] for S in start_nodes
                   for L in alpha_params[S]]
    opt = torch.optim.Adam(all_tensors, lr=lr)
    scheduler = (torch.optim.lr_scheduler.ExponentialLR(opt, lr_decay)
                 if lr_decay != 1.0 else None)

    last_op = gg['ops'][-1]
    w_t = torch.as_tensor(w_q, dtype=dtype, device=device)

    best_lb = -float('inf'); history = []
    best_bounds = {
        L: (torch.as_tensor(bbr_init[L][0], dtype=dtype, device=device).clone(),
            torch.as_tensor(bbr_init[L][1], dtype=dtype, device=device).clone())
        for L in bbr_init}
    # Adaptive early stop: if any of (eps_spec, eps_bounds) is not None,
    # run up to max_iters (default: n_iters fallback if max_iters is None)
    # and early-stop when both spec LB and bounds have plateaued for
    # `early_stop_patience` consecutive iters.
    _adaptive = (early_stop_eps_spec is not None
                  or early_stop_eps_bounds is not None)
    _loop_iters = max_iters if _adaptive else n_iters
    if _adaptive and max_iters is None:
        _loop_iters = max(n_iters, 50)
    # Per-iter Σ(bound improvement) — tracked only in adaptive mode
    bound_deltas = []

    for it in range(_loop_iters):
        opt.zero_grad()
        # 1. Compute intermediate bounds for each S ∈ intermediate_start_nodes.
        bbr_tensors = {L: (
            torch.as_tensor(bbr_init[L][0], dtype=dtype, device=device).clone(),
            torch.as_tensor(bbr_init[L][1], dtype=dtype, device=device).clone(),
        ) for L in bbr_init}
        for S in sorted(intermediate_start_nodes):
            alpha_for_S = alpha_params[S]
            start_op = _find_op_producing_relu_input(gg, S)
            un_S = unstable_indices[S]
            if not un_S or start_op is None: continue
            n_S = bbr_init[S][0].size
            n_un = len(un_S)
            # Chunked intermediate-bound backward. Building a single
            # (n_un, n_S) ew_init tensor and keeping all backward-pass
            # intermediates alive peaks in the GB range for resnet_large
            # (n_un~2300, n_S=65536 ⇒ ~600 MB per tensor, 2 GB transient).
            # Chunking over the unstable batch bounds the per-step peak;
            # with autograd on, each chunk retains the full backward graph
            # so 128 gives ~150 MB per intermediate tensor.
            chunk = min(n_un, 128)
            lb_parts = []
            ub_parts = []
            un_idx_all = torch.as_tensor(un_S, device=device)
            for start in range(0, n_un, chunk):
                end = min(start + chunk, n_un)
                kc = end - start
                ew_init_lb = torch.zeros(
                    kc, n_S, dtype=dtype, device=device)
                ew_init_lb[torch.arange(kc, device=device),
                           un_idx_all[start:end]] = 1.0
                lb_part, _ = _crown_backward_matrix(
                    gg, xl, xh, alpha_for_S, bbr_tensors,
                    start_op, ew_init_lb, device, dtype)
                neg_ub_part, _ = _crown_backward_matrix(
                    gg, xl, xh, alpha_for_S, bbr_tensors,
                    start_op, -ew_init_lb, device, dtype)
                lb_parts.append(lb_part)
                ub_parts.append(-neg_ub_part)
            lb_batch = torch.cat(lb_parts, dim=0)
            ub_batch = torch.cat(ub_parts, dim=0)
            un_t = torch.as_tensor(un_S, device=device, dtype=torch.long)
            # MAX(init, recomputed) for LB, MIN(init, recomputed) for UB.
            # Trade-off measured empirically:
            #   • MAX: prop_5 256x6 in pipeline (tight Phase-1-init bounds)
            #     reaches spec_lb = −265 after 10 joint α-CROWN iters.
            #     REPLACE (let bounds drift) regresses this to −995 because
            #     with tight init, recomputed bound is almost always looser
            #     → drift makes spec_lb worse → Adam can't recover.
            #   • REPLACE: matches ABC's α-CROWN to within 2% on ABC's
            #     plain-CROWN-loose init bounds (−1843 vs −1810). MAX caps
            #     us at −4132 there because gradient stops flowing when
            #     the init wins.
            # MAX is right for our pipeline because Phase 1 always tightens
            # before Phase 2.5; we never start from loose plain-CROWN bounds.
            # Tracked across iters via `best_bounds` (per-neuron, max/min).
            lo_new = bbr_tensors[S][0].scatter(
                0, un_t, torch.maximum(bbr_tensors[S][0][un_t], lb_batch))
            hi_new = bbr_tensors[S][1].scatter(
                0, un_t, torch.minimum(bbr_tensors[S][1][un_t], ub_batch))
            bbr_tensors[S] = (lo_new, hi_new)

        # 2. Spec LB with spec's α's on updated intermediate bounds.
        spec_alpha = alpha_params['spec']
        ew_init = w_t.unsqueeze(0)
        lb_batch, _ = _crown_backward_matrix(
            gg, xl, xh, spec_alpha, bbr_tensors,
            last_op['name'], ew_init, device, dtype)
        spec_lb = lb_batch[0] + float(b_q)

        loss = -spec_lb
        loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            for S in start_nodes:
                for L, alpha in alpha_params[S].items():
                    _, _, active, dead, _ = slopes_init_cache[L]
                    alpha.clamp_(0.0, 1.0)
                    alpha[active] = 1.0
                    alpha[dead] = 0.0
            iter_bound_delta = 0.0
            for L in bbr_tensors:
                lo_t, hi_t = bbr_tensors[L]
                old_lo, old_hi = best_bounds[L]
                new_lo = torch.maximum(old_lo, lo_t.detach())
                new_hi = torch.minimum(old_hi, hi_t.detach())
                if _adaptive:
                    iter_bound_delta += float(
                        (new_lo - old_lo).clamp(min=0).sum()
                        + (old_hi - new_hi).clamp(min=0).sum())
                best_bounds[L] = (new_lo, new_hi)
            if _adaptive:
                bound_deltas.append(iter_bound_delta)

        val = float(spec_lb.detach())
        history.append(val)
        if val > best_lb: best_lb = val

        if early_stop_on_positive and best_lb > 0:
            break

        if _adaptive and it + 1 > early_stop_patience:
            P = early_stop_patience
            spec_ok = True
            if early_stop_eps_spec is not None:
                recent = max(history[-P:])
                prior = max(history[:-P])
                spec_ok = (recent - prior) < early_stop_eps_spec
            bounds_ok = True
            if early_stop_eps_bounds is not None:
                bounds_ok = (sum(bound_deltas[-P:])
                              < early_stop_eps_bounds * P)
            if spec_ok and bounds_ok:
                break

    # Apply best-across-iters per-neuron bounds for the FINAL spec_lb
    # computation: even if mid-iter bounds drifted (REPLACE in the loop),
    # `best_bounds[L]` is the per-neuron tightest LB / UB over all iters.
    # Recompute spec_lb with those tightest bounds — they are valid bounds
    # (each entry was a valid CROWN-derived LB/UB at some iter), so this
    # is sound. Take max with iter-best for safety.
    with torch.no_grad():
        bbr_best = {L: (best_bounds[L][0].detach(), best_bounds[L][1].detach())
                    for L in best_bounds}
        spec_alpha_final = alpha_params['spec']
        lb_final, _ = _crown_backward_matrix(
            gg, xl, xh, spec_alpha_final, bbr_best,
            last_op['name'], w_t.unsqueeze(0), device, dtype)
        final_val = float(lb_final[0].detach() + float(b_q))
        if final_val > best_lb:
            best_lb = final_val
            history.append(final_val)

    return best_lb, alpha_params, best_bounds, history


def run_alpha_crown_fixed_intermediate(
        gg, xl, xh, bbr_init, w_q, b_q,
        device, dtype, n_iters=20, lr=0.25, lr_decay=0.98,
        early_stop_on_positive=True, time_left_fn=None,
        init_alpha=None):
    """α-CROWN with fixed intermediate bounds and spec-only α.

    Matches α,β-CROWN's effective `fix_intermediate_bounds=True` config: the
    intermediate pre-ReLU bounds are frozen to `bbr_init` throughout the
    Adam loop, and ONLY the spec-path α is trainable. This simplifies the
    optimization dramatically (single α tensor per ReLU layer instead of
    O(n_start_nodes × n_relu_layers)) and removes the per-iter chunked
    intermediate backward (45 → 1 `_crown_backward_matrix` call per iter
    on CIFAR100_resnet_large).

    The lr_decay (default 0.98) applies an ExponentialLR schedule matching
    α,β-CROWN's default. Without decay, Adam plateaus ~0.01 below 0 on
    borderline queries; with decay, it crosses 0 in comparable iter counts
    (5–10 for α,β-CROWN-provable cases).

    Signature intentionally differs from `run_alpha_crown`: no
    `intermediate_start_nodes` / `unstable_indices` / `max_iters` /
    adaptive early-stop, since those are tied to the joint-α path.

    Returns:
      best_lb (float): best spec LB seen across iters.
      alpha_params (dict): `{ 'spec': { L: tensor } }` — only the spec path.
        Shape compatible with `capture_ew_per_relu` and
        `build_dir_adaptive_alpha`.
      best_bounds (dict): `{L: (lo, hi)}` — unchanged copy of `bbr_init`
        (intermediate bounds are frozen in this path).
      history (list[float]): spec LB per iteration.
    """
    all_relu_layers = sorted(bbr_init.keys())

    # Freeze intermediate bounds to tensors on `device`. These are used by
    # `_crown_backward_matrix` every iter without modification.
    bbr_tensors_fixed = {
        L: (
            torch.as_tensor(bbr_init[L][0], dtype=dtype, device=device),
            torch.as_tensor(bbr_init[L][1], dtype=dtype, device=device),
        ) for L in bbr_init
    }

    # Initialize spec α — warm-start from `init_alpha` (e.g. Phase 0.5's
    # already-optimised slopes) when provided; otherwise min-area lo_s.
    spec_alpha = {}
    slopes_cache = {}
    for L in all_relu_layers:
        lo_t = bbr_tensors_fixed[L][0]
        hi_t = bbr_tensors_fixed[L][1]
        lo_s, up_s, up_t, active, dead, unstable = _make_slopes(lo_t, hi_t)
        slopes_cache[L] = (active, dead)
        if (init_alpha is not None and L in init_alpha
                and init_alpha[L].numel() == lo_t.numel()):
            alpha = init_alpha[L].detach().to(
                device=device, dtype=dtype).clone()
        else:
            alpha = torch.zeros_like(lo_t)
            alpha = alpha + active.to(dtype) * 1.0
            alpha = alpha + unstable.to(dtype) * lo_s
        alpha = alpha.clone().detach().requires_grad_(True)
        spec_alpha[L] = alpha

    opt = torch.optim.Adam(list(spec_alpha.values()), lr=lr)
    if lr_decay != 1.0:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, lr_decay)
    else:
        scheduler = None

    last_op = gg['ops'][-1]
    w_t = torch.as_tensor(w_q, dtype=dtype, device=device)
    ew_init = w_t.unsqueeze(0)

    best_lb = -float('inf')
    history = []

    for it in range(n_iters):
        if time_left_fn is not None and time_left_fn() <= 0:
            break
        opt.zero_grad()
        lb_batch, _ = _crown_backward_matrix(
            gg, xl, xh, spec_alpha, bbr_tensors_fixed,
            last_op['name'], ew_init, device, dtype)
        spec_lb = lb_batch[0] + float(b_q)

        loss = -spec_lb
        loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            for L, alpha in spec_alpha.items():
                active, dead = slopes_cache[L]
                alpha.clamp_(0.0, 1.0)
                alpha[active] = 1.0
                alpha[dead] = 0.0

        val = float(spec_lb.detach())
        history.append(val)
        if val > best_lb:
            best_lb = val
        if early_stop_on_positive and best_lb > 0:
            break

    best_bounds = {
        L: (bbr_tensors_fixed[L][0].clone(), bbr_tensors_fixed[L][1].clone())
        for L in bbr_tensors_fixed
    }
    return best_lb, {'spec': spec_alpha}, best_bounds, history


def run_alpha_crown_fixed_intermediate_batched(
        gg, xl, xh, bbr_init, w_qs, b_qs,
        device, dtype, n_iters=20, lr=0.25, lr_decay=0.98,
        early_stop_on_positive=True, sparse_alpha=False,
        hopeless_lb=None, hopeless_delta=0.5,
        per_spec_alpha=False, time_left_fn=None):
    """Batched version of `run_alpha_crown_fixed_intermediate`: shared spec α
    across multiple queries, spec backward batched over `ew_init` of shape
    `(n_q, n_out)`. Closed queries are excluded from the loss sum so α
    doesn't over-fit to them.

    sparse_alpha: if True, allocate α only for unstable neurons per layer
      (matches AB-CROWN's `sparse_features_alpha=True` default — see
      `auto_LiRPA/operators/relu.py:64`). Adam state shrinks ~5x on
      cifar_biasfield (10010 unstable / 45056 neurons).

    per_spec_alpha: if True, allocate a separate α per query (shape
      `(n_q, n_neurons)`) instead of one tensor shared across queries.
      Each query optimizes its own α, so a slope choice that closes one
      spec but loosens another is no longer a compromise — every spec
      can pick the slope that maximizes its own LB. Mirrors α,β-CROWN's
      per-(spec, layer) α (the strict per-spec convention; vibecheck's
      shared α is a compromise variant). Memory: n_q × n_neurons floats
      per layer (n_q × n_unstable when sparse_alpha=True).

    Returns:
      best_lbs (np.ndarray shape (n_q,))
      alpha_params (dict): { 'spec': { L: tensor } } — shared α (or
        per-spec α with leading dim n_q when per_spec_alpha=True).
      best_bounds (dict): { L: (lo, hi) } — unchanged copy of bbr_init.
      histories (list of list[float]): spec LB per iter per query.
    """
    n_q = int(w_qs.shape[0])
    assert b_qs.shape == (n_q,)
    all_relu_layers = sorted(bbr_init.keys())

    bbr_tensors_fixed = {
        L: (
            torch.as_tensor(bbr_init[L][0], dtype=dtype, device=device),
            torch.as_tensor(bbr_init[L][1], dtype=dtype, device=device),
        ) for L in bbr_init
    }

    spec_alpha = {}
    slopes_cache = {}
    unstable_idx_per_layer = {}
    for L in all_relu_layers:
        lo_t = bbr_tensors_fixed[L][0]
        hi_t = bbr_tensors_fixed[L][1]
        lo_s, up_s, up_t, active, dead, unstable = _make_slopes(lo_t, hi_t)
        slopes_cache[L] = (active, dead)
        if sparse_alpha:
            un_idx = torch.nonzero(
                unstable, as_tuple=False).flatten().to(
                device=device, dtype=torch.long)
            unstable_idx_per_layer[L] = un_idx
            alpha_un = lo_s[un_idx].clone().detach()
            if per_spec_alpha:
                # Per-spec sparse α: shape (n_q, n_unstable) — each query
                # optimizes its own α. `_crown_backward_matrix` densifies
                # via the same `index_copy` path; the leading n_q dim
                # broadcasts through the upstream `ep * eff_slope` mat.
                alpha_un = alpha_un.unsqueeze(0).expand(
                    n_q, -1).contiguous()
            alpha_un = alpha_un.requires_grad_(True)
            spec_alpha[L] = alpha_un
        else:
            alpha = torch.zeros_like(lo_t)
            alpha = alpha + active.to(dtype) * 1.0
            alpha = alpha + unstable.to(dtype) * lo_s
            alpha = alpha.clone().detach()
            if per_spec_alpha:
                # Per-spec dense α: shape (n_q, n_neurons). Broadcasts
                # through `_crown_backward_matrix` ReLU step where ep is
                # shape (n_q, n_neurons). Memory cost: n_q × n_neurons
                # floats per layer (≤ a few MB on mnist_fc).
                alpha = alpha.unsqueeze(0).expand(n_q, -1).contiguous()
            alpha = alpha.requires_grad_(True)
            spec_alpha[L] = alpha

    opt = torch.optim.Adam(list(spec_alpha.values()), lr=lr)
    if lr_decay != 1.0:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, lr_decay)
    else:
        scheduler = None

    last_op = gg['ops'][-1]
    w_ts = torch.as_tensor(w_qs, dtype=dtype, device=device)
    b_ts = torch.as_tensor(b_qs, dtype=dtype, device=device)

    best_lbs = np.full(n_q, -np.inf, dtype=np.float64)
    histories = [[] for _ in range(n_q)]
    sparse_at = unstable_idx_per_layer if sparse_alpha else None

    for it in range(n_iters):
        if time_left_fn is not None and time_left_fn() <= 0:
            break
        opt.zero_grad()
        lb_batch, _ = _crown_backward_matrix(
            gg, xl, xh, spec_alpha, bbr_tensors_fixed,
            last_op['name'], w_ts, device, dtype,
            unstable_at_layer=sparse_at)
        spec_lb_batch = lb_batch + b_ts

        with torch.no_grad():
            closed_mask = torch.as_tensor(
                best_lbs > 0, device=device, dtype=torch.bool)
        active_q = ~closed_mask
        if active_q.any():
            loss = -spec_lb_batch[active_q].sum()
            loss.backward()
            opt.step()
            if scheduler is not None:
                scheduler.step()

        with torch.no_grad():
            for L, alpha in spec_alpha.items():
                if sparse_alpha:
                    alpha.clamp_(0.0, 1.0)
                else:
                    active, dead = slopes_cache[L]
                    alpha.clamp_(0.0, 1.0)
                    if per_spec_alpha:
                        # alpha shape (n_q, n_neurons); broadcast index
                        # along the last dim.
                        alpha[:, active] = 1.0
                        alpha[:, dead] = 0.0
                    else:
                        alpha[active] = 1.0
                        alpha[dead] = 0.0

        vals = spec_lb_batch.detach().cpu().numpy().astype(np.float64)
        for q in range(n_q):
            histories[q].append(float(vals[q]))
            if vals[q] > best_lbs[q]:
                best_lbs[q] = float(vals[q])

        if early_stop_on_positive and np.all(best_lbs > 0):
            break
        # Hopeless-bound early-exit (mirrors AB-CROWN's "skip α-CROWN
        # when initial bound is far negative, branch instead"): bail
        # if every query's running best is < hopeless_lb AND average
        # 3-iter improvement is < hopeless_delta. Caller provides
        # hopeless_lb=None to disable (default).
        if (it >= 5 and hopeless_lb is not None
                and np.min(best_lbs) < hopeless_lb):
            recent = []
            for q in range(n_q):
                if len(histories[q]) >= 4:
                    recent.append(histories[q][-1] - histories[q][-4])
            if recent and float(np.mean(recent)) < hopeless_delta:
                if not np.all(best_lbs > 0):
                    break

    best_bounds = {
        L: (bbr_tensors_fixed[L][0].clone(), bbr_tensors_fixed[L][1].clone())
        for L in bbr_tensors_fixed
    }
    return best_lbs, {'spec': spec_alpha}, best_bounds, histories


def run_alpha_crown_batched(
        gg, xl, xh, bbr_init, w_qs, b_qs,
        intermediate_start_nodes, unstable_indices,
        device, dtype, n_iters=20, lr=0.25, lr_decay=1.0,
        early_stop_on_positive=False, sparse_alpha=False,
        hopeless_lb=None, hopeless_delta=0.5,
        dir_mode='auto', s_split_n=1, time_left_fn=None):
    """Batched α-CROWN across multiple spec directions (w_qs, b_qs).

    α is shared across queries (one (n_L,) tensor per (start_node, layer)
    pair). The spec backward is batched over queries via an (n_q, n_out)
    ew_init. The Adam loss is `-sum(spec_lb[q] for q in not-yet-closed)`
    so already-closed queries don't keep dragging α toward their direction.

    Differs from α,β-CROWN's per-(n_spec, n_L) α: weaker in theory (shared
    α is a compromise across queries) but much simpler to implement, and
    the early-stop-at-positive regime for easy cases closes in 1-2 iters
    before per-query α would materially diverge.

    sparse_alpha: if True, allocate α only for unstable neurons per layer
      (matches AB-CROWN's `sparse_features_alpha=True` default — see
      `auto_LiRPA/operators/relu.py:64`). The α tensor at layer L has
      shape (n_unstable_at_L,) instead of (n_neurons_at_L,). On
      cifar_biasfield this slashes the optimizer parameter count by
      ~10x and proportionally shrinks per-iter Adam state updates.

    Returns:
      best_lbs (np.ndarray shape (n_q,)): best spec LB per query.
      alpha_params (dict): {S: {L: tensor}} — shared α.
      best_bounds (dict): {L: (lo_tensor, hi_tensor)} — iter-best intermediate.
      histories (list of list[float]): spec LB per iteration per query.
    """
    n_q = int(w_qs.shape[0])
    assert b_qs.shape == (n_q,)
    all_relu_layers = sorted(bbr_init.keys())
    start_nodes = list(intermediate_start_nodes) + ['spec']

    # Initialize α per (S, L < S) at min-area's lo_s — shared across queries.
    alpha_params = {}
    slopes_init_cache = {}
    unstable_idx_per_layer = {}  # only populated when sparse_alpha=True
    for L in all_relu_layers:
        lo_t = torch.as_tensor(bbr_init[L][0], dtype=dtype, device=device)
        hi_t = torch.as_tensor(bbr_init[L][1], dtype=dtype, device=device)
        lo_s, up_s, up_t, active, dead, unstable = _make_slopes(lo_t, hi_t)
        slopes_init_cache[L] = (lo_s, up_s, active, dead, unstable)
        if sparse_alpha:
            unstable_idx_per_layer[L] = torch.nonzero(
                unstable, as_tuple=False).flatten().to(
                device=device, dtype=torch.long)

    for S in start_nodes:
        alpha_params[S] = {}
        S_val = 10**9 if S == 'spec' else S
        for L in all_relu_layers:
            if L >= S_val: continue
            lo_s, up_s, active, dead, unstable = slopes_init_cache[L]
            lo_t = torch.as_tensor(bbr_init[L][0], dtype=dtype, device=device)
            if sparse_alpha:
                un_idx = unstable_idx_per_layer[L]
                # Sparse α: only the unstable neurons get a learnable slope.
                # Initialise at min-area's lo_s (0/1 indicator of up_s > 0.5).
                alpha_un = lo_s[un_idx].clone().detach().requires_grad_(True)
                alpha_params[S][L] = alpha_un
            else:
                alpha = torch.zeros_like(lo_t)
                alpha = alpha + active.to(dtype) * 1.0
                alpha = alpha + unstable.to(dtype) * lo_s
                alpha = alpha.clone().detach().requires_grad_(True)
                alpha_params[S][L] = alpha

    all_tensors = [alpha_params[S][L] for S in start_nodes
                   for L in alpha_params[S]]
    opt = torch.optim.Adam(all_tensors, lr=lr)
    scheduler = (torch.optim.lr_scheduler.ExponentialLR(opt, lr_decay)
                 if lr_decay != 1.0 else None)

    last_op = gg['ops'][-1]
    w_ts = torch.as_tensor(w_qs, dtype=dtype, device=device)  # (n_q, n_out)
    b_ts = torch.as_tensor(b_qs, dtype=dtype, device=device)  # (n_q,)

    best_lbs = np.full(n_q, -np.inf, dtype=np.float64)
    histories = [[] for _ in range(n_q)]
    best_bounds = {
        L: (torch.as_tensor(bbr_init[L][0], dtype=dtype, device=device).clone(),
            torch.as_tensor(bbr_init[L][1], dtype=dtype, device=device).clone())
        for L in bbr_init}

    sparse_at = unstable_idx_per_layer if sparse_alpha else None
    base_chunk = 128

    def _do_pass(direction, s_split_n=1):
        """One Adam-iter pass, optionally split into `s_split_n` S-groups.

        direction ∈ {'both','lb','ub'} — see earlier docstring.

        `s_split_n`: split sorted(intermediate_start_nodes) into N groups.
        Each group: process its S's (with autograd), spec backward, loss,
        `.backward()` to free the group's graph. Cuts peak autograd
        retention to ~1/N at the cost of N spec backwards + a looser
        spec_lb per group (uses partial bbr updates).

        Returns (spec_lb_batch_values, bbr_tensors_with_combined_updates).
        Calls `.backward()` internally per group; caller invokes opt.step().
        """
        bbr_tensors = {L: (
            torch.as_tensor(bbr_init[L][0], dtype=dtype, device=device).clone(),
            torch.as_tensor(bbr_init[L][1], dtype=dtype, device=device).clone(),
        ) for L in bbr_init}
        sorted_S = sorted(intermediate_start_nodes)
        # Round-robin grouping so layer depths are distributed across groups.
        s_groups = [sorted_S[i::s_split_n] for i in range(s_split_n)] \
                    if s_split_n > 1 else [sorted_S]
        un_idx_t_cache = {S: torch.as_tensor(unstable_indices[S], device=device)
                           for S in sorted_S if unstable_indices[S]}

        per_group_spec_lb = []
        for group in s_groups:
            for S in group:
                alpha_for_S = alpha_params[S]
                start_op = _find_op_producing_relu_input(gg, S)
                un_S = unstable_indices[S]
                if not un_S or start_op is None:
                    continue
                n_S = bbr_init[S][0].size
                n_un = len(un_S)
                chunk = min(n_un, base_chunk)
                lb_parts = []; ub_parts = []
                un_idx_all = un_idx_t_cache[S]
                for start in range(0, n_un, chunk):
                    end = min(start + chunk, n_un)
                    kc = end - start
                    ew_init_lb = torch.zeros(
                        kc, n_S, dtype=dtype, device=device)
                    ew_init_lb[torch.arange(kc, device=device),
                               un_idx_all[start:end]] = 1.0
                    if direction in ('both', 'lb'):
                        lb_part, _ = _crown_backward_matrix(
                            gg, xl, xh, alpha_for_S, bbr_tensors,
                            start_op, ew_init_lb, device, dtype,
                            unstable_at_layer=sparse_at)
                        lb_parts.append(lb_part)
                    if direction in ('both', 'ub'):
                        neg_ub_part, _ = _crown_backward_matrix(
                            gg, xl, xh, alpha_for_S, bbr_tensors,
                            start_op, -ew_init_lb, device, dtype,
                            unstable_at_layer=sparse_at)
                        ub_parts.append(-neg_ub_part)
                un_t = torch.as_tensor(un_S, device=device, dtype=torch.long)
                if lb_parts:
                    lb_batch = torch.cat(lb_parts, dim=0)
                    lo_new = bbr_tensors[S][0].scatter(
                        0, un_t,
                        torch.maximum(bbr_tensors[S][0][un_t], lb_batch))
                    bbr_tensors[S] = (lo_new, bbr_tensors[S][1])
                if ub_parts:
                    ub_batch = torch.cat(ub_parts, dim=0)
                    hi_new = bbr_tensors[S][1].scatter(
                        0, un_t,
                        torch.minimum(bbr_tensors[S][1][un_t], ub_batch))
                    bbr_tensors[S] = (bbr_tensors[S][0], hi_new)

            # Group complete: spec backward with current bbr_tensors.
            spec_alpha = alpha_params['spec']
            lb_b, _ = _crown_backward_matrix(
                gg, xl, xh, spec_alpha, bbr_tensors,
                last_op['name'], w_ts, device, dtype,
                unstable_at_layer=sparse_at)
            spec_lb_batch_g = lb_b + b_ts
            with torch.no_grad():
                closed_mask = torch.as_tensor(
                    best_lbs > 0, device=device, dtype=torch.bool)
            active = ~closed_mask
            if active.any():
                loss_g = -spec_lb_batch_g[active].sum()
                loss_g.backward()
            per_group_spec_lb.append(spec_lb_batch_g.detach())
            # Detach this group's bbr updates so the next group sees them
            # as constants (no autograd retention across groups).
            if s_split_n > 1:
                for S in group:
                    if S in bbr_tensors:
                        bbr_tensors[S] = (bbr_tensors[S][0].detach(),
                                            bbr_tensors[S][1].detach())

        # Best (= max) spec_lb across groups; sound since each group's
        # spec backward uses correct (per-group-tightened) bbr.
        if len(per_group_spec_lb) == 1:
            spec_lb_batch = per_group_spec_lb[0]
        else:
            spec_lb_batch = per_group_spec_lb[0]
            for x in per_group_spec_lb[1:]:
                spec_lb_batch = torch.maximum(spec_lb_batch, x)
        return spec_lb_batch, bbr_tensors

    assert dir_mode in ('joint', 'split', 'auto'), dir_mode
    # Sticky downgrade ladder when dir_mode='auto':
    #   joint            (lowest wall, highest memory)
    #   split LB/UB      (~2× memory cut, ~1.5× wall)
    #   split + s_split=2 (~4× cut, ~3× wall — one half of S's per group)
    #   split + s_split=4 (~8× cut)
    #   ...
    _mode_active = ('joint' if dir_mode in ('joint', 'auto') else 'split')
    _s_split_n = int(s_split_n) if s_split_n is not None else 1

    def _run_split_pair(s_split):
        spec_lb_a, bbr_a = _do_pass('lb', s_split_n=s_split)
        spec_lb_b, bbr_b = _do_pass('ub', s_split_n=s_split)
        bbr_tensors_for_best = {}
        for L in bbr_a:
            lo_a, _hi_a = bbr_a[L]
            _lo_b, hi_b = bbr_b[L]
            bbr_tensors_for_best[L] = (lo_a, hi_b)
        with torch.no_grad():
            spec_lb_batch = torch.maximum(
                spec_lb_a.detach(), spec_lb_b.detach())
        return spec_lb_batch, bbr_tensors_for_best

    import gc as _gc
    for it in range(n_iters):
        # Honor caller-supplied deadline so the Adam loop stops cleanly
        # at the global time_left boundary (e.g. cascade refresh on a
        # resnet_large with little budget left).
        if time_left_fn is not None and time_left_fn() <= 0:
            break
        opt.zero_grad()
        spec_lb_batch = None
        bbr_tensors_for_best = None
        if _mode_active == 'joint':
            try:
                spec_lb_batch, bbr_tensors_for_best = _do_pass(
                    'both', s_split_n=_s_split_n)
            except torch.cuda.OutOfMemoryError:
                if dir_mode != 'auto':
                    raise
                # Aggressive release: gc.collect() forces Python to drop
                # references in the popped exception frame so CUDA blocks
                # can actually be reclaimed by empty_cache. Without this,
                # split-mode retries inherit ~2GB of leaked state and the
                # fallback fails even though split alone would fit.
                _gc.collect()
                torch.cuda.empty_cache()
                opt.zero_grad()
                _mode_active = 'split'  # sticky downgrade
        if _mode_active == 'split':
            while True:
                try:
                    spec_lb_batch, bbr_tensors_for_best = _run_split_pair(
                        _s_split_n)
                    break
                except torch.cuda.OutOfMemoryError:
                    # s_split halving fires for BOTH 'split' and 'auto' modes
                    # — only 'joint' suppresses all fallback. Halving the
                    # number of S-groups is algorithmically inert (same α
                    # updates accumulated, just looser per-group spec_lb),
                    # so it's safe to enable whenever we're already in
                    # split-mode rather than forcing the user to opt-in via
                    # 'auto'.
                    if dir_mode == 'joint':
                        raise
                    if _s_split_n >= max(1, len(intermediate_start_nodes)):
                        raise
                    _gc.collect()
                    torch.cuda.empty_cache()
                    opt.zero_grad()
                    _s_split_n = min(
                        _s_split_n * 2,
                        max(1, len(intermediate_start_nodes)))
        # Single opt.step() with accumulated grads from joint or split passes.
        opt.step()
        if scheduler is not None:
            scheduler.step()
        bbr_tensors = bbr_tensors_for_best

        with torch.no_grad():
            for S in start_nodes:
                for L, alpha in alpha_params[S].items():
                    if sparse_alpha:
                        # Sparse-α stores only unstable slopes; clamp to
                        # [0,1] is sufficient (no active/dead positions
                        # exist in the tensor by construction).
                        alpha.clamp_(0.0, 1.0)
                    else:
                        _, _, active_r, dead_r, _ = slopes_init_cache[L]
                        alpha.clamp_(0.0, 1.0)
                        alpha[active_r] = 1.0
                        alpha[dead_r] = 0.0
            for L in bbr_tensors:
                lo_t, hi_t = bbr_tensors[L]
                old_lo, old_hi = best_bounds[L]
                best_bounds[L] = (
                    torch.maximum(old_lo, lo_t.detach()),
                    torch.minimum(old_hi, hi_t.detach()))

        vals = spec_lb_batch.detach().cpu().numpy().astype(np.float64)
        for q in range(n_q):
            histories[q].append(float(vals[q]))
            if vals[q] > best_lbs[q]:
                best_lbs[q] = float(vals[q])

        if early_stop_on_positive and np.all(best_lbs > 0):
            break
        # Hopeless-bound early-exit (see fix_intermediate path comment).
        if (it >= 5 and hopeless_lb is not None
                and np.min(best_lbs) < hopeless_lb):
            recent = []
            for q in range(n_q):
                if len(histories[q]) >= 4:
                    recent.append(histories[q][-1] - histories[q][-4])
            if recent and float(np.mean(recent)) < hopeless_delta:
                if not np.all(best_lbs > 0):
                    break

    # Apply best-across-iters per-neuron bounds for FINAL spec_lb on each
    # query: REPLACE during the loop allows mid-iter bounds to drift, but
    # `best_bounds[L]` is the per-neuron tightest seen across iters. Recompute
    # spec_lb with these, take max with iter-best — sound (each best entry
    # was a valid CROWN-derived bound).
    with torch.no_grad():
        bbr_best = {L: (best_bounds[L][0].detach(), best_bounds[L][1].detach())
                    for L in best_bounds}
        spec_alpha_final = alpha_params['spec']
        lb_final, _ = _crown_backward_matrix(
            gg, xl, xh, spec_alpha_final, bbr_best,
            last_op['name'], w_ts, device, dtype,
            unstable_at_layer=sparse_at)
        final_vals = (lb_final + b_ts).detach().cpu().numpy().astype(np.float64)
        for q in range(n_q):
            if final_vals[q] > best_lbs[q]:
                best_lbs[q] = float(final_vals[q])

    return best_lbs, alpha_params, best_bounds, histories


# ---------------------------------------------------------------------------
# Direction-adaptive reconstruction + forward zonotope with per-neuron slopes
# ---------------------------------------------------------------------------

@torch.no_grad()
def capture_ew_per_relu(gg, xl, xh, alpha_spec, bbr, w_q, b_q, device, dtype):
    """Run one CROWN backward pass with the α-optimal `alpha_spec` and
    record the accumulated backward weight (ew) at each unstable ReLU.
    Returns (lb, ew_at_relu_dict). Used to pick per-neuron direction-adaptive
    slope choices for the forward zonotope."""
    ops = gg['ops']
    last_name = ops[-1]['name']
    ew_init = torch.as_tensor(w_q, dtype=dtype, device=device)
    ew_at = {last_name: ew_init}
    acc = torch.tensor(float(b_q), dtype=dtype, device=device)
    ew_at_relu = {}
    for op in reversed(ops):
        name = op['name']
        if name not in ew_at: continue
        ew = ew_at[name]; t = op['type']
        if t == 'conv':
            out_shape = op['out_shape']
            kernel = op['kernel'].to(dtype=dtype, device=device)
            bias = op['bias'].to(dtype=dtype, device=device)
            ew_4d = ew.reshape(1, *out_shape)
            acc = acc + (
                ew_4d.reshape(out_shape[0], -1).sum(dim=-1) * bias).sum()
            ew_back = F.conv_transpose2d(
                ew_4d, kernel, stride=op['stride'], padding=op['padding'],
                output_padding=op['output_padding']).flatten()
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
                lo_t = torch.as_tensor(bbr[L][0], dtype=dtype, device=device)
                hi_t = torch.as_tensor(bbr[L][1], dtype=dtype, device=device)
                lo_s, up_s, up_t, active, dead, unstable = _make_slopes(
                    lo_t, hi_t)
                alpha = alpha_spec.get(L)
                if alpha is None:
                    alpha = lo_s.clone()
                    alpha[active] = 1.0; alpha[dead] = 0.0
                eff_slope = torch.zeros_like(lo_t)
                eff_slope[active] = 1.0
                if alpha.numel() == lo_t.numel():
                    # Dense α: shape (n_neurons,)
                    eff_slope[unstable] = alpha[unstable]
                else:
                    # Sparse α: shape (n_unstable,) — alpha already
                    # corresponds to unstable neurons in index order.
                    eff_slope[unstable] = alpha
                ew_at_relu[L] = ew.clone()
                ep = ew.clamp(min=0); en = ew.clamp(max=0)
                acc = acc + (en * up_t).sum()
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
                    acc = acc + _bias_dot_ew(ew, bias, dtype, device)
                inp = op['inputs'][0]
                ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew)) + ew
        elif t == 'sub':
            bias = op.get('bias')
            if bias is not None:
                acc = acc - _bias_dot_ew(ew, bias, dtype, device)
            inp = op['inputs'][0]
            ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew)) + ew
        elif t == 'reshape':
            inp = op['inputs'][0]
            ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew)) + ew

        elif t == 'slice':
            flat_idx = op.get('flat_idx')
            in_shape_nd = op.get('in_shapes_nd', [None])[0]
            n_in = int(np.prod(in_shape_nd)) if in_shape_nd is not None else None
            if flat_idx is None or n_in is None:
                raise ValueError("slice backward missing flat_idx/in_shape")
            idx_t = torch.as_tensor(flat_idx, dtype=torch.long, device=device)
            ew_back = torch.zeros(n_in, dtype=ew.dtype, device=device)
            ew_back.index_copy_(-1, idx_t, ew)
            inp = op['inputs'][0]
            ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_back)) + ew_back

        elif t == 'concat':
            in_shapes = op.get('in_shapes_nd', [])
            offset = 0
            for inp, in_shape_nd in zip(op['inputs'], in_shapes):
                n_in = int(np.prod(in_shape_nd))
                ew_i = ew[..., offset:offset + n_in]
                ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_i)) + ew_i
                offset += n_in

        elif t == 'mul':
            ew_back = _mul_scale_backward(op, ew, dtype, device)
            inp = op['inputs'][0]
            ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_back)) + ew_back

        elif t in ('sigmoid', 'tanh'):
            L = op.get('layer_idx')
            lo_pre = torch.as_tensor(bbr[L][0], dtype=dtype, device=device)
            hi_pre = torch.as_tensor(bbr[L][1], dtype=dtype, device=device)
            lo_s, lo_t_b, up_s, up_t_b = _sigmoid_tanh_linear_bounds(
                lo_pre, hi_pre, t)
            ep = ew.clamp(min=0); en = ew.clamp(max=0)
            acc = acc + float((ep * lo_t_b).sum() + (en * up_t_b).sum())
            ew_back = ep * lo_s + en * up_s
            inp = op['inputs'][0]
            ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_back)) + ew_back

        else:
            raise NotImplementedError(
                f'per-query CROWN backward: unsupported op {t!r} '
                f'(name={name!r}). Silent skip would produce unsound bounds.')
    xl_t = xl.to(dtype=dtype, device=device)
    xh_t = xh.to(dtype=dtype, device=device)
    ew_inp = ew_at.get(gg['input_name'])
    lb = acc + ew_inp.clamp(min=0) @ xl_t + ew_inp.clamp(max=0) @ xh_t
    return float(lb), ew_at_relu


def build_dir_adaptive_alpha(alpha_spec, ew_at_relu, bbr, device, dtype):
    """Per unstable neuron k at layer L:
         λ_k = α_k     if ep_k > 0  (lower triangle line tight for ew > 0)
         λ_k = up_s_k  otherwise    (upper triangle line tight for ew ≤ 0)
       Stable-on: λ = 1. Dead: λ = 0. Returns dict {L: tensor}.
    """
    alpha_per_layer = {}
    for L, (lo, hi) in bbr.items():
        lo_t = torch.as_tensor(lo, dtype=dtype, device=device)
        hi_t = torch.as_tensor(hi, dtype=dtype, device=device)
        lo_s, up_s, up_t, active, dead, unstable = _make_slopes(lo_t, hi_t)
        lam = torch.zeros_like(lo_t)
        lam[active] = 1.0
        if L in ew_at_relu and L in alpha_spec:
            ew = ew_at_relu[L]
            pos = (ew > 0)
            alpha = alpha_spec[L]
            if alpha.numel() == lo_t.numel():
                lam[unstable & pos] = alpha[unstable & pos]
            else:
                # Sparse α: shape (n_unstable,). Densify into a full-shape
                # buffer first so subsequent index ops match (n_neurons,).
                alpha_full = torch.zeros_like(lo_t)
                un_idx = torch.nonzero(
                    unstable, as_tuple=False).flatten()
                alpha_full[un_idx] = alpha
                lam[unstable & pos] = alpha_full[unstable & pos]
            lam[unstable & ~pos] = up_s[unstable & ~pos]
        else:
            # Fall back to up_s (min-area) for layers without α.
            lam[unstable] = up_s[unstable]
        alpha_per_layer[L] = lam
    return alpha_per_layer


def forward_zono_dir_adaptive(xl, xh, gg, alpha_per_layer, bbr,
                                 device, dtype, settings=None,
                                 unstable_per_layer=None):
    """Forward zonotope using per-neuron λ from `alpha_per_layer`. At each
    unstable ReLU: λ, μ = max((1-λ)·hi/2, -λ·lo/2), shift = μ. Appends a new
    generator column for unstable neurons with μ ≠ 0.

    Returns (z_final, pre_relu_gpu) where pre_relu_gpu[L] = (c_gpu, G_gpu)
    tensors on `device` — usable for Phase 2.5 halfspace LP tightening.

    The pre_relu_gpu cache stores dense G even with patches mode; Phase 6
    will switch to a sparse ``nonzero_rows`` API for the halfspace-LP path.
    Direct ``z._gen_2d`` mutation in the per-neuron α path is preserved for
    Phase 4 (Phase 5 refactors it behind a method).
    """
    from .zonotope import make_input_zonotope
    per_layer = {}
    with torch.no_grad():
        for L, (lo, hi) in bbr.items():
            lo_t = torch.as_tensor(lo, dtype=dtype, device=device)
            hi_t = torch.as_tensor(hi, dtype=dtype, device=device)
            active = lo_t >= 0
            dead = hi_t <= 0
            unstable = (~active) & (~dead)
            alpha = alpha_per_layer.get(
                L, torch.zeros_like(lo_t)).to(dtype=dtype, device=device)
            lam = torch.zeros_like(lo_t)
            lam[active] = 1.0
            lam[unstable] = alpha[unstable]
            mu = torch.zeros_like(lo_t)
            if unstable.any():
                mu_un = torch.maximum(
                    (1 - alpha[unstable]) * hi_t[unstable] / 2.0,
                    -alpha[unstable] * lo_t[unstable] / 2.0)
                mu[unstable] = mu_un
            shift = mu.clone()
            per_layer[L] = (lam, mu, shift)

    z_init = make_input_zonotope(
        settings, xl, xh, device, dtype, in_shape=gg.get('input_shape'))
    zono_state = {gg['input_name']: z_init}
    gen_count = {gg['input_name']: z_init.n_gens}
    forks = gg['fork_points']
    pre_relu_gpu = {}
    last_use = {}
    for i, op2 in enumerate(gg['ops']):
        for inp in op2['inputs']: last_use[inp] = i

    def _get(n):
        return zono_state[n].copy() if n in forks else zono_state[n]

    for op_idx, op in enumerate(gg['ops']):
        name = op['name']; t = op['type']
        if t == 'conv':
            z = _get(op['inputs'][0])
            z.propagate_conv(op['kernel'], op['bias'], op['in_shape'],
                             op['stride'], op['padding'])
            zono_state[name] = z
        elif t == 'fc':
            z = _get(op['inputs'][0])
            z.propagate_fc(op['W'], op['bias'])
            zono_state[name] = z
        elif t == 'relu':
            z = _get(op['inputs'][0])
            if 'layer_idx' in op:
                L = op['layer_idx']
                # Snapshot pre-ReLU (c, G) — slim when unstable_per_layer
                # provided (uses nonzero_rows to skip full-G materialisation);
                # full G otherwise.
                if (unstable_per_layer is not None
                        and L in unstable_per_layer
                        and unstable_per_layer[L].numel() > 0):
                    un = unstable_per_layer[L].to(device)
                    c_slim = z.center[un].clone()
                    rid, cid, val = z.nonzero_rows(un)
                    K = z.n_gens
                    G_slim = torch.zeros(
                        un.numel(), K, dtype=z.center.dtype, device=device)
                    if rid.numel() > 0:
                        G_slim[rid, cid] = val
                    pre_relu_gpu[L] = (c_slim, G_slim)
                elif (unstable_per_layer is not None
                      and L in unstable_per_layer):
                    pre_relu_gpu[L] = (
                        torch.empty(
                            0, dtype=z.center.dtype, device=device),
                        torch.empty(
                            0, 0, dtype=z.center.dtype, device=device))
                else:
                    pre_relu_gpu[L] = (
                        z.center.clone(), z.generators.clone())
                lam, mu, shift = per_layer[L]
                z.apply_relu_custom(lam, mu, shift)
            else:
                z.apply_relu()
            zono_state[name] = z
        elif t == 'add':
            if op.get('is_merge'):
                z_a = _get(op['inputs'][0]); z_b = _get(op['inputs'][1])
                shared = _find_shared_gens_count(
                    op['inputs'][0], op['inputs'][1], gg, gen_count)
                zono_state[name] = z_a.add(z_b, shared)
            else:
                z = _get(op['inputs'][0]); bias = op.get('bias')
                if bias is not None:
                    bt = torch.tensor(
                        bias.flatten(), dtype=dtype, device=device)
                    z = z.copy()
                    z.center = z.center + bt
                zono_state[name] = z
        elif t == 'sub':
            z = _get(op['inputs'][0]); bias = op.get('bias')
            if bias is not None:
                bt = torch.tensor(
                    bias.flatten(), dtype=dtype, device=device)
                z = z.copy()
                z.center = z.center - bt
            zono_state[name] = z
        elif t == 'reshape':
            zono_state[name] = _get(op['inputs'][0])
        elif t == 'slice':
            from .verify_zono_bnb import TorchZonotope as _TZ
            z = _get(op['inputs'][0])
            flat_idx = op.get('flat_idx')
            idx_t = torch.as_tensor(flat_idx, dtype=torch.long, device=device)
            c_flat = z.center.reshape(-1)
            g_flat = z.generators.reshape(c_flat.numel(), -1)
            zono_state[name] = _TZ(
                c_flat.index_select(0, idx_t),
                g_flat.index_select(0, idx_t))
        elif t == 'concat':
            from .verify_zono_bnb import TorchZonotope as _TZ
            zs = [_get(inp) for inp in op['inputs']]
            n_gens = max(z.generators.shape[1] for z in zs)
            cs, gs = [], []
            for z in zs:
                c_flat = z.center.reshape(-1)
                g_flat = z.generators.reshape(c_flat.numel(), -1)
                if g_flat.shape[1] < n_gens:
                    pad = torch.zeros(c_flat.numel(),
                                       n_gens - g_flat.shape[1],
                                       dtype=g_flat.dtype, device=device)
                    g_flat = torch.cat([g_flat, pad], dim=1)
                cs.append(c_flat); gs.append(g_flat)
            zono_state[name] = _TZ(
                torch.cat(cs, dim=0), torch.cat(gs, dim=0))
        elif t == 'mul':
            z = _get(op['inputs'][0])
            new_c, new_g = _mul_scale_zono(op, z.center, z.generators,
                                              dtype, device)
            from .verify_zono_bnb import TorchZonotope as _TZ
            zono_state[name] = _TZ(new_c, new_g)
        elif t in ('sigmoid', 'tanh'):
            # Parallelogram-relax: y = α·x + β + γ·e_new where (α, β, γ)
            # come from `_sigmoid_tanh_chord_parallelogram` (sound +
            # ≥2× tighter than box on most intervals — preserves input
            # correlation through α slope; only γ is new slack).
            z = _get(op['inputs'][0])
            lo_pre, hi_pre = z.bounds()
            from .verify_zono_bnb import _sigmoid_tanh_chord_parallelogram
            alpha_p, beta_p, gamma_p = _sigmoid_tanh_chord_parallelogram(
                lo_pre, hi_pre, t)
            # c_out = α·c_in + β
            c_out = alpha_p * z.center + beta_p
            # G_out = α·G_in for old cols, append γ·diag for new slack.
            G_old_scaled = alpha_p.unsqueeze(-1) * z.generators
            new_gens = torch.diag(gamma_p)
            new_g = torch.cat([G_old_scaled, new_gens], dim=1)
            from .verify_zono_bnb import TorchZonotope as _TZ
            # Record pre-sigmoid zonotope center+gens in pre_relu_gpu so
            # downstream state_from_alpha_zono can read it (expects
            # (c_pre, G_pre) tensors).
            L = op.get('layer_idx')
            if L is not None and pre_relu_gpu is not None:
                pre_relu_gpu[L] = (z.center.clone(), z.generators.clone())
            zono_state[name] = _TZ(c_out, new_g)
        else:
            raise NotImplementedError(
                f'forward_zono_dir_adaptive: unsupported op {t!r} '
                f'(name={name!r}). Silent skip would propagate stale zono.')
        gen_count[name] = zono_state[name].n_gens
        for inp in op['inputs']:
            if last_use.get(inp) == op_idx and inp in zono_state:
                del zono_state[inp]
    return zono_state[gg['ops'][-1]['name']], pre_relu_gpu
