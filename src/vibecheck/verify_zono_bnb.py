"""Branch-and-Bound verification with zonotope forward + CROWN backward."""

import time
import numpy as np
import torch
import torch.nn.functional as F

from .settings import default_settings, resolve_torch
from .zonotope import TorchZonotope


def _sigmoid_tanh_linear_bounds(lo, hi, act_kind, n_iter=30):
    """Sound closed-form linear bounds for sigmoid/tanh on [lo, hi].

    Returns (lo_s, lo_t, up_s, up_t) such that for all x ∈ [lo, hi]:
        lo_s * x + lo_t ≤ σ(x) ≤ up_s * x + up_t

    Method (mirrors auto_LiRPA's `precompute_relaxation` in tanh.py).
    Sigmoid σ'' = σ'(1 - 2σ), so σ is **convex** on (-∞, 0) (σ < 1/2) and
    **concave** on (0, +∞) (σ > 1/2). Tanh has the same convexity sign
    pattern about 0.
      • Pure convex (hi ≤ 0): chord ABOVE σ → upper = chord.
          tangent below σ → lower = tangent at midpoint.
      • Pure concave (lo ≥ 0): chord BELOW σ → lower = chord.
          tangent above σ → upper = tangent at midpoint.
      • Mixed (lo < 0 < hi): σ convex on [lo, 0], concave on [0, hi].
          Lower: tangent at p ∈ [lo, 0] such that the line passes through
          (hi, σ(hi)). σ'(p)*(hi-p) + σ(p) = σ(hi). Binary-search the unique
          root (g(p) is monotone increasing in p on (lo, 0) since σ'' > 0
          on the convex half).
          Upper: tangent at q ∈ [0, hi] such that the line passes through
          (lo, σ(lo)). σ'(q)*(q-lo) − σ(q) + σ(lo) = 0. Mirror.

    Returns tensors with the same shape as lo/hi."""
    if act_kind == 'sigmoid':
        act = torch.sigmoid
        def dact(x):
            s = act(x); return s * (1 - s)
    elif act_kind == 'tanh':
        act = torch.tanh
        def dact(x):
            s = act(x); return 1 - s * s
    else:
        raise ValueError(f'unknown act_kind {act_kind!r}')

    s_lo = act(lo); s_hi = act(hi)
    width = (hi - lo).clamp(min=1e-12)
    chord_slope = (s_hi - s_lo) / width
    chord_b = s_lo - chord_slope * lo

    # Tangent at midpoint (used for pure cases).
    mid = (lo + hi) / 2
    s_mid = act(mid); ds_mid = dact(mid)
    tang_mid_s = ds_mid
    tang_mid_b = s_mid - ds_mid * mid

    # -- Mixed-case lower tangent at p1 ∈ [lo, 0] s.t. line(hi) == σ(hi). --
    # g(p) = σ'(p)*(hi-p) + σ(p) - σ(hi); g monotone increasing on [lo, 0].
    p_l = torch.minimum(lo, torch.zeros_like(lo))
    p_r = torch.zeros_like(lo)
    for _ in range(n_iter):
        p_m = (p_l + p_r) / 2
        g_m = dact(p_m) * (hi - p_m) + act(p_m) - s_hi
        mask = g_m > 0
        p_r = torch.where(mask, p_m, p_r)
        p_l = torch.where(mask, p_l, p_m)
    p1 = (p_l + p_r) / 2
    g_lo = dact(lo) * (hi - lo) + s_lo - s_hi
    g_at_0 = dact(torch.zeros_like(lo)) * hi + act(torch.zeros_like(lo)) - s_hi
    # Tangent point in [lo, 0] if root exists; else fall back later.
    lo_s_mixed = dact(p1)
    lo_t_mixed = act(p1) - lo_s_mixed * p1
    # If g(lo) > 0: no root in [lo, 0]; the tangent at lo would have line(hi) > σ(hi).
    # No tangent in [lo, hi] gives a sound lower bound — fall back to the
    # constant y = σ(lo) (sound since σ is monotone increasing).
    fallback_lower = g_lo > 0
    lo_s_mixed = torch.where(fallback_lower, torch.zeros_like(lo), lo_s_mixed)
    lo_t_mixed = torch.where(fallback_lower, s_lo, lo_t_mixed)
    # If g(0) ≤ 0: no root either; use tangent at 0 (slope σ'(0)).
    no_root_left = g_at_0 <= 0
    lo_s_mixed = torch.where(no_root_left, dact(torch.zeros_like(lo)), lo_s_mixed)
    lo_t_mixed = torch.where(no_root_left, act(torch.zeros_like(lo)), lo_t_mixed)

    # -- Mixed-case upper tangent at q1 ∈ [0, hi] s.t. line(lo) == σ(lo). --
    # h(q) = σ'(q)*(q-lo) - (σ(q) - σ(lo)); h monotone DECREASING on [0, hi].
    q_l = torch.zeros_like(hi)
    q_r = torch.maximum(hi, torch.zeros_like(hi))
    for _ in range(n_iter):
        q_m = (q_l + q_r) / 2
        h_m = dact(q_m) * (q_m - lo) - (act(q_m) - s_lo)
        mask = h_m > 0
        q_l = torch.where(mask, q_m, q_l)
        q_r = torch.where(mask, q_r, q_m)
    q1 = (q_l + q_r) / 2
    h_hi = dact(hi) * (hi - lo) - (s_hi - s_lo)
    h_at_0 = dact(torch.zeros_like(hi)) * (-lo) - (act(torch.zeros_like(hi)) - s_lo)
    up_s_mixed = dact(q1)
    up_t_mixed = act(q1) - up_s_mixed * q1
    # If h(hi) > 0 (σ'(hi)*(hi-lo) > σ(hi) - σ(lo)): no valid q in [0, hi].
    # Fall back to constant y = σ(hi).
    fallback_upper = h_hi > 0
    up_s_mixed = torch.where(fallback_upper, torch.zeros_like(hi), up_s_mixed)
    up_t_mixed = torch.where(fallback_upper, s_hi, up_t_mixed)
    no_root_right = h_at_0 <= 0
    up_s_mixed = torch.where(no_root_right, dact(torch.zeros_like(hi)), up_s_mixed)
    up_t_mixed = torch.where(no_root_right, act(torch.zeros_like(hi)), up_t_mixed)

    # Combine cases. Sigmoid/tanh: convex on x<0, concave on x>0.
    is_convex = hi <= 0   # entire interval in convex region
    is_concave = lo >= 0  # entire interval in concave region
    # Convex: lower = tangent at midpoint, upper = chord
    # Concave: lower = chord, upper = tangent at midpoint
    # Mixed: lower/upper from binary search
    lo_s = torch.where(is_convex, tang_mid_s,
            torch.where(is_concave, chord_slope, lo_s_mixed))
    lo_t = torch.where(is_convex, tang_mid_b,
            torch.where(is_concave, chord_b, lo_t_mixed))
    up_s = torch.where(is_convex, chord_slope,
            torch.where(is_concave, tang_mid_s, up_s_mixed))
    up_t = torch.where(is_convex, chord_b,
            torch.where(is_concave, tang_mid_b, up_t_mixed))
    return lo_s, lo_t, up_s, up_t


def _make_slopes(lo, hi):
    """Compute CROWN adaptive slopes for ReLU relaxation.

    Returns (lo_s, up_s, up_t, active_mask, dead_mask, unstable_mask).
    Works for both 1-D (n,) and (B, n) inputs — operations are
    elementwise so the batched form needs no separate implementation.
    """
    DT = lo.dtype
    lb_r = torch.clamp(lo, max=0)
    ub_r = torch.clamp(hi, min=0)
    ub_r = torch.maximum(ub_r, lb_r + 1e-8)
    up_s = ub_r / (ub_r - lb_r)
    up_t = -lb_r * up_s
    active = lo >= 0
    dead = hi <= 0
    unstable = ~active & ~dead
    lm = active.to(DT)
    um = dead.to(DT)
    lo_s = (up_s > 0.5).to(DT) * (1 - lm) * (1 - um) + lm
    return lo_s, up_s, up_t, active, dead, unstable


def _forward_batch_graph(x, gg):
    """Batched forward pass for PGD on graph networks (supports skip connections)."""
    batch = x.shape[0]
    act = {gg['input_name']: x}
    forks = gg['fork_points']

    for op in gg['ops']:
        name = op['name']
        t = op['type']

        if t == 'conv':
            a = act[op['inputs'][0]]
            ins = op['in_shape']
            a = F.conv2d(a.reshape(batch, *ins), op['kernel'],
                         bias=op['bias'], stride=op['stride'],
                         padding=op['padding']).reshape(batch, -1)
            act[name] = a

        elif t == 'fc':
            a = act[op['inputs'][0]]
            act[name] = a @ op['W'].T + op['bias']

        elif t == 'relu':
            act[name] = F.relu(act[op['inputs'][0]])

        elif t == 'add':
            if op.get('is_merge'):
                act[name] = act[op['inputs'][0]] + act[op['inputs'][1]]
            else:
                a = act[op['inputs'][0]]
                bias = op.get('bias')
                if bias is not None:
                    a = a + torch.tensor(bias.flatten(), dtype=a.dtype,
                                         device=a.device)
                act[name] = a

        elif t == 'sub':
            a = act[op['inputs'][0]]
            bias = op.get('bias')
            if bias is not None:
                a = a - torch.tensor(bias.flatten(), dtype=a.dtype,
                                     device=a.device)
            act[name] = a

        elif t == 'reshape':
            act[name] = act[op['inputs'][0]]

        elif t == 'conv_transpose':
            a = act[op['inputs'][0]]
            ins = op['in_shape']
            a = F.conv_transpose2d(
                a.reshape(batch, *ins), op['kernel'], bias=op['bias'],
                stride=op['stride'], padding=op['padding'],
                output_padding=op['output_padding']).reshape(batch, -1)
            act[name] = a

        elif t == 'bn':
            a = act[op['inputs'][0]]
            act[name] = a * op['factor'] + op['offset']

        elif t == 'upsample':
            a = act[op['inputs'][0]]
            in_shape = op['in_shape']
            sH, sW = op['scale']
            a4 = a.reshape(batch, *in_shape)
            a4 = F.interpolate(a4, scale_factor=(sH, sW), mode='nearest')
            act[name] = a4.reshape(batch, -1)

        elif t == 'sigmoid':
            act[name] = torch.sigmoid(act[op['inputs'][0]])

        elif t == 'tanh':
            act[name] = torch.tanh(act[op['inputs'][0]])

        elif t in ('avg_pool', 'max_pool'):
            a = act[op['inputs'][0]]
            in_shape = op['in_shape']
            a4 = a.reshape(batch, *in_shape)
            fn = F.avg_pool2d if t == 'avg_pool' else F.max_pool2d
            a4 = fn(a4, kernel_size=op['kernel'], stride=op['stride'],
                      padding=op['padding'])
            act[name] = a4.reshape(batch, -1)

        elif t == 'transpose':
            a = act[op['inputs'][0]]
            in_shape = op.get('in_shapes_nd', [None])[0]
            perm = op['perm']
            if in_shape and len(in_shape) + 1 == len(perm):
                # ONNX perm includes batch dim (typically perm[0]==0); convert
                # to N-D body perm by remapping.
                # Reshape (batch, n) -> (batch, *in_shape), permute, flatten.
                a_nd = a.reshape(batch, *in_shape)
                if perm[0] == 0:
                    body_perm = tuple(p - 1 for p in perm[1:])
                    a_nd = a_nd.permute(0, *(p + 1 for p in body_perm))
                else:
                    a_nd = a_nd.permute(*perm)
                act[name] = a_nd.reshape(batch, -1)
            else:
                act[name] = a  # passthrough fallback

        elif t == 'squeeze':
            act[name] = act[op['inputs'][0]]

        elif t == 'mul':
            a = act[op['inputs'][0]]
            scale = op.get('scale')
            if scale is not None:
                s = torch.as_tensor(np.asarray(scale).flatten(),
                                      dtype=a.dtype, device=a.device)
                act[name] = a * s
            else:
                act[name] = a

        elif t == 'mul_bilinear':
            a = act[op['inputs'][0]]
            b = act[op['inputs'][1]]
            act[name] = a * b

        elif t == 'matmul_bilinear':
            a = act[op['inputs'][0]]
            b = act[op['inputs'][1]]
            shapes = op.get('in_shapes_nd', [None, None])
            sh_a, sh_b = shapes[0], shapes[1]
            assert sh_a and sh_b and len(sh_a) >= 2 and len(sh_b) >= 2, \
                f'matmul_bilinear needs ≥2-D shapes; got {sh_a}, {sh_b}'
            a_nd = a.reshape(batch, *sh_a)
            b_nd = b.reshape(batch, *sh_b)
            out_nd = a_nd @ b_nd
            act[name] = out_nd.reshape(batch, -1)

        elif t == 'softmax':
            a = act[op['inputs'][0]]
            in_shape = op.get('in_shapes_nd', [None])[0]
            axis = op.get('axis', -1)
            if in_shape and len(in_shape) >= 1:
                a_nd = a.reshape(batch, *in_shape)
                # ONNX axis may include batch dim; if axis 0 → batch dim
                # which we don't want to softmax over. Normalize.
                if axis == 0:
                    axis = -1
                elif axis > 0:
                    axis = axis  # already counted including batch dim
                act[name] = F.softmax(a_nd, dim=axis).reshape(batch, -1)
            else:
                act[name] = F.softmax(a, dim=axis)

        else:
            raise ValueError(
                f'_forward_batch_graph: unknown op type {t!r} at {name!r}')

    return act[gg['ops'][-1]['name']]


def _forward_batch(x, fwd_data, nh):
    """Batched forward pass for PGD attack."""
    batch = x.shape[0]
    gpu_k = fwd_data['gpu_k']
    gpu_W_fwd = fwd_data['gpu_W_fwd']
    gpu_b_fwd = fwd_data['gpu_b_fwd']
    layer_types = fwd_data['layer_types']
    for l in range(nh + 1):
        lt, params = layer_types[l]
        if lt == 'conv':
            ins = params['input_shape']
            s = params['stride']
            p = params['padding']
            x = F.conv2d(x.reshape(batch, *ins), gpu_k[l],
                         bias=gpu_b_fwd[l], stride=s, padding=p
                         ).reshape(batch, -1)
        else:
            x = x @ gpu_W_fwd[l].T + gpu_b_fwd[l]
        if l < nh:
            x = F.relu(x)
    return x


def _pgd_attack(xl, xh, remaining_specs, pred, fwd_data, nh, settings):
    """Batched PGD with per-restart targets.

    Returns (is_sat, witness_np, best_adv_np).
    """
    DEV = xl.device
    DT = xl.dtype
    n_restarts = settings.pgd_restarts
    n_iter = settings.pgd_iter
    eps = (xh - xl) / 2
    step_size = eps * 0.2
    comps_list = sorted(remaining_specs)
    n_specs = len(comps_list)
    comps_t = torch.tensor(comps_list, device=DEV)
    target_idx = torch.arange(n_restarts, device=DEV) % n_specs
    target_comps = comps_t[target_idx]

    x_adv = xl + (xh - xl) * torch.rand(n_restarts, len(xl), dtype=DT,
                                         device=DEV)
    x_adv.requires_grad_(True)

    for _ in range(n_iter):
        out = _forward_batch(x_adv, fwd_data, nh)
        target_margins = (out[:, pred]
                          - out[torch.arange(n_restarts, device=DEV),
                                target_comps])
        all_margins = out[:, pred].unsqueeze(1) - out[:, comps_t]
        worst_per_sample = all_margins.min(dim=1).values
        if (worst_per_sample < 0).any():
            idx = worst_per_sample.argmin()
            return True, x_adv[idx].detach().cpu().numpy(), None
        loss = target_margins.sum()
        loss.backward()
        with torch.no_grad():
            x_new = x_adv - step_size * x_adv.grad.sign()
            x_adv = torch.clamp(x_new, xl, xh).clone().requires_grad_(True)

    with torch.no_grad():
        out = _forward_batch(x_adv, fwd_data, nh)
        all_margins = out[:, pred].unsqueeze(1) - out[:, comps_t]
        worst_per_sample = all_margins.min(dim=1).values
        if (worst_per_sample < 0).any():
            idx = worst_per_sample.argmin()
            return True, x_adv[idx].detach().cpu().numpy(), None
        best_idx = worst_per_sample.argmin()
        best_adv = x_adv[best_idx].detach().cpu().numpy()
    return False, None, best_adv


def _pgd_attack_graph(xl, xh, remaining_specs, pred, gg, settings):
    """Batched PGD on graph networks. Same interface as _pgd_attack."""
    DEV = xl.device
    DT = xl.dtype
    n_restarts = settings.pgd_restarts
    n_iter = settings.pgd_iter
    eps = (xh - xl) / 2
    step_size = eps * 0.2
    comps_list = sorted(remaining_specs)
    n_specs = len(comps_list)
    comps_t = torch.tensor(comps_list, device=DEV)
    target_idx = torch.arange(n_restarts, device=DEV) % n_specs
    target_comps = comps_t[target_idx]

    x_adv = xl + (xh - xl) * torch.rand(n_restarts, len(xl), dtype=DT,
                                         device=DEV)
    x_adv.requires_grad_(True)

    for _ in range(n_iter):
        out = _forward_batch_graph(x_adv, gg)
        target_margins = (out[:, pred]
                          - out[torch.arange(n_restarts, device=DEV),
                                target_comps])
        all_margins = out[:, pred].unsqueeze(1) - out[:, comps_t]
        worst_per_sample = all_margins.min(dim=1).values
        if (worst_per_sample < 0).any():
            idx = worst_per_sample.argmin()
            return True, x_adv[idx].detach().cpu().numpy(), None
        loss = target_margins.sum()
        loss.backward()
        with torch.no_grad():
            x_new = x_adv - step_size * x_adv.grad.sign()
            x_adv = torch.clamp(x_new, xl, xh).clone().requires_grad_(True)

    with torch.no_grad():
        out = _forward_batch_graph(x_adv, gg)
        all_margins = out[:, pred].unsqueeze(1) - out[:, comps_t]
        worst_per_sample = all_margins.min(dim=1).values
        if (worst_per_sample < 0).any():
            idx = worst_per_sample.argmin()
            return True, x_adv[idx].detach().cpu().numpy(), None
        best_idx = worst_per_sample.argmin()
        best_adv = x_adv[best_idx].detach().cpu().numpy()
    return False, None, best_adv


def _build_spec_ew(gpu_layers_list, pred, comps, device, dtype):
    """Precompute effective weight for spec backward pass.

    For the final layer, computes w_pred - w_comp and bias_pred - bias_comp.
    """
    spec_ew = {}
    final = gpu_layers_list[-1]
    if final['type'] == 'conv':
        in_shape = final['in_shape']
        n_prev = in_shape[0] * in_shape[1] * in_shape[2]
        kernel = final['kernel']
        bias = final['bias']
        out_shape = final['out_shape']
        n_out = final['n_out']
        for comp in comps:
            # Build one-hot for pred and comp, push through conv_transpose
            I_pred = torch.zeros(1, n_out, dtype=dtype, device=device)
            I_pred[0, pred] = 1.0
            wp = F.conv_transpose2d(
                I_pred.reshape(1, *out_shape), kernel,
                stride=final['stride'], padding=final['padding'],
                output_padding=final['output_padding']).flatten()
            I_comp = torch.zeros(1, n_out, dtype=dtype, device=device)
            I_comp[0, comp] = 1.0
            wc = F.conv_transpose2d(
                I_comp.reshape(1, *out_shape), kernel,
                stride=final['stride'], padding=final['padding'],
                output_padding=final['output_padding']).flatten()
            spatial = out_shape[1] * out_shape[2]
            b_diff = float(bias[pred // spatial]) - float(bias[comp // spatial])
            spec_ew[comp] = (wp - wc, b_diff)
    else:
        W = final['W']
        bias = final['bias']
        for comp in comps:
            spec_ew[comp] = (W[pred] - W[comp],
                             float(bias[pred]) - float(bias[comp]))
    return spec_ew


def _build_spec_ew_graph(gg, pred, comps, device, dtype):
    """Compute spec effective weights from gpu_graph's final linear layer."""
    # Find the last linear op (Conv or FC)
    last_linear = None
    for op in reversed(gg['ops']):
        if op['type'] in ('conv', 'fc'):
            last_linear = op
            break
    assert last_linear is not None, "No final linear layer found"

    spec_ew = {}
    if last_linear['type'] == 'fc':
        W = last_linear['W']
        bias = last_linear['bias']
        for comp in comps:
            spec_ew[comp] = (W[pred] - W[comp],
                             float(bias[pred]) - float(bias[comp]))
    else:
        kernel = last_linear['kernel']
        bias = last_linear['bias']
        out_shape = last_linear['out_shape']
        n_out = last_linear['n_out']
        for comp in comps:
            I_pred = torch.zeros(1, n_out, dtype=dtype, device=device)
            I_pred[0, pred] = 1.0
            wp = F.conv_transpose2d(
                I_pred.reshape(1, *out_shape), kernel,
                stride=last_linear['stride'],
                padding=last_linear['padding'],
                output_padding=last_linear['output_padding']).flatten()
            I_comp = torch.zeros(1, n_out, dtype=dtype, device=device)
            I_comp[0, comp] = 1.0
            wc = F.conv_transpose2d(
                I_comp.reshape(1, *out_shape), kernel,
                stride=last_linear['stride'],
                padding=last_linear['padding'],
                output_padding=last_linear['output_padding']).flatten()
            spatial = out_shape[1] * out_shape[2]
            b_diff = float(bias[pred // spatial]) - float(bias[comp // spatial])
            spec_ew[comp] = (wp - wc, b_diff)
    return spec_ew


@torch.no_grad()
def _forward_zonotope_graph(xl, xh, gg, device, dtype, settings=None,
                             rec_zono=None, tight_bounds=None):
    """Graph-aware zonotope forward pass (supports skip connections).

    Args:
        xl, xh: input bounds (flat torch tensors)
        gg: gpu_graph dict from ComputeGraph.gpu_graph()
        settings: optional settings DotMap. When provided AND
            `settings.zono_impl == 'patches'` AND the input shape is
            image-like (C, H, W), the initial zonotope is built as a
            `PatchesZonotope` instead of a `TorchZonotope`. On
            TinyImageNet ResNet (3×56×56 = 9 408 input pixels) the
            dense path needs ~700 MB just for the input zonotope's
            generator-identity matrix and OOMs the RTX 3080 inside the
            first conv; the patches path uses ~0.6 MB for the same
            input. Defaults to dense when `settings is None` for
            backward compatibility with callers that don't carry the
            settings (e.g. unit tests, BaB-leaf shortcuts).
        rec_zono: optional dict to populate with ``{gen_rows_by_layer,
            col_origin, n_input}`` harvested at each layer's pre-ReLU.
            Same protocol as ``_forward_zonotope_interleaved`` —
            downstream Phase 7 (`state_from_phase1`) consumes this to
            avoid the multi-GB ``precompute_gen_state`` allocation.
            When None, behaves as before.
        tight_bounds: optional ``{layer_idx: (lo_np, hi_np)}`` dict of
            externally-computed (e.g. cascade-tightened) pre-activation
            bounds. When provided, ``apply_relu`` uses the intersection
            of ``z.bounds()`` with these tight bounds for the relaxation
            (sound). ``rec_zono`` entries also record the intersected
            (lo, hi), keeping the parametrization consistent so
            ``state_from_phase1``'s LP triangle constraints use the
            same (lo, hi) as the recorded μ, λ.

    Returns:
        sb: dict mapping layer_idx -> (lo, hi) bounds at each ReLU
        z_final: final zonotope (after last op, before output)
    """
    if settings is not None and str(getattr(
            settings, 'zono_impl', 'dense')) == 'patches':
        from .zonotope import make_input_zonotope
        in_shape = getattr(gg, 'input_shape', None) or gg.get('input_shape')
        z_init = make_input_zonotope(
            settings, xl, xh, device, dtype, in_shape=in_shape)
    else:
        z_init = TorchZonotope.from_input_bounds(xl, xh, device, dtype)
    zono_state = {gg['input_name']: z_init}
    gen_count = {gg['input_name']: z_init.n_gens}
    forks = gg['fork_points']
    sb = {}

    if rec_zono is not None:
        rec_zono.setdefault('gen_rows_by_layer', {})
        rec_zono.setdefault('col_origin', {})
        rec_zono['n_input'] = z_init.n_gens

    # Precompute last consumer index for each op name → free memory eagerly
    last_use = {}
    for i, op2 in enumerate(gg['ops']):
        for inp in op2['inputs']:
            last_use[inp] = i

    def _get(name):
        if name in forks:
            return zono_state[name].copy()
        return zono_state[name]

    for op_idx, op in enumerate(gg['ops']):
        name = op['name']
        t = op['type']

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
            layer_idx = op.get('layer_idx')
            # Build the (lo, hi) the relaxation will use: intersect z's
            # own bounds with any externally-supplied tight bounds. We
            # record this same (lo, hi) into rec_zono so the LP triangle
            # constraints in state_from_phase1 match the parametrization.
            need_pre_bounds = (
                rec_zono is not None and layer_idx is not None
            ) or (tight_bounds is not None and layer_idx in (tight_bounds or {}))
            if need_pre_bounds:
                pre_lo_z, pre_hi_z = z.bounds()
                if tight_bounds is not None and layer_idx in tight_bounds:
                    tlo_np, thi_np = tight_bounds[layer_idx]
                    tlo = torch.as_tensor(tlo_np, dtype=dtype, device=device)
                    thi = torch.as_tensor(thi_np, dtype=dtype, device=device)
                    pre_lo = torch.maximum(pre_lo_z, tlo)
                    pre_hi = torch.minimum(pre_hi_z, thi)
                else:
                    pre_lo, pre_hi = pre_lo_z, pre_hi_z
                if rec_zono is not None and layer_idx is not None:
                    from .verify_graph import _record_zono_pre_relu_rows
                    _record_zono_pre_relu_rows(
                        z, layer_idx,
                        (pre_lo.cpu().numpy(), pre_hi.cpu().numpy()),
                        rec_zono)
                lo, hi = z.apply_relu(tight_lo=pre_lo, tight_hi=pre_hi)
            else:
                lo, hi = z.apply_relu()
            if layer_idx is not None:
                sb[layer_idx] = (lo.clone(), hi.clone())
            zono_state[name] = z

        elif t == 'add':
            if op.get('is_merge'):
                z_a = _get(op['inputs'][0])
                z_b = _get(op['inputs'][1])
                # Find shared generators: use the deepest common fork point
                shared = _find_shared_gens_count(
                    op['inputs'][0], op['inputs'][1], gg, gen_count)
                zono_state[name] = z_a.add(z_b, shared)
            else:
                z = _get(op['inputs'][0])
                bias = op.get('bias')
                if bias is not None:
                    z = TorchZonotope(z.center + torch.tensor(
                        bias.flatten(), dtype=dtype, device=device),
                        z.generators.clone())
                zono_state[name] = z

        elif t == 'sub':
            z = _get(op['inputs'][0])
            bias = op.get('bias')
            if bias is not None:
                z = TorchZonotope(
                    z.center - torch.tensor(bias.flatten(), dtype=dtype,
                                            device=device),
                    z.generators.clone())
            zono_state[name] = z

        elif t == 'reshape':
            zono_state[name] = _get(op['inputs'][0])

        elif t in ('sigmoid', 'tanh'):
            # Nonlinear activation: collapse to box. Center = midpoint of
            # the activation's range over [lo, hi]; one new gen per cell
            # with magnitude (hi - lo)/2. Record pre-act bounds for CROWN.
            z = _get(op['inputs'][0])
            lo_pre, hi_pre = z.bounds()
            act = torch.sigmoid if t == 'sigmoid' else torch.tanh
            s_lo = act(lo_pre); s_hi = act(hi_pre)
            c_out = (s_lo + s_hi) / 2
            mu = (s_hi - s_lo) / 2
            n = c_out.numel()
            # New zonotope: zero old gens (no preserved correlation), add
            # diag(mu) for the n new noise variables.
            new_g = torch.diag(mu)
            zono_state[name] = TorchZonotope(c_out, new_g)
            layer_idx = op.get('layer_idx')
            if layer_idx is not None:
                sb[layer_idx] = (lo_pre.clone(), hi_pre.clone())

        elif t == 'mul':
            # Constant scalar / per-channel multiply: y = scale * x.
            z = _get(op['inputs'][0])
            scale_t = op.get('scale')
            if scale_t is None:
                raise ValueError("mul op missing 'scale' for forward zono")
            if isinstance(scale_t, np.ndarray):
                scale_t = torch.from_numpy(scale_t).to(device=device, dtype=dtype)
            elif not isinstance(scale_t, torch.Tensor):
                scale_t = torch.tensor(scale_t, dtype=dtype, device=device)
            else:
                scale_t = scale_t.to(device=device, dtype=dtype)
            sflat = scale_t.flatten()
            n = z.center.numel()
            if sflat.numel() == 1:
                new_c = z.center * sflat
                new_g = z.generators * sflat
            elif sflat.numel() == n:
                new_c = z.center * sflat
                new_g = z.generators * sflat.unsqueeze(-1)
            else:
                in_shape = op.get('in_shapes_nd', [None])[0]
                if in_shape is None or len(in_shape) != 3:
                    raise ValueError(
                        f'mul: scale shape {sflat.shape} incompatible with '
                        f'input ({n}); no spatial shape')
                C, H, W = in_shape
                assert sflat.numel() == C
                scale_4d = sflat.view(1, C, 1, 1).expand(
                    1, C, H, W).reshape(-1)
                new_c = z.center * scale_4d
                new_g = z.generators * scale_4d.unsqueeze(-1)
            zono_state[name] = TorchZonotope(new_c, new_g)

        gen_count[name] = zono_state[name].n_gens

        # Free zonotopes that are no longer needed
        for inp in op['inputs']:
            if last_use.get(inp) == op_idx and inp in zono_state:
                del zono_state[inp]

    # Find last op's zonotope
    last_name = gg['ops'][-1]['name']
    return sb, zono_state[last_name]


def _find_shared_gens_count(name_a, name_b, gg, gen_count):
    """Find shared generator count at fork point for two merging branches.

    Walks backward through gpu_graph ops to find the deepest common ancestor
    that is a fork point.
    """
    forks = gg['fork_points']
    input_name = gg['input_name']

    # Build predecessor map from ops
    pred_map = {}
    for op in gg['ops']:
        pred_map[op['name']] = op['inputs']

    def _ancestors(name):
        visited = []
        stack = [name]
        seen = set()
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            visited.append(n)
            if n in pred_map:
                for inp in pred_map[n]:
                    stack.append(inp)
        return visited

    anc_a = _ancestors(name_a)
    anc_b_set = set(_ancestors(name_b))
    for anc in anc_a:
        if anc in anc_b_set and anc in forks:
            return gen_count.get(anc, 0)
    return gen_count.get(input_name, 0)


@torch.no_grad()
def _spec_backward_graph(tight, xl, xh, gg, spec_ew,
                          remaining_specs, nh, device, dtype,
                          return_ew=False, return_input_linear=False):
    """Graph-aware spec backward pass for networks with skip connections.

    spec_ew maps query_id -> (w, bias) where w is in OUTPUT space.
    Propagates backward through ALL ops including the final linear layer.

    Returns (spec_lbs, still_open). Optional flags add tuple tails:
      - return_ew=True: appends ew_at_relu (qid -> {layer_idx -> ew_numpy})
      - return_input_linear=True: appends input_linear
        (qid -> (ew_inp_numpy, acc_float)), the linear lower-bound
        coefficients in input space such that for all x in [xl, xh]:
            spec(x) >= ew_inp · x + acc
        Used by `_input_split_fast_leaf`'s joint-AND infeasibility LP.
    """
    ops = gg['ops']

    spec_lbs = {}
    all_ew_at_relu = {} if return_ew else None
    input_linear = {} if return_input_linear else None
    for qid in remaining_specs:
        ew_init, b_spec = spec_ew[qid]
        ew_at = {}
        acc = b_spec
        qid_ew_at_relu = {} if return_ew else None

        # Seed ew at the output of the last op
        last_name = ops[-1]['name']
        ew_at[last_name] = ew_init.clone()

        # Walk backward through ALL ops
        for op in reversed(ops):
            name = op['name']
            if name not in ew_at:
                continue
            ew = ew_at[name]
            t = op['type']

            if t == 'conv':
                acc += float(
                    ew.reshape(1, *op['out_shape']).reshape(
                        op['out_shape'][0], -1).sum(dim=1) @ op['bias'])
                ew_back = F.conv_transpose2d(
                    ew.reshape(1, *op['out_shape']), op['kernel'],
                    stride=op['stride'], padding=op['padding'],
                    output_padding=op['output_padding']).flatten()
                inp = op['inputs'][0]
                ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_back)) + ew_back

            elif t == 'fc':
                acc += float(ew @ op['bias'])
                ew_back = ew @ op['W']
                inp = op['inputs'][0]
                ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_back)) + ew_back

            elif t == 'relu':
                if 'layer_idx' in op:
                    if return_ew:
                        qid_ew_at_relu[op['layer_idx']] = ew.cpu().numpy()
                    lo_k, hi_k = tight[op['layer_idx']]
                    lo_s, up_s, up_t, _, _, _ = _make_slopes(lo_k, hi_k)
                    ep = ew.clamp(min=0)
                    en = ew.clamp(max=0)
                    acc += float((en * up_t).sum())
                    ew_back = ep * lo_s + en * up_s
                else:
                    ew_back = ew
                inp = op['inputs'][0]
                ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_back)) + ew_back

            elif t == 'add':
                # Add backward: ew goes to both inputs unchanged
                for inp in op['inputs']:
                    ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew)) + ew

            elif t == 'sub':
                # Sub backward: ew passes through, bias contributes to acc
                bias = op.get('bias')
                if bias is not None:
                    b_t = torch.tensor(bias.flatten(), dtype=ew.dtype,
                                       device=ew.device)
                    acc -= float((ew * b_t).sum())
                inp = op['inputs'][0]
                ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew)) + ew

            elif t == 'reshape':
                inp = op['inputs'][0]
                ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew)) + ew

            elif t == 'mul':
                # y = scale * x → ew_back = ew * scale.
                scale_t = op.get('scale')
                if isinstance(scale_t, np.ndarray):
                    scale_t = torch.from_numpy(scale_t).to(
                        device=ew.device, dtype=ew.dtype)
                elif not isinstance(scale_t, torch.Tensor):
                    scale_t = torch.tensor(scale_t, dtype=ew.dtype,
                                            device=ew.device)
                else:
                    scale_t = scale_t.to(device=ew.device, dtype=ew.dtype)
                sflat = scale_t.flatten()
                n = ew.numel()
                if sflat.numel() == 1 or sflat.numel() == n:
                    ew_back = ew * sflat
                else:
                    in_shape = op.get('in_shapes_nd', [None])[0]
                    C, H, W = in_shape
                    assert sflat.numel() == C
                    scale_4d = sflat.view(1, C, 1, 1).expand(
                        1, C, H, W).reshape(-1)
                    ew_back = ew * scale_4d
                inp = op['inputs'][0]
                ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_back)) + ew_back

            elif t in ('sigmoid', 'tanh'):
                # CROWN backward: closed-form linear slopes via the
                # same `_sigmoid_tanh_linear_bounds` helper used by the
                # batched pipeline. Pre-activation bounds from `tight`.
                L = op.get('layer_idx')
                lo_pre, hi_pre = tight[L]
                lo_s, lo_t, up_s, up_t = _sigmoid_tanh_linear_bounds(
                    lo_pre, hi_pre, t)
                ep = ew.clamp(min=0); en = ew.clamp(max=0)
                acc += float((ep * lo_t).sum() + (en * up_t).sum())
                ew_back = ep * lo_s + en * up_s
                inp = op['inputs'][0]
                ew_at[inp] = ew_at.get(inp, torch.zeros_like(ew_back)) + ew_back

        # At input: interval bound
        input_name = gg['input_name']
        ew_inp = ew_at.get(input_name, torch.zeros_like(xl))
        spec_lbs[qid] = acc + float(
            ew_inp.clamp(min=0) @ xl + ew_inp.clamp(max=0) @ xh)
        if return_ew:
            all_ew_at_relu[qid] = qid_ew_at_relu
        if return_input_linear:
            input_linear[qid] = (
                ew_inp.detach().cpu().numpy().astype(np.float64),
                float(acc))

    still_open = {c for c in remaining_specs if spec_lbs[c] <= 0}
    if return_input_linear and return_ew:
        return spec_lbs, still_open, all_ew_at_relu, input_linear
    if return_input_linear:
        return spec_lbs, still_open, input_linear
    if return_ew:
        return spec_lbs, still_open, all_ew_at_relu
    return spec_lbs, still_open


# ---------------------------------------------------------------------------
# Batched forward zono + spec-backward CROWN for input-split BaB.
# Each batch element is an INDEPENDENT input box [xl[b], xh[b]] processed
# through the SAME network graph. Centers / generators carry a leading
# batch dim B; intermediate tensors are (B, n, K) for generators and
# (B, n) for centers/bounds. After each ReLU we append `n_layer` new
# generator columns (one per neuron, with mu padded to zero for stable
# neurons) — keeps gen-count uniform across the batch so ops stay
# vectorized. For cersyve (200 ReLU total, input dim 4) this caps gens
# at ~260 → at batch=4096 the largest intermediate is ~850 MB (fits 10
# GB GPU).
#
# Not supported: conv, add-merge with non-trivially-shared gens, patches
# zonotope. Add-merge with shared_gens equal to fork K is supported by
# concat. The driver `_input_split_batched` falls back to the scalar
# path when the graph requires unsupported ops.
# ---------------------------------------------------------------------------


@torch.no_grad()
def _forward_zonotope_graph_batched(xl, xh, gg, device, dtype):
    """Batched forward zonotope on the graph.

    Args:
        xl, xh: (B, n_in) input bounds — one box per batch element.
        gg: gpu_graph dict.

    Returns:
        sb: dict layer_idx → (lo, hi) shape (B, n_layer) per ReLU.
        z_final: (c, G) tuple where c is (B, n_out) and G is (B, n_out, K).

    Raises:
        ValueError on unsupported op types (conv, add-merge with extras).
    """
    B, n_in = xl.shape
    c = (xl + xh) / 2
    radii = (xh - xl) / 2
    # G is (B, n_in, n_in): diagonal of radii. nonzero radii get an own
    # gen column; zero-radius inputs contribute zero columns (still
    # presented in the tensor for shape uniformity — they don't add
    # spurious mass since the matrix entry is 0).
    G = torch.diag_embed(radii)
    state = {gg['input_name']: (c, G)}
    gen_count = {gg['input_name']: G.shape[2]}
    forks = gg['fork_points']
    sb = {}

    last_use = {}
    for i, op2 in enumerate(gg['ops']):
        for inp in op2['inputs']:
            last_use[inp] = i

    def _get(name):
        # `forks` means the value is consumed twice — clone (cheap on GPU)
        c_, G_ = state[name]
        if name in forks:
            return c_.clone(), G_.clone()
        return c_, G_

    for op_idx, op in enumerate(gg['ops']):
        name = op['name']
        t = op['type']

        if t == 'fc':
            c_in, G_in = _get(op['inputs'][0])
            W = op['W']  # (n_out, n_in_layer)
            bias = op['bias']  # (n_out,)
            c_out = c_in @ W.T + bias  # (B, n_out)
            # G_out[b, o, k] = sum_i W[o, i] * G_in[b, i, k]
            G_out = torch.einsum('oi,bik->bok', W, G_in)
            state[name] = (c_out, G_out)

        elif t == 'relu':
            c_in, G_in = _get(op['inputs'][0])
            abs_sum = G_in.abs().sum(dim=2)  # (B, n)
            lo = c_in - abs_sum
            hi = c_in + abs_sum
            ust = (lo < 0) & (hi > 0)
            dead = hi <= 0
            lam = torch.where(ust, hi / (hi - lo),
                               torch.where(dead, torch.zeros_like(hi),
                                            torch.ones_like(hi)))  # (B, n)
            mu = torch.where(ust, -hi * lo / (2 * (hi - lo)),
                              torch.zeros_like(hi))  # (B, n)
            c_out = lam * c_in + mu
            G_scaled = G_in * lam.unsqueeze(-1)  # (B, n, K)
            # Compact gen append: only one new column per UNSTABLE neuron
            # (stable neurons have mu=0; full diag was 800MB+ on cGAN at
            # n=28800). Per-batch unstable counts may differ; pad with
            # zeros to max across batch.
            ust_cnt = ust.sum(dim=1)  # (B,)
            max_K = int(ust_cnt.max().item())
            if max_K == 0:
                G_out = G_scaled
            else:
                new_gens = torch.zeros(B, c_in.shape[1], max_K,
                                          dtype=dtype, device=device)
                # k-index within each batch's unstable list
                ust_rank = ust.long().cumsum(dim=1) - 1  # (B, n)
                b_idx = torch.arange(B, device=device).unsqueeze(-1).expand(
                    -1, c_in.shape[1])  # (B, n)
                r_idx = torch.arange(c_in.shape[1],
                                       device=device).unsqueeze(0).expand(
                    B, -1)  # (B, n)
                new_gens[b_idx[ust], r_idx[ust], ust_rank[ust]] = mu[ust]
                G_out = torch.cat([G_scaled, new_gens], dim=2)
            state[name] = (c_out, G_out)
            if 'layer_idx' in op:
                sb[op['layer_idx']] = (lo.clone(), hi.clone())

        elif t == 'add':
            if op.get('is_merge'):
                c_a, G_a = _get(op['inputs'][0])
                c_b, G_b = _get(op['inputs'][1])
                shared = _find_shared_gens_count(
                    op['inputs'][0], op['inputs'][1], gg, gen_count)
                K_a, K_b = G_a.shape[2], G_b.shape[2]
                assert 0 <= shared <= K_a and 0 <= shared <= K_b
                if K_b == shared:
                    # Fast path: mutate a's first `shared` cols.
                    G_a[:, :, :shared] = G_a[:, :, :shared] + G_b[:, :, :shared]
                    state[name] = (c_a + c_b, G_a)
                else:
                    K_out = K_a + K_b - shared
                    n = c_a.shape[1]
                    G_out = torch.empty(B, n, K_out, dtype=dtype, device=device)
                    G_out[:, :, :shared] = G_a[:, :, :shared] + G_b[:, :, :shared]
                    if K_a > shared:
                        G_out[:, :, shared:K_a] = G_a[:, :, shared:]
                    if K_b > shared:
                        G_out[:, :, K_a:] = G_b[:, :, shared:]
                    state[name] = (c_a + c_b, G_out)
            else:
                c_in, G_in = _get(op['inputs'][0])
                bias = op.get('bias')
                if bias is not None:
                    bt = torch.as_tensor(bias.flatten(),
                                          dtype=dtype, device=device)
                    c_in = c_in + bt  # broadcast across batch
                state[name] = (c_in, G_in)

        elif t == 'sub':
            c_in, G_in = _get(op['inputs'][0])
            bias = op.get('bias')
            if bias is not None:
                bt = torch.as_tensor(bias.flatten(),
                                      dtype=dtype, device=device)
                c_in = c_in - bt
            state[name] = (c_in, G_in)

        elif t == 'reshape':
            state[name] = _get(op['inputs'][0])

        elif t == 'conv_transpose':
            c_in, G_in = _get(op['inputs'][0])
            kernel = op['kernel']
            bias = op['bias']
            in_shape = op['in_shape']  # (C_in, H_in, W_in)
            stride = op['stride']
            padding = op['padding']
            output_padding = op['output_padding']
            n_in = c_in.shape[1]
            assert n_in == in_shape[0] * in_shape[1] * in_shape[2]
            # Center: (B, n_in) → (B, C_in, H, W) → conv_transpose → flatten
            c_4d = c_in.reshape(B, *in_shape)
            c_out_4d = F.conv_transpose2d(
                c_4d, kernel, bias=bias, stride=stride, padding=padding,
                output_padding=output_padding)
            c_out = c_out_4d.reshape(B, -1)
            # Generators: (B, n_in, K) → (B*K, C_in, H, W) per K → reshape
            K = G_in.shape[2]
            if K == 0:
                G_out = torch.zeros(B, c_out.shape[1], 0,
                                       dtype=dtype, device=device)
            else:
                # permute to (B, K, n_in) → (B*K, C, H, W)
                g_perm = G_in.permute(0, 2, 1).reshape(B * K, *in_shape)
                g_out = F.conv_transpose2d(
                    g_perm, kernel, bias=None, stride=stride, padding=padding,
                    output_padding=output_padding)
                n_out = c_out.shape[1]
                G_out = g_out.reshape(B, K, n_out).permute(0, 2, 1).contiguous()
            state[name] = (c_out, G_out)

        elif t == 'bn':
            c_in, G_in = _get(op['inputs'][0])
            factor = op['factor']  # (n,)
            offset = op['offset']  # (n,)
            c_out = c_in * factor + offset  # (B, n)
            G_out = G_in * factor.unsqueeze(-1)  # (B, n, K)
            state[name] = (c_out, G_out)

        elif t in ('sigmoid', 'tanh'):
            c_in, G_in = _get(op['inputs'][0])
            abs_sum = G_in.abs().sum(dim=2)
            lo_pre = c_in - abs_sum
            hi_pre = c_in + abs_sum
            act = torch.sigmoid if t == 'sigmoid' else torch.tanh
            s_lo = act(lo_pre); s_hi = act(hi_pre)
            c_out = (s_lo + s_hi) / 2
            mu = (s_hi - s_lo) / 2
            if 'layer_idx' in op:
                sb[op['layer_idx']] = (lo_pre.clone(), hi_pre.clone())
            # Collapse old gens (no preserved correlation through nonlinearity).
            # Compact: only add gen columns for neurons with non-zero slack.
            G_scaled = torch.zeros(B, c_in.shape[1], G_in.shape[2],
                                      dtype=dtype, device=device)
            nonzero_mask = mu.abs() > 1e-9
            ust_cnt = nonzero_mask.sum(dim=1)
            max_K = int(ust_cnt.max().item())
            if max_K == 0:
                G_out = G_scaled  # (B, n, K_old) zeros
            else:
                new_gens = torch.zeros(B, c_in.shape[1], max_K,
                                          dtype=dtype, device=device)
                rank = nonzero_mask.long().cumsum(dim=1) - 1
                b_idx = torch.arange(B, device=device).unsqueeze(-1).expand(
                    -1, c_in.shape[1])
                r_idx = torch.arange(c_in.shape[1],
                                       device=device).unsqueeze(0).expand(
                    B, -1)
                new_gens[b_idx[nonzero_mask], r_idx[nonzero_mask],
                          rank[nonzero_mask]] = mu[nonzero_mask]
                G_out = torch.cat([G_scaled, new_gens], dim=2)
            state[name] = (c_out, G_out)

        elif t == 'upsample':
            c_in, G_in = _get(op['inputs'][0])
            in_shape = op['in_shape']
            sH, sW = op['scale']
            n_in_layer = in_shape[0] * in_shape[1] * in_shape[2]
            assert c_in.shape[1] == n_in_layer
            c_4d = c_in.reshape(B, *in_shape)
            c_out_4d = F.interpolate(c_4d, scale_factor=(sH, sW),
                                       mode='nearest')
            c_out = c_out_4d.reshape(B, -1)
            K = G_in.shape[2]
            if K == 0:
                G_out = torch.zeros(B, c_out.shape[1], 0,
                                       dtype=dtype, device=device)
            else:
                g_perm = G_in.permute(0, 2, 1).reshape(B * K, *in_shape)
                g_out = F.interpolate(g_perm, scale_factor=(sH, sW),
                                        mode='nearest')
                n_out_layer = c_out.shape[1]
                G_out = g_out.reshape(B, K, n_out_layer).permute(0, 2, 1).contiguous()
            state[name] = (c_out, G_out)

        elif t == 'conv':
            c_in, G_in = _get(op['inputs'][0])
            kernel = op['kernel']
            bias = op['bias']
            in_shape = op['in_shape']
            stride = op['stride']
            padding = op['padding']
            n_in_layer = in_shape[0] * in_shape[1] * in_shape[2]
            assert c_in.shape[1] == n_in_layer
            c_4d = c_in.reshape(B, *in_shape)
            c_out_4d = F.conv2d(c_4d, kernel, bias=bias,
                                  stride=stride, padding=padding)
            c_out = c_out_4d.reshape(B, -1)
            K = G_in.shape[2]
            if K == 0:
                G_out = torch.zeros(B, c_out.shape[1], 0,
                                       dtype=dtype, device=device)
            else:
                g_perm = G_in.permute(0, 2, 1).reshape(B * K, *in_shape)
                g_out = F.conv2d(g_perm, kernel, bias=None,
                                   stride=stride, padding=padding)
                n_out_layer = c_out.shape[1]
                G_out = g_out.reshape(B, K, n_out_layer).permute(0, 2, 1).contiguous()
            state[name] = (c_out, G_out)

        elif t == 'avg_pool':
            # avg_pool is linear: y = (1/k^2) * sum over window. Apply
            # F.avg_pool2d to center and each generator column. No bound
            # loss.
            c_in, G_in = _get(op['inputs'][0])
            in_shape = op['in_shape']
            kH, kW = op['kernel']
            sH, sW = op['stride']
            pH, pW = op['padding']
            n_in_layer = in_shape[0] * in_shape[1] * in_shape[2]
            assert c_in.shape[1] == n_in_layer
            c_4d = c_in.reshape(B, *in_shape)
            c_out_4d = F.avg_pool2d(c_4d, kernel_size=(kH, kW),
                                       stride=(sH, sW), padding=(pH, pW))
            c_out = c_out_4d.reshape(B, -1)
            K = G_in.shape[2]
            if K == 0:
                G_out = torch.zeros(B, c_out.shape[1], 0,
                                       dtype=dtype, device=device)
            else:
                g_perm = G_in.permute(0, 2, 1).reshape(B * K, *in_shape)
                g_out = F.avg_pool2d(g_perm, kernel_size=(kH, kW),
                                       stride=(sH, sW), padding=(pH, pW))
                n_out_layer = c_out.shape[1]
                G_out = g_out.reshape(B, K, n_out_layer).permute(
                    0, 2, 1).contiguous()
            state[name] = (c_out, G_out)

        elif t == 'max_pool':
            # max_pool is nonlinear. Box approximation: per-cell bounds
            # are lo_out=max(lo_in over window), hi_out=max(hi_in over
            # window). Collapse correlations into a new gen column per
            # cell with non-zero slack. Sound but loose; suffices for
            # cgan small_transformer's attention (4 MaxPool ops).
            c_in, G_in = _get(op['inputs'][0])
            in_shape = op['in_shape']
            kH, kW = op['kernel']
            sH, sW = op['stride']
            pH, pW = op['padding']
            n_in_layer = in_shape[0] * in_shape[1] * in_shape[2]
            abs_sum = G_in.abs().sum(dim=2)
            lo_pre = (c_in - abs_sum).reshape(B, *in_shape)
            hi_pre = (c_in + abs_sum).reshape(B, *in_shape)
            lo_out = F.max_pool2d(lo_pre, (kH, kW), stride=(sH, sW),
                                     padding=(pH, pW))
            hi_out = F.max_pool2d(hi_pre, (kH, kW), stride=(sH, sW),
                                     padding=(pH, pW))
            n_out_layer = lo_out.shape[1] * lo_out.shape[2] * lo_out.shape[3]
            lo_flat = lo_out.reshape(B, n_out_layer)
            hi_flat = hi_out.reshape(B, n_out_layer)
            c_out = (lo_flat + hi_flat) / 2
            mu = (hi_flat - lo_flat) / 2
            # Compact gen append (mirrors sigmoid/tanh).
            nonzero_mask = mu.abs() > 1e-9
            ust_cnt = nonzero_mask.sum(dim=1)
            max_K = int(ust_cnt.max().item())
            G_zeros = torch.zeros(B, n_out_layer, G_in.shape[2],
                                     dtype=dtype, device=device)
            if max_K == 0:
                G_out = G_zeros
            else:
                new_gens = torch.zeros(B, n_out_layer, max_K,
                                          dtype=dtype, device=device)
                rank = nonzero_mask.long().cumsum(dim=1) - 1
                b_idx = torch.arange(B, device=device).unsqueeze(-1).expand(
                    -1, n_out_layer)
                r_idx = torch.arange(n_out_layer,
                                       device=device).unsqueeze(0).expand(B, -1)
                new_gens[b_idx[nonzero_mask], r_idx[nonzero_mask],
                          rank[nonzero_mask]] = mu[nonzero_mask]
                G_out = torch.cat([G_zeros, new_gens], dim=2)
            state[name] = (c_out, G_out)
            # Record pre-act bounds for backward (used by box CROWN).
            sb[op['name'] + '__maxpool_box'] = (lo_flat.clone(),
                                                   hi_flat.clone())

        elif t == 'mul':
            # Constant scalar/per-channel multiply: y = scale * x.
            c_in, G_in = _get(op['inputs'][0])
            scale_t = op.get('scale')
            if scale_t is None:
                raise ValueError("mul op missing 'scale' for forward zono")
            if isinstance(scale_t, np.ndarray):
                scale_t = torch.from_numpy(scale_t).to(device=device, dtype=dtype)
            elif not isinstance(scale_t, torch.Tensor):
                scale_t = torch.tensor(scale_t, dtype=dtype, device=device)
            else:
                scale_t = scale_t.to(device=device, dtype=dtype)
            sflat = scale_t.flatten()
            # Broadcast: per-channel or scalar. Per-channel must match
            # spatial layout; assume scalar or matches c_in.shape[1].
            if sflat.numel() == 1:
                c_out = c_in * sflat
                G_out = G_in * sflat
            elif sflat.numel() == c_in.shape[1]:
                c_out = c_in * sflat.unsqueeze(0)
                G_out = G_in * sflat.unsqueeze(0).unsqueeze(-1)
            else:
                # Per-channel broadcast over spatial: use op's input
                # shape to expand.
                in_shape = op.get('in_shapes_nd', [None])[0]
                if in_shape is None or len(in_shape) != 3:
                    raise ValueError(
                        f'mul: scale shape {sflat.shape} incompatible with '
                        f'input ({c_in.shape[1]}); no spatial shape known')
                C, H, W = in_shape
                assert sflat.numel() == C, (
                    f'mul per-channel scale {sflat.numel()} != C={C}')
                scale_4d = sflat.view(1, C, 1, 1).expand(1, C, H, W).reshape(1, -1)
                c_out = c_in * scale_4d
                G_out = G_in * scale_4d.unsqueeze(-1)
            state[name] = (c_out, G_out)

        elif t in ('mul_bilinear', 'matmul_bilinear', 'softmax'):
            # Nonlinear / variable-x-variable ops: collapse to box.
            # Forward computes interval bounds and emits a single new
            # gen column per non-zero-slack cell. Center = midpoint.
            c_a, G_a = _get(op['inputs'][0])
            abs_a = G_a.abs().sum(dim=2)
            lo_a = c_a - abs_a; hi_a = c_a + abs_a
            if t == 'mul_bilinear':
                c_b, G_b = _get(op['inputs'][1])
                abs_b = G_b.abs().sum(dim=2)
                lo_b = c_b - abs_b; hi_b = c_b + abs_b
                # Sound bound for x*y where x in [lo_a, hi_a], y in [lo_b, hi_b].
                corners = torch.stack(
                    [lo_a * lo_b, lo_a * hi_b, hi_a * lo_b, hi_a * hi_b], dim=-1)
                lo_out = corners.min(dim=-1).values
                hi_out = corners.max(dim=-1).values
            elif t == 'matmul_bilinear':
                # (B, .., M, K) @ (B, .., K, N) -> (B, .., M, N).
                # Reshape to N-D via op['in_shapes_nd'] and ['out_shape_nd'].
                c_b, G_b = _get(op['inputs'][1])
                abs_b = G_b.abs().sum(dim=2)
                lo_b = c_b - abs_b; hi_b = c_b + abs_b
                sh_a = op['in_shapes_nd'][0]
                sh_b = op['in_shapes_nd'][1]
                sh_o = op['out_shape_nd']
                lo_a_nd = lo_a.reshape(B, *sh_a)
                hi_a_nd = hi_a.reshape(B, *sh_a)
                lo_b_nd = lo_b.reshape(B, *sh_b)
                hi_b_nd = hi_b.reshape(B, *sh_b)
                # For y = a @ b with each a_ij in [lo_a_ij, hi_a_ij] and
                # b_jk in [lo_b_jk, hi_b_jk], element y_ik = sum_j a_ij * b_jk.
                # Bound sum_j of min/max over corners.
                # Per-pair corners:
                #   p_jk^lo = min(lo_a_ij*lo_b_jk, lo_a_ij*hi_b_jk,
                #                  hi_a_ij*lo_b_jk, hi_a_ij*hi_b_jk)
                # Sum over j gives sound lower bound. (Conservative but
                # straightforward; auto_LiRPA does tighter via McCormick.)
                # Implementation via four matmuls + min/max.
                pp = lo_a_nd.unsqueeze(-1) * lo_b_nd.unsqueeze(-3)  # (B, ..., M, K, N)
                pn = lo_a_nd.unsqueeze(-1) * hi_b_nd.unsqueeze(-3)
                np_ = hi_a_nd.unsqueeze(-1) * lo_b_nd.unsqueeze(-3)
                nn = hi_a_nd.unsqueeze(-1) * hi_b_nd.unsqueeze(-3)
                cmin = torch.minimum(torch.minimum(pp, pn),
                                       torch.minimum(np_, nn))  # (B,...,M,K,N)
                cmax = torch.maximum(torch.maximum(pp, pn),
                                       torch.maximum(np_, nn))
                lo_out_nd = cmin.sum(dim=-2)  # (B, ..., M, N)
                hi_out_nd = cmax.sum(dim=-2)
                n_out_layer = 1
                for d in sh_o:
                    n_out_layer *= d
                lo_out = lo_out_nd.reshape(B, n_out_layer)
                hi_out = hi_out_nd.reshape(B, n_out_layer)
            else:  # softmax
                # auto_LiRPA's interval bound:
                #   lower = exp(lo - shift) / (sum exp(hi - shift)
                #                              - exp(hi - shift) + exp(lo - shift) + eps)
                #   upper = exp(hi - shift) / (sum exp(lo - shift)
                #                              - exp(lo - shift) + exp(hi - shift) + eps)
                # where shift = max(hi) per row.
                axis = int(op.get('axis', -1))
                # Reshape to (B, ..., n_axis) per op's in_shape.
                sh_a = op['in_shapes_nd'][0]
                if sh_a is None:
                    raise ValueError('softmax requires in_shape_nd')
                # Normalize axis to the reshaped (B, *sh_a) tensor.
                ax = axis if axis >= 0 else axis + 1 + len(sh_a)
                lo_nd = lo_a.reshape(B, *sh_a)
                hi_nd = hi_a.reshape(B, *sh_a)
                shift = hi_nd.max(dim=ax, keepdim=True).values
                exp_lo = torch.exp(lo_nd - shift)
                exp_hi = torch.exp(hi_nd - shift)
                sum_hi = exp_hi.sum(dim=ax, keepdim=True)
                sum_lo = exp_lo.sum(dim=ax, keepdim=True)
                eps = 1e-12
                lo_out_nd = exp_lo / (sum_hi - exp_hi + exp_lo + eps)
                hi_out_nd = exp_hi / (sum_lo - exp_lo + exp_hi + eps)
                lo_out = lo_out_nd.reshape(B, -1)
                hi_out = hi_out_nd.reshape(B, -1)
            n_out_layer = lo_out.shape[1]
            c_out = (lo_out + hi_out) / 2
            mu = (hi_out - lo_out) / 2
            nonzero_mask = mu.abs() > 1e-9
            ust_cnt = nonzero_mask.sum(dim=1)
            max_K = int(ust_cnt.max().item())
            G_zeros = torch.zeros(B, n_out_layer, G_a.shape[2],
                                     dtype=dtype, device=device)
            if max_K == 0:
                G_out = G_zeros
            else:
                new_gens = torch.zeros(B, n_out_layer, max_K,
                                          dtype=dtype, device=device)
                rank = nonzero_mask.long().cumsum(dim=1) - 1
                b_idx = torch.arange(B, device=device).unsqueeze(-1).expand(
                    -1, n_out_layer)
                r_idx = torch.arange(n_out_layer,
                                       device=device).unsqueeze(0).expand(B, -1)
                new_gens[b_idx[nonzero_mask], r_idx[nonzero_mask],
                          rank[nonzero_mask]] = mu[nonzero_mask]
                G_out = torch.cat([G_zeros, new_gens], dim=2)
            state[name] = (c_out, G_out)
            sb[op['name'] + f'__{t}_box'] = (lo_out.clone(), hi_out.clone())

        elif t == 'transpose':
            # Linear permutation of dims (excluding the batch dim).
            # Equivalent to a permutation of the flat indices.
            c_in, G_in = _get(op['inputs'][0])
            sh_in = op['in_shapes_nd'][0]
            sh_out = op['out_shape_nd']
            perm = op['perm']
            # Perm is given over (1, *sh_in) (with leading 1 for batch).
            # Strip the leading 0 and shift the rest by -1 to apply on
            # (B, *sh_in).
            perm_b = [0] + [p for p in perm if p != 0]
            c_nd = c_in.reshape(B, *sh_in)
            c_out = c_nd.permute(*perm_b).reshape(B, -1).contiguous()
            K = G_in.shape[2]
            if K == 0:
                G_out = torch.zeros(B, c_out.shape[1], 0,
                                       dtype=dtype, device=device)
            else:
                # Permute each gen column the same way.
                g_perm = G_in.permute(0, 2, 1).reshape(
                    B * K, *sh_in)
                g_out = g_perm.permute(*perm_b).reshape(
                    B * K, -1).reshape(B, K, -1).permute(0, 2, 1).contiguous()
                G_out = g_out
            state[name] = (c_out, G_out)

        elif t == 'squeeze':
            # Reshape-only: data unchanged.
            c_in, G_in = _get(op['inputs'][0])
            state[name] = (c_in, G_in)

        else:
            raise ValueError(
                f'batched zono forward: unknown op type {t!r}')

        gen_count[name] = state[name][1].shape[2]
        for inp in op['inputs']:
            if last_use.get(inp) == op_idx and inp in state:
                del state[inp]

    last_name = gg['ops'][-1]['name']
    return sb, state[last_name]


def _spec_backward_graph_batched(tight, xl, xh, gg, spec_ew, device, dtype,
                                   return_input_linear=False,
                                   alpha_at_layer=None,
                                   seed_ew_at=None, seed_acc=None):
    """Batched CROWN spec backward.

    Args:
        tight: {layer_idx: (lo, hi)} where lo, hi are (B, n_layer).
        xl, xh: (B, n_in) per-batch input bounds.
        spec_ew: dict {qid: (w, bias)} with w (n_out,), bias scalar.
            Same w/bias applied across the batch.
        return_input_linear: if True, also returns the linear lower
            bound coefficients in input space:
              A: (B, Q, n_in) such that spec_q(x) >= A[b,q] · x + acc[b,q]
              acc: (B, Q) per-batch per-query bias
            Used by domain clipping in `_input_split_batched`.
        alpha_at_layer: optional dict {layer_idx: alpha} where alpha is
            (B, n_layer) tensor with values in [0, 1] giving the
            per-(leaf, neuron) lower slope for unstable ReLUs.
            Stable+ neurons use slope 1, stable- use 0, unstable use
            `alpha`. When provided, gradients flow through alpha →
            enables α-CROWN optimization. When None, falls back to
            min-area: lower slope = (up_s > 0.5). Caller controls
            torch.no_grad / torch.enable_grad as appropriate.

    Returns:
        spec_lbs: (B, Q) per-batch per-query lower bound on
        `w_q · y(x) + bias_q` over x in the batch's box.
        If return_input_linear: (spec_lbs, A, acc).
    """
    ops = gg['ops']
    B, n_in = xl.shape

    if seed_ew_at is not None:
        # Caller-provided seed: used for intermediate-layer tightening
        # (start the backward at an arbitrary op rather than the spec).
        ew_at = {name: seed.clone() for name, seed in seed_ew_at.items()}
        if seed_acc is None:
            any_seed = next(iter(seed_ew_at.values()))
            Q = any_seed.shape[1]
            acc = torch.zeros(B, Q, dtype=dtype, device=device)
        else:
            acc = seed_acc.clone()
    else:
        qids = sorted(spec_ew.keys())
        Q = len(qids)
        W_q = torch.stack([spec_ew[qid][0].flatten() for qid in qids])  # (Q, n_out)
        b_q = torch.tensor([float(spec_ew[qid][1]) for qid in qids],
                            dtype=dtype, device=device)  # (Q,)
        # Seed ew at output: (B, Q, n_out) — broadcast queries across batch.
        last_name = ops[-1]['name']
        ew_at = {last_name: W_q.unsqueeze(0).expand(B, -1, -1).clone()}
        acc = b_q.unsqueeze(0).expand(B, -1).clone()  # (B, Q)

    for op in reversed(ops):
        name = op['name']
        if name not in ew_at:
            continue
        ew = ew_at[name]  # (B, Q, n)
        t = op['type']

        if t == 'fc':
            W = op['W']
            bias = op['bias']
            acc = acc + ew @ bias  # (B, Q)
            ew_back = ew @ W  # (B, Q, n_in_layer)
            inp = op['inputs'][0]
            existing = ew_at.get(inp)
            ew_at[inp] = ew_back if existing is None else existing + ew_back

        elif t == 'relu':
            if 'layer_idx' in op:
                L = op['layer_idx']
                lo, hi = tight[L]  # (B, n)
                lo_s_def, up_s, up_t, active, dead, unstable = _make_slopes(
                    lo, hi)
                ep = ew.clamp(min=0)  # (B, Q, n)
                en = ew.clamp(max=0)
                # acc += (en * up_t).sum over n, per (b, q)
                acc = acc + (en * up_t.unsqueeze(1)).sum(dim=-1)
                # α-CROWN: replace default lower slope with α[L] for
                # unstable neurons; stable+ → 1, stable- → 0 (already
                # the case via masks).
                if alpha_at_layer is not None and L in alpha_at_layer:
                    alpha_L = alpha_at_layer[L]  # (B, n) or (B, Q, n)
                    DT = lo.dtype
                    if alpha_L.dim() == 2:
                        # shared α across queries
                        lo_s = (active.to(DT)
                                + unstable.to(DT) * alpha_L).unsqueeze(1)
                    else:
                        # per-query α: (B, Q, n)
                        lo_s = (active.to(DT).unsqueeze(1)
                                + unstable.to(DT).unsqueeze(1) * alpha_L)
                    ew_back = ep * lo_s + en * up_s.unsqueeze(1)
                else:
                    lo_s = lo_s_def
                    ew_back = ep * lo_s.unsqueeze(1) + en * up_s.unsqueeze(1)
            else:
                ew_back = ew
            inp = op['inputs'][0]
            existing = ew_at.get(inp)
            ew_at[inp] = ew_back if existing is None else existing + ew_back

        elif t == 'add':
            if op.get('is_merge'):
                # Skip connection: y = z_a + z_b. ew flows to both
                # inputs unchanged; no constant contribution to acc.
                for inp in op['inputs']:
                    existing = ew_at.get(inp)
                    ew_at[inp] = ew.clone() if existing is None else existing + ew
            else:
                # Constant bias-add: y = z + bias. CROWN backward
                # passes ew through to z but must accumulate the bias
                # contribution `(ew · bias).sum` into acc. Dropping it
                # is silently UNSOUND when the bias contribution
                # changes sign across α-CROWN iters (acasxu prop_8:
                # plain CROWN sound by accident, α-CROWN explodes to
                # +6 when true min is +0.03). Same pattern as the
                # bias-drop bugs fixed in `verify_milp.py`.
                bias = op.get('bias')
                if bias is not None:
                    bt = torch.as_tensor(bias.flatten(),
                                          dtype=dtype, device=device)
                    acc = acc + (ew * bt).sum(dim=-1)
                inp = op['inputs'][0]
                existing = ew_at.get(inp)
                ew_at[inp] = ew.clone() if existing is None else existing + ew

        elif t == 'sub':
            bias = op.get('bias')
            if bias is not None:
                bt = torch.as_tensor(bias.flatten(),
                                      dtype=dtype, device=device)
                # acc -= (ew * bt).sum over n
                acc = acc - (ew * bt).sum(dim=-1)
            inp = op['inputs'][0]
            existing = ew_at.get(inp)
            ew_at[inp] = ew.clone() if existing is None else existing + ew

        elif t == 'reshape':
            inp = op['inputs'][0]
            existing = ew_at.get(inp)
            ew_at[inp] = ew.clone() if existing is None else existing + ew

        elif t == 'conv':
            # Backward of conv2d is conv_transpose2d. ew shape (B, Q, n_out).
            kernel = op['kernel']
            bias = op['bias']
            out_shape = op['out_shape']
            in_shape = op['in_shape']
            stride = op['stride']
            padding = op['padding']
            output_padding = op['output_padding']
            # acc += sum over neurons of ew * bias_per_neuron.
            # bias is per-channel, broadcast over spatial.
            C_out, H_out, W_out = out_shape
            spatial = H_out * W_out
            bias_flat = bias.repeat_interleave(spatial)  # (C_out * spatial,)
            acc = acc + (ew * bias_flat).sum(dim=-1)
            # ew_back: reshape ew to (B*Q, C_out, H_out, W_out), apply
            # conv_transpose2d with kernel, flatten back.
            ew_4d = ew.reshape(B * Q, *out_shape)
            ew_back_4d = F.conv_transpose2d(
                ew_4d, kernel, bias=None, stride=stride, padding=padding,
                output_padding=output_padding)
            n_in_layer = in_shape[0] * in_shape[1] * in_shape[2]
            ew_back = ew_back_4d.reshape(B, Q, n_in_layer)
            inp = op['inputs'][0]
            existing = ew_at.get(inp)
            ew_at[inp] = ew_back if existing is None else existing + ew_back

        elif t == 'conv_transpose':
            # Backward of conv_transpose2d is conv2d. ew shape (B, Q, n_out).
            kernel = op['kernel']
            bias = op['bias']
            out_shape = op['out_shape']
            in_shape = op['in_shape']
            stride = op['stride']
            padding = op['padding']
            C_out, H_out, W_out = out_shape
            spatial = H_out * W_out
            bias_flat = bias.repeat_interleave(spatial)
            acc = acc + (ew * bias_flat).sum(dim=-1)
            ew_4d = ew.reshape(B * Q, *out_shape)
            ew_back_4d = F.conv2d(
                ew_4d, kernel, bias=None, stride=stride, padding=padding)
            n_in_layer = in_shape[0] * in_shape[1] * in_shape[2]
            assert ew_back_4d.shape[2] * ew_back_4d.shape[3] * ew_back_4d.shape[1] == n_in_layer, \
                (f"conv_transpose backward shape mismatch: got "
                 f"{ew_back_4d.shape} expected total {n_in_layer}")
            ew_back = ew_back_4d.reshape(B, Q, n_in_layer)
            inp = op['inputs'][0]
            existing = ew_at.get(inp)
            ew_at[inp] = ew_back if existing is None else existing + ew_back

        elif t == 'bn':
            # Per-channel affine y = factor * x + offset.
            # Backward: ew_back = ew * factor; acc += (ew * offset).sum.
            factor = op['factor']  # (n,)
            offset = op['offset']  # (n,)
            acc = acc + (ew * offset).sum(dim=-1)
            ew_back = ew * factor  # broadcast across (B, Q, n)
            inp = op['inputs'][0]
            existing = ew_at.get(inp)
            ew_at[inp] = ew_back if existing is None else existing + ew_back

        elif t in ('sigmoid', 'tanh'):
            # CROWN backward through sigmoid/tanh — closed-form linear
            # bounds from `_sigmoid_tanh_linear_bounds`.
            # Pre-activation bounds come from `tight[L]`, recorded by
            # the forward zono.
            L = op['layer_idx']
            lo_pre, hi_pre = tight[L]
            lo_s, lo_t, up_s, up_t = _sigmoid_tanh_linear_bounds(
                lo_pre, hi_pre, t)
            ep = ew.clamp(min=0)
            en = ew.clamp(max=0)
            acc = acc + (ep * lo_t.unsqueeze(1)).sum(dim=-1) + \
                    (en * up_t.unsqueeze(1)).sum(dim=-1)
            ew_back = ep * lo_s.unsqueeze(1) + en * up_s.unsqueeze(1)
            inp = op['inputs'][0]
            existing = ew_at.get(inp)
            ew_at[inp] = ew_back if existing is None else existing + ew_back

        elif t == 'upsample':
            # Nearest-mode upsample y[c, h*sH+a, w*sW+b] = x[c, h, w] for
            # all (a, b) in [0, sH)×[0, sW). Adjoint sums over the
            # repeated output cells per input cell: avg_pool2d with
            # divisor_override=1.
            in_shape = op['in_shape']
            out_shape = op['out_shape']
            sH, sW = op['scale']
            ew_4d = ew.reshape(B * Q, *out_shape)
            ew_back_4d = F.avg_pool2d(
                ew_4d, kernel_size=(sH, sW), stride=(sH, sW),
                divisor_override=1)
            n_in_layer = in_shape[0] * in_shape[1] * in_shape[2]
            ew_back = ew_back_4d.reshape(B, Q, n_in_layer)
            inp = op['inputs'][0]
            existing = ew_at.get(inp)
            ew_at[inp] = ew_back if existing is None else existing + ew_back

        elif t == 'avg_pool':
            # avg_pool y = (1/(kH*kW)) * sum window. Adjoint = depthwise
            # conv_transpose with (1/(kH*kW))-uniform kernel. For
            # non-overlapping (stride==kernel) just F.conv_transpose2d
            # works. For overlapping windows the same call gives the
            # correct sum-of-broadcasted-gradients.
            in_shape = op['in_shapes_nd'][0]
            out_shape = op['out_shape_nd']
            C, H_in, W_in = in_shape
            kH, kW = op['kernel']
            sH, sW = op['stride']
            pH, pW = op['padding']
            ew_4d = ew.reshape(B * Q, *out_shape)
            w_avg = torch.full((C, 1, kH, kW), 1.0 / (kH * kW),
                                  dtype=dtype, device=device)
            ew_back_4d = F.conv_transpose2d(
                ew_4d, w_avg, bias=None, stride=(sH, sW),
                padding=(pH, pW), groups=C)
            # Crop or pad to match input shape exactly.
            if ew_back_4d.shape[2] != H_in or ew_back_4d.shape[3] != W_in:
                ew_back_4d = ew_back_4d[:, :, :H_in, :W_in]
                if ew_back_4d.shape[2] < H_in or ew_back_4d.shape[3] < W_in:
                    pad_h = H_in - ew_back_4d.shape[2]
                    pad_w = W_in - ew_back_4d.shape[3]
                    ew_back_4d = F.pad(ew_back_4d, (0, pad_w, 0, pad_h))
            n_in_layer = C * H_in * W_in
            ew_back = ew_back_4d.reshape(B, Q, n_in_layer)
            inp = op['inputs'][0]
            existing = ew_at.get(inp)
            ew_at[inp] = ew_back if existing is None else existing + ew_back

        elif t == 'mul':
            # y = scale * x (constant scale). Backward: ew_back = ew * scale.
            scale_t = op.get('scale')
            if isinstance(scale_t, np.ndarray):
                scale_t = torch.from_numpy(scale_t).to(
                    device=device, dtype=dtype)
            elif not isinstance(scale_t, torch.Tensor):
                scale_t = torch.tensor(scale_t, dtype=dtype, device=device)
            else:
                scale_t = scale_t.to(device=device, dtype=dtype)
            sflat = scale_t.flatten()
            n_in_layer = ew.shape[-1]
            if sflat.numel() == 1:
                ew_back = ew * sflat
            elif sflat.numel() == n_in_layer:
                ew_back = ew * sflat.unsqueeze(0).unsqueeze(0)
            else:
                in_shape = op['in_shapes_nd'][0]
                C, H, W = in_shape
                assert sflat.numel() == C
                scale_4d = sflat.view(1, C, 1, 1).expand(
                    1, C, H, W).reshape(1, 1, -1)
                ew_back = ew * scale_4d
            inp = op['inputs'][0]
            existing = ew_at.get(inp)
            ew_at[inp] = ew_back if existing is None else existing + ew_back

        elif t in ('max_pool', 'mul_bilinear', 'matmul_bilinear', 'softmax'):
            # Box-relaxation CROWN: y has constant bounds (lo_out, hi_out)
            # stamped at forward time. Linear lower bound is the constant
            # lo_out (slope 0); the contribution to acc is
            # sum_n max(0, ew[n]) * lo_out[n] + sum_n min(0, ew[n]) * hi_out[n].
            # No backward signal to inputs.
            key = op['name'] + f'__{t}_box'
            if key not in tight:
                raise ValueError(
                    f'batched backward: missing box bounds for {key}')
            lo_box, hi_box = tight[key]  # (B, n_out)
            ep = ew.clamp(min=0)
            en = ew.clamp(max=0)
            acc = acc + (ep * lo_box.unsqueeze(1)).sum(dim=-1) + \
                    (en * hi_box.unsqueeze(1)).sum(dim=-1)
            # No ew_back: inputs aren't propagated (box loses correlation).

        elif t == 'transpose':
            # Inverse-permutation of the gen layout.
            sh_in = op['in_shapes_nd'][0]
            sh_out = op['out_shape_nd']
            perm = op['perm']
            perm_b = [0] + [p for p in perm if p != 0]
            # Inverse permutation.
            inv_perm = [0] * len(perm_b)
            for i, p in enumerate(perm_b):
                inv_perm[p] = i
            ew_nd = ew.reshape(B, Q, *sh_out)
            # ew is (B, Q, *sh_out). We want to permute the sh_out axes
            # by inv_perm. ew has 2 leading dims (B, Q) — shift inv_perm
            # by +1 to account for the Q dim while keeping B at 0.
            perm_eq = [0, 1] + [p + 1 for p in inv_perm if p != 0]
            ew_back_nd = ew_nd.permute(*perm_eq).contiguous()
            n_in_layer = ew_back_nd.numel() // (B * Q)
            ew_back = ew_back_nd.reshape(B, Q, n_in_layer)
            inp = op['inputs'][0]
            existing = ew_at.get(inp)
            ew_at[inp] = ew_back if existing is None else existing + ew_back

        elif t == 'squeeze':
            # No-op: shape change only, data unchanged.
            inp = op['inputs'][0]
            existing = ew_at.get(inp)
            ew_at[inp] = ew if existing is None else existing + ew

        else:
            raise ValueError(f'batched spec backward: unknown op {t!r}')

    input_name = gg['input_name']
    ew_inp = ew_at.get(input_name)
    if ew_inp is None:
        if return_input_linear:
            zeros = torch.zeros(B, Q, n_in, dtype=dtype, device=device)
            return acc, zeros, acc
        return acc
    # Per-batch interval bound: spec_lb[b, q] = acc[b, q]
    # + sum_i [pos(ew[b, q, i]) * xl[b, i] + neg(ew[b, q, i]) * xh[b, i]]
    pos = ew_inp.clamp(min=0)
    neg = ew_inp.clamp(max=0)
    spec_lbs = (acc
                 + (pos * xl.unsqueeze(1)).sum(dim=-1)
                 + (neg * xh.unsqueeze(1)).sum(dim=-1))
    if return_input_linear:
        # spec_q(x) >= ew_inp[b,q] · x + acc[b,q]
        return spec_lbs, ew_inp, acc
    return spec_lbs


def _run_alpha_crown_inputsplit_batched(xl, xh, gg, spec_ew, device, dtype,
                                           n_iters=10, lr=0.25, lr_decay=0.98,
                                           early_stop_eps=1e-6):
    """Batched α-CROWN for input-split BaB boundary leaves.

    Optimizes per-(leaf, layer, neuron) lower-slope α to maximize
    per-query spec lb across a BATCH of leaves on GPU. Uses Adam.

    Args:
        xl, xh: (B, n_in) input bounds per leaf.
        gg: gpu_graph dict.
        spec_ew: dict {qid: (w (n_out,), bias float)} — same query
            family for all leaves; spec_lbs returned as (B, Q).
        n_iters: max Adam iters.
        lr, lr_decay: optimizer schedule.
        early_stop_eps: if no leaf's spec_lb improves by more than
            `eps` for one full iter, stop early.

    Returns:
        best_spec_lbs: (B, Q) — best spec lb seen across iterations.

    Notes:
      - α is initialized to min-area choice (1.0 where up_s > 0.5, else
        0.0). First iter equals plain CROWN.
      - α is clamped to [0, 1] after each Adam step.
      - Loss = -sum over (b, q) of spec_lbs (maximize lb).
      - For ACASXU 6-layer × 50-neuron net at B=100: ~10 ms per iter.
    """
    B, n_in = xl.shape
    with torch.no_grad():
        sb_init, _ = _forward_zonotope_graph_batched(
            xl, xh, gg, device, dtype)
    alpha_at_layer = {}
    for L, (lo, hi) in sb_init.items():
        _, up_s, _, active, dead, unstable = _make_slopes(lo, hi)
        init_alpha = ((up_s > 0.5).to(dtype) * unstable.to(dtype))
        alpha_at_layer[L] = init_alpha.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam(
        [alpha_at_layer[L] for L in alpha_at_layer], lr=lr)
    best_spec_lbs = None
    prev_max = -float('inf')
    for it in range(n_iters):
        optimizer.zero_grad()
        spec_lbs = _spec_backward_graph_batched(
            sb_init, xl, xh, gg, spec_ew, device, dtype,
            alpha_at_layer=alpha_at_layer)
        with torch.no_grad():
            if best_spec_lbs is None:
                best_spec_lbs = spec_lbs.detach().clone()
            else:
                best_spec_lbs = torch.maximum(best_spec_lbs, spec_lbs.detach())
            curr_max = float(best_spec_lbs.max().item())
        if (best_spec_lbs > 0).all().item():
            break
        loss = -spec_lbs.sum()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for L in alpha_at_layer:
                alpha_at_layer[L].clamp_(0.0, 1.0)
        if it > 0 and curr_max - prev_max < early_stop_eps:
            break
        prev_max = curr_max
        for g in optimizer.param_groups:
            g['lr'] *= lr_decay
    return best_spec_lbs


def _alpha_crown_layerwise_tighten(xl, xh, gg, device, dtype,
                                     n_iters_per_layer=50, lr=0.25,
                                     lr_decay=0.98, early_stop_eps=1e-6,
                                     per_query_alpha=True, verbose=False):
    """Per-layer α-CROWN intermediate-bound tightening.

    For each ReLU layer L (in order), solve α-CROWN to tighten the
    pre-ReLU bounds of layer L's neurons, using α optimization over the
    UPSTREAM ReLU layers k < L. Update `tight[L]` with the elementwise
    max/min of the previous and the newly computed bounds.

    The tightened bounds at layer k cascade into layer k+1's
    optimization (via `tight[k]` used for ReLU slopes in the backward).

    Args:
        xl, xh: (B, n_in) input bounds (B=1 for root use).
        gg: gpu_graph dict.
        n_iters_per_layer: Adam iters per layer.
        lr, lr_decay: optimizer schedule (reset per layer).
        verbose: if True, return a per-layer log.

    Returns:
        tight: {layer_idx: (lo, hi)} dict, each (B, n_layer), tightened.
        log: list of dicts (only if verbose); one per layer.
    """
    B, n_in = xl.shape
    with torch.no_grad():
        sb_init, _ = _forward_zonotope_graph_batched(
            xl, xh, gg, device, dtype)
    tight = {L: (lo.clone(), hi.clone()) for L, (lo, hi) in sb_init.items()}
    layer_order = sorted(tight.keys())

    # Map layer_idx → ReLU op and its feed op (the linear/add producing
    # pre-activations consumed by this ReLU).
    relu_op_by_L = {}
    for op in gg['ops']:
        if op['type'] == 'relu' and 'layer_idx' in op:
            relu_op_by_L[op['layer_idx']] = op

    log = []
    for L in layer_order:
        relu_op = relu_op_by_L[L]
        feed_name = relu_op['inputs'][0]
        lo_old, hi_old = tight[L]
        n_layer = lo_old.shape[1]
        ust_before = ((lo_old < 0) & (hi_old > 0)).sum().item()

        # Seed: 2*n_layer queries with W rows = +I (rows 0..n-1) and -I
        # (rows n..2n-1). lb of +e_k gives new lb of y_L[k]; lb of -e_k
        # gives -new_ub of y_L[k].
        I = torch.eye(n_layer, dtype=dtype, device=device)
        W_q = torch.cat([I, -I], dim=0)  # (2n, n_layer)
        seed_ew = {feed_name: W_q.unsqueeze(0).expand(B, -1, -1)}
        seed_acc = torch.zeros(B, 2 * n_layer, dtype=dtype, device=device)

        # α params for layers k < L only (downstream layers not touched
        # by this backward). Init to min-area slope. When per_query_alpha
        # is True, α has shape (B, 2*n_layer, n_k) so each query gets
        # its own α (closes the dual gap of shared-α; per-neuron α can
        # reach the LP triangle bound).
        Q = 2 * n_layer
        alpha_at_layer = {}
        for k in layer_order:
            if k >= L:
                break
            lo_k, hi_k = tight[k]
            _, up_s_k, _, _, _, unstable_k = _make_slopes(lo_k, hi_k)
            init_alpha = ((up_s_k > 0.5).to(dtype) * unstable_k.to(dtype))
            if per_query_alpha:
                # broadcast (B, n_k) → (B, Q, n_k)
                init_alpha = init_alpha.unsqueeze(1).expand(-1, Q, -1).contiguous()
            alpha_at_layer[k] = (init_alpha.detach().clone()
                                  .requires_grad_(True))

        if alpha_at_layer:
            optimizer = torch.optim.Adam(
                [alpha_at_layer[k] for k in alpha_at_layer], lr=lr)
        else:
            optimizer = None

        best_lbs = None
        prev_max = -float('inf')
        for it in range(max(1, n_iters_per_layer)):
            if optimizer is not None:
                optimizer.zero_grad()
            spec_lbs = _spec_backward_graph_batched(
                tight, xl, xh, gg, None, device, dtype,
                alpha_at_layer=alpha_at_layer if alpha_at_layer else None,
                seed_ew_at=seed_ew, seed_acc=seed_acc)
            with torch.no_grad():
                if best_lbs is None:
                    best_lbs = spec_lbs.detach().clone()
                else:
                    best_lbs = torch.maximum(best_lbs, spec_lbs.detach())
                # Sum-of-best — captures progress on ANY query, not
                # just the worst. Using best.max saturates instantly
                # when there are many independent queries (per-layer
                # tightening = 2*n_layer queries).
                curr_sig = float(best_lbs.sum().item())
            if optimizer is None:
                break
            loss = -spec_lbs.sum()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for k in alpha_at_layer:
                    alpha_at_layer[k].clamp_(0.0, 1.0)
            if it > 0 and curr_sig - prev_max < early_stop_eps:
                break
            prev_max = curr_sig
            for g in optimizer.param_groups:
                g['lr'] *= lr_decay

        # best_lbs[:, :n_layer] = new_lo; best_lbs[:, n_layer:] = -new_hi.
        with torch.no_grad():
            new_lo = best_lbs[:, :n_layer]
            new_hi = -best_lbs[:, n_layer:]
            lo_t = torch.maximum(lo_old, new_lo)
            hi_t = torch.minimum(hi_old, new_hi)
            # Numerical safety: keep lo <= hi.
            lo_t = torch.minimum(lo_t, hi_t)
        tight[L] = (lo_t, hi_t)

        if verbose:
            ust_after = ((lo_t < 0) & (hi_t > 0)).sum().item()
            log.append({
                'layer': L,
                'n_layer': n_layer,
                'ust_before': ust_before,
                'ust_after': ust_after,
                'iters': it + 1,
            })

    if verbose:
        return tight, log
    return tight


def _alpha_crown_layerwise_tighten_chunked(xl, xh, gg, device, dtype,
                                              chunk_size=512,
                                              min_chunk_size=8,
                                              **kwargs):
    """OOM-resilient chunked wrapper around
    `_alpha_crown_layerwise_tighten`.

    Splits the input batch into chunks of `chunk_size`, runs the tighten
    on each chunk separately, and concatenates results. On
    `torch.cuda.OutOfMemoryError`, halves `chunk_size` and retries until
    either the chunk succeeds or `min_chunk_size` is reached.

    Empirically (acasxu 6×50, RTX 3080 10 GB): peak throughput at
    chunk≈512 (~168 leaves/sec) with per-query α; larger chunks add no
    throughput but consume memory. Default 512.

    Args:
        xl, xh: (B, n_in) input bounds — any B.
        chunk_size: target chunk size; auto-halves on OOM.
        min_chunk_size: floor; raises if reached without success.
        **kwargs: forwarded to `_alpha_crown_layerwise_tighten`.

    Returns:
        tight: {layer_idx: (lo, hi)} concatenated along batch dim.
        (No `verbose` log return — inner verbose is ignored.)
    """
    kwargs.pop('verbose', None)  # log doesn't compose across chunks
    B = xl.shape[0]
    out_tight = None
    i = 0
    cur_chunk = max(1, min(int(chunk_size), B))
    while i < B:
        end = min(i + cur_chunk, B)
        chunk_xl = xl[i:end]
        chunk_xh = xh[i:end]
        try:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            t = _alpha_crown_layerwise_tighten(
                chunk_xl, chunk_xh, gg, device, dtype, **kwargs)
            # accumulate
            if out_tight is None:
                out_tight = {L: ([lo], [hi]) for L, (lo, hi) in t.items()}
            else:
                for L, (lo, hi) in t.items():
                    out_tight[L][0].append(lo)
                    out_tight[L][1].append(hi)
            i = end
        except torch.cuda.OutOfMemoryError:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            if cur_chunk <= min_chunk_size:
                raise
            cur_chunk = max(min_chunk_size, cur_chunk // 2)
    # concat
    return {L: (torch.cat(los, dim=0), torch.cat(his, dim=0))
             for L, (los, his) in out_tight.items()}


@torch.no_grad()
def _evaluate_region(xl, xh, remaining_specs, gpu_layers_list, spec_ew,
                     pred, nh, device, dtype):
    """Three-phase evaluation: forward zonotope, backward tighten, spec backward.

    Returns (spec_lbs, still_open, split_dim).
    """
    # Phase 1: Forward zonotope
    z = TorchZonotope.from_input_bounds(xl, xh, device, dtype)
    sb = {}
    for l in range(nh):
        gl = gpu_layers_list[l]
        if gl['type'] == 'conv':
            z.propagate_conv(gl['kernel'], gl['bias'], gl['in_shape'],
                             gl['stride'], gl['padding'])
        else:
            z.propagate_fc(gl['W'], gl['bias'])
        lo, hi = z.apply_relu()
        sb[l] = (lo.clone(), hi.clone())

    # Phase 2: Backward tighten unstable neurons
    if nh > 0:
        tight = {0: (sb[0][0].clone(), sb[0][1].clone())}
    else:
        tight = {}
    for l in range(1, nh):
        lo_std, hi_std = sb[l]
        ust_idx = torch.where((lo_std < 0) & (hi_std > 0))[0]
        n_ust = len(ust_idx)
        if n_ust == 0:
            tight[l] = (lo_std.clone(), hi_std.clone())
            continue

        # Precompute layer info for backward pass
        layer_info = {}
        for k in range(l):
            lo_k, hi_k = tight[k]
            lo_s, up_s, up_t, active, dead, ust_k = _make_slopes(lo_k, hi_k)
            act_idx = torch.where(active)[0]
            ust_k_idx = torch.where(ust_k)[0]
            dead_idx = torch.where(dead)[0]
            pct = len(ust_k_idx) / len(lo_k) if len(lo_k) > 0 else 1.0
            glk = gpu_layers_list[k]
            info = {
                'act_idx': act_idx, 'ust_idx': ust_k_idx,
                'dead_idx': dead_idx,
                'up_s_ust': up_s[ust_k_idx], 'up_t_ust': up_t[ust_k_idx],
                'lo_s_ust': lo_s[ust_k_idx],
                'is_conv': glk['type'] == 'conv', 'glk': glk,
                'lo_s_full': lo_s, 'up_s_full': up_s, 'up_t_full': up_t,
                'pct': pct,
            }
            if not info['is_conv']:
                info['W_act'] = glk['W'][act_idx]
                info['b_act'] = glk['bias'][act_idx]
                info['W_ust'] = glk['W'][ust_k_idx]
                info['b_ust'] = glk['bias'][ust_k_idx]
            layer_info[k] = info

        gl = gpu_layers_list[l]
        lbs = torch.empty(n_ust, dtype=dtype, device=device)
        ubs = torch.empty(n_ust, dtype=dtype, device=device)

        for cs in range(0, n_ust, 512):
            ce = min(cs + 512, n_ust)
            cidx = ust_idx[cs:ce]
            batch = len(cidx)

            if gl['type'] == 'conv':
                I_p = torch.zeros(batch, gl['n_out'], dtype=dtype,
                                  device=device)
                I_p[torch.arange(batch, device=device), cidx] = 1.0
                EW = F.conv_transpose2d(
                    I_p.reshape(batch, *gl['out_shape']), gl['kernel'],
                    stride=gl['stride'], padding=gl['padding'],
                    output_padding=gl['output_padding']).reshape(batch, -1)
                spatial = gl['out_shape'][1] * gl['out_shape'][2]
                bi = gl['bias'][cidx // spatial]
            else:
                EW = gl['W'][cidx].clone()
                bi = gl['bias'][cidx].clone()

            bias_lb = bi.clone()
            bias_ub = bi.clone()
            EW_lb = EW.clone()
            EW_ub = EW.clone()

            for k in range(l - 1, -1, -1):
                info = layer_info[k]
                if info['is_conv']:
                    if info['pct'] < 0.5:
                        ust_k_idx = info['ust_idx']
                        dead_idx = info['dead_idx']
                        EW_lb[:, dead_idx] = 0
                        EW_ub[:, dead_idx] = 0
                        ew_u = EW_lb[:, ust_k_idx]
                        ep = ew_u.clamp(min=0)
                        en = ew_u.clamp(max=0)
                        bias_lb += (en * info['up_t_ust']).sum(dim=1)
                        EW_lb[:, ust_k_idx] = (ep * info['lo_s_ust']
                                               + en * info['up_s_ust'])
                        ew_u = EW_ub[:, ust_k_idx]
                        ep = ew_u.clamp(min=0)
                        en = ew_u.clamp(max=0)
                        bias_ub += (ep * info['up_t_ust']).sum(dim=1)
                        EW_ub[:, ust_k_idx] = (ep * info['up_s_ust']
                                               + en * info['lo_s_ust'])
                    else:
                        ep = EW_lb.clamp(min=0)
                        en = EW_lb.clamp(max=0)
                        bias_lb += (en * info['up_t_full']).sum(dim=1)
                        EW_lb = (ep * info['lo_s_full']
                                 + en * info['up_s_full'])
                        ep = EW_ub.clamp(min=0)
                        en = EW_ub.clamp(max=0)
                        bias_ub += (ep * info['up_t_full']).sum(dim=1)
                        EW_ub = (ep * info['up_s_full']
                                 + en * info['lo_s_full'])
                    glk = info['glk']
                    os_k = glk['out_shape']
                    bias_lb += (EW_lb.reshape(batch, *os_k).sum(dim=(2, 3))
                                @ glk['bias'])
                    bias_ub += (EW_ub.reshape(batch, *os_k).sum(dim=(2, 3))
                                @ glk['bias'])
                    EW_lb = F.conv_transpose2d(
                        EW_lb.reshape(batch, *os_k), glk['kernel'],
                        stride=glk['stride'], padding=glk['padding'],
                        output_padding=glk['output_padding']
                    ).reshape(batch, -1)
                    EW_ub = F.conv_transpose2d(
                        EW_ub.reshape(batch, *os_k), glk['kernel'],
                        stride=glk['stride'], padding=glk['padding'],
                        output_padding=glk['output_padding']
                    ).reshape(batch, -1)
                else:
                    act_idx = info['act_idx']
                    ust_k_idx = info['ust_idx']
                    n_act = len(act_idx)
                    n_ust_k = len(ust_k_idx)
                    out_dim = (info['W_act'].shape[1] if n_act > 0
                               else info['W_ust'].shape[1])
                    EW_lb_new = torch.zeros(batch, out_dim, dtype=dtype,
                                            device=device)
                    EW_ub_new = torch.zeros_like(EW_lb_new)
                    if n_act > 0:
                        EW_lb_new += EW_lb[:, act_idx] @ info['W_act']
                        bias_lb += EW_lb[:, act_idx] @ info['b_act']
                        EW_ub_new += EW_ub[:, act_idx] @ info['W_act']
                        bias_ub += EW_ub[:, act_idx] @ info['b_act']
                    if n_ust_k > 0:
                        ep = EW_lb[:, ust_k_idx].clamp(min=0)
                        en = EW_lb[:, ust_k_idx].clamp(max=0)
                        bias_lb += (en * info['up_t_ust']).sum(dim=1)
                        ew_a = ep * info['lo_s_ust'] + en * info['up_s_ust']
                        EW_lb_new += ew_a @ info['W_ust']
                        bias_lb += ew_a @ info['b_ust']
                        ep = EW_ub[:, ust_k_idx].clamp(min=0)
                        en = EW_ub[:, ust_k_idx].clamp(max=0)
                        bias_ub += (ep * info['up_t_ust']).sum(dim=1)
                        ew_a = ep * info['up_s_ust'] + en * info['lo_s_ust']
                        EW_ub_new += ew_a @ info['W_ust']
                        bias_ub += ew_a @ info['b_ust']
                    EW_lb = EW_lb_new
                    EW_ub = EW_ub_new

            lbs[cs:ce] = (bias_lb + EW_lb.clamp(min=0) @ xl
                          + EW_lb.clamp(max=0) @ xh)
            ubs[cs:ce] = (bias_ub + EW_ub.clamp(min=0) @ xh
                          + EW_ub.clamp(max=0) @ xl)

        new_lo = lo_std.clone()
        new_hi = hi_std.clone()
        new_lo[ust_idx] = torch.maximum(lo_std[ust_idx], lbs)
        new_hi[ust_idx] = torch.minimum(hi_std[ust_idx], ubs)
        tight[l] = (new_lo, new_hi)

    # Phase 3: Spec backward
    spec_lbs = {}
    input_weights = {}
    for comp in remaining_specs:
        ew, b_spec = spec_ew[comp]
        ew = ew.clone()
        acc = b_spec
        for k in range(nh - 1, -1, -1):
            lo_k, hi_k = tight[k]
            lo_s, up_s, up_t, _, _, _ = _make_slopes(lo_k, hi_k)
            ep = ew.clamp(min=0)
            en = ew.clamp(max=0)
            acc += float((en * up_t).sum())
            ew = ep * lo_s + en * up_s
            glk = gpu_layers_list[k]
            if glk['type'] == 'conv':
                os_k = glk['out_shape']
                ew_4d = ew.reshape(1, *os_k)
                acc += float(
                    ew_4d.reshape(os_k[0], -1).sum(dim=1) @ glk['bias'])
                ew = F.conv_transpose2d(
                    ew_4d, glk['kernel'], stride=glk['stride'],
                    padding=glk['padding'],
                    output_padding=glk['output_padding']).flatten()
            else:
                acc += float(ew @ glk['bias'])
                ew = ew @ glk['W']
        spec_lbs[comp] = acc + float(
            ew.clamp(min=0) @ xl + ew.clamp(max=0) @ xh)
        input_weights[comp] = ew.detach()

    still_open = {c for c in remaining_specs if spec_lbs[c] <= 0}
    if still_open:
        w = (xh - xl).cpu().numpy()
        score = np.zeros(len(w))
        for comp in still_open:
            score += np.abs(input_weights[comp].cpu().numpy())
        split_dim = int(np.argmax(score * w))
    else:
        split_dim = -1
    return spec_lbs, still_open, split_dim


@torch.no_grad()
def _spec_backward(tight, xl, xh, gpu_layers_list, spec_ew,
                   remaining_specs, nh, device, dtype):
    """Spec backward pass using provided tight bounds.

    Returns (spec_lbs, still_open) without split_dim computation.
    """
    spec_lbs = {}
    for comp in remaining_specs:
        ew, b_spec = spec_ew[comp]
        ew = ew.clone()
        acc = b_spec
        for k in range(nh - 1, -1, -1):
            lo_k, hi_k = tight[k]
            lo_s, up_s, up_t, _, _, _ = _make_slopes(lo_k, hi_k)
            ep = ew.clamp(min=0)
            en = ew.clamp(max=0)
            acc += float((en * up_t).sum())
            ew = ep * lo_s + en * up_s
            glk = gpu_layers_list[k]
            if glk['type'] == 'conv':
                os_k = glk['out_shape']
                ew_4d = ew.reshape(1, *os_k)
                acc += float(
                    ew_4d.reshape(os_k[0], -1).sum(dim=1) @ glk['bias'])
                ew = F.conv_transpose2d(
                    ew_4d, glk['kernel'], stride=glk['stride'],
                    padding=glk['padding'],
                    output_padding=glk['output_padding']).flatten()
            else:
                acc += float(ew @ glk['bias'])
                ew = ew @ glk['W']
        spec_lbs[comp] = acc + float(
            ew.clamp(min=0) @ xl + ew.clamp(max=0) @ xh)
    still_open = {c for c in remaining_specs if spec_lbs[c] <= 0}
    return spec_lbs, still_open


def _fmt_eta(seconds):
    """Format ETA for display."""
    if seconds < 60:
        return '%.1fs' % seconds
    if seconds < 3600:
        return '%dm%02ds' % (int(seconds) // 60, int(seconds) % 60)
    if seconds < 86400:
        return '%dh%02dm' % (int(seconds) // 3600,
                             (int(seconds) % 3600) // 60)
    days = seconds / 86400
    if days > 99:
        return '>99days'
    return '%dd%02dh' % (int(days), int((seconds % 86400) / 3600))


def _run_bnb(evaluate_fn, pgd_fn, x_lo, x_hi, comps, settings):
    """Queue-based Branch-and-Bound loop.

    Returns ('verified', 'unknown', or 'sat', details_dict).
    """
    mode = settings.bnb_order
    print_progress = settings.print_progress
    timeout = settings.bnb_timeout

    if print_progress:
        print('=== BnB: %s + PGD-guided + progress tracking ===' % mode.upper())

    t_wall = time.perf_counter()

    # Initial PGD
    t0 = time.perf_counter()
    is_sat, witness, best_adv = pgd_fn(x_lo, x_hi, set(comps))
    t_pgd_init = time.perf_counter() - t0

    if is_sat:
        if print_progress:
            print('SAT found by initial PGD in %.1fms!' % (t_pgd_init * 1000))
        return 'sat', {'witness': witness, 'n_evals': 0,
                        'time': time.perf_counter() - t_wall}

    if print_progress:
        print('Initial PGD: UNSAT (%.1fms)' % (t_pgd_init * 1000))

    queue = [(x_lo.copy(), x_hi.copy(), set(comps), 0)]
    n_evals = 0
    n_verified = 0
    max_depth = 0
    volume_proven = 0.0
    depth_sum = 0
    depth_count = 0

    t_bab_start = time.perf_counter()

    while queue:
        if mode == 'dfs':
            x_l, x_h, remaining, depth = queue.pop(-1)
        else:
            x_l, x_h, remaining, depth = queue.pop(0)

        max_depth = max(max_depth, depth)

        if depth >= settings.bnb_max_depth:
            if print_progress:
                print('MAX DEPTH %d reached, giving up on this branch'
                      % settings.bnb_max_depth)
            continue

        spec_lbs, still_open, split_dim = evaluate_fn(x_l, x_h, remaining)
        n_evals += 1

        if not still_open:
            n_verified += 1
            volume_proven += 2.0 ** (-depth) if depth < 1024 else 0.0
            depth_sum += depth
            depth_count += 1
            if print_progress:
                avg_depth = depth_sum / depth_count
                elapsed = time.perf_counter() - t_bab_start
                eta = (elapsed * (1.0 - volume_proven) / volume_proven
                       if 0 < volume_proven < 1.0 else 0)
                print('UNSAT leaf d=%d | proven=%.1f%% | q=%d | evals=%d'
                      ' | elapsed=%.1fs | avg_d=%.1f | ETA=%s' % (
                          depth, volume_proven * 100, len(queue), n_evals,
                          elapsed, avg_depth, _fmt_eta(eta)))
            continue

        # PGD attack on subregion
        is_sat, witness, best_adv = pgd_fn(x_l, x_h, still_open)

        if is_sat:
            if print_progress:
                print('\nSAT! Counterexample at eval %d, depth %d'
                      % (n_evals, depth))
            return 'sat', {'witness': witness, 'n_evals': n_evals,
                            'time': time.perf_counter() - t_wall}

        # Split
        mid = (x_l[split_dim] + x_h[split_dim]) / 2
        xh1 = x_h.copy()
        xh1[split_dim] = mid
        xl2 = x_l.copy()
        xl2[split_dim] = mid

        adv_in_left = best_adv is not None and best_adv[split_dim] < mid

        if mode == 'dfs':
            if adv_in_left:
                queue.append((xl2, x_h, still_open, depth + 1))
                queue.append((x_l, xh1, still_open, depth + 1))
            else:
                queue.append((x_l, xh1, still_open, depth + 1))
                queue.append((xl2, x_h, still_open, depth + 1))
        else:
            if adv_in_left:
                queue.append((x_l, xh1, still_open, depth + 1))
                queue.append((xl2, x_h, still_open, depth + 1))
            else:
                queue.append((xl2, x_h, still_open, depth + 1))
                queue.append((x_l, xh1, still_open, depth + 1))

        if print_progress and (n_evals <= 5 or n_evals % 10 == 0):
            elapsed = time.perf_counter() - t_bab_start
            worst = min(spec_lbs[c] for c in still_open)
            print('split d=%d dim=%d | open=%d worst=%.4f | q=%d evals=%d'
                  ' | elapsed=%.1fs' % (depth, split_dim, len(still_open),
                                        worst, len(queue), n_evals, elapsed))

        if time.perf_counter() - t_bab_start > timeout:
            if print_progress:
                print('\nTIMEOUT %.0fs' % timeout)
            break

    t_total = time.perf_counter() - t_wall

    if print_progress:
        print('\nEvals: %d, Verified: %d, MaxDepth: %d, Queue: %d'
              % (n_evals, n_verified, max_depth, len(queue)))
        print('Volume proven: %.2f%%' % (volume_proven * 100))
        print('Total: %.1fms' % (t_total * 1000))

    if not queue and volume_proven >= 1.0 - 1e-9:
        return 'verified', {'n_evals': n_evals, 'time': t_total,
                             'volume_proven': volume_proven}
    return 'unknown', {'n_evals': n_evals, 'time': t_total,
                        'volume_proven': volume_proven, 'queue_remaining': len(queue)}


def zonotope_bnb_verify(graph, spec, settings=None):
    """BnB verification: forward zonotope + CROWN backward + input splitting.

    Args:
        graph: ComputeGraph loaded from ONNX
        spec: VNNSpec with input bounds and pairwise constraints
        settings: DotMap settings (or None for defaults)

    Returns:
        (result, details) where result is 'verified', 'unknown', or 'sat'
    """
    if settings is None:
        settings = default_settings()
    device, dtype = resolve_torch(settings)

    torch.set_num_threads(1)

    pw = spec.as_pairwise()
    assert pw is not None, (
        "BnB verification requires pairwise constraints (Y_comp >= Y_pred)")
    pred, comps = pw

    gpu_layers_list, fwd_data = graph.gpu_layers(device, dtype)
    nh = len(gpu_layers_list) - 1

    spec_ew = _build_spec_ew(gpu_layers_list, pred, comps, device, dtype)

    x_lo_np = spec.x_lo.astype(np.float32 if settings.bits == 32
                                else np.float64)
    x_hi_np = spec.x_hi.astype(np.float32 if settings.bits == 32
                                else np.float64)

    xl_g = torch.tensor(x_lo_np, dtype=dtype, device=device)
    xh_g = torch.tensor(x_hi_np, dtype=dtype, device=device)

    # Warmup
    _evaluate_region(xl_g, xh_g, set(comps), gpu_layers_list, spec_ew,
                     pred, nh, device, dtype)
    _pgd_attack(xl_g, xh_g, set(comps), pred, fwd_data, nh, settings)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    def evaluate_fn(x_l, x_h, remaining):
        xl_t = torch.tensor(x_l, dtype=dtype, device=device)
        xh_t = torch.tensor(x_h, dtype=dtype, device=device)
        return _evaluate_region(xl_t, xh_t, remaining, gpu_layers_list,
                                spec_ew, pred, nh, device, dtype)

    def pgd_fn(x_l, x_h, remaining):
        xl_t = torch.tensor(x_l, dtype=dtype, device=device)
        xh_t = torch.tensor(x_h, dtype=dtype, device=device)
        return _pgd_attack(xl_t, xh_t, remaining, pred, fwd_data, nh,
                           settings)

    return _run_bnb(evaluate_fn, pgd_fn, x_lo_np, x_hi_np, comps, settings)
