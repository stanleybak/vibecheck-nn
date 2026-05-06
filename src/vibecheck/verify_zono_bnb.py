"""Branch-and-Bound verification with zonotope forward + CROWN backward."""

import time
import numpy as np
import torch
import torch.nn.functional as F

from .settings import default_settings, resolve_torch
from .zonotope import TorchZonotope


def _make_slopes(lo, hi):
    """Compute CROWN adaptive slopes for ReLU relaxation.

    Returns (lo_s, up_s, up_t, active_mask, dead_mask, unstable_mask).
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
def _forward_zonotope_graph(xl, xh, gg, device, dtype):
    """Graph-aware zonotope forward pass (supports skip connections).

    Args:
        xl, xh: input bounds (flat torch tensors)
        gg: gpu_graph dict from ComputeGraph.gpu_graph()

    Returns:
        sb: dict mapping layer_idx -> (lo, hi) bounds at each ReLU
        z_final: final TorchZonotope (after last op, before output)
    """
    z_init = TorchZonotope.from_input_bounds(xl, xh, device, dtype)
    zono_state = {gg['input_name']: z_init}
    gen_count = {gg['input_name']: z_init.n_gens}
    forks = gg['fork_points']
    sb = {}

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
            lo, hi = z.apply_relu()
            if 'layer_idx' in op:
                sb[op['layer_idx']] = (lo.clone(), hi.clone())
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
                          return_ew=False):
    """Graph-aware spec backward pass for networks with skip connections.

    spec_ew maps query_id -> (w, bias) where w is in OUTPUT space.
    Propagates backward through ALL ops including the final linear layer.

    Returns (spec_lbs, still_open) or (spec_lbs, still_open, ew_at_relu)
    if return_ew=True. ew_at_relu maps qid -> {layer_idx -> ew_numpy}.
    """
    ops = gg['ops']

    spec_lbs = {}
    all_ew_at_relu = {} if return_ew else None
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

        # At input: interval bound
        input_name = gg['input_name']
        ew_inp = ew_at.get(input_name, torch.zeros_like(xl))
        spec_lbs[qid] = acc + float(
            ew_inp.clamp(min=0) @ xl + ew_inp.clamp(max=0) @ xh)
        if return_ew:
            all_ew_at_relu[qid] = qid_ew_at_relu

    still_open = {c for c in remaining_specs if spec_lbs[c] <= 0}
    if return_ew:
        return spec_lbs, still_open, all_ew_at_relu
    return spec_lbs, still_open


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
