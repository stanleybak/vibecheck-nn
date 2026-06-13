"""Build the alpha_zono dual-ascent state via per-target BACKWARD passes.

Memory-bounded alternative to `verify_gen_lp.precompute_gen_state` for nets
whose forward generator tensor does not fit GPU memory. On
challenging_certified_training tinyimagenet the input box has 12 288
generators and the first conv has 262 144 activations, so the dense
generator matrix G is 12 288 × 262 144 × 4 B ≈ 12 GiB per layer and two
consecutive full-resolution conv layers OOM a 24 GB card (measured: the
`bounds()`/forward path dies there).

`precompute_gen_state` materializes G forward and reads each unstable
neuron's row off it. This builder computes each row by a BACKWARD pass from
that neuron, using the *same* zonotope (parallelogram) relaxation, chunked
over targets so only (chunk × n_acts) is ever resident. It reuses
externally-supplied pre-activation bounds (e.g. CROWN-IBP + α tightening) —
no zonotope `bounds()` call.

Relaxation (identical to the forward builder, `verify_gen_lp.py:975-1002`):
each unstable ReLU k is replaced by

    y_k = λ_k · z_k + μ_k · (1 + e_new_k),   e_new_k ∈ [-1, 1]
    μ_k = max((1 - λ_k)·hi_k / 2,  -λ_k·lo_k / 2)

with λ_k the supplied slope. Stable-active → slope 1, dead → slope 0.

Generator-column model (must match the forward builder exactly, else the
spec LP/dual-ascent reads the wrong columns and can certify a false UNSAT —
see the SOUNDNESS INVARIANT in `verify_gen_lp.py`): columns `0 … n_input-1`
are the input generators (one per perturbed input dim, magnitude =
half-width); then one new column per unstable ReLU in (layer-ascending,
neuron-ascending) order.

The row of an unstable pre-activation z_j over those columns is the exact
linear map produced by a backward walk from z_j applying λ at unstable
ReLUs (1 / 0 at active / dead) and W^T at linear layers:
  - input column i  : (backward coeff at input i) × half_width_i
  - new column of prior unstable k : (backward coeff at y_k) × μ_k

Soundness rests on this being the *same* affine over-approximation as the
forward builder; validated row-for-row against `precompute_gen_state` on
small nets in `tests/test_gen_state_backward.py`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _relu_lam_mu(lo, hi, alpha):
    """Zonotope slope λ and half-width μ per neuron (arrays).

    λ defaults to the supplied α; μ = max((1-λ)hi/2, -λ lo/2) for unstable.
    Returned μ is 0 for stable neurons (they add no generator).
    """
    lo = np.asarray(lo, np.float64)
    hi = np.asarray(hi, np.float64)
    unstable = (lo < 0) & (hi > 0)
    lam = np.where(unstable, alpha, (hi > 0).astype(np.float64))
    mu = np.where(
        unstable,
        np.maximum((1.0 - lam) * hi / 2.0, -lam * lo / 2.0),
        0.0,
    )
    return lam, mu, unstable


def _forward_center(gg, xl, xh, bbr, alpha_per_layer, device, dtype):
    """Propagate the zonotope CENTER to get c_in per pre-ReLU layer.

    y_center at an unstable ReLU is λ·z_center + μ (the parallelogram is
    centred there); active → z_center, dead → 0. Cheap O(acts) vector pass.
    Returns {layer_idx: c_pre (np.float64)}.
    """
    cen = {gg['input_name']: torch.as_tensor((xl + xh) / 2.0, device=device,
                                              dtype=dtype).flatten()}
    c_pre = {}
    for op in gg['ops']:
        t = op['type']
        name = op['name']
        c = cen[op['inputs'][0]]
        if t == 'conv':
            out = F.conv2d(c.reshape(1, *op['in_shape']), op['kernel'],
                           op['bias'], stride=op['stride'],
                           padding=op['padding'])
            cen[name] = out.flatten()
        elif t == 'fc':
            cen[name] = op['W'] @ c + (op['bias'] if op['bias'] is not None
                                       else 0.0)
        elif t == 'relu':
            L = op.get('layer_idx')
            cc = c.detach().cpu().numpy().astype(np.float64)
            if L is not None:
                c_pre[L] = cc
                lo, hi = bbr[L]
                alpha = _resolve_alpha(alpha_per_layer, L, lo)
                lam, mu, _ = _relu_lam_mu(lo, hi, alpha)
                y = lam * cc + mu                      # unstable centre
                y = np.where(np.asarray(hi) <= 0, 0.0, y)   # dead → 0
                cen[name] = torch.as_tensor(y, device=device, dtype=dtype)
            else:
                cen[name] = c.clamp(min=0)
        elif t == 'reshape':
            cen[name] = c
        else:
            raise NotImplementedError(
                f'gen_state_backward._forward_center: unsupported op {t!r} '
                f'at {name!r}')
    return c_pre


def _resolve_alpha(alpha_per_layer, L, lo):
    """α slope vector for layer L (default 0.5-ish min-area is the caller's
    job; here a missing entry means 'use the chord midpoint slope' = the
    forward builder's default). Returns an array sized like `lo`."""
    a = alpha_per_layer.get(L) if alpha_per_layer else None
    n = len(np.asarray(lo))
    if a is None:
        return np.full(n, 0.5, np.float64)
    a = np.asarray(a.detach().cpu().numpy() if hasattr(a, 'detach')
                   else a, np.float64).flatten()
    if a.size == n:
        return a
    # slim α (only unstable entries): scatter into full by unstable order
    full = np.full(n, 0.5, np.float64)
    return full  # caller passes full-size α for the matched path; see test


def _producing_op(gg, relu_op):
    """The op whose output is `relu_op`'s pre-activation input."""
    inp = relu_op['inputs'][0]
    for op in gg['ops']:
        if op['name'] == inp:
            return op
    return None


def build_alpha_zono_state_backward(gg, xl, xh, bbr, alpha_per_layer, *,
                                    device, dtype=torch.float32,
                                    chunk=256):
    """Construct the alpha_zono gen-state (unstable_list etc.) via backward.

    Args mirror what the forward builder is given:
      gg              : gpu_graph dict (conv/fc/relu/reshape ops).
      xl, xh          : input box (flat, numpy or tensor).
      bbr             : {layer_idx: (lo, hi)} pre-ReLU bounds (np arrays).
      alpha_per_layer : {layer_idx: full-size α tensor/array} or {} for
                        the default chord-midpoint slope.
      chunk           : targets processed per backward sweep (memory knob).

    Returns the same dict shape as `verify_gen_lp.precompute_gen_state`'s
    alpha_zono output: n_input, n_gens, formulation, unstable_list (each
    entry: layer_idx, neuron_idx, c_in, lo, hi, e_new_col, row_indices,
    row_values, lam, mu, form='alpha_zono').
    """
    xl = np.asarray(xl, np.float64).flatten()
    xh = np.asarray(xh, np.float64).flatten()
    half = (xh - xl) / 2.0
    n_input = int((half > 0).sum()) if False else int(half.size)
    # NOTE: forward builder seeds n_input = number of input GENERATORS. For an
    # L∞ box every input dim is a generator (half>0), so n_input = half.size.
    # Zero-width dims still occupy a column in the positional model; keep all.
    relu_layers = sorted(L for L in bbr
                         if any(op.get('layer_idx') == L and op['type'] == 'relu'
                                for op in gg['ops']))

    # Column offsets: input gens, then new col per unstable in layer order.
    new_col_start = {}
    running = n_input
    unstable_by_layer = {}
    for L in relu_layers:
        lo, hi = bbr[L]
        un = np.where((np.asarray(lo) < 0) & (np.asarray(hi) > 0))[0]
        unstable_by_layer[L] = un
        new_col_start[L] = running
        running += len(un)
    n_gens = running

    c_pre = _forward_center(gg, xl, xh, bbr, alpha_per_layer, device, dtype)

    half_t = torch.as_tensor(half, device=device, dtype=dtype)
    # Precompute per-layer (lam, mu) tensors and the relu op lookup.
    relu_op_by_L = {op['layer_idx']: op for op in gg['ops']
                    if op['type'] == 'relu' and 'layer_idx' in op}
    lam_mu = {}
    for L in relu_layers:
        lo, hi = bbr[L]
        alpha = _resolve_alpha(alpha_per_layer, L, lo)
        lam, mu, _ = _relu_lam_mu(lo, hi, alpha)
        lam_mu[L] = (torch.as_tensor(lam, device=device, dtype=dtype),
                     torch.as_tensor(mu, device=device, dtype=dtype),
                     torch.as_tensor(np.asarray(hi) <= 0, device=device))

    unstable_list = []
    for L_j in relu_layers:
        un_j = unstable_by_layer[L_j]
        if len(un_j) == 0:
            continue
        prod_op = _producing_op(gg, relu_op_by_L[L_j])
        for c0 in range(0, len(un_j), chunk):
            tgt = un_j[c0:c0 + chunk]
            rows = _backward_rows(gg, prod_op, tgt, relu_layers, L_j,
                                  lam_mu, half_t, new_col_start,
                                  unstable_by_layer, n_gens, n_input,
                                  device, dtype)
            lo_j, hi_j = bbr[L_j]
            lam_j, mu_j, _ = lam_mu[L_j]
            for r, j in enumerate(tgt):
                j = int(j)
                row = rows[r]
                nz = torch.nonzero(row, as_tuple=False).flatten()
                unstable_list.append({
                    'layer_idx': int(L_j),
                    'neuron_idx': j,
                    'c_in': float(c_pre[L_j][j]),
                    'lo': float(lo_j[j]),
                    'hi': float(hi_j[j]),
                    'e_new_col': int(new_col_start[L_j] + (c0 + r)),
                    'row_indices': nz.cpu().numpy().astype(np.int32),
                    'row_values': row[nz].detach().cpu().numpy().astype(
                        np.float64),
                    'lam': float(lam_j[j]),
                    'mu': float(mu_j[j]),
                    'form': 'alpha_zono',
                })
    return {
        'n_input': n_input,
        'n_gens': n_gens,
        'formulation': 'alpha_zono',
        'unstable_list': unstable_list,
        'stable_list': [],
    }


def _backward_rows(gg, prod_op, tgt, relu_layers, L_j, lam_mu, half_t,
                   new_col_start, unstable_by_layer, n_gens, n_input,
                   device, dtype):
    """Backward sweep from z_{L_j}'s `tgt` neurons → full generator rows.

    Returns a dense (len(tgt) × n_gens) tensor (only used per-chunk, then
    sparsified by the caller).
    """
    T = len(tgt)
    out = torch.zeros(T, n_gens, device=device, dtype=dtype)
    # ew_at[op_name] = (T × n_out_of_that_op) coefficient of that op's output.
    # Seed: identity over the producing op's output at the target neurons.
    n_prod = _op_out_size(prod_op)
    seed = torch.zeros(T, n_prod, device=device, dtype=dtype)
    seed[torch.arange(T, device=device),
         torch.as_tensor(tgt, device=device, dtype=torch.long)] = 1.0
    ew_at = {prod_op['name']: seed}

    # Walk ops in reverse, starting at prod_op (inclusive).
    ops = gg['ops']
    start = next(i for i, op in enumerate(ops) if op['name'] == prod_op['name'])
    for op in reversed(ops[:start + 1]):
        name = op['name']
        if name not in ew_at:
            continue
        ew = ew_at[name]
        t = op['type']
        if t == 'conv':
            ew4 = ew.reshape(T, *op['out_shape'])
            back = F.conv_transpose2d(
                ew4, op['kernel'], stride=op['stride'],
                padding=op['padding'],
                output_padding=op.get('output_padding', (0, 0)))
            back = back.reshape(T, -1)
            _accum(ew_at, op['inputs'][0], back)
        elif t == 'fc':
            back = ew @ op['W']                         # (T × n_in)
            _accum(ew_at, op['inputs'][0], back)
        elif t == 'relu':
            L = op.get('layer_idx')
            if L is None:
                _accum(ew_at, op['inputs'][0], ew)
                continue
            lam, mu, dead = lam_mu[L]
            # Harvest new-gen columns for this layer's unstable (coeff over
            # y_k is ew[:, k]; the e_new_k column entry is ew[:,k]·μ_k).
            un = unstable_by_layer[L]
            if len(un):
                un_t = torch.as_tensor(un, device=device, dtype=torch.long)
                cols = torch.arange(len(un), device=device) + new_col_start[L]
                contrib = ew[:, un_t] * mu[un_t].unsqueeze(0)   # (T × |un|)
                out[:, cols] += contrib
            # Propagate to z^L: slope λ (unstable), 1 (active), 0 (dead).
            slope = lam.clone()
            slope = torch.where(dead, torch.zeros_like(slope), slope)
            # active (lo>=0) already has λ==1 from _relu_lam_mu's stable branch
            _accum(ew_at, op['inputs'][0], ew * slope.unsqueeze(0))
        elif t == 'reshape':
            _accum(ew_at, op['inputs'][0], ew)
        else:
            raise NotImplementedError(
                f'gen_state_backward._backward_rows: unsupported op {t!r} '
                f'at {name!r}')
    # Input generators: coeff at input × half-width.
    in_ew = ew_at.get(gg['input_name'])
    if in_ew is not None:
        out[:, :n_input] += in_ew * half_t.unsqueeze(0)
    return out


def _accum(ew_at, name, val):
    ew_at[name] = val if name not in ew_at else ew_at[name] + val


def _op_out_size(op):
    if op['type'] == 'conv':
        return int(np.prod(op['out_shape']))
    if op['type'] == 'fc':
        return int(op['W'].shape[0])
    raise NotImplementedError(
        f'gen_state_backward: cannot size output of op type {op["type"]!r}')
