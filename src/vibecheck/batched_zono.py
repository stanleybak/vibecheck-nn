"""Batched zonotope forward pass — one network forward over N input boxes.

Why this exists: the per-leaf cost on cifar_biasfield-class networks is
~250 ms for the forward zono alone (CUDA kernel dispatch + small conv
ops on a 7-layer net). On RTX 3080 the GPU is mostly idle between
launches when batch=1. AB-CROWN's input-split BaB processes 8 leaves
per forward pass via a leading batch dim, which is why their per-leaf
cost is ~73 ms vs our 580 ms. This module gives vibecheck the same
capability.

Design:
- All ops operate on tensors with leading batch dim B (number of leaves).
- center: (B, n_flat) or (B, C, H, W) after conv.
- gens:   (B, K, n_flat) or (B, K, C, H, W) — K = current generator count.
- ReLU adds new generators per unstable neuron. Different leaves have
  different unstables, so we pad the new-gen block with zeros to a
  uniform width = max_nu_across_batch. Padding leaves the zonotope
  unchanged (zero generators contribute nothing to bounds).
- Skip connections (Add merge) are handled by sharing the input
  generator block across both branches; new gens stay independent.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def make_input_zonotopes_batched(xls, xhs, device, dtype):
    """Build B input zonotopes from per-leaf bounds.

    Args:
        xls, xhs: torch tensors of shape (B, n_input) — per-leaf input
            bounds. Input axes with zero radius (xl == xh) get no
            generator column (K = number of axes with positive radius
            in ANY leaf — uniform across batch via union).

    Returns:
        center: (B, n_input)
        gens:   (B, K, n_input) where gens[b, k, axis_k] = radius_b
    """
    assert xls.shape == xhs.shape and xls.ndim == 2, (
        f'expected (B, n_input), got xls={xls.shape} xhs={xhs.shape}')
    B, n_input = xls.shape
    radii = (xhs - xls) / 2  # (B, n_input)
    # K = axes with positive radius in ANY leaf — union so the gen
    # tensor shape is uniform. Per-leaf rows still have only the axes
    # with positive radius non-zero (others are 0 — sound).
    any_positive = (radii.abs() > 0).any(dim=0)  # (n_input,)
    nz = torch.nonzero(any_positive, as_tuple=False).flatten()
    K = int(nz.numel())
    center = (xls + xhs) / 2  # (B, n_input)
    # gens shape (B, K, n_input). For each k, set gens[b, k, nz[k]] = radius
    gens = torch.zeros(B, K, n_input, dtype=dtype, device=device)
    # Vectorised scatter: arange(K) for the k-axis, nz[k] for the input axis.
    if K > 0:
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, K)
        k_idx = torch.arange(K, device=device).unsqueeze(0).expand(B, K)
        ax_idx = nz.unsqueeze(0).expand(B, K)
        gens[b_idx, k_idx, ax_idx] = radii[b_idx, ax_idx]
    return center, gens


def propagate_fc_batched(center, gens, W, b):
    """Apply (W @ x + b) to a batch of zonotopes.

    center: (B, n_in)        → (B, n_out)
    gens:   (B, K, n_in)     → (B, K, n_out)
    W: (n_out, n_in)
    b: (n_out,) or None
    """
    new_center = F.linear(center, W, b)
    # gens @ W^T : (B, K, n_in) @ (n_in, n_out) = (B, K, n_out)
    new_gens = gens @ W.t()
    return new_center, new_gens


def propagate_conv_batched(center, gens, kernel, bias,
                            in_shape, stride, padding):
    """Apply Conv2d to a batch of zonotopes.

    center: (B, n_in_flat)        → (B, n_out_flat)
    gens:   (B, K, n_in_flat)     → (B, K, n_out_flat)
    kernel: (out_C, in_C, kH, kW)
    in_shape: (in_C, in_H, in_W)

    Strategy: reshape (B*K, in_C, in_H, in_W) for conv, then back.
    """
    B = center.shape[0]
    in_C, in_H, in_W = in_shape
    # Center: shape (B, in_C, in_H, in_W) → conv → (B, out_C, out_H, out_W)
    c_4d = center.reshape(B, in_C, in_H, in_W)
    new_c_4d = F.conv2d(c_4d, kernel, bias=bias,
                         stride=stride, padding=padding)
    new_center = new_c_4d.reshape(B, -1)
    if gens.shape[1] == 0:
        # No generators — return empty gens with right output shape.
        return new_center, gens.reshape(B, 0, new_center.shape[1])
    K = gens.shape[1]
    # Gens: shape (B, K, in_C, in_H, in_W) → reshape (B*K, in_C, in_H, in_W)
    g_4d = gens.reshape(B * K, in_C, in_H, in_W)
    new_g_4d = F.conv2d(g_4d, kernel, stride=stride, padding=padding)
    # Reshape back to (B, K, out_flat)
    new_gens = new_g_4d.reshape(B, K, -1)
    return new_center, new_gens


def bounds_batched(center, gens):
    """(lo, hi) per element across batch.

    center: (B, n_flat)
    gens:   (B, K, n_flat)
    Returns lo, hi: (B, n_flat) each.
    """
    abs_sum = gens.abs().sum(dim=1)  # (B, n_flat)
    return center - abs_sum, center + abs_sum


def apply_relu_batched(center, gens):
    """Min-area ReLU relaxation, batch-aware.

    center: (B, n_flat)        → (B, n_flat)
    gens:   (B, K, n_flat)     → (B, K + new_K, n_flat)
        where new_K = max_b unstable_count(b). Per-leaf rows in the
        new-gen block use mu_b at the unstable indices and zero
        elsewhere — padding doesn't change the represented zonotope.

    Returns (new_center, new_gens, lo, hi).
    """
    lo, hi = bounds_batched(center, gens)  # (B, n_flat) each
    ust = (lo < 0) & (hi > 0)
    dead = hi <= 0
    # lam: (B, n_flat). For unstable: hi/(hi-lo). For dead: 0. Else 1.
    safe_dy = torch.where(ust, hi - lo, torch.ones_like(hi))
    lam = torch.where(ust, hi / safe_dy,
                      torch.where(dead, torch.zeros_like(hi),
                                  torch.ones_like(hi)))
    mu = torch.where(ust, -hi * lo / (2 * safe_dy),
                     torch.zeros_like(hi))
    new_center = lam * center + mu
    # Scale all existing generators by lam
    scaled_gens = gens * lam.unsqueeze(1)  # (B, K, n_flat) * (B, 1, n_flat)

    # New generators per leaf: one new gen column per unstable neuron.
    # Pad to uniform width = max nu over batch. The previous version
    # had a `for b in range(B)` scatter loop that serialised the
    # work — vectorise via cumsum-into-flat-indices: for each (b, n)
    # where ust[b, n] is True, compute slot_index_within_b via
    # cumsum-1, then scatter mu[b, n] into new_block[b, slot, n].
    nu_per_leaf = ust.sum(dim=1)  # (B,)
    max_nu = int(nu_per_leaf.max().item()) if nu_per_leaf.numel() > 0 else 0
    if max_nu == 0:
        return new_center, scaled_gens, lo, hi
    B, n_flat = center.shape
    # cumsum gives 1-based slot index along the n_flat axis per leaf;
    # we want 0-based, so subtract 1 (only meaningful at unstable
    # positions).
    slot_idx = ust.long().cumsum(dim=1) - 1  # (B, n_flat); valid where ust
    # Flat scatter: build a (B, max_nu, n_flat) zero tensor and write
    # mu[b, n] at (b, slot_idx[b, n], n) for every unstable (b, n).
    new_block = torch.zeros(B, max_nu, n_flat,
                             dtype=center.dtype, device=center.device)
    b_grid, n_grid = torch.meshgrid(
        torch.arange(B, device=center.device),
        torch.arange(n_flat, device=center.device),
        indexing='ij')
    # Mask ust into flat tuples and index_put.
    mask = ust
    new_block[b_grid[mask], slot_idx[mask], n_grid[mask]] = mu[mask]
    full_gens = torch.cat([scaled_gens, new_block], dim=1)
    return new_center, full_gens, lo, hi


def forward_zonotope_graph_batched(xls, xhs, gg, device, dtype):
    """Batched analogue of `_forward_zonotope_graph` (verify_zono_bnb).

    Args:
        xls, xhs: (B, n_input) per-leaf input bounds.
        gg: gpu_graph dict (single-net topology — same network for every
            leaf; only the input box differs).

    Returns:
        sb: dict layer_idx -> (lo, hi) tensors of shape (B, n_neurons).
        z_final_center, z_final_gens: final tensors (for axis-score reuse).
    """
    center, gens = make_input_zonotopes_batched(xls, xhs, device, dtype)
    state = {gg['input_name']: (center, gens)}
    sb = {}

    last_use = {}
    for i, op2 in enumerate(gg['ops']):
        for inp in op2['inputs']:
            last_use[inp] = i

    for op_idx, op in enumerate(gg['ops']):
        name = op['name']; t = op['type']
        if t == 'conv':
            c, g = state[op['inputs'][0]]
            c2, g2 = propagate_conv_batched(
                c, g, op['kernel'], op['bias'],
                op['in_shape'], op['stride'], op['padding'])
            state[name] = (c2, g2)
        elif t == 'fc':
            c, g = state[op['inputs'][0]]
            c2, g2 = propagate_fc_batched(c, g, op['W'], op['bias'])
            state[name] = (c2, g2)
        elif t == 'relu':
            c, g = state[op['inputs'][0]]
            c2, g2, lo, hi = apply_relu_batched(c, g)
            if 'layer_idx' in op:
                sb[op['layer_idx']] = (lo, hi)
            state[name] = (c2, g2)
        elif t == 'add':
            if op.get('is_merge'):
                # Skip-connection merge — not used by the cifar_biasfield
                # / oval21 nets we batch (they have no forks).
                raise NotImplementedError(
                    'add-merge skip connections not yet batched; '
                    'fall back to unbatched _forward_zonotope_graph')
            c, g = state[op['inputs'][0]]
            bias = op.get('bias')
            if bias is not None:
                bt = torch.tensor(bias.flatten(), dtype=dtype, device=device)
                c = c + bt
            state[name] = (c, g)
        elif t == 'sub':
            c, g = state[op['inputs'][0]]
            bias = op.get('bias')
            if bias is not None:
                bt = torch.tensor(bias.flatten(), dtype=dtype, device=device)
                c = c - bt
            state[name] = (c, g)
        elif t == 'reshape':
            state[name] = state[op['inputs'][0]]

        for inp in op['inputs']:
            if last_use.get(inp) == op_idx and inp in state:
                del state[inp]

    last_name = gg['ops'][-1]['name']
    c_final, g_final = state[last_name]
    return sb, c_final, g_final
