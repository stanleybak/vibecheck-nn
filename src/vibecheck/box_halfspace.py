"""Closed-form LP for the box + 1-halfspace polytope.

Primal:
    min / max  d · e + c0
    s.t.       e ∈ [-1, 1]^n
               a · e ≤ β

Lagrangian dual (λ ≥ 0):
    g(λ) = c0 − λ β − Σ_i |d_i + λ a_i|

g is piecewise-linear concave in λ with breakpoints at λ*_i = −d_i / a_i
(only for a_i ≠ 0 and λ*_i > 0). The max is at λ = 0 if g'(0+) ≤ 0;
otherwise at the first breakpoint where the slope flips from positive
to non-positive. Runs in O(n log n) per solve (dominated by sorting
the breakpoints).

Equivalent to Clip-and-Verify's "tightest axis-aligned contraction" for
the special case of a single linear constraint, but here we compute the
tight LP value on an arbitrary linear objective `d · e + c0` directly
instead of tightening each e_i componentwise.
"""

import numpy as np
import torch


def lagrangian_min(d, c0, a, beta):
    """Return min_{e ∈ [-1,1]^n, a·e ≤ β} d·e + c0.

    All inputs cast to float64. O(n log n).
    """
    d = np.asarray(d, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    c0 = float(c0)
    beta = float(beta)

    # g(0) = box min over d
    g0 = c0 - float(np.sum(np.abs(d)))
    # Right-derivative at λ=0+: sign(d_i) when d_i ≠ 0, else sign(a_i)
    eff_sign = np.where(d != 0, np.sign(d), np.sign(a))
    gprime0_plus = -beta - float(np.sum(a * eff_sign))
    if gprime0_plus <= 0:
        return g0

    # Breakpoints where a coefficient's sign flips: λ*_i = -d_i / a_i.
    # Only λ*_i > 0 and d_i ≠ 0 matter (d_i = 0 cases absorbed into eff_sign).
    valid = (a != 0) & (d != 0)
    if valid.any():
        d_v = d[valid]; a_v = a[valid]
        lam_v = -d_v / a_v
        pos = lam_v > 0
        lam_pos = lam_v[pos]; a_pos = a_v[pos]
        decr = 2.0 * np.abs(a_pos)
        order = np.argsort(lam_pos)
        lam_sorted = lam_pos[order]
        decr_sorted = decr[order]
    else:
        lam_sorted = np.empty(0, dtype=np.float64)
        decr_sorted = np.empty(0, dtype=np.float64)

    lam_prev = 0.0
    slope = gprime0_plus
    g_val = g0
    for k in range(lam_sorted.size):
        lam_k = lam_sorted[k]
        g_val = g_val + slope * (lam_k - lam_prev)
        new_slope = slope - decr_sorted[k]
        if slope > 0 and new_slope <= 0:
            return g_val
        slope = new_slope
        lam_prev = lam_k
    # All breakpoints exhausted with slope still positive — halfspace is
    # infeasible wrt the box (min a·e > β). Primal infeasible; dual
    # unbounded. Return +inf to signal vacuous (no feasible point).
    return float('inf')


def lagrangian_max(d, c0, a, beta):
    """Return max_{e ∈ [-1,1]^n, a·e ≤ β} d·e + c0."""
    d = np.asarray(d, dtype=np.float64)
    c0 = float(c0)
    return -lagrangian_min(-d, -c0, a, beta)


def tighten_layer(c_L, G_L, lo, hi, a, beta, n_gens=None):
    """Closed-form lb/ub tightening for every unstable neuron at a layer.

    c_L: (n_out,) pre-ReLU centers (numpy float64 or castable).
    G_L: (n_out, n_gens_L) pre-ReLU generator rows.
    lo, hi: (n_out,) current pre-ReLU bounds.
    a, beta: the single halfspace `a · e ≤ β` in full-generator-space.
    n_gens: if provided and greater than G_L.shape[1], rows are padded
        with zeros to match a's length (newer generators appended after
        this layer don't affect z_L[j]).

    Returns (new_lo, new_hi) with bounds only tightened (never loosened)
    for neurons with (lo < 0) & (hi > 0). Stable neurons untouched.
    """
    c_L = np.asarray(c_L, dtype=np.float64)
    G_L = np.asarray(G_L, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64).copy()
    hi = np.asarray(hi, dtype=np.float64).copy()
    a = np.asarray(a, dtype=np.float64)
    beta = float(beta)

    if n_gens is not None and G_L.shape[1] < n_gens:
        G_L = np.hstack([
            G_L,
            np.zeros((G_L.shape[0], n_gens - G_L.shape[1]), dtype=np.float64),
        ])

    un = np.where((lo < 0) & (hi > 0))[0]
    for j in un:
        j = int(j)
        row = G_L[j]
        cij = float(c_L[j])
        new_lo = lagrangian_min(row, cij, a, beta)
        new_hi = lagrangian_max(row, cij, a, beta)
        if np.isfinite(new_lo):
            lo[j] = max(lo[j], new_lo)
        if np.isfinite(new_hi):
            hi[j] = min(hi[j], new_hi)
    return lo, hi


def tighten_all_layers(pre_relu_gpu, c_out, G_out, w_q, b_q, bbr, layers,
                        device, dtype):
    """Tighten unstable pre-ReLU bounds at each specified layer using the
    box + spec-halfspace polytope.

    pre_relu_gpu: dict layer_idx -> (c_gpu, G_gpu) torch tensors (any dtype).
    c_out, G_out: final output zono (numpy float64).
    w_q, b_q: the target query's spec weight and bias (numpy, float).
    bbr: dict layer_idx -> (lo, hi) numpy arrays.
    layers: iterable of layer_idx to tighten.
    device, dtype: torch device/dtype for the GPU pre_relu tensors.

    Returns (result, stats) where:
      result: {layer_idx: (new_lo, new_hi)} — intersected with original bbr.
      stats:  {'per_layer': {L: {un, flipped, shrink, t_xfer, t_LP}},
               'n_flipped': int, 'total_shrink': float}.
    """
    n_gens = int(G_out.shape[1])
    a = G_out.T @ w_q
    beta = float(-(w_q @ c_out) - b_q)

    import time as _time

    result = {}
    stats = {'per_layer': {}, 'n_flipped': 0, 'total_shrink': 0.0}

    for L in layers:
        c_L_gpu, G_L_gpu = pre_relu_gpu[L]
        lo = np.asarray(bbr[L][0], dtype=np.float64).copy()
        hi = np.asarray(bbr[L][1], dtype=np.float64).copy()
        un = np.where((lo < 0) & (hi > 0))[0]
        if un.size == 0:
            result[L] = (lo, hi)
            stats['per_layer'][L] = {'un': 0, 'flipped': 0, 'shrink': 0.0,
                                      't_xfer': 0.0, 't_LP': 0.0}
            continue

        t0 = _time.perf_counter()
        # Two supported snapshot shapes (decided by `_forward_keep_pre_gpu`):
        # (A) Full-G: c_L shape (n_flat,), G_L shape (n_flat, K) — legacy.
        # (B) Slim-G: c_L shape (n_unstable,), G_L shape (n_unstable, K),
        #     already aligned with `un` (the bbr-computed unstable set). Any
        #     subsequent tightening pass still sees the same `un` set since
        #     this halfspace LP only shrinks bounds, never creates new
        #     unstables. If un.size doesn't match the slim size we fall back
        #     to treating it as full-G.
        slim = (c_L_gpu.shape[0] == un.size
                and (G_L_gpu.ndim == 2 and G_L_gpu.shape[0] == un.size))
        if slim:
            G_un = G_L_gpu.detach().cpu().numpy().astype(np.float64)
            c_un = c_L_gpu.detach().cpu().numpy().astype(np.float64)
        else:
            un_t = torch.as_tensor(un, device=device, dtype=torch.long)
            G_un = G_L_gpu[un_t].detach().cpu().numpy().astype(np.float64)
            c_un = c_L_gpu[un_t].detach().cpu().numpy().astype(np.float64)
        if G_un.shape[1] < n_gens:
            G_un = np.hstack([
                G_un,
                np.zeros((G_un.shape[0], n_gens - G_un.shape[1]),
                         dtype=np.float64),
            ])
        t_xfer = _time.perf_counter() - t0

        t0 = _time.perf_counter()
        flipped = 0
        shrink = 0.0
        for k in range(un.size):
            j = int(un[k])
            row = G_un[k]
            cij = float(c_un[k])
            old_w = hi[j] - lo[j]
            new_lo = lagrangian_min(row, cij, a, beta)
            new_hi = lagrangian_max(row, cij, a, beta)
            if np.isfinite(new_lo):
                lo[j] = max(lo[j], new_lo)
            if np.isfinite(new_hi):
                hi[j] = min(hi[j], new_hi)
            shrink += (old_w - (hi[j] - lo[j]))
            if lo[j] >= -1e-9 or hi[j] <= 1e-9:
                flipped += 1
        t_LP = _time.perf_counter() - t0

        result[L] = (lo, hi)
        stats['per_layer'][L] = {'un': int(un.size), 'flipped': flipped,
                                  'shrink': shrink, 't_xfer': t_xfer,
                                  't_LP': t_LP}
        stats['n_flipped'] += flipped
        stats['total_shrink'] += shrink

    return result, stats


def tighten_all_layers_with_halfspace(pre_relu_gpu, a, beta, n_gens, bbr,
                                       layers, device, dtype):
    """Same as `tighten_all_layers` but takes a raw halfspace `a · e ≤ β`
    instead of building it from a spec direction. Used for BaB-style splits
    on a single neuron's pre-activation: child A halfspace is `−g_N · e ≤ c_N`
    and child B is `g_N · e ≤ −c_N`.

    a: numpy float64 array, shape (n_gens,) — halfspace normal.
    beta: float — halfspace offset.
    n_gens: total generator count (for padding G_un to a's dim).
    """
    import time as _time

    a = np.asarray(a, dtype=np.float64)
    beta = float(beta)
    if a.shape[0] < n_gens:
        # pad halfspace with zeros for trailing gens (don't affect a·e ≤ β)
        a = np.concatenate([a, np.zeros(n_gens - a.shape[0],
                                          dtype=np.float64)])

    result = {}
    stats = {'per_layer': {}, 'n_flipped': 0, 'total_shrink': 0.0}

    for L in layers:
        c_L_gpu, G_L_gpu = pre_relu_gpu[L]
        lo = np.asarray(bbr[L][0], dtype=np.float64).copy()
        hi = np.asarray(bbr[L][1], dtype=np.float64).copy()
        un = np.where((lo < 0) & (hi > 0))[0]
        if un.size == 0:
            result[L] = (lo, hi)
            stats['per_layer'][L] = {'un': 0, 'flipped': 0, 'shrink': 0.0,
                                      't_xfer': 0.0, 't_LP': 0.0}
            continue

        t0 = _time.perf_counter()
        slim = (c_L_gpu.shape[0] == un.size
                and (G_L_gpu.ndim == 2 and G_L_gpu.shape[0] == un.size))
        if slim:
            G_un = G_L_gpu.detach().cpu().numpy().astype(np.float64)
            c_un = c_L_gpu.detach().cpu().numpy().astype(np.float64)
        else:
            un_t = torch.as_tensor(un, device=device, dtype=torch.long)
            G_un = G_L_gpu[un_t].detach().cpu().numpy().astype(np.float64)
            c_un = c_L_gpu[un_t].detach().cpu().numpy().astype(np.float64)
        if G_un.shape[1] < n_gens:
            G_un = np.hstack([
                G_un,
                np.zeros((G_un.shape[0], n_gens - G_un.shape[1]),
                         dtype=np.float64),
            ])
        t_xfer = _time.perf_counter() - t0

        t0 = _time.perf_counter()
        flipped = 0
        shrink = 0.0
        for k in range(un.size):
            j = int(un[k])
            row = G_un[k]
            cij = float(c_un[k])
            old_w = hi[j] - lo[j]
            new_lo = lagrangian_min(row, cij, a, beta)
            new_hi = lagrangian_max(row, cij, a, beta)
            if np.isfinite(new_lo):
                lo[j] = max(lo[j], new_lo)
            if np.isfinite(new_hi):
                hi[j] = min(hi[j], new_hi)
            shrink += (old_w - (hi[j] - lo[j]))
            if lo[j] >= -1e-9 or hi[j] <= 1e-9:
                flipped += 1
        t_LP = _time.perf_counter() - t0

        result[L] = (lo, hi)
        stats['per_layer'][L] = {'un': int(un.size), 'flipped': flipped,
                                  'shrink': shrink, 't_xfer': t_xfer,
                                  't_LP': t_LP}
        stats['n_flipped'] += flipped
        stats['total_shrink'] += shrink

    return result, stats
