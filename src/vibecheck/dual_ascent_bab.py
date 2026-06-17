"""GPU-batched Lagrangian dual-ascent BaB verifier.

Drop-in alternative to Phase 8 MILP racing (`verify_gen_lp.parallel_query_racing`).
Solves the per-query α-zonotope LP via custom dual ascent (no Gurobi), batches
all open BaB nodes on GPU per-layer, and uses the substitution-form encoding
that eliminates `e_new_k` on each branch (matches Gurobi's y=0/y=z LP exactly).

Per node ~80–250 µs on RTX 3080 with K=1 hard iter cap. Verifies prop_4260 in
0.29 s (vs Gurobi LP-only BaB: 105 s, vs production MILP racing: 70 s).

Key design choices:
- Substitution: y_k = 0 (OFF) or y_k = z_k (ON) is folded into per-node
  `d_path` and `c0_path` corrections. The e_new_k column of A is zeroed.
  Each branch adds 2 halfspaces: `z ≤ 0`/`z ≥ -2μ/λ` for OFF;
  `z ≥ 0`/`z ≤ 2μ/(1-λ)` for ON.
- Hard iter cap K (default 1): no patience-based stalling. If best_g > 0 in
  K iters → certify; else find primal witness via greedy projection; else
  split. K=1 is the sweet spot empirically (extra splits are cheap).
- Dual ascent direction: subgradient s_proj = A·x*(rc) - b masked at λ=0;
  exact line search via sort + cumulative-slope sweep.
- Primal repair (greedy single-halfspace projection clipped to box) catches
  primal_unsafe witnesses that would otherwise become safety_cap.

Soundness: for any λ ≥ 0, g(λ) = c0 + min_box(rc·x) - λ·b ≤ LP_min by weak
duality. We only certify when computed best_g > 0 strictly. Sanity-checked
against Gurobi LP for 250 nodes on prop_4260 — 100% decision match, 0
unsoundness.

Limitations:
- BFS frontier can blow up if too few nodes certify early (caps at 16384).
- Uses a single spec direction (qw, qb); for multi-disjunct specs the caller
  loops over disjuncts.
"""
from __future__ import annotations
import time
from typing import Optional, Callable
import numpy as np
import torch

from .nonlinear_split_planes import op_planes, split_point
from .nonlinear_split_dual import (backward_sensitivity,
                                   band_change_correction, split_halfspace)

# ---------------------------------------------------------------------------
# Defaults — tuned on prop_4260 (TinyImageNet ResNet medium). K=1 wins on
# wall, K=2 matches Gurobi node count, K=5 gives tightest bounds. Increase
# REPAIR_STEPS to find primal witnesses faster on harder instances.
# ---------------------------------------------------------------------------
_DEFAULT_K = 1
_DEFAULT_REPAIR_STEPS = 5
# Chunk a layer's frontier into sub-batches sized so A_batch + d_path fits.
# Conservative target: 1 GB per chunk for A + d (bytes_per_elem * B * m * n
# + B * n). Auto-adjust below.
_DEFAULT_CHUNK_BYTES_BUDGET = 1.5 * 1024 * 1024 * 1024  # 1.5 GB
_TOL = 1e-9
_FEAS_TOL = 1e-5
_ROWS_PER_SPLIT = 2  # OFF: z≤0, z≥-2μ/λ.  ON: z≥0, z≤2μ/(1-λ).


def _precompute_state_geometry(state, device, dtype):
    """Precompute QUERY-INDEPENDENT pieces of the substitution-form LP.

    Most of `_build_substitution_caches` only depends on the *state* (box
    bounds, per-unstable `lam`/`mu`/`c_in`/`row_*`/`e_new_col`), not on the
    spec direction `(qw, qb)`. On CIFAR100 we call dual-ascent BaB once per
    disjunct query (up to 99 queries per spec), so the original code was
    repeating ~50–200 s of geometric work per case. Cache it on the state
    dict and gather per query.

    Cached fields (all keyed by global unstable index = position in
    `state['unstable_list']`):
      - n_gens, n_input, device, dtype                  (sanity)
      - e_lb, e_hi, width                               box bounds (torch)
      - lam_pu, mu_pu, c_in_pu                          [n_u] numpy float64
      - e_new_col_pu                                    [n_u] numpy int64
      - a_pu_np                                         [n_u, n_gens] np float64,
                                                          e_new_col zeroed
      - a_pu_t                                          [n_u, n_gens] torch
      - z_lo_pu, z_hi_pu                                [n_u] np float64
      - obj_G_out_np, obj_c_out_np                      cached float64
    """
    n_gens = int(state['n_gens'])
    n_input = int(state['n_input'])
    unstable_list = state['unstable_list']
    n_u = len(unstable_list)

    # Box bounds
    e_lb_np = np.zeros(n_gens, dtype=np.float64)
    e_hi_np = np.zeros(n_gens, dtype=np.float64)
    e_lb_np[:n_input] = -1.0
    e_hi_np[:n_input] = 1.0
    for u in unstable_list:
        c = int(u['e_new_col'])
        e_lb_np[c] = -1.0
        e_hi_np[c] = 1.0
    width_np = e_hi_np - e_lb_np

    # Per-unstable scalar arrays
    lam_pu = np.empty(n_u, dtype=np.float64)
    mu_pu = np.empty(n_u, dtype=np.float64)
    c_in_pu = np.empty(n_u, dtype=np.float64)
    e_new_col_pu = np.empty(n_u, dtype=np.int64)
    a_pu_np = np.zeros((n_u, n_gens), dtype=np.float64)
    for j, u in enumerate(unstable_list):
        lam_pu[j] = float(u['lam'])
        mu_pu[j] = float(u['mu'])
        c_in_pu[j] = float(u['c_in'])
        e_new_col_pu[j] = int(u['e_new_col'])
        row_idx = np.asarray(u['row_indices'], dtype=np.int64)
        row_val = np.asarray(u['row_values'], dtype=np.float64)
        # Safety: e_new_col must not collide with a row coefficient, or our
        # optimization (zeroing a_pu[e_new_col] before forming d_corr) would
        # drop the term that the legacy code's `d_corr[row_idx] = ...; d_corr[e_new_col] -= ...`
        # pair set then decremented. Assert disjointness.
        assert e_new_col_pu[j] not in row_idx, (
            f'unstable {(u["layer_idx"], u["neuron_idx"])}: e_new_col '
            f'{e_new_col_pu[j]} overlaps with row_indices — geometric cache '
            f'would be incorrect.')
        a_pu_np[j, row_idx] = row_val
        a_pu_np[j, e_new_col_pu[j]] = 0.0  # explicit; original sets h1[e_new_col]=0

    # z_lo, z_hi (scalar per unstable)
    safe_mu = mu_pu > 1e-12
    safe_lam = lam_pu > 1e-12
    safe_1ml = (1.0 - lam_pu) > 1e-12
    z_lo_pu = np.where(safe_mu & safe_lam,
                        2.0 * mu_pu / np.maximum(lam_pu, 1e-30) + c_in_pu,
                        1e9)
    z_hi_pu = np.where(safe_mu & safe_1ml,
                        2.0 * mu_pu / np.maximum(1.0 - lam_pu, 1e-30) - c_in_pu,
                        1e9)

    geo = {
        'n_gens': n_gens, 'n_input': n_input,
        'device': device, 'dtype': dtype,
        'e_lb_t': torch.tensor(e_lb_np, device=device, dtype=dtype),
        'e_hi_t': torch.tensor(e_hi_np, device=device, dtype=dtype),
        'width_t': torch.tensor(width_np, device=device, dtype=dtype),
        'lam_pu': lam_pu, 'mu_pu': mu_pu, 'c_in_pu': c_in_pu,
        'safe_mu': safe_mu,
        'e_new_col_pu': e_new_col_pu,
        'a_pu_np': a_pu_np,
        'a_pu_t': torch.tensor(a_pu_np, device=device, dtype=dtype),
        'z_lo_pu': z_lo_pu, 'z_hi_pu': z_hi_pu,
        'obj_G_out_np': state['obj_G_out_csr'].toarray().astype(np.float64),
        'obj_c_out_np': np.asarray(state['obj_c_out'], dtype=np.float64),
        'unstable_idx_by_key': {(u['layer_idx'], u['neuron_idx']): j
                                  for j, u in enumerate(unstable_list)},
    }
    return geo


def _compute_query_caches(geo, scored_keys, qw, qb, device, dtype):
    """Per-query: gather geometry and apply qw-dependent corrections.

    Bit-equivalent to the legacy `_build_substitution_caches` (within
    float32 conversion). All heavy ops happen on GPU using the cached
    `a_pu_t` to avoid a CPU→GPU transfer of the per-query hs_A.
    """
    n_gens = geo['n_gens']
    # Map scored_keys -> global unstable indices.
    idx_by_key = geo['unstable_idx_by_key']
    scored_idx = np.fromiter(
        (idx_by_key[k] for k in scored_keys),
        dtype=np.int64, count=len(scored_keys))
    n_splits = scored_idx.shape[0]

    # Per-query objective on host (cheap dense matvec).
    qw_np = np.asarray(qw, dtype=np.float64)
    d_np = qw_np @ geo['obj_G_out_np']
    c0 = float(qw_np @ geo['obj_c_out_np'] + qb)

    # Gather per-split scalar fields from precomputed numpy.
    lam = geo['lam_pu'][scored_idx]
    mu = geo['mu_pu'][scored_idx]
    c_in = geo['c_in_pu'][scored_idx]
    e_new_col = geo['e_new_col_pu'][scored_idx]
    z_lo = geo['z_lo_pu'][scored_idx]
    z_hi = geo['z_hi_pu'][scored_idx]
    safe_mu = geo['safe_mu'][scored_idx]
    d_at_e_new = d_np[e_new_col]                       # [n_splits]

    inv_mu_safe = 1.0 / np.where(safe_mu, mu, 1.0)
    ratio_off_np = np.where(safe_mu, -(lam * inv_mu_safe) * d_at_e_new, 0.0)
    ratio_on_np = np.where(safe_mu, ((1.0 - lam) * inv_mu_safe) * d_at_e_new, 0.0)
    c0_off_np = np.where(safe_mu,
                          -(1.0 + lam * c_in * inv_mu_safe) * d_at_e_new,
                          -d_at_e_new)
    c0_on_np = np.where(safe_mu,
                         ((1.0 - lam) * c_in * inv_mu_safe - 1.0) * d_at_e_new,
                         -d_at_e_new)

    # GPU side: gather a_pu and form hs_A / d_corr without CPU temps.
    scored_idx_t = torch.as_tensor(scored_idx, device=device, dtype=torch.long)
    e_new_col_t = torch.as_tensor(e_new_col, device=device, dtype=torch.long)
    a_gathered = geo['a_pu_t'].index_select(0, scored_idx_t)  # [n_splits, n_gens]
    # signs: OFF=(+a, -a), ON=(-a, +a)
    signs = torch.tensor([[+1.0, -1.0], [-1.0, +1.0]], device=device, dtype=dtype)
    hs_A = signs.view(1, 2, 2, 1) * a_gathered.view(n_splits, 1, 1, n_gens)
    hs_b = torch.empty(n_splits, 2, 2, device=device, dtype=dtype)
    hs_b[:, 0, 0] = torch.as_tensor(-c_in, device=device, dtype=dtype)
    hs_b[:, 0, 1] = torch.as_tensor(z_lo, device=device, dtype=dtype)
    hs_b[:, 1, 0] = torch.as_tensor(c_in, device=device, dtype=dtype)
    hs_b[:, 1, 1] = torch.as_tensor(z_hi, device=device, dtype=dtype)

    ratio_off_t = torch.as_tensor(ratio_off_np, device=device, dtype=dtype)
    ratio_on_t = torch.as_tensor(ratio_on_np, device=device, dtype=dtype)
    d_at_e_new_t = torch.as_tensor(d_at_e_new, device=device, dtype=dtype)
    d_corr_off = ratio_off_t.view(n_splits, 1) * a_gathered    # a has e_new col=0
    d_corr_on = ratio_on_t.view(n_splits, 1) * a_gathered
    rows = torch.arange(n_splits, device=device, dtype=torch.long)
    d_corr_off[rows, e_new_col_t] = -d_at_e_new_t
    d_corr_on[rows, e_new_col_t] = -d_at_e_new_t
    d_corr = torch.stack([d_corr_off, d_corr_on], dim=1)  # [n_splits, 2, n_gens]
    c0_corr = torch.stack([
        torch.as_tensor(c0_off_np, device=device, dtype=dtype),
        torch.as_tensor(c0_on_np, device=device, dtype=dtype),
    ], dim=1)
    d_t = torch.as_tensor(d_np, device=device, dtype=dtype)
    return (n_gens, geo['e_lb_t'], geo['e_hi_t'], geo['width_t'],
            d_t, c0, hs_A, hs_b, d_corr, c0_corr)


def _build_substitution_caches(
    state, qw, qb, scored_keys, device, dtype,
):
    """Pre-compute per-split caches for the substitution-form LP.

    For each split in `scored_keys`, populate:
      - hs_A[i, side, row, :n_gens]   halfspace coefficients
      - hs_b[i, side, row]            rhs values
      - d_corr[i, side, :n_gens]      additive d-correction from substitution
      - c0_corr[i, side]              additive c0-correction from substitution

    Returns: (n_gens, e_lb, e_hi, width, d_t, c0, hs_A, hs_b, d_corr, c0_corr)
    """
    n_gens = int(state['n_gens'])
    n_input = int(state['n_input'])

    # Box bounds: input gens on [-1, 1]; non-input gens default 0; unstable
    # neurons add e_new on [-1, 1].
    e_lb_np = np.zeros(n_gens, dtype=np.float64)
    e_hi_np = np.zeros(n_gens, dtype=np.float64)
    e_lb_np[:n_input] = -1.0
    e_hi_np[:n_input] = 1.0
    unstable_list = state['unstable_list']
    unstable_by_key = {(u['layer_idx'], u['neuron_idx']): u
                       for u in unstable_list}
    for u in unstable_list:
        c = int(u['e_new_col'])
        e_lb_np[c] = -1.0
        e_hi_np[c] = 1.0
    width_np = e_hi_np - e_lb_np

    # Objective: c0 + d·e where d = qw @ G_out
    obj_G_out = state['obj_G_out_csr'].toarray().astype(np.float64)
    obj_c_out = np.asarray(state['obj_c_out'], dtype=np.float64)
    qw_np = np.asarray(qw, dtype=np.float64)
    d_np = qw_np @ obj_G_out
    c0 = float(qw_np @ obj_c_out + qb)

    e_lb_t = torch.tensor(e_lb_np, device=device, dtype=dtype)
    e_hi_t = torch.tensor(e_hi_np, device=device, dtype=dtype)
    width_t = torch.tensor(width_np, device=device, dtype=dtype)
    d_t = torch.tensor(d_np, device=device, dtype=dtype)

    # max_depth removed in favor of time_limit. Use full scored_keys length.
    n_splits = len(scored_keys)
    hs_A = torch.zeros(n_splits, 2, _ROWS_PER_SPLIT, n_gens,
                       device=device, dtype=dtype)
    hs_b = torch.zeros(n_splits, 2, _ROWS_PER_SPLIT,
                       device=device, dtype=dtype)
    d_corr = torch.zeros(n_splits, 2, n_gens, device=device, dtype=dtype)
    c0_corr = torch.zeros(n_splits, 2, device=device, dtype=dtype)

    for i in range(n_splits):
        key = scored_keys[i]
        u = unstable_by_key[key]
        c_in = float(u['c_in'])
        lam_k = float(u['lam'])
        mu_k = float(u['mu'])
        row_idx = np.asarray(u['row_indices'], dtype=np.int64)
        row_val = np.asarray(u['row_values'], dtype=np.float64)
        e_new_col = int(u['e_new_col'])
        d_at_e_new = float(d_np[e_new_col])
        a_k = np.zeros(n_gens, dtype=np.float64)
        a_k[row_idx] = row_val

        # OFF branch: z ≤ 0, z ≥ -2μ/λ (the e_new_k ∈ [-1, 1] box-feasibility
        # bound after substituting e_new_k = -1 - (λ/μ)z).
        h1 = a_k.copy()
        h1[e_new_col] = 0.0
        z_lo = (2.0 * mu_k / lam_k + c_in
                if (mu_k > 1e-12 and lam_k > 1e-12) else 1e9)
        h2 = -a_k.copy()
        h2[e_new_col] = 0.0
        hs_A[i, 0, 0] = torch.tensor(h1, device=device, dtype=dtype)
        hs_b[i, 0, 0] = -c_in
        hs_A[i, 0, 1] = torch.tensor(h2, device=device, dtype=dtype)
        hs_b[i, 0, 1] = float(z_lo)

        # OFF d-correction: d_off[i] += -(λ/μ)·d[e_new]·a_k[i] for i in row_idx
        #                  d_off[e_new] = 0 (substituted out)
        d_corr_off = np.zeros(n_gens, dtype=np.float64)
        if mu_k > 1e-12:
            d_corr_off[row_idx] = -(lam_k / mu_k) * d_at_e_new * row_val
        d_corr_off[e_new_col] -= d_at_e_new
        d_corr[i, 0] = torch.tensor(d_corr_off, device=device, dtype=dtype)
        c0_corr[i, 0] = (-(1.0 + lam_k * c_in / mu_k) * d_at_e_new
                         if mu_k > 1e-12 else -d_at_e_new)

        # ON branch: z ≥ 0, z ≤ 2μ/(1-λ).
        h1 = -a_k.copy()
        h1[e_new_col] = 0.0
        z_hi = (2.0 * mu_k / (1.0 - lam_k) - c_in
                if (mu_k > 1e-12 and (1.0 - lam_k) > 1e-12) else 1e9)
        h2 = a_k.copy()
        h2[e_new_col] = 0.0
        hs_A[i, 1, 0] = torch.tensor(h1, device=device, dtype=dtype)
        hs_b[i, 1, 0] = c_in
        hs_A[i, 1, 1] = torch.tensor(h2, device=device, dtype=dtype)
        hs_b[i, 1, 1] = float(z_hi)

        # ON d-correction.
        d_corr_on = np.zeros(n_gens, dtype=np.float64)
        if mu_k > 1e-12:
            d_corr_on[row_idx] = ((1.0 - lam_k) / mu_k) * d_at_e_new * row_val
        d_corr_on[e_new_col] -= d_at_e_new
        d_corr[i, 1] = torch.tensor(d_corr_on, device=device, dtype=dtype)
        c0_corr[i, 1] = (((1.0 - lam_k) * c_in / mu_k - 1.0) * d_at_e_new
                         if mu_k > 1e-12 else -d_at_e_new)

    return n_gens, e_lb_t, e_hi_t, width_t, d_t, c0, hs_A, hs_b, d_corr, c0_corr


def build_nonlinear_split_caches(splittables, d_np, c0, n_gens, device, dtype,
                                 alpha_l=None, alpha_r=None):
    """Dual-ascent caches for NONLINEAR splittable neurons (sigmoid/tanh/pow/
    sin/cos), in the SAME (hs_A, hs_b, d_corr, c0_corr) layout the ReLU
    substitution path produces — so the batched BaB consumes a mix transparently.

    Each `splittables[i]` dict carries the neuron's parent geometry:
        relax       a ScalarNonlinearRelax (Sigmoid/Tanh/Sin/Cos/Pow…)
        op_type     graph op string (drives the zero-vs-midpoint split point)
        lo, hi      parent pre-activation interval
        c_in        pre-activation center (z = c_in + a·e)
        row_indices, row_values   the a row over error symbols (e_new excluded)
        e_new_col   column of the op's fresh error symbol in d
        lam, mu, delta            parent band (λ z + μ ± δ); delta = e_new mag

    For each side (0=left z≤p, 1=right z≥p) we re-tighten the band on the
    sub-interval and fold the change into the objective via
    `band_change_correction`; one halfspace pins the side's pre-activation.
    The op stays a band (NOT substituted), so the e_new column survives with a
    smaller δ'. _ROWS_PER_SPLIT=2 is kept uniform with ReLU by padding row 1
    with a no-op (0·e ≤ 1, never binds; the dual sets its λ=0).

    `alpha_l`/`alpha_r` (default None) pick the CHILD slope. None → inherit the
    PARENT slope λ: then g(x)=f(x)−λx has a SUBSET range on the (contained)
    sub-interval, so the child's offset band [μ'−δ', μ'+δ'] ⊆ the parent's and
    the split is GUARANTEED to (weakly) tighten — the BaB always makes progress.
    A slope CHANGE (alpha given) instead rescales the input generators g_k·λ'·a
    and can loosen at a bad α; supply alphas only when they're being optimised
    (α-CROWN), initialised at the parent slope so they never regress.

    Returns (hs_A, hs_b, d_corr, c0_corr) shaped like `_compute_query_caches`.
    Soundness: see nonlinear_split_dual — fixed downstream relaxations over the
    parent interval remain valid on the (contained) child sub-interval.
    """
    n_splits = len(splittables)
    hs_A = torch.zeros(n_splits, 2, _ROWS_PER_SPLIT, n_gens,
                       device=device, dtype=dtype)
    hs_b = torch.zeros(n_splits, 2, _ROWS_PER_SPLIT, device=device, dtype=dtype)
    d_corr = torch.zeros(n_splits, 2, n_gens, device=device, dtype=dtype)
    c0_corr = torch.zeros(n_splits, 2, device=device, dtype=dtype)
    # no-op pad row: 0·e ≤ 1 (always feasible, never binds).
    hs_b[:, :, 1] = 1.0

    d_np = np.asarray(d_np, dtype=np.float64)
    for i, sp in enumerate(splittables):
        relax = sp['relax']
        lo = float(sp['lo']); hi = float(sp['hi']); c_in = float(sp['c_in'])
        ridx = np.asarray(sp['row_indices'], dtype=np.int64)
        rval = np.asarray(sp['row_values'], dtype=np.float64)
        e_new_col = int(sp['e_new_col'])
        assert e_new_col not in ridx, (
            f"nonlinear split {sp.get('op_type')}: e_new_col {e_new_col} "
            f"overlaps row_indices — d_corr/halfspace would double-count.")
        lam = float(sp['lam']); mu = float(sp['mu']); delta = float(sp['delta'])
        p = float(split_point(sp['op_type'], lo, hi))
        g_k = float(backward_sensitivity(float(d_np[e_new_col]), delta))

        for side, (clo, chi, a_side, name) in enumerate((
                (lo, p, alpha_l, 'left'), (p, hi, alpha_r, 'right'))):
            clo_t = torch.tensor(clo, dtype=torch.float64)
            chi_t = torch.tensor(chi, dtype=torch.float64)
            if a_side is None:
                # inherit parent slope -> guaranteed-tightening offset re-fit.
                lam_n, mu_n, delta_n = (float(t) for t in relax.affine_band(
                    clo_t, chi_t,
                    lam=torch.tensor(lam, dtype=torch.float64)))
            else:
                # α-chosen slope (optimisable; may loosen at a bad α).
                sL, tL, sU, tU = op_planes(
                    relax, clo_t, chi_t,
                    torch.tensor(a_side, dtype=torch.float64), 'band')
                lam_n = float(sL); mu_n = 0.5 * float(tL + tU)
                delta_n = 0.5 * float(tU - tL)
            dcr, dce, c0c = band_change_correction(
                g_k, c_in, rval, lam, mu, delta, lam_n, mu_n, delta_n)
            d_corr[i, side, e_new_col] = float(dce)
            d_corr[i, side].index_add_(
                0, torch.as_tensor(ridx, device=device, dtype=torch.long),
                torch.as_tensor(np.asarray(dcr, dtype=np.float64),
                                device=device, dtype=dtype))
            c0_corr[i, side] = float(c0c)
            hs_row, hs_rhs = split_halfspace(c_in, rval, p, name)
            hs_A[i, side, 0].index_add_(
                0, torch.as_tensor(ridx, device=device, dtype=torch.long),
                torch.as_tensor(np.asarray(hs_row, dtype=np.float64),
                                device=device, dtype=dtype))
            hs_b[i, side, 0] = float(hs_rhs)
    return hs_A, hs_b, d_corr, c0_corr


def _batched_dual_ascent(
    d_batch, c0_batch, A_batch, b_batch, lam_batch, alive_mask,
    e_lb, e_hi, width, *, max_iter: int, repair_steps: int,
    feas_tol: float, tol: float, dtype, device,
):
    """One batched K-iter dual-ascent pass on B nodes simultaneously.

    Returns (best_g [B], lam [B,m], reason [B] int8)
      reason: 0=dual_safe (best_g>0), 1=primal_unsafe (witness found, must
              split), 2=no-progress (split), 3=safety_cap (split).
    """
    B, m, n = A_batch.shape
    e_lb_b = e_lb.unsqueeze(0)
    e_hi_b = e_hi.unsqueeze(0)
    rc = d_batch + torch.einsum('bmn,bm->bn', A_batch, lam_batch)
    best_g = torch.full((B,), -float('inf'), device=device, dtype=dtype)
    reason = torch.full((B,), -1, dtype=torch.int8, device=device)
    # Stash the primal witness x for primal_unsafe nodes — caller can map
    # x[:n_input] back to input space and run the real NN to look for an
    # actual counterexample.
    witness = torch.zeros(B, n, device=device, dtype=dtype)
    pending = alive_mask.clone()
    for _k in range(max_iter):
        x_star = torch.where(rc < 0,
                             e_hi_b.expand(B, n), e_lb_b.expand(B, n))
        g_lam = (c0_batch + (rc * x_star).sum(-1)
                 - (lam_batch * b_batch).sum(-1))
        better = g_lam > best_g
        best_g = torch.where(better & pending, g_lam, best_g)
        safe = pending & (best_g > 0)
        reason = torch.where(safe,
                             torch.tensor(0, dtype=torch.int8, device=device),
                             reason)
        pending = pending & ~safe

        Ax = torch.einsum('bmn,bn->bm', A_batch, x_star)
        s = Ax - b_batch
        max_s = s.max(-1).values
        p_eval = c0_batch + (d_batch * x_star).sum(-1)
        raw_unsafe = pending & (max_s <= feas_tol) & (p_eval <= 0)
        # Stash x_star as witness for raw_unsafe nodes
        witness = torch.where(raw_unsafe.unsqueeze(-1), x_star, witness)
        reason = torch.where(raw_unsafe,
                             torch.tensor(1, dtype=torch.int8, device=device),
                             reason)
        pending = pending & ~raw_unsafe

        # Greedy primal repair: project onto most-violated halfspace, clip box.
        x_rep = x_star.clone()
        for _ in range(repair_steps):
            s_rep = torch.einsum('bmn,bn->bm', A_batch, x_rep) - b_batch
            max_s_rep, j_rep = s_rep.max(-1)
            still_violating = pending & (max_s_rep > feas_tol)
            a_sel = A_batch[torch.arange(B, device=device), j_rep, :]
            denom = (a_sel * a_sel).sum(-1).clamp_min(1e-30)
            step = (max_s_rep / denom).unsqueeze(-1) * a_sel
            x_rep = torch.where(still_violating.unsqueeze(-1),
                                x_rep - step, x_rep)
            x_rep = torch.where(still_violating.unsqueeze(-1),
                                torch.clamp(x_rep, e_lb_b, e_hi_b), x_rep)
        s_after = torch.einsum('bmn,bn->bm', A_batch, x_rep) - b_batch
        feas_rep = s_after.max(-1).values <= feas_tol
        p_rep = c0_batch + (d_batch * x_rep).sum(-1)
        rep_unsafe = pending & feas_rep & (p_rep <= 0)
        witness = torch.where(rep_unsafe.unsqueeze(-1), x_rep, witness)
        reason = torch.where(rep_unsafe,
                             torch.tensor(1, dtype=torch.int8, device=device),
                             reason)
        pending = pending & ~rep_unsafe

        # Ascent direction: s_proj is the projected subgradient.
        zero_mask = (lam_batch <= tol) & (s < 0)
        s_proj = torch.where(zero_mask, torch.zeros_like(s), s)
        da = torch.einsum('bmn,bm->bn', A_batch, s_proj)
        slope0 = (s_proj * s).sum(-1)

        # Breakpoints: η_i = -rc_i / da_i where the sign flip happens.
        positive = ((da > 0) & (rc < 0)) | ((da < 0) & (rc > 0))
        etas = torch.where(positive, -rc / da,
                           torch.tensor(float('inf'),
                                        device=device, dtype=dtype))
        etas_sorted, idx_sorted = etas.sort(dim=-1)
        rc_sorted = torch.gather(rc, -1, idx_sorted)
        da_sorted = torch.gather(da, -1, idx_sorted)
        width_sorted = width.unsqueeze(0).expand(B, n).gather(-1, idx_sorted)

        # Slope decreases monotonically in η; find first η where slope ≤ 0.
        decr = torch.where(rc_sorted < 0,
                           width_sorted * da_sorted,
                           -width_sorted * da_sorted)
        decr = torch.where(torch.isfinite(etas_sorted),
                           decr, torch.zeros_like(decr))
        cumdec = decr.cumsum(dim=-1)
        slope_after = slope0.unsqueeze(-1) - cumdec
        below = slope_after <= tol
        any_below = below.any(-1)
        first_below = below.float().argmax(-1)
        eta_star = torch.gather(etas_sorted, -1,
                                first_below.unsqueeze(-1)).squeeze(-1)
        finite_mask = torch.isfinite(etas_sorted)
        last_finite_idx = (finite_mask.cumsum(-1)
                           * finite_mask).argmax(-1)
        eta_fallback = torch.gather(etas_sorted, -1,
                                    last_finite_idx.unsqueeze(-1)).squeeze(-1)
        eta_star = torch.where(any_below, eta_star, eta_fallback)
        eta_star = torch.where(torch.isfinite(eta_star), eta_star,
                               torch.zeros_like(eta_star))

        step_mask = pending.unsqueeze(-1).to(dtype)
        lam_prev = lam_batch
        lam_batch = (lam_batch
                     + (eta_star.unsqueeze(-1) * s_proj) * step_mask
                     ).clamp_min(0.0)
        # Re-sync rc with the CLAMPED λ. The clamp changes λ by a different
        # amount than eta·s_proj whenever a multiplier hits the λ≥0 boundary;
        # the old incremental `rc += eta·da` assumed the unclamped step, so rc
        # drifted from d+Aᵀλ and g_lam (computed from rc next iter) became an
        # INVALID — possibly too-high — bound that could falsely certify (seen
        # for K≥2 with warm-started λ; cold-start K=1 never clamps so it was
        # masked). Apply the delta for the ACTUAL λ change instead.
        rc = rc + torch.einsum('bmn,bm->bn', A_batch, lam_batch - lam_prev)

    reason = torch.where(
        (reason == -1) & alive_mask,
        torch.tensor(3, dtype=torch.int8, device=device), reason)
    return best_g, lam_batch, reason, witness


def verify_query_dual_ascent_bab(
    state, qw, qb, scored_keys, *,
    time_limit: float = 60.0,
    max_iter: int = _DEFAULT_K,
    repair_steps: int = _DEFAULT_REPAIR_STEPS,
    feas_tol: float = _FEAS_TOL,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    print_progress: bool = False,
    time_left_fn: Optional[Callable[[], float]] = None,
    witness_check_fn: Optional[Callable[[np.ndarray], Optional[np.ndarray]]] = None,
):
    """Verify a single (qw, qb) query via GPU layered-BFS BaB with dual ascent.

    Args:
        state: gen-LP state dict from `verify_gen_lp.state_from_alpha_zono`.
        qw, qb: spec direction (qw·output ≤ qb means counterexample).
        scored_keys: list of (layer_idx, neuron_idx) tuples in branching order
            (highest priority first). Depth is naturally capped at the length
            of this list.
        time_limit: wall clock budget in seconds (default 60). Sole exit
            criterion alongside frontier exhaustion — no depth/frontier cap.
        max_iter: hard iteration cap per BaB node (default 1 — fastest).
        repair_steps: greedy primal-repair steps per dual iter (default 5).
        feas_tol: feasibility tolerance for primal repair (default 1e-5).
        device: torch device (default 'cuda' if available else 'cpu').
        dtype: torch dtype (default float32 — sound; bfloat16 needs threshold).
        print_progress: per-layer breakdown print.
        time_left_fn: callable returning remaining seconds; if provided,
            overrides time_limit.

    Returns:
        (verdict, info) where:
          verdict in {'unsat', 'unknown'}
          info: dict with 'wall', 'nodes', 'max_depth_seen', 'exit_counts',
                'reason' (if unknown), 'final_open' (frontier size at exit).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Env-gated dump of the per-query box+halfspace BnB instance (one pkl per
    # query) for the local optimizer harness. Off unless VC_DUMP_BNB_DIR set.
    import os as _os
    _dump_dir = _os.environ.get('VC_DUMP_BNB_DIR', '')
    if _dump_dir:
        from .fast_dual_ascent.fast_verify_dual import _dump_bnb_instance
        _dump_bnb_instance(state, qw, qb, scored_keys, _dump_dir)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    # Disable autograd for the entire BaB. Standalone runs are 9× faster
    # than production BaB on identical state; suspect autograd state from
    # prior phases is propagating tracking. Force no_grad.
    # Use try/finally so an exception or early return cannot leak the
    # disabled-grad state into the caller (caller's α-CROWN backward
    # would silently fail with "element 0 does not require grad").
    _grad_was = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        return _verify_dab_impl(
            state, qw, qb, scored_keys, time_limit, max_iter, repair_steps,
            feas_tol, device, dtype,
            print_progress, time_left_fn, witness_check_fn)
    finally:
        torch.set_grad_enabled(_grad_was)


def _verify_dab_impl(state, qw, qb, scored_keys, time_limit, max_iter,
                      repair_steps, feas_tol,
                      device, dtype, print_progress, time_left_fn,
                      witness_check_fn):
    t_start = time.perf_counter()
    if time_left_fn is None:
        deadline = t_start + time_limit
        def _left():
            return deadline - time.perf_counter()
    else:
        _left = time_left_fn

    # Early exit if the deadline has already passed — skip per-query setup
    # (which costs ~0.5–2 s on cifar100 even when BaB itself would bail).
    if _left() <= 0:
        return 'unknown', {'wall': time.perf_counter() - t_start,
                            'nodes': 0, 'max_depth_seen': 0,
                            'exit_counts': {'dual_safe': 0, 'primal_unsafe': 0,
                                            'safety_cap': 0, 'no_progress': 0},
                            'final_open': 0, 'root_g': float('nan'),
                            'reason': 'deadline-on-entry'}

    # Filter scored_keys to those actually present in the per-query state's
    # unstable list. (per-query rebuilds with tightened bounds can drop
    # neurons that became stable; original ranking may include them.)
    _state_keys = {(u['layer_idx'], u['neuron_idx']) for u in state['unstable_list']}
    scored_keys = [k for k in scored_keys if k in _state_keys]
    n_splits = len(scored_keys)
    if n_splits == 0:
        # No splits available — just evaluate root LP.
        n_gens = int(state['n_gens'])
        n_input = int(state['n_input'])
        e_lb_np = np.zeros(n_gens); e_hi_np = np.zeros(n_gens)
        e_lb_np[:n_input] = -1.0; e_hi_np[:n_input] = 1.0
        for u in state['unstable_list']:
            c = int(u['e_new_col']); e_lb_np[c] = -1.0; e_hi_np[c] = 1.0
        obj_G_out = state['obj_G_out_csr'].toarray().astype(np.float64)
        obj_c_out = np.asarray(state['obj_c_out'], dtype=np.float64)
        qw_np = np.asarray(qw, dtype=np.float64)
        d_np = qw_np @ obj_G_out
        c0 = float(qw_np @ obj_c_out + qb)
        x_star_root = np.where(d_np < 0, e_hi_np, e_lb_np)
        g_root = c0 + float(d_np @ x_star_root)
        verdict = 'unsat' if g_root > 0 else 'unknown'
        return verdict, {'wall': time.perf_counter() - t_start,
                         'nodes': 1, 'max_depth_seen': 0,
                         'exit_counts': {'dual_safe': 1 if g_root > 0 else 0,
                                         'primal_unsafe': 0,
                                         'safety_cap': 0,
                                         'no_progress': 0},
                         'final_open': 0,
                         'root_g': g_root}

    n_input = int(state['n_input'])
    # Build / fetch state-level geometric cache (LRU-1 keyed by id(state)
    # to bound GPU memory when state_by_qi populates many distinct states).
    geo = state.get('_dab_geom')
    if (geo is None
            or geo['device'] != device
            or geo['dtype'] != dtype
            or geo['n_gens'] != int(state['n_gens'])):
        geo = _precompute_state_geometry(state, device, dtype)
        state['_dab_geom'] = geo
    (n_gens, e_lb, e_hi, width, d_t, c0, hs_A, hs_b, d_corr, c0_corr) = \
        _compute_query_caches(geo, scored_keys, qw, qb, device, dtype)

    # Root LP (no halfspaces): box minimum gives root g.
    x_star_root = torch.where(d_t < 0, e_hi, e_lb)
    g_root = c0 + (d_t * x_star_root).sum().item()
    n_total = 1
    exit_counts = {'dual_safe': 0, 'primal_unsafe': 0,
                   'safety_cap': 0, 'no_progress': 0}
    if g_root > 0:
        exit_counts['dual_safe'] = 1
        return 'unsat', {'wall': time.perf_counter() - t_start,
                         'nodes': n_total, 'max_depth_seen': 0,
                         'exit_counts': exit_counts, 'final_open': 0,
                         'root_g': g_root}

    # Layered BFS frontier.
    open_paths = torch.tensor([[0], [1]], device=device, dtype=torch.int8)
    open_lam = torch.zeros(2, _ROWS_PER_SPLIT, device=device, dtype=dtype)
    max_depth_seen = 0
    info_extra = {}
    depth = 1
    while open_paths.shape[0] > 0 and depth < n_splits:
        if _left() <= 0:
            info_extra['reason'] = f'TIMEOUT at depth {depth}'
            break
        max_depth_seen = max(max_depth_seen, depth)
        B = open_paths.shape[0]
        n_total += B

        m_dim = depth * _ROWS_PER_SPLIT
        if open_lam.shape[1] < m_dim:
            pad = m_dim - open_lam.shape[1]
            open_lam = torch.cat(
                [open_lam, torch.zeros(B, pad, device=device, dtype=dtype)],
                dim=1)

        # Chunk the frontier so A_batch + d_path fits in budget.
        # bytes_per_chunk ≈ chunk_size * (m * n + n) * dtype_bytes.
        bytes_per_elem = 4 if dtype == torch.float32 else 2
        per_node_bytes = bytes_per_elem * (m_dim * n_gens + n_gens) * 8  # 8x slack for working buffers
        chunk_size = max(1, min(B, int(_DEFAULT_CHUNK_BYTES_BUDGET
                                        / max(1, per_node_bytes))))

        pl_full = open_paths.long()
        reason_all = torch.empty(B, dtype=torch.int8, device=device)
        lam_out_all = torch.empty(B, m_dim, device=device, dtype=dtype)
        timed_out_mid_layer = False
        for chunk_start in range(0, B, chunk_size):
            if _left() <= 0:
                info_extra['reason'] = (f'TIMEOUT mid-layer at depth {depth} '
                                         f'(chunk {chunk_start}/{B})')
                timed_out_mid_layer = True
                break
            chunk_end = min(chunk_start + chunk_size, B)
            cs = chunk_end - chunk_start
            pl = pl_full[chunk_start:chunk_end]
            A_batch = torch.empty(cs, m_dim, n_gens,
                                   device=device, dtype=dtype)
            b_batch = torch.empty(cs, m_dim, device=device, dtype=dtype)
            d_path = d_t.unsqueeze(0).expand(cs, n_gens).clone()
            c0_path = torch.full((cs,), c0, device=device, dtype=dtype)
            for j in range(depth):
                sides_j = pl[:, j]
                A_batch[:, j * _ROWS_PER_SPLIT:(j + 1) * _ROWS_PER_SPLIT, :] = \
                    hs_A[j, sides_j]
                b_batch[:, j * _ROWS_PER_SPLIT:(j + 1) * _ROWS_PER_SPLIT] = \
                    hs_b[j, sides_j]
                d_path += d_corr[j, sides_j]
                c0_path += c0_corr[j, sides_j]
            lam_chunk = open_lam[chunk_start:chunk_end].contiguous()
            alive = torch.ones(cs, dtype=torch.bool, device=device)
            _, lam_out_chunk, reason_chunk, witness_chunk = _batched_dual_ascent(
                d_path, c0_path, A_batch, b_batch, lam_chunk, alive,
                e_lb, e_hi, width,
                max_iter=max_iter, repair_steps=repair_steps,
                feas_tol=feas_tol, tol=_TOL, dtype=dtype, device=device)
            reason_all[chunk_start:chunk_end] = reason_chunk
            lam_out_all[chunk_start:chunk_end] = lam_out_chunk
            # Stash any primal_unsafe witnesses' input portions + their
            # primal value (so we can pick the worst-case = most adversarial
            # = most negative p — a few of those is much cheaper to forward
            # than thousands of mediocre ones).
            if witness_check_fn is not None:
                w_mask = (reason_chunk == 1)
                if w_mask.any():
                    w = witness_chunk[w_mask][:, :n_input].cpu().numpy()
                    # Compute primal value at these witnesses
                    p = (c0_path[w_mask].float()
                         + (d_path[w_mask].float() * witness_chunk[w_mask].float()).sum(-1)).cpu().numpy()
                    if 'witness_inputs' not in info_extra:
                        info_extra['witness_inputs'] = []
                        info_extra['witness_p'] = []
                    info_extra['witness_inputs'].append(w)
                    info_extra['witness_p'].append(p)
            del A_batch, b_batch, d_path, c0_path
        if timed_out_mid_layer:
            break
        reason = reason_all
        lam_out = lam_out_all
        if device.type == 'cuda':
            torch.cuda.synchronize()
        # Attack: try each accumulated witness against the real NN. If any
        # falsifies, return 'sat' early. We accumulate per-layer to avoid
        # excessive callback invocations on small batches.
        if witness_check_fn is not None and 'witness_inputs' in info_extra:
            ws = info_extra.pop('witness_inputs')
            ps = info_extra.pop('witness_p', None)
            if ws:
                w_all = np.concatenate(ws, axis=0)
                # Take only TOP-K worst-case (most negative primal) witnesses.
                # Forward-passing all 1000s of them is the production
                # bottleneck (3574: 12086 witnesses × ~ms = 120s slowdown).
                K_WITNESS = 5
                if ps is not None and len(w_all) > K_WITNESS:
                    p_all = np.concatenate(ps, axis=0)
                    top_idx = np.argpartition(p_all, K_WITNESS)[:K_WITNESS]
                    w_all = w_all[top_idx]
                cex = witness_check_fn(w_all)
                if cex is not None:
                    elapsed = time.perf_counter() - t_start
                    info = {'wall': elapsed, 'nodes': n_total,
                            'max_depth_seen': max_depth_seen,
                            'exit_counts': exit_counts,
                            'final_open': int(open_paths.shape[0]),
                            'root_g': g_root,
                            'sat_witness_depth': depth,
                            'sat_witness': cex}
                    return 'sat', info

        # Frontier expansion: split survivors into off/on children.
        must_split = (reason != 0) & (reason != -1)
        survivors_idx = must_split.nonzero(as_tuple=True)[0]
        ns = survivors_idx.shape[0]
        survivor_paths = open_paths[survivors_idx]
        survivor_lams = lam_out[survivors_idx]
        new_paths = torch.zeros(ns * 2, depth + 1,
                                device=device, dtype=torch.int8)
        new_paths[0::2, :depth] = survivor_paths; new_paths[0::2, depth] = 0
        new_paths[1::2, :depth] = survivor_paths; new_paths[1::2, depth] = 1
        new_lams = torch.zeros(ns * 2, m_dim + _ROWS_PER_SPLIT,
                               device=device, dtype=dtype)
        new_lams[0::2, :m_dim] = survivor_lams
        new_lams[1::2, :m_dim] = survivor_lams
        open_paths = new_paths
        open_lam = new_lams

        r_np = reason.cpu().numpy()
        exit_counts['dual_safe'] += int((r_np == 0).sum())
        exit_counts['primal_unsafe'] += int((r_np == 1).sum())
        exit_counts['no_progress'] += int((r_np == 2).sum())
        exit_counts['safety_cap'] += int((r_np == 3).sum())

        if print_progress:
            print(f'    depth={depth} B={B} m={m_dim} → '
                  f'safe={int((r_np == 0).sum())} '
                  f'uns={int((r_np == 1).sum())} '
                  f'cap={int((r_np == 3).sum())}')
        depth += 1

    elapsed = time.perf_counter() - t_start
    final_open = int(open_paths.shape[0])
    if final_open == 0:
        verdict = 'unsat'
    else:
        verdict = 'unknown'
        if 'reason' not in info_extra:
            if depth >= n_splits:
                info_extra['reason'] = (f'exhausted all {n_splits} splits '
                                         f'with {final_open} open nodes')
            else:
                info_extra['reason'] = f'aborted at depth {depth}'

    info = {'wall': elapsed, 'nodes': n_total,
            'max_depth_seen': max_depth_seen, 'exit_counts': exit_counts,
            'final_open': final_open, 'root_g': g_root}
    info.update(info_extra)
    return verdict, info


def verify_queries_batched(
    state, query_specs, *,
    time_limit: float = 60.0,
    max_iter: int = _DEFAULT_K,
    repair_steps: int = _DEFAULT_REPAIR_STEPS,
    feas_tol: float = _FEAS_TOL,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    print_progress: bool = False,
    time_left_fn: Optional[Callable[[], float]] = None,
    witness_check_fn: Optional[Callable[[np.ndarray], Optional[np.ndarray]]] = None,
):
    """Verify multiple (qw, qb) queries simultaneously via cross-query
    batching of the dual-ascent BaB kernel.

    `query_specs` is a list of `(qw, qb, scored_keys)` tuples. All queries
    share the same `state`. Returns a list `[(verdict, info), ...]` in the
    same order.

    Per-layer: each active query has its own frontier and scored_keys; we
    pad halfspace counts to `max(m_dim)` across active queries and run
    ONE `_batched_dual_ascent` call on the concatenated frontier. This
    amortizes kernel-launch + scheduling overhead across queries.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    _grad_was = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        return _verify_queries_batched_impl(
            state, query_specs, time_limit, max_iter, repair_steps,
            feas_tol, device, dtype, print_progress, time_left_fn,
            witness_check_fn)
    finally:
        torch.set_grad_enabled(_grad_was)


def _root_lp_only(state, geo, qw, qb, e_lb, e_hi, device, dtype, t_start):
    """Evaluate root LP (no halfspaces) for a query — used when scored_keys
    is empty (no splits possible). Same semantics as the n_splits==0 branch
    in _verify_dab_impl."""
    qw_np = np.asarray(qw, dtype=np.float64)
    d_np = qw_np @ geo['obj_G_out_np']
    c0_val = float(qw_np @ geo['obj_c_out_np'] + qb)
    d_t_q = torch.as_tensor(d_np, device=device, dtype=dtype)
    x_star = torch.where(d_t_q < 0, e_hi, e_lb)
    g_root = c0_val + (d_t_q * x_star).sum().item()
    verdict = 'unsat' if g_root > 0 else 'unknown'
    info = {'wall': time.perf_counter() - t_start, 'nodes': 1,
            'max_depth_seen': 0,
            'exit_counts': {'dual_safe': 1 if g_root > 0 else 0,
                             'primal_unsafe': 0, 'safety_cap': 0, 'no_progress': 0},
            'final_open': 0, 'root_g': g_root}
    return verdict, info


def _verify_queries_batched_impl(
    state, query_specs, time_limit, max_iter, repair_steps,
    feas_tol, device, dtype, print_progress, time_left_fn,
    witness_check_fn,
):
    t_start = time.perf_counter()
    if time_left_fn is None:
        deadline = t_start + time_limit
        def _left():
            return deadline - time.perf_counter()
    else:
        _left = time_left_fn

    Q = len(query_specs)
    if _left() <= 0 or Q == 0:
        return [('unknown', {'wall': 0.0, 'nodes': 0, 'max_depth_seen': 0,
                              'exit_counts': {'dual_safe': 0, 'primal_unsafe': 0,
                                              'safety_cap': 0, 'no_progress': 0},
                              'final_open': 0, 'root_g': float('nan'),
                              'reason': 'deadline-on-entry'}) for _ in range(Q)]

    # Shared state geometry (LRU-1 on state).
    geo = state.get('_dab_geom')
    if (geo is None or geo['device'] != device or geo['dtype'] != dtype
            or geo['n_gens'] != int(state['n_gens'])):
        geo = _precompute_state_geometry(state, device, dtype)
        state['_dab_geom'] = geo
    n_gens = geo['n_gens']
    n_input = int(state['n_input'])
    e_lb, e_hi, width = geo['e_lb_t'], geo['e_hi_t'], geo['width_t']
    _state_keys = {(u['layer_idx'], u['neuron_idx']) for u in state['unstable_list']}

    # Per-query state. `verdict=None` => still active.
    Q_state = [None] * Q
    for qi in range(Q):
        qw, qb, scored_keys = query_specs[qi]
        sk = [k for k in scored_keys if k in _state_keys]
        if len(sk) == 0:
            v, info = _root_lp_only(state, geo, qw, qb, e_lb, e_hi, device, dtype, t_start)
            Q_state[qi] = {'verdict': v, 'info': info, 'open_paths': None}
            continue
        cache = _compute_query_caches(geo, sk, qw, qb, device, dtype)
        _, _, _, _, d_t_q, c0_q, hs_A_q, hs_b_q, d_corr_q, c0_corr_q = cache
        # Root LP first.
        x_star_root = torch.where(d_t_q < 0, e_hi, e_lb)
        g_root = c0_q + (d_t_q * x_star_root).sum().item()
        if g_root > 0:
            Q_state[qi] = {'verdict': 'unsat',
                            'info': {'wall': time.perf_counter() - t_start,
                                      'nodes': 1, 'max_depth_seen': 0,
                                      'exit_counts': {'dual_safe': 1, 'primal_unsafe': 0,
                                                      'safety_cap': 0, 'no_progress': 0},
                                      'final_open': 0, 'root_g': g_root},
                            'open_paths': None}
            continue
        Q_state[qi] = {
            'verdict': None,
            'd_t': d_t_q, 'c0': c0_q,
            'hs_A': hs_A_q, 'hs_b': hs_b_q,
            'd_corr': d_corr_q, 'c0_corr': c0_corr_q,
            'n_splits': hs_A_q.shape[0],
            'open_paths': torch.tensor([[0], [1]], device=device, dtype=torch.int8),
            'open_lam': torch.zeros(2, _ROWS_PER_SPLIT, device=device, dtype=dtype),
            'depth': 1, 'n_total': 1, 'max_depth_seen': 0,
            'exit_counts': {'dual_safe': 0, 'primal_unsafe': 0,
                             'safety_cap': 0, 'no_progress': 0},
            'witness_inputs': [], 'witness_p': [],
            'root_g': g_root, 'reason': None,
            'sat_witness': None, 'sat_witness_depth': None,
        }

    # Layered BFS — synchronized across active queries, padded m_dim.
    while True:
        if _left() <= 0:
            break
        active = [qi for qi in range(Q)
                   if Q_state[qi]['verdict'] is None
                   and Q_state[qi]['open_paths'] is not None
                   and Q_state[qi]['open_paths'].shape[0] > 0]
        if not active:
            break
        max_m = max(Q_state[qi]['depth'] * _ROWS_PER_SPLIT for qi in active)

        # Build per-query (A, b, d, c0, lam) padded to max_m, then concat
        # across queries along the node dim. Chunk along the COMBINED node
        # dim using the same memory budget.
        # Pre-compute total_B and per-query slices.
        slices = []  # list of (qi, start, end, depth_q, m_q, B_q)
        cursor = 0
        for qi in active:
            qs = Q_state[qi]
            B_q = qs['open_paths'].shape[0]
            slices.append((qi, cursor, cursor + B_q, qs['depth'],
                            qs['depth'] * _ROWS_PER_SPLIT, B_q))
            cursor += B_q
        total_B = cursor

        # Chunk size based on max_m (worst case per node).
        bytes_per_elem = 4 if dtype == torch.float32 else 2
        per_node_bytes = bytes_per_elem * (max_m * n_gens + n_gens) * 8
        chunk_size = max(1, min(total_B, int(_DEFAULT_CHUNK_BYTES_BUDGET
                                              / max(1, per_node_bytes))))

        reason_all = torch.empty(total_B, dtype=torch.int8, device=device)
        lam_out_all = torch.zeros(total_B, max_m, device=device, dtype=dtype)
        # NOTE: no full-batch `witness_all` tensor — would be total_B × n_gens
        # (~6 GB at total_B=1M, n_gens=11414) and isn't needed since witness
        # stashing happens INSIDE the chunk loop below.
        # Per-query witness inputs/primals captured INSIDE the chunk loop,
        # since we no longer materialize the per-query A_q ahead of time.
        layer_witness_per_q = {qi: ([], []) for qi, *_ in slices}

        # Build an index helping us map combined positions -> per-query.
        # slices is already ordered by combined cursor.
        slice_starts = np.array([s[1] for s in slices], dtype=np.int64)
        slice_ends = np.array([s[2] for s in slices], dtype=np.int64)

        timed_out_mid = False
        for chunk_start in range(0, total_B, chunk_size):
            if _left() <= 0:
                timed_out_mid = True; break
            chunk_end = min(chunk_start + chunk_size, total_B)
            cs = chunk_end - chunk_start
            # Build the chunk's (A, b, d, c0, lam) by walking the slices
            # that intersect [chunk_start, chunk_end). Per-query segment
            # rows of A get the gathered hs_A; other rows stay zero (which
            # are inactive constraints since b=0 ≥ 0·e is trivially feasible
            # and lam=0 contributes nothing to rc or g).
            A_chunk = torch.zeros(cs, max_m, n_gens, device=device, dtype=dtype)
            b_chunk = torch.zeros(cs, max_m, device=device, dtype=dtype)
            d_chunk = torch.zeros(cs, n_gens, device=device, dtype=dtype)
            c0_chunk = torch.zeros(cs, device=device, dtype=dtype)
            lam_chunk = torch.zeros(cs, max_m, device=device, dtype=dtype)
            # Track which slice indices contributed to this chunk (for the
            # witness-stashing pass below).
            contributing = []
            for sidx, (qi, qs_st, qs_en, depth_q, m_q, B_q) in enumerate(slices):
                seg_st = max(chunk_start, qs_st)
                seg_en = min(chunk_end, qs_en)
                if seg_st >= seg_en:
                    continue
                local_st = seg_st - qs_st
                local_en = seg_en - qs_st
                cl_st = seg_st - chunk_start
                cl_en = seg_en - chunk_start
                qs = Q_state[qi]
                pl = qs['open_paths'][local_st:local_en].long()
                d_chunk[cl_st:cl_en] = qs['d_t']
                c0_chunk[cl_st:cl_en] = qs['c0']
                for j in range(depth_q):
                    sides_j = pl[:, j]
                    A_chunk[cl_st:cl_en,
                             j*_ROWS_PER_SPLIT:(j+1)*_ROWS_PER_SPLIT, :] = qs['hs_A'][j, sides_j]
                    b_chunk[cl_st:cl_en,
                             j*_ROWS_PER_SPLIT:(j+1)*_ROWS_PER_SPLIT] = qs['hs_b'][j, sides_j]
                    d_chunk[cl_st:cl_en] += qs['d_corr'][j, sides_j]
                    c0_chunk[cl_st:cl_en] += qs['c0_corr'][j, sides_j]
                lam_chunk[cl_st:cl_en, :m_q] = qs['open_lam'][local_st:local_en]
                contributing.append((qi, cl_st, cl_en))

            alive = torch.ones(cs, dtype=torch.bool, device=device)
            _, lam_chunk_out, reason_chunk, witness_chunk = _batched_dual_ascent(
                d_chunk, c0_chunk, A_chunk, b_chunk, lam_chunk, alive,
                e_lb, e_hi, width,
                max_iter=max_iter, repair_steps=repair_steps,
                feas_tol=feas_tol, tol=_TOL, dtype=dtype, device=device)
            reason_all[chunk_start:chunk_end] = reason_chunk
            lam_out_all[chunk_start:chunk_end] = lam_chunk_out

            # Capture primal_unsafe witnesses per query within this chunk.
            if witness_check_fn is not None:
                for qi, cl_st, cl_en in contributing:
                    seg_reason = reason_chunk[cl_st:cl_en]
                    w_mask = (seg_reason == 1)
                    if not w_mask.any():
                        continue
                    seg_witness = witness_chunk[cl_st:cl_en][w_mask][:, :n_input].cpu().numpy()
                    seg_d = d_chunk[cl_st:cl_en][w_mask].float()
                    seg_c0 = c0_chunk[cl_st:cl_en][w_mask].float()
                    seg_w_full = witness_chunk[cl_st:cl_en][w_mask].float()
                    seg_p = (seg_c0 + (seg_d * seg_w_full).sum(-1)).cpu().numpy()
                    ws, ps = layer_witness_per_q[qi]
                    ws.append(seg_witness)
                    ps.append(seg_p)

            del A_chunk, b_chunk, d_chunk, c0_chunk, lam_chunk
        if device.type == 'cuda':
            torch.cuda.synchronize()
        if timed_out_mid:
            break

        # Split back per query and expand frontier.
        for qi, st, en, depth_q, m_q, B_q in slices:
            qs = Q_state[qi]
            reason_q = reason_all[st:en]
            lam_out_q = lam_out_all[st:en, :m_q]
            r_np = reason_q.cpu().numpy()
            qs['exit_counts']['dual_safe'] += int((r_np == 0).sum())
            qs['exit_counts']['primal_unsafe'] += int((r_np == 1).sum())
            qs['exit_counts']['no_progress'] += int((r_np == 2).sum())
            qs['exit_counts']['safety_cap'] += int((r_np == 3).sum())
            qs['n_total'] += B_q
            qs['max_depth_seen'] = max(qs['max_depth_seen'], depth_q)

            if witness_check_fn is not None:
                ws, ps = layer_witness_per_q[qi]
                if ws:
                    qs['witness_inputs'].extend(ws)
                    qs['witness_p'].extend(ps)

            # Frontier expansion for non-closed survivors.
            must_split = (reason_q != 0) & (reason_q != -1)
            survivors_idx = must_split.nonzero(as_tuple=True)[0]
            ns = int(survivors_idx.shape[0])
            survivor_paths = qs['open_paths'][survivors_idx]
            survivor_lams = lam_out_q[survivors_idx]
            new_paths = torch.zeros(ns * 2, depth_q + 1, device=device, dtype=torch.int8)
            new_paths[0::2, :depth_q] = survivor_paths; new_paths[0::2, depth_q] = 0
            new_paths[1::2, :depth_q] = survivor_paths; new_paths[1::2, depth_q] = 1
            new_lams = torch.zeros(ns * 2, m_q + _ROWS_PER_SPLIT,
                                    device=device, dtype=dtype)
            new_lams[0::2, :m_q] = survivor_lams
            new_lams[1::2, :m_q] = survivor_lams
            qs['open_paths'] = new_paths
            qs['open_lam'] = new_lams
            qs['depth'] += 1

        # Witness attack check (per-query, after the layer).
        if witness_check_fn is not None:
            for qi in active:
                qs = Q_state[qi]
                if qs['verdict'] is not None:
                    continue
                ws = qs.pop('witness_inputs')
                ps = qs.pop('witness_p')
                qs['witness_inputs'] = []
                qs['witness_p'] = []
                if ws:
                    w_all = np.concatenate(ws, axis=0)
                    K_WITNESS = 5
                    if ps and len(w_all) > K_WITNESS:
                        p_all = np.concatenate(ps, axis=0)
                        top_idx = np.argpartition(p_all, K_WITNESS)[:K_WITNESS]
                        w_all = w_all[top_idx]
                    cex = witness_check_fn(w_all)
                    if cex is not None:
                        qs['verdict'] = 'sat'
                        qs['sat_witness'] = cex
                        qs['sat_witness_depth'] = qs['depth'] - 1
                        qs['open_paths'] = None

        # Close queries whose frontier emptied or whose depth exhausted splits.
        for qi in active:
            qs = Q_state[qi]
            if qs['verdict'] is not None:
                continue
            if qs['open_paths'] is None or qs['open_paths'].shape[0] == 0:
                qs['verdict'] = 'unsat'
            elif qs['depth'] >= qs['n_splits']:
                qs['verdict'] = 'unknown'
                qs['reason'] = (f'exhausted all {qs["n_splits"]} splits '
                                 f'with {qs["open_paths"].shape[0]} open nodes')

    # Build results (any still-active query becomes unknown with TIMEOUT).
    elapsed = time.perf_counter() - t_start
    results = []
    for qi in range(Q):
        qs = Q_state[qi]
        if 'd_t' not in qs:
            # Pure root-LP path: info already complete.
            results.append((qs['verdict'], qs['info']))
            continue
        v = qs['verdict'] or 'unknown'
        final_open = (int(qs['open_paths'].shape[0])
                       if qs['open_paths'] is not None else 0)
        info = {'wall': elapsed, 'nodes': qs['n_total'],
                'max_depth_seen': qs['max_depth_seen'],
                'exit_counts': qs['exit_counts'],
                'final_open': final_open, 'root_g': qs['root_g']}
        if v == 'unknown':
            info['reason'] = qs.get('reason') or f'TIMEOUT at depth {qs["depth"]}'
        if qs.get('sat_witness') is not None:
            info['sat_witness'] = qs['sat_witness']
            info['sat_witness_depth'] = qs['sat_witness_depth']
        results.append((v, info))
    return results


__all__ = ['verify_query_dual_ascent_bab', 'verify_queries_batched']
