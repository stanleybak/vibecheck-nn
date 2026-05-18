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

# ---------------------------------------------------------------------------
# Defaults — tuned on prop_4260 (TinyImageNet ResNet medium). K=1 wins on
# wall, K=2 matches Gurobi node count, K=5 gives tightest bounds. Increase
# REPAIR_STEPS to find primal witnesses faster on harder instances.
# ---------------------------------------------------------------------------
_DEFAULT_K = 1
_DEFAULT_REPAIR_STEPS = 5
_DEFAULT_MAX_DEPTH = 200  # let time_limit be the primary cap; 200 covers any realistic case
_DEFAULT_FRONTIER_CAP = 16384
# Chunk a layer's frontier into sub-batches sized so A_batch + d_path fits.
# Conservative target: 1 GB per chunk for A + d (bytes_per_elem * B * m * n
# + B * n). Auto-adjust below.
_DEFAULT_CHUNK_BYTES_BUDGET = 1.5 * 1024 * 1024 * 1024  # 1.5 GB
_TOL = 1e-9
_FEAS_TOL = 1e-5
_ROWS_PER_SPLIT = 2  # OFF: z≤0, z≥-2μ/λ.  ON: z≥0, z≤2μ/(1-λ).


def _build_substitution_caches(
    state, qw, qb, scored_keys, device, dtype, max_depth: int,
):
    """Pre-compute per-split caches for the substitution-form LP.

    For each of the first `max_depth` splits in `scored_keys`, populate:
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
        lam_batch = lam_batch + (eta_star.unsqueeze(-1) * s_proj) * step_mask
        lam_batch = lam_batch.clamp_min(0.0)
        rc = rc + (eta_star.unsqueeze(-1) * da) * step_mask

    reason = torch.where(
        (reason == -1) & alive_mask,
        torch.tensor(3, dtype=torch.int8, device=device), reason)
    return best_g, lam_batch, reason, witness


def verify_query_dual_ascent_bab(
    state, qw, qb, scored_keys, *,
    time_limit: float = 60.0,
    max_iter: int = _DEFAULT_K,
    repair_steps: int = _DEFAULT_REPAIR_STEPS,
    max_depth: int = _DEFAULT_MAX_DEPTH,
    frontier_cap: int = _DEFAULT_FRONTIER_CAP,
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
            (highest priority first). Length cap: max_depth.
        time_limit: wall clock budget in seconds (default 60).
        max_iter: hard iteration cap per BaB node (default 1 — fastest).
        repair_steps: greedy primal-repair steps per dual iter (default 5).
        max_depth: max BaB depth before declaring INCONCLUSIVE (default 40).
        frontier_cap: cap on layer-frontier size (default 16384). If exceeded,
            returns 'unknown' early.
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
            max_depth, frontier_cap, feas_tol, device, dtype,
            print_progress, time_left_fn, witness_check_fn)
    finally:
        torch.set_grad_enabled(_grad_was)


def _verify_dab_impl(state, qw, qb, scored_keys, time_limit, max_iter,
                      repair_steps, max_depth, frontier_cap, feas_tol,
                      device, dtype, print_progress, time_left_fn,
                      witness_check_fn):
    t_start = time.perf_counter()
    if time_left_fn is None:
        deadline = t_start + time_limit
        def _left():
            return deadline - time.perf_counter()
    else:
        _left = time_left_fn

    # Filter scored_keys to those actually present in the per-query state's
    # unstable list. (per-query rebuilds with tightened bounds can drop
    # neurons that became stable; original ranking may include them.)
    _state_keys = {(u['layer_idx'], u['neuron_idx']) for u in state['unstable_list']}
    scored_keys = [k for k in scored_keys if k in _state_keys]
    # max_depth removed in favor of time_limit. Use full scored_keys length.
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
    # Build substitution caches (one-time per query).
    (n_gens, e_lb, e_hi, width, d_t, c0, hs_A, hs_b, d_corr, c0_corr) = \
        _build_substitution_caches(state, qw, qb, scored_keys,
                                    device, dtype, max_depth)

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
        if B > frontier_cap:
            info_extra['reason'] = (f'frontier cap {frontier_cap} '
                                     f'exceeded at depth {depth}')
            break

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
        for chunk_start in range(0, B, chunk_size):
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
                info_extra['reason'] = (f'reached max_depth={n_splits} '
                                         f'with {final_open} open nodes')
            else:
                info_extra['reason'] = f'aborted at depth {depth}'

    info = {'wall': elapsed, 'nodes': n_total,
            'max_depth_seen': max_depth_seen, 'exit_counts': exit_counts,
            'final_open': final_open, 'root_g': g_root}
    info.update(info_extra)
    return verdict, info


__all__ = ['verify_query_dual_ascent_bab']
