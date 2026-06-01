"""Hybrid α-CROWN BaB pipeline for ACASXU. 186/186 on the regular track.

Pipeline:
  1. Root PGD: simple sign-gradient PGD (10K restarts × 50 iters) over the
     input box. Multi-disjunct DNF aware. ~100 ms on RTX 3080; catches 42/47
     SAT cases by itself.
  2. Initial fan-out: split root box into INIT_LEAVES=32 by widest dim
     (depth-5 splits).
  3. Batched 32-leaf α-CROWN freeze: per-query α-CROWN (Q=2·n_layer at each
     ReLU layer; Q=Q_spec for the spec). 100 Adam iters with sum-based
     early-stop. Stores per-leaf α tensors as "α groups" for replay.
  4. BaB loop: pop batch (up to 4096), grouped by α-group. For each group:
       a. Forward zonotope (min-area parallelogram) → tight bounds per layer
       b. CROWN backward with FROZEN layer α — intersect with forward zono
       c. Per-leaf spec α-opt (10 Adam iters, group α as warmstart)
     Close leaves with worst-disjunct best spec_lb > 0. Open leaves split
     on widest dim; children inherit parent's α group.
  5. Between BaB rounds: every iter, take K=5 worst-spec-lb leaves and run
     simple PGD (1K restarts × 50 iters) on each. Catches narrow SAT
     witnesses the root PGD misses (1_5/1_6/3_2/5_3 prop_2).
  6. Stop on empty worklist (UNSAT) or timeout (UNKNOWN).

Returns dict with verdict, time, telemetry. See
`docs/benchmarks/acasxu_2023.md` for the design rationale.
"""
import os, time, numpy as np, torch
from collections import defaultdict

from vibecheck.network import ComputeGraph
from vibecheck.vnnlib_loader import load_vnnlib
from vibecheck.verify_zono_bnb import (
    _forward_zonotope_graph_batched, _spec_backward_graph_batched,
    _forward_batch_graph, _make_slopes,
)
from vibecheck.pgd import pgd_attack_general
from vibecheck.settings import default_settings


def _simple_pgd(xl, xh, spec, gg, n_output, device, dtype,
                  n_restarts=10000, n_iter=50, lr=0.1, seed=0,
                  min_restarts=64):
    """Simple sign-gradient PGD over the input box. Multi-disjunct DNF:
    SAT iff ∃ disjunct d such that all its constraints' margins ≤ 0.
    Per-restart loss = min_d(max_c margin_dc); SAT if min ≤ 0.

    On `torch.cuda.OutOfMemoryError` the call is retried with restarts
    halved (down to `min_restarts`). Larger models like cGAN imgSz64
    require fewer per-restart samples to fit in GPU memory; without
    this halving we silently miss every imgSz64 SAT case.

    Returns (sat: bool, witness: np.ndarray or None)."""
    xl0 = xl.flatten(); xh0 = xh.flatten()
    n_in = xl0.numel()
    Ws = []; bs = []
    for conj in spec.disjuncts:
        n_c = len(conj.constraints)
        W = torch.zeros(n_c, n_output, dtype=dtype, device=device)
        b = torch.zeros(n_c, dtype=dtype, device=device)
        for i, c in enumerate(conj.constraints):
            if hasattr(c, 'pred'):
                W[i, c.pred] = 1.0; W[i, c.comp] = -1.0
            elif c.op == '>=':
                W[i, c.index] = -1.0; b[i] = float(c.value)
            elif c.op == '<=':
                W[i, c.index] = 1.0; b[i] = -float(c.value)
            else:
                return False, None
        Ws.append(W); bs.append(b)
    cur_restarts = n_restarts
    while cur_restarts >= min_restarts:
        try:
            torch.manual_seed(seed)
            x = xl0 + (xh0 - xl0) * torch.rand(cur_restarts, n_in, device=device, dtype=dtype)
            width = xh0 - xl0
            x = x.detach().requires_grad_(True)
            for _ in range(n_iter):
                out = _forward_batch_graph(x, gg)
                max_per_disj = []
                for W, b in zip(Ws, bs):
                    m = out @ W.T + b
                    max_per_disj.append(m.max(dim=1).values)
                stacked = torch.stack(max_per_disj, dim=1)
                min_m, _ = stacked.min(dim=1)
                with torch.no_grad():
                    sat_mask = min_m <= 1e-6
                    if sat_mask.any():
                        idx = int(sat_mask.nonzero()[0].item())
                        return True, x[idx].detach().cpu().numpy()
                loss = min_m.sum()
                loss.backward()
                with torch.no_grad():
                    step = lr * x.grad.sign() * width
                    x = (x - step).clamp(min=xl0, max=xh0)
                x = x.detach().requires_grad_(True)
            return False, None
        except torch.cuda.OutOfMemoryError:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            cur_restarts //= 2
    return False, None


def _simple_pgd_batched(xl, xh, spec, gg, n_output, device, dtype,
                          n_restarts=128, n_iter=50, lr=0.1, seed=0):
    """Sign-gradient PGD over MANY distinct input boxes at once.

    `xl`, `xh`: (B, n_in) — one box per leaf. Samples `n_restarts` points
    INSIDE each box (each sample clamped to its own box), so it finds a narrow
    SAT witness that fills a large fraction of a deep input-split leaf — exactly
    the case `_simple_pgd` on the wide root box misses. SAT iff some sample
    satisfies some disjunct (all its constraints' margins <= 0). Mirrors
    `_simple_pgd`'s loss but vectorised across leaves.

    Returns (sat: bool, witness: np.ndarray or None).
    """
    B, n_in = xl.shape
    Ws = []; bs = []
    for conj in spec.disjuncts:
        n_c = len(conj.constraints)
        W = torch.zeros(n_c, n_output, dtype=dtype, device=device)
        b = torch.zeros(n_c, dtype=dtype, device=device)
        for i, c in enumerate(conj.constraints):
            if hasattr(c, 'pred'):
                W[i, c.pred] = 1.0; W[i, c.comp] = -1.0
            elif c.op == '>=':
                W[i, c.index] = -1.0; b[i] = float(c.value)
            elif c.op == '<=':
                W[i, c.index] = 1.0; b[i] = -float(c.value)
            else:
                return False, None
        Ws.append(W); bs.append(b)
    torch.manual_seed(seed)
    R = n_restarts
    # (B, R, n_in) samples, then flattened to (B*R, n_in) for the graph forward.
    lo = xl.unsqueeze(1).expand(B, R, n_in).reshape(B * R, n_in)
    hi = xh.unsqueeze(1).expand(B, R, n_in).reshape(B * R, n_in)
    x = lo + (hi - lo) * torch.rand(B * R, n_in, device=device, dtype=dtype)
    x = x.detach().requires_grad_(True)
    width = hi - lo
    for _ in range(n_iter):
        out = _forward_batch_graph(x, gg)  # (B*R, n_out)
        max_per_disj = []
        for W, b in zip(Ws, bs):
            m = out @ W.T + b
            max_per_disj.append(m.max(dim=1).values)
        stacked = torch.stack(max_per_disj, dim=1)
        min_m, _ = stacked.min(dim=1)
        with torch.no_grad():
            sat_mask = min_m <= 1e-6
            if sat_mask.any():
                idx = int(sat_mask.nonzero()[0].item())
                return True, x[idx].detach().cpu().numpy()
        loss = min_m.sum()
        loss.backward()
        with torch.no_grad():
            step = lr * x.grad.sign() * width
            x = torch.maximum(torch.minimum(x - step, hi), lo)
        x = x.detach().requires_grad_(True)
    return False, None


def _full_freeze(xl, xh, gg, spec_ew, device, dtype,
                 n_iters_per_layer=100, n_iters_spec=100, lr=0.25,
                 deadline=None):
    """Per-query α-CROWN layerwise tighten + spec α-CROWN.
    Returns (tight, alpha_per_layer, alpha_spec, best_spec).
    Per-leaf α (each leaf in batch B gets independent α tensors)."""
    B = xl.shape[0]
    sb_init, _ = _forward_zonotope_graph_batched(xl, xh, gg, device, dtype)
    tight = {L: (lo.clone(), hi.clone()) for L, (lo, hi) in sb_init.items()}
    layer_order = sorted(tight.keys())
    relu_op_by_L = {op['layer_idx']: op for op in gg['ops']
                     if op['type'] == 'relu' and 'layer_idx' in op}
    alpha_per_layer = {}
    for L in layer_order:
        # Respect the budget: if the freeze itself would overrun, stop tightening
        # (remaining layers keep their sound forward-zono bounds). Without this
        # the freeze ignored the timeout and overran to 150-180s.
        if deadline is not None and time.perf_counter() > deadline:
            alpha_per_layer[L] = {}
            continue
        feed = relu_op_by_L[L]['inputs'][0]
        n_layer = tight[L][0].shape[1]
        Q = 2 * n_layer
        I = torch.eye(n_layer, dtype=dtype, device=device)
        W_q = torch.cat([I, -I], dim=0)
        seed_ew = {feed: W_q.unsqueeze(0).expand(B, -1, -1)}
        seed_acc = torch.zeros(B, Q, dtype=dtype, device=device)
        alpha_at = {}
        for k in layer_order:
            if k >= L: break
            lo_k, hi_k = tight[k]
            _, up_s_k, _, _, _, unstable_k = _make_slopes(lo_k, hi_k)
            init = ((up_s_k > 0.5).to(dtype) * unstable_k.to(dtype))
            init = init.unsqueeze(1).expand(-1, Q, -1).contiguous()
            alpha_at[k] = init.detach().clone().requires_grad_(True)
        if alpha_at:
            opt = torch.optim.Adam([alpha_at[k] for k in alpha_at], lr=lr)
            best = None; prev = -float('inf')
            for it in range(n_iters_per_layer):
                opt.zero_grad()
                sl = _spec_backward_graph_batched(
                    tight, xl, xh, gg, None, device, dtype,
                    alpha_at_layer=alpha_at,
                    seed_ew_at=seed_ew, seed_acc=seed_acc)
                with torch.no_grad():
                    best = sl.detach().clone() if best is None else torch.maximum(best, sl.detach())
                    cur = float(best.sum().item())
                loss = -sl.sum(); loss.backward(); opt.step()
                with torch.no_grad():
                    for k in alpha_at:
                        alpha_at[k].clamp_(0.0, 1.0)
                if it > 0 and cur - prev < 1e-6:
                    break
                prev = cur
                for gp in opt.param_groups:
                    gp['lr'] *= 0.98
            new_lo = best[:, :n_layer]; new_hi = -best[:, n_layer:]
            lo_t = torch.maximum(tight[L][0], new_lo)
            hi_t = torch.minimum(tight[L][1], new_hi)
            lo_t = torch.minimum(lo_t, hi_t)
            tight[L] = (lo_t, hi_t)
        alpha_per_layer[L] = {k: a.detach().clone() for k, a in alpha_at.items()}
    Q_spec = len(spec_ew)
    alpha_spec = {}
    for L in layer_order:
        lo, hi = tight[L]
        _, up_s, _, _, _, unstable = _make_slopes(lo, hi)
        init = ((up_s > 0.5).to(dtype) * unstable.to(dtype))
        init = init.unsqueeze(1).expand(-1, Q_spec, -1).contiguous()
        alpha_spec[L] = init.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([alpha_spec[L] for L in alpha_spec], lr=lr)
    best_spec = None; prev = -float('inf')
    for it in range(n_iters_spec):
        if (deadline is not None and it > 0
                and time.perf_counter() > deadline):
            break
        opt.zero_grad()
        sl = _spec_backward_graph_batched(
            tight, xl, xh, gg, spec_ew, device, dtype,
            alpha_at_layer=alpha_spec)
        with torch.no_grad():
            best_spec = sl.detach().clone() if best_spec is None else torch.maximum(best_spec, sl.detach())
            cur = float(best_spec.sum().item())
        loss = -sl.sum(); loss.backward(); opt.step()
        with torch.no_grad():
            for L in alpha_spec:
                alpha_spec[L].clamp_(0.0, 1.0)
        if it > 0 and cur - prev < 1e-6:
            break
        prev = cur
        for gp in opt.param_groups:
            gp['lr'] *= 0.98
    alpha_spec_frozen = {k: a.detach().clone() for k, a in alpha_spec.items()}
    return tight, alpha_per_layer, alpha_spec_frozen, best_spec.detach()


def _replay_batched(xl_b, xh_b, gg, spec_ew, alpha_layer_b, alpha_spec_b,
                     device, dtype, spec_iters=10, spec_lr=0.1):
    """Forward zono + frozen-α layer tighten + per-leaf spec α-opt.
    Layer α frozen, spec α warmstarted from group's frozen spec α then
    Adam-opt'd for `spec_iters` iters.
    Returns (best_spec_lbs, best_A)."""
    B = xl_b.shape[0]
    with torch.no_grad():
        sb, _ = _forward_zonotope_graph_batched(xl_b, xh_b, gg, device, dtype)
        tight = {L: (lo.clone(), hi.clone()) for L, (lo, hi) in sb.items()}
        layer_order = sorted(tight.keys())
        relu_op_by_L = {op['layer_idx']: op for op in gg['ops']
                         if op['type'] == 'relu' and 'layer_idx' in op}
        for L in layer_order:
            if not alpha_layer_b[L]:
                continue
            feed = relu_op_by_L[L]['inputs'][0]
            n_layer = tight[L][0].shape[1]
            Q = 2 * n_layer
            I = torch.eye(n_layer, dtype=dtype, device=device)
            W_q = torch.cat([I, -I], dim=0)
            seed_ew = {feed: W_q.unsqueeze(0).expand(B, -1, -1)}
            seed_acc = torch.zeros(B, Q, dtype=dtype, device=device)
            alpha_b = {k: a.expand(B, -1, -1).contiguous()
                        for k, a in alpha_layer_b[L].items()}
            sl = _spec_backward_graph_batched(
                tight, xl_b, xh_b, gg, None, device, dtype,
                alpha_at_layer=alpha_b,
                seed_ew_at=seed_ew, seed_acc=seed_acc)
            new_lo = sl[:, :n_layer]; new_hi = -sl[:, n_layer:]
            lo_t = torch.maximum(tight[L][0], new_lo)
            hi_t = torch.minimum(tight[L][1], new_hi)
            lo_t = torch.minimum(lo_t, hi_t)
            tight[L] = (lo_t, hi_t)
    # per-leaf spec α opt warmstart from frozen
    alpha_spec = {k: a.expand(B, -1, -1).contiguous().clone()
                       .detach().requires_grad_(True)
                     for k, a in alpha_spec_b.items()}
    opt = torch.optim.Adam([alpha_spec[k] for k in alpha_spec], lr=spec_lr)
    best = None; best_A = None
    for it in range(max(1, spec_iters)):
        opt.zero_grad()
        sl, A, acc = _spec_backward_graph_batched(
            tight, xl_b, xh_b, gg, spec_ew, device, dtype,
            alpha_at_layer=alpha_spec, return_input_linear=True)
        with torch.no_grad():
            if best is None:
                best = sl.detach().clone(); best_A = A.detach().clone()
            else:
                imp = sl.detach() > best
                best = torch.where(imp, sl.detach(), best)
                leaf_imp = imp.any(dim=1)
                if leaf_imp.any():
                    best_A[leaf_imp] = A.detach()[leaf_imp]
        loss = -sl.sum(); loss.backward(); opt.step()
        with torch.no_grad():
            for k in alpha_spec:
                alpha_spec[k].clamp_(0.0, 1.0)
    return best, best_A


def verify_hybrid(graph, spec, settings=None, timeout=120.0,
                    init_leaves=32, k_freeze=5, batch=4096, verbose=False,
                    pgd_between_every=1, pgd_between_k=5,
                    pgd_between_restarts=1000, pgd_between_iter=50):
    """Returns dict: {verdict, time, ...}. verdict in {'sat','unsat','unknown'}."""
    if settings is None:
        settings = default_settings()
    settings.total_timeout = timeout
    # Soundness probe: when sat-finding is disabled, skip ALL PGD (Phase 0 +
    # between-rounds). A SAT case must then come back 'unknown' (the BaB can't
    # verify it), never a false 'unsat'.
    _disable_sat = bool(getattr(settings, 'disable_sat_finding', False))
    # The between-rounds PGD is pure waste on UNSAT cases (139 of 186). Let the
    # config dial it down (acasxu: every-3 × 500 restarts → ~28% faster on hard
    # UNSAT cases, SAT still found). Same for the freeze depth.
    pgd_between_every = int(getattr(
        settings, 'hybrid_pgd_between_every', pgd_between_every))
    pgd_between_restarts = int(getattr(
        settings, 'hybrid_pgd_between_restarts', pgd_between_restarts))
    freeze_iters = int(getattr(settings, 'hybrid_freeze_iters', 100))

    t_start = time.perf_counter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    gg = graph.gpu_graph(device, dtype)

    n_output = None
    for op in reversed(gg['ops']):
        if op['type'] == 'fc':
            n_output = int(op['W'].shape[0]); break
    queries = spec.as_linear_queries(n_output)
    spec_ew = {qi: (torch.as_tensor(w, dtype=dtype, device=device).flatten(),
                     float(b))
                for qi, (di, w, b) in enumerate(queries)}
    disj_q_idx = defaultdict(list)
    for qi, (di, w, b) in enumerate(queries):
        disj_q_idx[di].append(qi)

    xl_root = torch.as_tensor(spec.x_lo, dtype=dtype, device=device).flatten()[None, :]
    xh_root = torch.as_tensor(spec.x_hi, dtype=dtype, device=device).flatten()[None, :]

    # --- Phase 0: root PGD (simple direct PGD, sign-gradient, 10K restarts) ---
    if bool(getattr(settings, 'pgd_phase0_enabled', True)) and not _disable_sat:
        sat, witness = _simple_pgd(
            xl_root, xh_root, spec, gg, n_output, device, dtype,
            n_restarts=10000, n_iter=50, lr=0.1)
        if sat:
            return {'verdict': 'sat', 'time': time.perf_counter() - t_start,
                     'phase': 'root_pgd', 'witness': witness}
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # (root α was wasted overhead — only fed an early-exit check that
    # doesn't fire on cases reaching this Phase 1 escalation. Dropped.)

    # --- Phase 2: initial fan-out + batched depth-K freeze ---
    boxes = [(xl_root, xh_root)]
    while len(boxes) < init_leaves:
        new = []
        for xl, xh in boxes:
            w = xh[0] - xl[0]
            d = int(w.argmax().item())
            mid = ((xl[0, d] + xh[0, d]) / 2).item()
            a_xh = xh.clone(); a_xh[0, d] = mid
            b_xl = xl.clone(); b_xl[0, d] = mid
            new.append((xl, a_xh)); new.append((b_xl, xh))
            if len(new) >= init_leaves: break
        boxes = new
    boxes = boxes[:init_leaves]
    xl_init = torch.stack([b[0][0] for b in boxes])
    xh_init = torch.stack([b[1][0] for b in boxes])
    try:
        tight_K, alpha_layer_K, alpha_spec_K, best_K = _full_freeze(
            xl_init, xh_init, gg, spec_ew, device, dtype,
            n_iters_per_layer=freeze_iters, n_iters_spec=freeze_iters,
            deadline=t_start + 0.8 * timeout)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {'verdict': 'unknown', 'time': time.perf_counter() - t_start,
                 'phase': 'freeze_oom'}
    # depth-K closure
    per_disj_K = []
    for di, qids in disj_q_idx.items():
        qs = sorted(qids)
        per_disj_K.append(best_K[:, qs].max(dim=1).values)
    per_disj_K = torch.stack(per_disj_K, dim=1)
    worst_K = per_disj_K.min(dim=1).values

    # α groups: one per init leaf
    alpha_groups = []
    for i in range(init_leaves):
        layer_i = {L: {k: a[i:i+1].contiguous()
                         for k, a in alpha_layer_K[L].items()}
                    for L in alpha_layer_K}
        spec_i = {k: a[i:i+1].contiguous()
                    for k, a in alpha_spec_K.items()}
        alpha_groups.append((layer_i, spec_i))

    open_idx_K = (worst_K <= 0).nonzero(as_tuple=True)[0].tolist()
    worklist = []
    for i in open_idx_K:
        worklist.append((xl_init[i], xh_init[i], i))
    n_evaluated = init_leaves
    n_closed = init_leaves - len(open_idx_K)
    if verbose:
        print(f'  depth-K closure: {n_closed}/{init_leaves}  '
              f'open={len(open_idx_K)}  '
              f'worst@K={float(worst_K.min()):+.4f}')

    # --- Phase 3: BaB loop ---
    cur_batch = batch
    iters = 0
    while worklist:
        if time.perf_counter() - t_start > timeout - 0.5:
            return {'verdict': 'unknown', 'time': time.perf_counter() - t_start,
                     'phase': 'bab_timeout', 'residual': len(worklist),
                     'evaluated': n_evaluated, 'closed': n_closed}
        take = min(cur_batch, len(worklist))
        batch_entries = worklist[-take:]
        del worklist[-take:]
        groups = defaultdict(list)
        for idx, (xl_, xh_, gid) in enumerate(batch_entries):
            groups[gid].append((idx, xl_, xh_))
        n_in = xl_root.shape[1]
        all_best = torch.empty(len(batch_entries), len(spec_ew),
                                 dtype=dtype, device=device)
        all_A = torch.empty(len(batch_entries), len(spec_ew),
                              n_in, dtype=dtype, device=device)
        try:
            for gid, entries in groups.items():
                idxs = [e[0] for e in entries]
                xl_g = torch.stack([e[1] for e in entries])
                xh_g = torch.stack([e[2] for e in entries])
                layer_g, spec_g = alpha_groups[gid]
                best_g, A_g = _replay_batched(
                    xl_g, xh_g, gg, spec_ew, layer_g, spec_g, device, dtype)
                idx_t = torch.tensor(idxs, device=device, dtype=torch.long)
                all_best[idx_t] = best_g
                all_A[idx_t] = A_g
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            worklist.extend(batch_entries)
            if cur_batch <= 8:
                return {'verdict': 'unknown', 'time': time.perf_counter() - t_start,
                         'phase': 'bab_oom'}
            cur_batch = max(8, cur_batch // 2)
            continue
        n_evaluated += len(batch_entries)
        if verbose and iters % 10 == 0:
            elapsed = time.perf_counter() - t_start
            print(f'    iter {iters} t={elapsed:.1f} wl={len(worklist)} '
                  f'eval={n_evaluated} closed={n_closed}')
        # closure
        per_disj_best = []
        for di, qids in disj_q_idx.items():
            qs = sorted(qids)
            per_disj_best.append(all_best[:, qs].max(dim=1).values)
        pdb = torch.stack(per_disj_best, dim=1)
        worst_per_leaf, _ = pdb.min(dim=1)
        closed_mask = worst_per_leaf > 0
        n_closed += int(closed_mask.sum().item())
        open_idx = (~closed_mask).nonzero(as_tuple=True)[0]

        # PGD between rounds — target top-K worst (most negative spec_lb)
        # open leaves. Cheap per call; SAT exit short-circuits everything.
        if (pgd_between_every > 0 and not _disable_sat
                and open_idx.numel() > 0
                and iters % pgd_between_every == 0):
            wv = worst_per_leaf[open_idx]
            k_atk = min(pgd_between_k, open_idx.numel())
            top_k = wv.argsort()[:k_atk]
            for j in top_k.tolist():
                j_global = int(open_idx[j].item())
                xl_a = batch_entries[j_global][0][None, :]
                xh_a = batch_entries[j_global][1][None, :]
                sat, witness = _simple_pgd(
                    xl_a, xh_a, spec, gg, n_output, device, dtype,
                    n_restarts=pgd_between_restarts, n_iter=pgd_between_iter)
                if sat:
                    return {'verdict': 'sat',
                             'time': time.perf_counter() - t_start,
                             'phase': 'between_pgd', 'witness': witness}

        if open_idx.numel() > 0:
            # split widest dim per-leaf
            xl_b = torch.stack([batch_entries[i][0] for i in open_idx.tolist()])
            xh_b = torch.stack([batch_entries[i][1] for i in open_idx.tolist()])
            widths = xh_b - xl_b
            split_dims = widths.argmax(dim=1)
            mids = (xl_b.gather(1, split_dims.unsqueeze(1)).squeeze(1)
                    + xh_b.gather(1, split_dims.unsqueeze(1)).squeeze(1)) / 2
            for j_local, j_global in enumerate(open_idx.tolist()):
                d = int(split_dims[j_local].item())
                m = float(mids[j_local].item())
                xl_p, xh_p, gid_p = batch_entries[j_global]
                a_xh = xh_p.clone(); a_xh[d] = m
                b_xl = xl_p.clone(); b_xl[d] = m
                worklist.append((xl_p, a_xh, gid_p))
                worklist.append((b_xl, xh_p, gid_p))
        iters += 1

    return {'verdict': 'unsat', 'time': time.perf_counter() - t_start,
             'phase': 'bab_complete',
             'evaluated': n_evaluated, 'closed': n_closed,
             'bab_iters': iters}


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    onnx_p = sys.argv[1]
    vnn_p = sys.argv[2]
    to = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0
    g = ComputeGraph.from_onnx(onnx_p, dtype=np.float32)
    spec = load_vnnlib(vnn_p)
    res = verify_hybrid(g, spec, timeout=to, verbose=True)
    print(res)
