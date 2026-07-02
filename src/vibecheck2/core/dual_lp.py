"""Dual-ascent LP leaf certifier (design 3.1 dual_lp): the vc2 adapter for
v1's `fast_dual_ascent` GPU-compiled BaB verifier (log-bucket line search,
far-probe infeasibility cert, torch.compile-fused kernels; measured ~50x
over the Gurobi-backed racing in v1, hundreds of thousands to millions of
node bounds per second compiled).

The verifier consumes the alpha-zonotope LP state of one query: the query
value as c0 + d . e over generator coordinates e in [-1,1]^n, plus per
unstable-relu substitution data (pre-activation row, band coefficients,
fresh column). vc2's forward zonotope already carries exactly this
(ZonoState with symbol provenance); `build_state` snapshots it into the v1
schema.

COLUMN ORDER SOUNDNESS: v1's parser boxes only the first `n_input` columns
and the relu fresh columns to [-1,1] and PINS everything else to 0. Any
non-relu fresh generator (sigmoid/tanh bands, bilinear boxes) is therefore
moved INTO the leading free block, and `n_input` covers the whole block.
"""
from __future__ import annotations

import numpy as np
import torch

from . import forward as fwd


def build_state_backward(net, lo, hi, inter, slopes=None, device='cpu'):
    """v1 reverse_g ported onto the vc2 IR: the alpha-zono LP state built by
    BACKWARD passes (one seeded at each relu layer's unstable neurons, one at
    the output), with NO forward zonotope. Memory-bounded: only unstable
    rows materialize, seeds chunk through the memory service, and every
    layout detail lives in LinMap.lin_t (patches-ready by construction).

    The walk is the SLOPE-LINEAR adjoint (y = lam*z + mu + mu*e_new at each
    relu: scale by lam, deposit mu on the fresh column), NOT the sign-split
    CROWN planes; `slopes` optionally overrides lam per relu (any value in
    [0,1] is sound; default DeepZ h/(h-l)).
    """
    import scipy.sparse as sp

    from . import memory
    from .relax import REL
    dev = torch.device(device)
    dt = torch.float32
    lo2 = lo.reshape(1, -1).to(dev, dt)
    hi2 = hi.reshape(1, -1).to(dev, dt)
    n_in = net.n_in
    radii = ((hi2 - lo2) / 2)[0]

    # per-relu band coefficients from the (refined) bounds
    relu_ops = [nm for nm in net.order
                if net.ops[nm].kind == 'nonlin' and net.ops[nm].fn == 'relu']
    lam_L, mu_L, ust_L, pre_lh = {}, {}, {}, {}
    col = n_in
    e_col = {}
    for nm in relu_ops:
        l, h = inter[nm]
        l, h = l[0].to(dev, dt), h[0].to(dev, dt)
        if slopes and nm in slopes:
            a = slopes[nm].reshape(-1).to(dev, dt).clamp(0.0, 1.0)
            lam = torch.where(l >= 0, torch.ones_like(l),
                              torch.where(h <= 0, torch.zeros_like(l), a))
            mu = torch.where((l < 0) & (h > 0),
                             torch.maximum((1 - lam) * h, -lam * l) / 2,
                             torch.zeros_like(l))
        else:
            lam, mu, _d = REL['relu'].band(l.unsqueeze(0), h.unsqueeze(0))
            lam, mu = lam[0], mu[0]
        lam_L[nm], mu_L[nm], pre_lh[nm] = lam, mu, (l, h)
        u = torch.nonzero((l < 0) & (h > 0), as_tuple=False).flatten()
        ust_L[nm] = u
        for j in u.tolist():
            e_col[(nm, j)] = col
            col += 1
    n_gens = col

    # centers by one slope-linear forward point pass (relu -> lam*z + mu)
    center = {net.input_name: ((lo2 + hi2) / 2)[0]}
    pre_center = {}
    for name in net.order:
        op = net.ops[name]
        if op.kind == 'linmap':
            center[name] = op.lm.point(center[op.inputs[0]].unsqueeze(0))[0]
        elif op.kind == 'nonlin' and op.fn == 'relu':
            z = center[op.inputs[0]]
            pre_center[name] = z
            center[name] = lam_L[name] * z + mu_L[name]
        elif op.kind == 'add':
            center[name] = center[op.inputs[0]] + center[op.inputs[1]]
        elif op.kind == 'concat':
            out = torch.as_tensor(op.params['base'], device=dev,
                                  dtype=dt).clone()
            for src, pos in zip(op.inputs, op.params['positions']):
                out[torch.as_tensor(pos, device=dev)] = center[src]
            center[name] = out
        else:
            raise NotImplementedError(
                f'state_backward center: {op.kind}/{op.fn} (relu nets only)')

    def backward_rows(seed_edge, seed_idx, self_relu):
        """(len(seed_idx), n_gens) generator rows of the seeded neurons."""
        ns = len(seed_idx)
        rowG = torch.zeros(ns, n_gens, device=dev, dtype=dt)
        sens = {seed_edge: torch.zeros(ns, net.ops[seed_edge].n,
                                       device=dev, dtype=dt)}
        sens[seed_edge][torch.arange(ns, device=dev),
                        torch.as_tensor(seed_idx, device=dev)] = 1.0
        for name in reversed(net.order):
            if name not in sens:
                continue
            sx = sens.pop(name)
            op = net.ops[name]
            if op.kind == 'linmap':
                add = op.lm.lin_t(sx)
            elif op.kind == 'nonlin' and op.fn == 'relu':
                if name != self_relu:
                    u = ust_L[name]
                    if u.numel():
                        cols = torch.as_tensor(
                            [e_col[(name, int(j))] for j in u.tolist()],
                            device=dev)
                        rowG[:, cols] += sx[:, u] * mu_L[name][u].unsqueeze(0)
                    sx = sx * lam_L[name].unsqueeze(0)
                add = sx
            elif op.kind == 'add':
                sens[op.inputs[1]] = sens.get(op.inputs[1], 0) + sx
                add = sx
            elif op.kind == 'concat':
                for src, pos in zip(op.inputs, op.params['positions']):
                    p = torch.as_tensor(pos, device=dev)
                    sens[src] = sens.get(src, 0) + sx[:, p]
                continue
            else:
                raise NotImplementedError(
                    f'state_backward: {op.kind}/{op.fn}')
            sens[op.inputs[0]] = sens.get(op.inputs[0], 0) + add
        s_in = sens.get(net.input_name)
        if s_in is not None:
            rowG[:, :n_in] += s_in * radii.unsqueeze(0)
        return rowG

    widest = max(net.ops[o].n for o in net.order)
    per_row = widest * 4 * 6
    unstable_list = []
    for nm in relu_ops:
        u = ust_L[nm]
        if not u.numel():
            continue
        pre_edge = net.ops[nm].inputs[0]
        rows_out = []

        def take(sel, _pre=pre_edge, _nm=nm, _acc=rows_out):
            _acc.append(backward_rows(_pre, sel.tolist(), _nm))

        memory.chunked_indices(take, u, per_row)
        rowG = torch.cat(rows_out).cpu().numpy()
        for i, j in enumerate(u.tolist()):
            nz = np.nonzero(rowG[i])[0]
            unstable_list.append({
                'layer_idx': nm, 'neuron_idx': int(j),
                'lam': float(lam_L[nm][j]), 'mu': float(mu_L[nm][j]),
                'c_in': float(pre_center[nm][j]),
                'e_new_col': e_col[(nm, int(j))],
                'row_indices': nz.tolist(),
                'row_values': rowG[i, nz].astype(np.float64).tolist(),
            })
    obj_rows = []

    def take_out(sel, _acc=obj_rows):
        _acc.append(backward_rows(net.output_name, sel.tolist(), None))

    memory.chunked_indices(take_out, torch.arange(net.n_out, device=dev),
                           per_row)
    obj_G = torch.cat(obj_rows).cpu().numpy()
    state = {
        'n_gens': int(n_gens), 'n_input': int(n_in),
        'unstable_list': unstable_list,
        'obj_G_out_csr': sp.csr_matrix(obj_G.astype(np.float64)),
        'obj_c_out': center[net.output_name].cpu().numpy().astype(np.float64),
    }
    keys = [(u['layer_idx'], u['neuron_idx']) for u in unstable_list]
    return state, keys


def build_state(net, lo, hi, inter=None, slopes=None):
    """One recorded zonotope pass -> the v1 gen-state dict (single box).
    `inter` (CROWN-refined pre-activation bounds) clamps every band, which
    is what makes the LP state competitive with v1's tightened states.

    Returns (state, scored_key_universe) where the universe lists every
    splittable (relu_name, neuron) key present in the state.
    """
    import scipy.sparse as sp
    record = {}
    clamp = None
    if inter is not None:
        clamp = {k: (v[0], v[1]) for k, v in inter.items()
                 if len(v) == 2}
    _lo, _hi, zstate = fwd.zono(net, lo, hi, return_state=True,
                                record=record, clamp_bounds=clamp,
                                slope_override=slopes)
    out = zstate[net.output_name]
    final_sym = out.sym
    n_gens = len(final_sym)

    # permute columns: all always-free generators (input + non-relu bands)
    # first, relu fresh columns last (see module docstring)
    relu_names = set(record)
    free_cols = [i for i, s in enumerate(final_sym) if s[0] not in relu_names]
    relu_cols = [i for i, s in enumerate(final_sym) if s[0] in relu_names]
    perm = free_cols + relu_cols
    colmap = {final_sym[i]: k for k, i in enumerate(perm)}
    n_input = len(free_cols)

    G_out = out.G[0].cpu().numpy()[:, perm]
    obj_c_out = out.c[0].cpu().numpy().astype(np.float64)

    unstable_list = []
    for name, rec in record.items():
        c_pre = rec['c_pre'][0].cpu().numpy()
        G_pre = rec['G_pre'][0].cpu().numpy()
        lam = rec['lam'][0].cpu().numpy()
        mu = rec['mu'][0].cpu().numpy()
        # columns of this snapshot in final coordinates
        cols = np.array([colmap[s] for s in rec['sym']], dtype=np.int64)
        fresh = [(j, colmap.get((name, j))) for j in range(c_pre.shape[0])]
        for j, col in fresh:
            if col is None:
                continue                      # stable neuron: no fresh col
            row = G_pre[j]
            nz = np.nonzero(row)[0]
            unstable_list.append({
                'layer_idx': name, 'neuron_idx': int(j),
                'lam': float(lam[j]), 'mu': float(mu[j]),
                'c_in': float(c_pre[j]), 'e_new_col': int(col),
                'row_indices': cols[nz].tolist(),
                'row_values': row[nz].astype(np.float64).tolist(),
            })
    state = {
        'n_gens': int(n_gens), 'n_input': int(n_input),
        'unstable_list': unstable_list,
        'obj_G_out_csr': sp.csr_matrix(G_out.astype(np.float64)),
        'obj_c_out': obj_c_out,
    }
    keys = [(u['layer_idx'], u['neuron_idx']) for u in unstable_list]
    return state, keys


_VERIFIER = {}


def _verifier(device):
    """One compiled Verifier per device (kernel warm-up is reused)."""
    if device not in _VERIFIER:
        from vibecheck.fast_dual_ascent import Verifier
        _VERIFIER[device] = Verifier(device=device,
                                     compile=(torch.device(device).type
                                              == 'cuda'))
    return _VERIFIER[device]


def score_keys(net, lo, hi, W_open, inter, keys):
    """BaBSR split order for the state's keys: |pre-activation adjoint| x
    triangle intercept from one collected-adjoint crown pass, descending."""
    from . import backward
    adj = {}
    backward.crown(net, lo, hi, W_open, inter, collect_adjoints=adj)
    scores = {}
    for name, j in keys:
        l, h = inter[name]
        icpt = float((-h[0, j] * l[0, j]
                      / max(float(h[0, j] - l[0, j]), 1e-30)))
        a = adj.get(name)
        w = float(a[0, :, j].abs().max()) if a is not None else 1.0
        scores[(name, j)] = w * max(icpt, 0.0)
    return sorted(keys, key=lambda k: -scores[k])


def certify_queries(net, spec, W, bias, disj_idx, lo, hi, inter, open_d,
                    deadline, device='cpu', log=lambda m: None):
    """Refute the still-open disjuncts with the dual-ascent BaB, one query
    row at a time (sibling rows of the disjunct join as extra halfspaces).
    Returns the set of disjuncts refuted."""
    import time
    from . import backward
    # per-edge CROWN refinement of the pre-activation bounds first: the LP
    # state's bands inherit them, which is what makes the dual competitive
    # with v1's tightened states
    inter = backward.intermediates_crown(net, lo, hi, base_inter=inter)
    # per-query direction-adaptive slopes (v1 build_dir_adaptive_alpha):
    # per neuron, the OPTIMIZED alpha where the query's adjoint ew > 0
    # (lower plane binds) and the chord slope h/(h-l) where ew <= 0, so the
    # single-slope state reproduces the backward alpha-CROWN bound
    open_rows = [r for d in open_d
                 for r in torch.nonzero(disj_idx == d,
                                        as_tuple=False).flatten().tolist()]
    W_open = W[open_rows]
    _lb, alpha = backward.alpha_crown(net, lo, hi, W_open, inter,
                                      iters=60, thresholds=-bias[open_rows],
                                      return_alpha=True)
    adj = {}
    backward.crown(net, lo, hi, W_open, inter, alpha=alpha,
                   collect_adjoints=adj)
    row_pos = {r: i for i, r in enumerate(open_rows)}

    def dir_adaptive_slopes(row_i):
        slopes = {}
        for nm, a in alpha.items():
            if a.dim() != 3 or nm not in adj:
                continue                       # relu alphas only
            l, h = inter[nm]
            chord = (h[0] / (h[0] - l[0]).clamp_min(1e-30)).clamp(0.0, 1.0)
            ew = adj[nm][0, row_i]
            slopes[nm] = torch.where(ew > 0, a[0, row_i], chord)
        return slopes
    refuted = set()
    dev = str(torch.device(device))
    ver = _verifier(dev)
    state_cache = {}
    gamma_inter = {}
    for d in open_d:
        rows = torch.nonzero(disj_idx == d, as_tuple=False).flatten().tolist()
        left = deadline - time.time()
        if left <= 1.0:
            break
        per_q = max(2.0, left / max(1, len(open_d)) / max(1, len(rows)))
        for r in rows:
            qw = W[r].cpu().numpy()
            qb = float(bias[r])
            # NOTE: naively reusing CROWN alphas as zonotope slopes makes
            # the state LOOSER (measured: img96 dual went unsat 3.5s ->
            # frontier OOM); DeepZ slopes + refined bounds win. State comes
            # from the BACKWARD builder (v1 reverse_g port): no forward
            # zonotope, unstable rows only, LinMap-generic.
            if r not in state_cache:
                sl = dir_adaptive_slopes(row_pos[r])
                try:
                    state_cache[r] = build_state_backward(
                        net, lo, hi, inter, device=device, slopes=sl)
                except NotImplementedError:
                    # nets with non-slope-linear ops (mul, sigmoid): the
                    # forward-recorded builder handles them as free-block
                    # generators; slopes still apply to the relus
                    state_cache[r] = build_state(
                        net, lo, hi, inter=inter,
                        slopes={k: v.unsqueeze(0) for k, v in sl.items()})
            state, keys = state_cache[r]
            if not keys:
                continue
            sk = score_keys(net, lo, hi, W[r:r + 1], inter, keys)
            extra = [(W[r2].cpu().numpy(), float(bias[r2]))
                     for r2 in rows if r2 != r]
            verdict, info = ver.verify_query(
                state, qw, qb, sk, time_limit=min(per_q,
                                                  deadline - time.time()),
                extra_hs=extra)
            if (verdict != 'unsat'
                    and info.get('reason') != 'splits_exhausted'
                    and deadline - time.time() > 5.0):
                # gamma retry: refine THIS disjunct's intermediates under
                # its own output rows (INVPROP; conditional on the CE
                # region, so scoped strictly to this disjunct) and rerun
                if d not in gamma_inter:
                    Wg = W[rows].cpu().numpy()
                    bg = bias[rows].cpu().numpy()
                    gamma_inter[d] = backward.intermediates_crown(
                        net, lo, hi, base_inter=inter,
                        gamma_rows=(Wg, bg))
                g_state, g_keys = build_state_backward(
                    net, lo, hi, gamma_inter[d], device=device,
                    slopes=dir_adaptive_slopes(row_pos[r]))
                sk2 = score_keys(net, lo, hi, W[r:r + 1], gamma_inter[d],
                                 g_keys)
                verdict, info = ver.verify_query(
                    g_state, qw, qb, sk2,
                    time_limit=min(per_q, deadline - time.time()),
                    extra_hs=extra)
                log(f'[vc2/dual]   gamma retry: {verdict} '
                    f'nodes={info.get("nodes")} '
                    f'wall={info.get("wall", 0):.2f}s')
            log(f'[vc2/dual] disj {d} row {r}: {verdict} '
                f'nodes={info.get("nodes")} wall={info.get("wall", 0):.2f}s '
                f'reason={info.get("reason", "-")} open={info.get("open", 0)}')
            if verdict == 'unsat':
                refuted.add(d)
                break
            if info.get('reason') == 'splits_exhausted':
                # every relu split used and the frontier is still open: the
                # slack lives in unsplittable (free-block) generators, so no
                # other row of this state can close either -- hand the time
                # back to the outer BaB (which shrinks those generators)
                log('[vc2/dual] splits exhausted; state too loose, bailing')
                return refuted
    return refuted
