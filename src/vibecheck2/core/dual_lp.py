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


def build_state(net, lo, hi, inter=None):
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
                                record=record, clamp_bounds=clamp)
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
    state, keys = build_state(net, lo, hi, inter=inter)
    if not keys:
        return set()
    refuted = set()
    dev = str(torch.device(device))
    ver = _verifier(dev)
    for d in open_d:
        rows = torch.nonzero(disj_idx == d, as_tuple=False).flatten().tolist()
        left = deadline - time.time()
        if left <= 1.0:
            break
        per_q = max(2.0, left / max(1, len(open_d)) / max(1, len(rows)))
        for r in rows:
            qw = W[r].cpu().numpy()
            qb = float(bias[r])
            sk = score_keys(net, lo, hi, W[r:r + 1], inter, keys)
            extra = [(W[r2].cpu().numpy(), float(bias[r2]))
                     for r2 in rows if r2 != r]
            verdict, info = ver.verify_query(
                state, qw, qb, sk, time_limit=min(per_q,
                                                  deadline - time.time()),
                extra_hs=extra)
            log(f'[vc2/dual] disj {d} row {r}: {verdict} '
                f'nodes={info.get("nodes")} wall={info.get("wall", 0):.2f}s '
                f'reason={info.get("reason", "-")} open={info.get("open", 0)}')
            if verdict == 'unsat':
                refuted.add(d)
                break
    return refuted
