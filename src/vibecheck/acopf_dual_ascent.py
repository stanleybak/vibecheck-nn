"""Dual-ascent nonlinear-split BaB for ml4acopf-style nets (sigmoid/tanh/sin/
cos/pow + bilinear physics).

Unlike the re-forward trig BaB (`verify_graph._verify_trig_nonlinear_split`,
which re-runs the whole zonotope forward per leaf), this builds the LP state
ONCE — the output zonotope objective `G_out` plus per-nonlinear-neuron geometry
captured during a single forward (`op_geom_out`) — then each BaB node is a cheap
dual-ascent solve over the fixed error-symbol box with band corrections +
pre-activation halfspaces (`nonlinear_split_dual` / `dual_ascent_bab`). No
re-forward, so it scales to many nodes.

Soundness: the root LP min in a spec direction equals the (verified) forward
zonotope's projection — sound by construction. Each split re-tightens one
neuron's band on a sub-interval (parent slope, guaranteed-tightening) and pins
its pre-activation with z≥clo & z≤chi; children cover the parent. The parent
band (λ,μ,δ) is the EXACT one the forward used (captured via
`zono_affine_transform(return_band=True)` / the pow α-band), so the dual
sensitivity g_k = d[e_new]/δ is exact. A disjunct (conjunction of unsafe
constraints) is refuted when some leaf query is always-positive; the spec is
unsat iff every disjunct is refuted. Never certifies a SAT instance (validated
on 14_ieee prop1).
"""
from __future__ import annotations
import time
from collections import defaultdict
import numpy as np
import torch

from .nonlinear_relax import REGISTRY
from .nl_pow import PowRelax
from .nonlinear_split_dual import (backward_sensitivity,
                                   band_change_correction, split_halfspace)
from . import dual_ascent_bab as dab

_SPLIT = ('pow', 'sigmoid', 'tanh', 'sin', 'cos')
_RELAX_NAME = {'sigmoid': 'Sigmoid', 'tanh': 'Tanh', 'sin': 'Sin', 'cos': 'Cos'}
# Refute a leaf only when PROVABLY safe (dual bound > 0); the tiny epsilon is
# float slack, NOT the spec tolerance. We do NOT refute at >-tol — that would
# falsely call a case with a genuine (margin<=0) counterexample 'unsat'. The
# spec `tol` is used ONLY to accept a within-tolerance near-miss as a last-resort
# SAT witness, never to relax UNSAT (which must stay sound).
_REFUTE_EPS = 1e-9


def build_acopf_dual_state(graph, spec, settings, device='cpu',
                           dtype=torch.float64, pow_alpha=0.5,
                           onnx_path=None):
    """Run one capturing forward and assemble the dual-ascent state:
    objective (G_out, c_out) + per-element splittable list with the EXACT band.
    Pow ops are forced onto the α-band path (relu_lambdas) so their (λ,μ,δ) is
    captured; sigmoid/tanh/sin/cos use the affine-band path (band captured for
    any α)."""
    from .verify_zono_bnb import _forward_zonotope_graph
    if settings is not None:
        settings.sigmoid_relaxation = 'affine_band'
    gg = graph.gpu_graph(device=device, dtype=dtype)
    xl0 = torch.tensor(np.asarray(spec.x_lo).flatten(), dtype=dtype,
                       device=device)
    xh0 = torch.tensor(np.asarray(spec.x_hi).flatten(), dtype=dtype,
                       device=device)
    relu_lambdas = {op['name']: torch.tensor(pow_alpha, dtype=dtype,
                                             device=device)
                    for op in gg['ops'] if op['type'] == 'pow'}
    ob, cids, ofi, geom = {}, {}, {}, {}
    with torch.no_grad():
        _, zf = _forward_zonotope_graph(
            xl0, xh0, gg, device, dtype, settings=settings,
            relu_lambdas=relu_lambdas, op_bounds=ob, col_ids_out=cids,
            op_fresh_ids=ofi, op_geom_out=geom)
    G_out = zf.generators.detach().cpu().numpy().astype(np.float64)
    c_out = zf.center.detach().cpu().numpy().astype(np.float64)
    n_gens = G_out.shape[1]
    out_ids = cids[gg['ops'][-1]['name']]
    final_pos = {gid: i for i, gid in enumerate(out_ids)}

    splittables = []
    for op in gg['ops']:
        nm, t = op['name'], op['type']
        if t not in _SPLIT or nm not in geom or 'band' not in geom[nm]:
            continue
        g = geom[nm]
        c_in = g['c_in'].cpu().numpy().astype(np.float64)
        gens = g['gens'].cpu().numpy().astype(np.float64)
        in_ids = g['in_ids']
        lam, mu, delta = (b.cpu().numpy().astype(np.float64) for b in g['band'])
        lo, hi = (x.cpu().numpy().astype(np.float64) for x in ob[nm])
        a_start, _ = ofi[nm]
        relax = (PowRelax(int(op.get('exponent', 2))) if t == 'pow'
                 else REGISTRY[_RELAX_NAME[t]]())
        # map this op's input error-symbol IDs to final-objective column positions
        if any(gid not in final_pos for gid in in_ids):
            continue                                   # column dropped (rare)
        col_pos = np.array([final_pos[gid] for gid in in_ids], dtype=np.int64)
        n_elem = c_in.shape[0]
        for i in range(n_elem):
            row = gens[i] if gens.ndim == 2 else np.zeros(len(in_ids))
            nz = np.nonzero(row)[0]
            e_gid = a_start + i
            if e_gid not in final_pos:
                continue
            splittables.append(dict(
                op_type=t, relax=relax, name=nm, elem=i,
                c_in=float(c_in[i]), lo=float(lo[i]), hi=float(hi[i]),
                row_indices=col_pos[nz], row_values=row[nz],
                e_new_col=final_pos[e_gid],
                lam=float(lam[i]), mu=float(mu[i]), delta=float(delta[i])))
    # input-noise columns: from_input_bounds allocates ONE column per VARYING
    # dim (column k <-> dim nz[k]); map each to its final-objective position so
    # a witness can be turned back into a network input.
    xl_np = np.asarray(spec.x_lo).flatten().astype(np.float64)
    xh_np = np.asarray(spec.x_hi).flatten().astype(np.float64)
    nz = np.nonzero(xh_np - xl_np > 0)[0]
    # global ID of the k-th input-noise column is k (from_input_bounds order);
    # keep aligned with nz (-1 => that column didn't survive to the objective).
    in_noise_pos = [int(final_pos.get(k, -1)) for k in range(len(nz))]
    return dict(G_out=G_out, c_out=c_out, n_gens=n_gens,
                splittables=splittables, n_out=c_out.shape[0],
                gg=gg, xl0=xl0, xh0=xh0, n_input=int(xl0.numel()),
                xl_np=xl_np, xh_np=xh_np, nz=nz, in_noise_pos=in_noise_pos,
                onnx_path=onnx_path, device=device, dtype=dtype)


def _disj_margin_at(spec, x, y):
    """min over disjuncts (whose X-subrange contains x) of conj.margin(y,y):
    <= 0 means the unsafe conjunction HOLDS at (x,y) -> counterexample; the
    magnitude is how far past the constraint (compared against the tolerance)."""
    ms = [c.margin(y, y) for c in spec.disjuncts if c.x_satisfied(x)]
    return min(ms) if ms else float('inf')


def _make_ce_checker(state, spec):
    """Return (check_witness, margin_at_input, witness_to_input):
      - check_witness(w)   -> (margin, x, y) from a dual-ascent primal witness;
      - margin_at_input(x) -> (margin, y) for a raw network input (used to
        confirm a PGD-found point);
      - witness_to_input(w)-> x.
    Evaluation is on **onnxruntime CPU** on the ORIGINAL onnx (the VNNCOMP
    reference executor) when ``onnx_path`` is set, else the exact zono-center
    forward. `margin <= 0` means a counterexample."""
    xl, xh, nz = state['xl_np'], state['xh_np'], state['nz']
    in_noise_pos = state['in_noise_pos']
    onnx_path = state.get('onnx_path')
    _sess = _iname = _ishape = None
    if onnx_path:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        _sess = ort.InferenceSession(onnx_path, sess_options=so,
                                     providers=['CPUExecutionProvider'])
        _iname = _sess.get_inputs()[0].name
        _ishape = _sess.get_inputs()[0].shape

    def witness_to_input(witness_np):
        x = xl.copy()
        w = np.clip(np.asarray(witness_np, dtype=np.float64), -1.0, 1.0)
        for k, dim in enumerate(nz):
            pos = in_noise_pos[k]
            if pos >= 0:
                x[dim] = xl[dim] + (xh[dim] - xl[dim]) * 0.5 * (w[pos] + 1.0)
        return x

    def margin_at_input(x):
        x = np.asarray(x, dtype=np.float64).flatten()
        if _sess is not None:
            shp = [d if isinstance(d, int) else 1 for d in _ishape]
            y = np.asarray(_sess.run(
                None, {_iname: x.reshape(shp).astype(np.float32)})[0]
            ).flatten().astype(np.float64)
        else:
            from .verify_zono_bnb import _forward_zonotope_graph
            xt = torch.tensor(x, dtype=state['dtype'], device=state['device'])
            with torch.no_grad():
                _, zc = _forward_zonotope_graph(xt, xt, state['gg'],
                                                state['device'], state['dtype'])
            y = zc.center.cpu().numpy()
        return _disj_margin_at(spec, x, y), y

    def check_witness(witness_np):
        x = witness_to_input(witness_np)
        m, y = margin_at_input(x)
        return m, x, y
    return check_witness, margin_at_input, witness_to_input


def _node_lp(state, w, bias, restr, sp_by_idx):
    """Build (d, c0, A_rows, b_rows) for the leaf defined by `restr`
    (dict splittable_idx -> (clo, chi)): apply each neuron's parent->[clo,chi]
    band correction (parent slope) + two pre-activation halfspaces."""
    G = state['G_out']
    d = w @ G                                    # (n_gens,)
    c0 = float(w @ state['c_out'] + bias)
    rows, rhs = [], []
    n_gens = state['n_gens']
    for idx, (clo, chi) in restr.items():
        sp = sp_by_idx[idx]
        relax = sp['relax']
        g_k = float(backward_sensitivity(d[sp['e_new_col']], sp['delta']))
        lam_n, mu_n, delta_n = (float(t) for t in relax.affine_band(
            torch.tensor(clo, dtype=torch.float64),
            torch.tensor(chi, dtype=torch.float64),
            lam=torch.tensor(sp['lam'], dtype=torch.float64)))
        dcr, dce, c0c = band_change_correction(
            g_k, sp['c_in'], sp['row_values'], sp['lam'], sp['mu'], sp['delta'],
            lam_n, mu_n, delta_n)
        d = d.copy()
        d[sp['row_indices']] += np.asarray(dcr)
        d[sp['e_new_col']] += float(dce)
        c0 += float(c0c)
        for p, side in ((clo, 'right'), (chi, 'left')):   # z>=clo, z<=chi
            hr, hb = split_halfspace(sp['c_in'], sp['row_values'], p, side)
            full = np.zeros(n_gens)
            full[sp['row_indices']] = np.asarray(hr)
            rows.append(full); rhs.append(float(hb))
    A = np.stack(rows) if rows else np.zeros((1, n_gens))
    b = np.asarray(rhs) if rhs else np.array([1.0])      # no-op row
    return d, c0, A, b


def _best_g(d, c0, A, b, n_gens, device, dtype, max_iter=20):
    m = A.shape[0]
    bg, _, reason, witness = dab._batched_dual_ascent(
        torch.as_tensor(d, dtype=dtype, device=device).unsqueeze(0),
        torch.tensor([c0], dtype=dtype, device=device),
        torch.as_tensor(A, dtype=dtype, device=device).reshape(1, m, n_gens),
        torch.as_tensor(b, dtype=dtype, device=device).reshape(1, m),
        torch.zeros(1, m, dtype=dtype, device=device),
        torch.ones(1, dtype=torch.bool, device=device),
        torch.full((n_gens,), -1.0, dtype=dtype, device=device),
        torch.full((n_gens,), 1.0, dtype=dtype, device=device),
        torch.full((n_gens,), 2.0, dtype=dtype, device=device),
        max_iter=max_iter, repair_steps=3, feas_tol=1e-9, tol=1e-12,
        dtype=dtype, device=device)
    # witness is meaningful only when the leaf found a feasible primal at p<=0
    w = (witness[0].cpu().numpy() if int(reason[0]) == 1 else None)
    return float(bg[0]), w


def verify_acopf_dual_ascent(state, spec, deadline, device='cpu',
                             dtype=torch.float64, max_depth=60,
                             max_leaves_per_dj=4000, print_progress=False,
                             find_sat=True, tol=1e-4,
                             pgd_restarts=256, pgd_iter=120):
    """Refute every disjunct via nonlinear-split dual-ascent BaB.

    Verdicts (a witness is CONFIRMED on the network — onnxruntime CPU when
    ``state['onnx_path']`` is set, else the exact zono-center forward — and its
    spec margin classified; margin <= 0 means a genuine violation):
      * 'sat'    — a confirmed witness with margin <= 0 (a REAL counterexample).
        Returned immediately; this is the goal.
      * 'unsat'  — every disjunct PROVABLY safe (query min bound > 0). Sound:
        we never refute at >-tol, so a genuine within-tol CE is never mislabelled
        unsat.
      * 'sat' (within_tol) — only a near-miss was found: a SAFE point within `tol`
        of the unsafe boundary (0 < margin <= tol), and we could neither find a
        real CE nor prove unsat before the deadline. Stored on first sighting
        (its witness goes in the result/CE file), but we KEEP searching; reported
        only as the last-resort fallback, flagged `within_tol=True` + logged.
      * 'unknown' — budget/timeout with no witness at all.

    With ``find_sat`` (default) witnesses are checked; vary the build α
    (`pow_alpha`) to surface different witnesses across calls."""
    from .nonlinear_split_planes import split_point
    sp_list = state['splittables']
    sp_by_idx = {i: sp for i, sp in enumerate(sp_list)}
    n_gens = state['n_gens']
    ce_check = margin_at_input = witness_to_input = None
    if find_sat:
        ce_check, margin_at_input, witness_to_input = _make_ce_checker(
            state, spec)
    queries = spec.as_linear_queries(state['n_out'])
    by_dj = defaultdict(list)
    for (di, w, bias) in queries:
        by_dj[di].append((np.asarray(w, dtype=np.float64), float(bias)))
    total_nodes = 0
    marginal = None        # best within-tol confirmed witness (x, margin)

    # Phase 0 — PGD-from-witness SAT finder. Seed float64 PGD on the network
    # with the dual-ascent root primal witnesses (LP-informed, near the binding
    # region — far better than random starts in high-dim ACOPF boxes), confirm
    # the result on onnxruntime CPU, and classify by tol. Cracks SAT cases the
    # BaB alone returns 'unknown' on. (Sound: only acts on an ORT-confirmed CE.)
    if find_sat and state.get('onnx_path') and pgd_restarts > 0:
        from .onnx_torch_runner import pgd_via_onnx
        seeds = []
        # collect root witnesses as PGD seeds, but cap the work (one dual solve
        # per query is slow when a spec has hundreds of disjuncts) and respect
        # the deadline.
        seed_budget = min(deadline, time.perf_counter()
                          + 0.25 * (deadline - time.perf_counter()))
        for di, qs in by_dj.items():
            if time.perf_counter() >= seed_budget or len(seeds) >= 64:
                break
            for w, bias in qs:
                d, c0, A, b = _node_lp(state, w, bias, {}, sp_by_idx)
                _g, wit = _best_g(d, c0, A, b, n_gens, device, dtype)
                if wit is not None:
                    seeds.append(witness_to_input(wit))
        # give PGD at most half the remaining budget so the BaB (UNSAT) also
        # gets time; pgd_via_onnx honours the deadline per-iteration.
        now = time.perf_counter()
        pgd_deadline = min(deadline, now + max(5.0, 0.5 * (deadline - now)))
        try:
            sat, wpgd = pgd_via_onnx(
                state['onnx_path'], spec, n_restarts=pgd_restarts,
                n_iter=pgd_iter, lr=0.2, device=torch.device('cpu'),
                dtype=torch.float64, deadline=pgd_deadline,
                seeds=(np.stack(seeds) if seeds else None))
        except (RuntimeError, ValueError):
            sat, wpgd = False, None
        if sat and wpgd is not None:
            m, _y = margin_at_input(wpgd)          # ORT final validation
            x = np.asarray(wpgd, dtype=np.float64).flatten()
            if m <= 0:                             # genuine violation => real CE
                return 'sat', {'method': 'acopf_dual+pgd', 'within_tol': False,
                               'margin': m, 'witness': x, 'nodes': 0}
            if m <= tol:                           # safe but within tol; keep
                marginal = (x, m)                  # going for a real CE / unsat

    def _maybe_marginal_sat(reason):
        if marginal is not None:
            if print_progress:
                print(f'[acopf_dual] SAT within tolerance (margin='
                      f'{marginal[1]:.3e} > -{tol:g}); could not prove unsat '
                      f'({reason})', flush=True)
            return 'sat', {'method': 'acopf_dual', 'within_tol': True,
                           'margin': marginal[1], 'witness': marginal[0],
                           'reason': reason, 'nodes': total_nodes}
        return 'unknown', {'method': 'acopf_dual', 'reason': reason,
                           'nodes': total_nodes}

    for di, qs in by_dj.items():
        # BFS: a leaf is refuted (safe up to tol) if some query bound > -tol.
        queue = [({}, 0)]
        leaves = 0
        while queue:
            if time.perf_counter() >= deadline:
                return _maybe_marginal_sat('timeout')
            restr, depth = queue.pop()
            total_nodes += 1
            best_q_g = -float('inf')
            d0 = None
            for w, bias in qs:
                d, c0, A, b = _node_lp(state, w, bias, restr, sp_by_idx)
                g, wit = _best_g(d, c0, A, b, n_gens, device, dtype)
                if ce_check is not None and wit is not None:
                    margin, x, _y = ce_check(wit)
                    if margin <= 0:                   # genuine violation => CE
                        return 'sat', {'method': 'acopf_dual', 'witness': x,
                                       'margin': margin, 'within_tol': False,
                                       'disjunct': di, 'nodes': total_nodes}
                    if margin <= tol and (marginal is None
                                          or margin < marginal[1]):
                        marginal = (x, margin)        # safe within tol; keep
                if g > best_q_g:
                    best_q_g, d0 = g, d
            if best_q_g > _REFUTE_EPS:
                continue                              # leaf PROVABLY safe (>0)
            if depth >= max_depth or leaves > max_leaves_per_dj:
                return _maybe_marginal_sat('budget')
            # split the splittable with the largest band-slack * width.
            best_i, best_score, best_iv = None, 0.0, None
            for i, sp in sp_by_idx.items():
                clo, chi = restr.get(i, (sp['lo'], sp['hi']))
                width = chi - clo
                if width <= 1e-9:
                    continue
                score = abs(float(d0[sp['e_new_col']])) * width
                if score > best_score:
                    best_i, best_score, best_iv = i, score, (clo, chi)
            if best_i is None:
                return _maybe_marginal_sat('no_split')
            clo, chi = best_iv
            sp = sp_by_idx[best_i]
            p = float(split_point(sp['op_type'], clo, chi))
            if not (clo < p < chi):
                p = 0.5 * (clo + chi)
            for sub in ((clo, p), (p, chi)):
                r2 = dict(restr); r2[best_i] = sub
                queue.append((r2, depth + 1))
            leaves += 1
        # disjunct exhausted with every leaf refuted (safe up to tol)
    return 'unsat', {'method': 'acopf_dual', 'nodes': total_nodes,
                     'n_splittables': len(sp_list), 'tol': tol}
