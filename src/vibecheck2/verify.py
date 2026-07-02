"""vibecheck2 verification entry point.

Thin orchestration for the current milestone: load net + spec (v1 front end),
forward bounds for intermediates, alpha-CROWN on the spec query rows, verdict
by per-disjunct refutation. Grows into the scheduler of design 2.5; the
results-file discipline matches v1 (the file is the verdict authority).

Disjunct semantics (v1 spec.py): a counterexample must satisfy EVERY
constraint of SOME disjunct. A disjunct is refuted when ANY of its query
rows w.y + b has a positive proven lower bound; `unsat` when every disjunct
is refuted.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from .core import backward, forward
from .core.graph import load as load_net


def _spec_queries(spec, n_out, dtype=torch.float32):
    """(W (q, n_out), bias (q,), disj_idx (q,)) from the v1 VNNSpec."""
    rows = spec.as_linear_queries(n_out)
    W = torch.tensor(np.stack([w for _, w, _ in rows]), dtype=dtype)
    b = torch.tensor([bias for _, _, bias in rows], dtype=dtype)
    di = torch.tensor([d for d, _, _ in rows])
    return W, b, di


def _verdict_from_lbs(lb_plus_bias, disj_idx, n_disjuncts):
    """'unsat' iff every disjunct has some strictly-positive query row."""
    refuted = set()
    for d in range(n_disjuncts):
        rows = lb_plus_bias[disj_idx == d]
        if rows.numel() and rows.max() > 0:
            refuted.add(d)
    open_d = [d for d in range(n_disjuncts) if d not in refuted]
    return ('unsat' if not open_d else 'unknown'), open_d


def _subbox_groups(spec):
    """Group disjuncts by their per-conjunct input subbox (acasxu prop_6,
    nn4sys lindex). Returns [(x_lo, x_hi, [disjunct indices])]; a single
    group with the global box when no disjunct declares one."""
    groups = {}
    for i, c in enumerate(spec.disjuncts):
        if c.input_lo is not None:
            key = (tuple(np.asarray(c.input_lo).ravel()),
                   tuple(np.asarray(c.input_hi).ravel()))
        else:
            key = None
        groups.setdefault(key, []).append(i)
    out = []
    for key, idxs in groups.items():
        if key is None:
            out.append((spec.x_lo, spec.x_hi, idxs))
        else:
            out.append((np.asarray(key[0]), np.asarray(key[1]), idxs))
    return out


def _log_flush(m):
    print(m, flush=True)


def verify(onnx_path, vnnlib_path, timeout=60.0, device='cpu',
           alpha_iters=20, pgd_budget=5.0, log=_log_flush):
    """Returns (verdict, details); details carries 'witness' for 'sat'.

    Disjuncts carrying their own input subboxes (acasxu prop_6) decompose
    into independent sub-instances: 'sat' if any, 'unsat' iff all."""
    from vibecheck.spec import VNNSpec
    from vibecheck.vnnlib_loader import load_vnnlib
    t0 = time.time()
    try:
        net = load_net(onnx_path)
    except Exception as e:                    # noqa: BLE001 - see re-raise
        # a net the graph loader cannot model: try the discrete-grid
        # handler (cctsdb); if the instance is not discrete either,
        # re-raise the ORIGINAL load error (never silently swallowed)
        from .handlers.discrete_enum import try_discrete_enum
        log(f'[vc2] graph load failed ({type(e).__name__}: {str(e)[:80]}); '
            f'trying discrete-enum handler')
        try:
            return try_discrete_enum(onnx_path, vnnlib_path, timeout, log)
        except NotImplementedError:
            raise e
    spec = load_vnnlib(vnnlib_path)
    log(f'[vc2] {net}')

    groups = _subbox_groups(spec)
    if len(groups) == 1:
        return _verify_one(net, spec, onnx_path, timeout, device,
                           alpha_iters, pgd_budget, log, t0)
    log(f'[vc2] {len(groups)} input-subbox groups (per-disjunct boxes)')
    if len(groups) > 16:
        # mega-disjunct screening (nn4sys-style): one batched CROWN pass
        # over ALL subboxes refutes the easy mass; only survivors get the
        # full per-group pipeline
        groups = _screen_subbox_groups(net, spec, groups, device, log)
        log(f'[vc2] {len(groups)} groups open after batched screening')
    share = ((timeout - (time.time() - t0)) / max(1, len(groups))
             if groups else 0.0)
    for glo, ghi, idxs in groups:
        sub = VNNSpec(x_lo=np.asarray(glo, dtype=np.float64),
                      x_hi=np.asarray(ghi, dtype=np.float64),
                      disjuncts=[spec.disjuncts[i] for i in idxs])
        verdict, details = _verify_one(net, sub, onnx_path, share, device,
                                       alpha_iters, pgd_budget, log,
                                       time.time())
        if verdict != 'unsat':
            details['time'] = time.time() - t0
            return verdict, details
    return 'unsat', {'time': time.time() - t0}


def _screen_subbox_groups(net, spec, groups, device, log):
    """Batched-CROWN refutation screen over subbox groups; returns the
    still-open subset. Sound: only provably-refuted groups are dropped."""
    from .core import backward, memory
    dev = torch.device(device)
    W_all, b_all, di_all = _spec_queries(spec, net.n_out)
    W_all, b_all = W_all.to(dev), b_all.to(dev)
    open_groups = []
    widest = max(net.ops[o].n for o in net.order)
    per_dom = W_all.shape[0] * widest * 4 * 8
    cs = memory.chunk_size(len(groups), per_dom, dev)
    for i in range(0, len(groups), cs):
        chunk = groups[i:i + cs]
        lo = torch.tensor(np.stack([g[0] for g in chunk]),
                          dtype=torch.float32, device=dev)
        hi = torch.tensor(np.stack([g[1] for g in chunk]),
                          dtype=torch.float32, device=dev)
        lbq = backward.crown(net, lo, hi, W_all) + b_all
        for k, (glo, ghi, idxs) in enumerate(chunk):
            refuted = all(
                bool((lbq[k][di_all == d] > 0).any()) for d in idxs)
            if not refuted:
                open_groups.append((glo, ghi, idxs))
    return open_groups


def _verify_one(net, spec, onnx_path, timeout, device, alpha_iters,
                pgd_budget, log, t0):
    from .core import attack
    from .core.budget import Budget, OutOfTime
    budget = Budget(timeout, margin=0.0)
    budget.t0 = t0
    budget.deadline = t0 + timeout - 2.0

    # Phase A: falsification first (cheap, decides most sat instances).
    # A candidate is only a 'sat' after the ORT chokepoint accepts it.
    if pgd_budget > 0:
        w, _info = attack.pgd(net, spec, device=device, restarts=100,
                              init='osi', time_budget=pgd_budget, log=log)
        if w is not None:
            ok, vinfo = attack.validate(onnx_path, spec, w)
            if ok:
                w_emit = vinfo.get('witness_inbox', w)
                return 'sat', {'witness': np.asarray(w_emit),
                               'time': time.time() - t0}
            log('[vc2] pgd candidate rejected by ORT chokepoint; continuing')

    dev = torch.device(device)
    lo = torch.tensor(spec.x_lo, dtype=torch.float32, device=dev).unsqueeze(0)
    hi = torch.tensor(spec.x_hi, dtype=torch.float32, device=dev).unsqueeze(0)
    W, b, di = _spec_queries(spec, net.n_out)
    W, b = W.to(dev), b.to(dev)

    try:
        inter = backward.intermediates(net, lo, hi)
    except OutOfTime:
        return 'timeout', {'time': time.time() - t0}
    lb0 = backward.crown(net, lo, hi, W, inter)[0]
    verdict, open_d = _verdict_from_lbs(lb0 + b, di, len(spec.disjuncts))
    log(f'[vc2] crown: worst={float((lb0 + b).min()):.4f} '
        f'open={len(open_d)}/{len(spec.disjuncts)}')
    if verdict != 'unsat':
        from .core import memory
        from .core.backward import _zono_cost_bytes
        if (_zono_cost_bytes(net, 1)
                >= memory.free_bytes(lo.device) * memory.SAFETY):
            # big net: the interval intermediates were the bottleneck;
            # recompute them by per-edge backward CROWN (chunked)
            try:
                inter = backward.intermediates_crown(net, lo, hi,
                                                     budget=budget)
            except OutOfTime:
                return 'timeout', {'time': time.time() - t0}
            lb0 = torch.maximum(lb0, backward.crown(net, lo, hi, W, inter)[0])
            verdict, open_d = _verdict_from_lbs(lb0 + b, di,
                                                len(spec.disjuncts))
            log(f'[vc2] crown-inter: worst={float((lb0 + b).min()):.4f} '
                f'open={len(open_d)}/{len(spec.disjuncts)}')
    if verdict != 'unsat' and alpha_iters > 0:
        lb = backward.alpha_crown(net, lo, hi, W, inter,
                                  iters=alpha_iters, thresholds=-b,
                                  budget=budget)[0]
        lb = torch.maximum(lb, lb0)
        verdict, open_d = _verdict_from_lbs(lb + b, di, len(spec.disjuncts))
        log(f'[vc2] alpha-crown: worst={float((lb + b).min()):.4f} '
            f'open={len(open_d)}/{len(spec.disjuncts)}')
        worst = float((lb + b).min())
        if verdict != 'unsat' and -1.0 < worst <= 0 and budget.remaining() > 20:
            # near-zero gap: a longer, lower-lr polish often closes it
            # outright (abcrown runs ~100 root iters; the quick pass is 20)
            lb2 = backward.alpha_crown(net, lo, hi, W, inter, iters=150,
                                       lr=0.1, thresholds=-b,
                                       budget=budget)[0]
            lb = torch.maximum(lb, lb2)
            verdict, open_d = _verdict_from_lbs(lb + b, di,
                                                len(spec.disjuncts))
            log(f'[vc2] alpha-polish: worst={float((lb + b).min()):.4f} '
                f'open={len(open_d)}/{len(spec.disjuncts)}')
    if verdict != 'unsat':
        # dual-ascent LP certifier (compiled GPU BaB over the alpha-zono
        # state, ported v1 fast_dual_ascent): the strongest per-query
        # refuter. The state builds BACKWARD (unstable rows only, chunked),
        # so no forward-zonotope gate; survivors fall through to BaB.
        from .core.dual_lp import certify_queries
        refuted = certify_queries(
            net, spec, W, b, di, lo, hi, inter, open_d,
            deadline=t0 + timeout - 2.0, device=device, log=log)
        open_d = [d for d in open_d if d not in refuted]
        log(f'[vc2] dual-lp: {len(refuted)} disjuncts refuted, '
            f'{len(open_d)} open')
        if not open_d:
            return 'unsat', {'time': time.time() - t0}
    if verdict != 'unsat':
        # branch and bound: input splits for low-dimensional inputs, relu
        # phase splits otherwise (unified scoring across both is the design
        # target; the two loops share bound/attack machinery meanwhile)
        from .core.search import input_split_bab, relu_split_bab
        kw = {}
        if net.n_in <= 32:
            bab = input_split_bab
        else:
            bab = relu_split_bab
            kw['root_inter'] = inter        # the crown-refined root bounds
        verdict, binfo = bab(
            net, spec, W, b, di, lo[0], hi[0],
            deadline=t0 + timeout - 2.0, device=device,
            onnx_path=onnx_path, log=log, **kw)
        log(f'[vc2] {bab.__name__}: {verdict} '
            f'{ {k: v for k, v in binfo.items() if k != "witness"} }')
        if verdict == 'sat':
            return 'sat', {'witness': binfo['witness'],
                           'time': time.time() - t0}
        if verdict == 'timeout':
            verdict = 'unknown'
    return verdict, {'open_disjuncts': open_d, 'time': time.time() - t0}


def main(argv=None):
    """Minimal CLI mirroring v1's verdict conventions for parity harnesses."""
    import argparse
    p = argparse.ArgumentParser(prog='vibecheck2')
    p.add_argument('--net', required=True)
    p.add_argument('--spec', required=True)
    p.add_argument('--timeout', type=float, default=60.0)
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--results-file', default=None)
    a = p.parse_args(argv)
    if a.net.lstrip().startswith('['):
        # network-pair instance (isomorphic/monotonic acasxu): reuse the v1
        # front end to merge the pair into one onnx + v1 spec (exact,
        # ORT-oracle-gated), then verify normally (design: frontends port)
        from vibecheck import network_pair as npair
        a.net, a.spec = npair.build_merged_instance(a.net, a.spec)
    else:
        # nonlinear v2 spec (adaptive_cruise): v1's ORT-oracle-gated
        # transpile to an augmented onnx + linear v1 spec. NOTE: an unsat on
        # the augmented instance is sound for the original; a sat witness is
        # re-validated by the chokepoint on the AUGMENTED net here, and the
        # strict original-spec disposition is handler work (v1
        # _sat_disposition), so borderline CEs may differ from v1 for now.
        from vibecheck import nonlinear_augment as nla
        try:
            text = nla._read_vnnlib_text(a.spec)
        except (OSError, ValueError):
            text = ''
        if text and nla.is_nonlinear_v2_spec(text):
            a.net, a.spec = nla.build_augmented_instance(a.net, a.spec)
    if a.results_file:                        # pre-seed like v1
        with open(a.results_file, 'w') as f:
            f.write('timeout\n')
    try:
        verdict, details = verify(a.net, a.spec, a.timeout, a.device)
    except BaseException as e:                # crash -> 'error' (v1 discipline)
        import traceback
        traceback.print_exc()
        if a.results_file:
            with open(a.results_file, 'w') as f:
                f.write(f'error\n{type(e).__name__}: {str(e)[:300]}\n')
        return 2
    if a.results_file:
        ce = None
        if verdict == 'sat' and details.get('witness') is not None:
            # v1's CE formatting: version/io names resolved from the spec,
            # Y recomputed by the same ORT forward the scorer replays
            from vibecheck.main import (_counterexample_sexpr,
                                        _resolve_cex_io_meta,
                                        _vnnlib_version)
            from vibecheck.vnnlib_loader import load_vnnlib
            ce = _counterexample_sexpr(
                a.net, load_vnnlib(a.spec), details['witness'],
                version=_vnnlib_version(a.spec),
                io_meta=_resolve_cex_io_meta(a.spec))
        tmp = a.results_file + '.tmp'
        with open(tmp, 'w') as f:
            f.write(verdict + '\n')
            if ce is not None:
                f.write(ce + '\n')
        os.replace(tmp, a.results_file)
    print(f'[vc2] verdict: {verdict}  ({details["time"]:.2f}s)')
    return 0 if verdict == 'unsat' else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
