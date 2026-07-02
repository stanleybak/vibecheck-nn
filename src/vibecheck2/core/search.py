"""The BaB search: one batched frontier over one domain type (design 2.3).

Milestone M4a: input-dimension splitting. A domain is an input subbox plus
its per-disjunct openness; the whole frontier lives in flat tensors, a
memory-budgeted batch of the worst domains is bounded per iteration
(forward intermediates + backward CROWN, batched over domains), verified
domains are dropped, the rest split on the best-scoring action. ReLU/
nonlinear clamp actions join the same ranking in M4b, sharing this loop.

Scoring: the action's estimated lb improvement from the SAME backward pass
that produced the bound: input dim k scores |A_in[:, k]| * width_k / 2
(Smart-Branching), giving the unified currency later action types share.

Falsification interleaves: the attack engine runs on the worst domain's
subbox every few rounds with the frontier's worst point as a seed; any
validated hit ends the search with 'sat'.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from . import attack, backward, memory


def input_split_bab(net, spec, W, bias, disj_idx, lo, hi, deadline,
                    device='cpu', batch=4096, split_dims=2, alpha_iters=8,
                    onnx_path=None, attack_every=8, log=lambda m: None):
    """Returns (verdict, info): 'unsat' | 'sat' (+witness) | 'timeout'.

    W (q, n_out), bias (q,), disj_idx (q,): the spec query rows.
    lo, hi: (n_in,) root box. Each open domain splits its top `split_dims`
    scoring dims simultaneously (2^k children); domains whose plain-CROWN
    bound lands near zero get a short per-batch alpha pass before splitting.
    """
    dev = torch.device(device)
    dt = torch.float32
    W = W.to(dev, dt)
    bias = bias.to(dev, dt)
    D = int(disj_idx.max()) + 1 if disj_idx.numel() else 0
    q = W.shape[0]
    # per-disjunct row selector (D, q) for the batched refutation check
    sel = torch.zeros(D, q, device=dev, dtype=torch.bool)
    sel[disj_idx, torch.arange(q)] = True

    f_lo = lo.reshape(1, -1).to(dev, dt)
    f_hi = hi.reshape(1, -1).to(dev, dt)
    f_worst = torch.full((1,), -torch.inf, device=dev)
    n_bounded = n_split = rounds = 0
    t0 = time.time()

    def domain_refuted(lbq):
        """(B, q) query lbs -> (B, D) refutation matrix."""
        pos = (lbq + bias) > 0                       # refuting rows
        refuted = torch.zeros(lbq.shape[0], D, device=dev, dtype=torch.bool)
        for dd in range(D):
            refuted[:, dd] = (pos & sel[dd]).any(dim=1)
        return refuted

    def clip_domains(olo, ohi, A, lbb, rows):
        """ABC-style input clipping: every row r in `rows` must satisfy its
        linear lower form L_r(x) = A_r.x + d_r <= 0 for a counterexample to
        exist in the domain, so tighten the box to those halfspaces per dim.
        A (B, q, n); lbb (B, q) = lbq + bias = min of L_r over the box."""
        for r in rows:
            Ar = A[:, r]                                    # (B, n)
            mn = Ar.clamp(max=0) * ohi + Ar.clamp(min=0) * olo
            d = lbb[:, r].unsqueeze(1) - mn.sum(dim=1, keepdim=True)
            slack = -d - (mn.sum(dim=1, keepdim=True) - mn)  # max A_k x_k
            new_hi = torch.minimum(ohi, slack / Ar.clamp_min(1e-30))
            new_lo = torch.maximum(olo, slack / Ar.clamp_max(-1e-30))
            ohi = torch.where(Ar > 0, new_hi, ohi)
            olo = torch.where(Ar < 0, new_lo, olo)
        return olo, ohi

    while f_lo.shape[0]:
        if time.time() > deadline:
            return 'timeout', {'frontier': int(f_lo.shape[0]),
                               'bounded': n_bounded, 'splits': n_split}
        rounds += 1
        # pick the worst-first batch, sized by the memory budget
        per_dom = q * max(net.ops[o].n for o in net.order) * 4 * 8
        bs = min(batch, memory.chunk_size(f_lo.shape[0], per_dom, dev))
        order = torch.argsort(f_worst)                # least-verified first
        take, keep = order[:bs], order[bs:]
        blo, bhi = f_lo[take], f_hi[take]
        f_lo, f_hi, f_worst = f_lo[keep], f_hi[keep], f_worst[keep]

        # per-edge CROWN-refined intermediates: the decisive tightener for
        # input-split domains (zono intermediates alone leave ~85% of deep
        # acasxu domains open; refined ones close them, measured 2026-07)
        inter = backward.intermediates_crown(net, blo, bhi)
        lbq, Ain = backward.crown(net, blo, bhi, W, inter,
                                  return_input_adjoint=True)
        n_bounded += blo.shape[0]
        refuted = domain_refuted(lbq)
        open_mask = ~refuted.all(dim=1)
        if open_mask.any() and alpha_iters > 0:
            # short alpha pass on the still-open domains only
            oi = torch.nonzero(open_mask, as_tuple=False).flatten()
            inter_o = {k2: tuple(t[oi] for t in v)
                       for k2, v in inter.items()}
            lb_a = backward.alpha_crown(net, blo[oi], bhi[oi], W, inter_o,
                                        iters=alpha_iters, thresholds=-bias)
            lbq[oi] = torch.maximum(lbq[oi], lb_a)
            refuted = domain_refuted(lbq)
            open_mask = ~refuted.all(dim=1)
        if open_mask.any():
            olo, ohi = blo[open_mask], bhi[open_mask]
            oA = Ain[open_mask]
            # clip each single-open-disjunct domain to the halfspaces its
            # counterexample must satisfy (sound; often empties the box)
            oref = refuted[open_mask]
            single = (~oref).sum(dim=1) == 1
            if single.any():
                dd_open = (~oref[single]).float().argmax(dim=1)
                lbb = (lbq + bias)[open_mask][single]
                slo, shi = olo[single], ohi[single]
                for dd in dd_open.unique().tolist():
                    m = dd_open == dd
                    rows = torch.nonzero(sel[dd], as_tuple=False).flatten()
                    clo, chi = clip_domains(slo[m], shi[m], oA[single][m],
                                            lbb[m], rows.tolist())
                    slo[m], shi[m] = clo, chi
                olo[single], ohi[single] = slo, shi
                nonempty = (ohi - olo).min(dim=1).values >= 0
                olo, ohi, oA = olo[nonempty], ohi[nonempty], oA[nonempty]
                lbq_open = (lbq + bias)[open_mask][nonempty]
            else:
                lbq_open = (lbq + bias)[open_mask]
            if not olo.shape[0]:
                continue
            # Smart-Branching: estimated improvement per input dim; split the
            # top `split_dims` dims simultaneously -> 2^k children
            score = oA.abs().sum(dim=1) * (ohi - olo) / 2
            kdims = min(split_dims, olo.shape[1])
            topk = score.topk(kdims, dim=1).indices          # (B, kdims)
            ch_lo, ch_hi = olo, ohi
            for j in range(kdims):
                reps = ch_lo.shape[0] // olo.shape[0]
                kk = topk[:, j].repeat(reps).unsqueeze(1)
                mid = (ch_lo.gather(1, kk) + ch_hi.gather(1, kk)) / 2
                left_hi = ch_hi.clone()
                left_hi.scatter_(1, kk, mid)
                right_lo = ch_lo.clone()
                right_lo.scatter_(1, kk, mid)
                ch_lo = torch.cat([ch_lo, right_lo])
                ch_hi = torch.cat([left_hi, ch_hi])
            f_lo = torch.cat([f_lo, ch_lo])
            f_hi = torch.cat([f_hi, ch_hi])
            w = lbq_open.min(dim=1).values
            f_worst = torch.cat([f_worst,
                                 w.repeat(ch_lo.shape[0] // w.shape[0])])
            n_split += int(w.shape[0])

            if onnx_path is not None and rounds % attack_every == 1:
                # attack the worst open subboxes as one batched-box PGD
                widx = torch.argsort(w)[:64]
                cand, _ = attack.pgd(net, spec, lo=olo[widx], hi=ohi[widx],
                                     restarts=256, iters=60, device=device,
                                     time_budget=1.5, seed=rounds)
                if cand is not None:
                    ok, vinfo = attack.validate(onnx_path, spec, cand)
                    if ok:
                        return 'sat', {'witness': np.asarray(
                            vinfo.get('witness_inbox', cand))}
        if rounds % 32 == 0:
            log(f'[vc2/bab] round={rounds} frontier={int(f_lo.shape[0])} '
                f'bounded={n_bounded} t={time.time() - t0:.1f}s')
    return 'unsat', {'bounded': n_bounded, 'splits': n_split,
                     'rounds': rounds}


def relu_split_bab(net, spec, W, bias, disj_idx, lo, hi, deadline,
                   device='cpu', batch=256, beta_iters=12, onnx_path=None,
                   attack_every=16, root_inter=None, log=lambda m: None):
    """ReLU-phase splitting BaB (no-reforward): intermediates stay ROOT
    bounds; each domain carries sign clamps, and the bound comes from
    alpha+beta CROWN under those clamps (v1 _crown_bab_noreforward / abcrown
    beta-CROWN style). Action score is BaBSR: |pre-act adjoint| x triangle
    intercept, from the same backward pass that produced the bound.

    Domains are (worst_lb, splits) with splits a tuple of
    (relu_name, neuron, sign); clamps materialize densely per batch.
    """
    import heapq
    dev = torch.device(device)
    dt = torch.float32
    W = W.to(dev, dt)
    bias = bias.to(dev, dt)
    q = W.shape[0]
    D = int(disj_idx.max()) + 1 if disj_idx.numel() else 0
    sel = torch.zeros(D, q, device=dev, dtype=torch.bool)
    sel[disj_idx, torch.arange(q)] = True
    lo1 = lo.reshape(1, -1).to(dev, dt)
    hi1 = hi.reshape(1, -1).to(dev, dt)

    if root_inter is None:
        root_inter = backward.intermediates(net, lo1, hi1)
    relu_edges = [nm for nm in net.order
                  if net.ops[nm].kind == 'nonlin' and net.ops[nm].fn == 'relu']

    heap = [(-float('inf'), 0, ())]           # (worst_lb, tiebreak, splits)
    tick = 1
    n_bounded = rounds = 0
    t0 = time.time()

    def refuted_of(lbq):
        pos = (lbq + bias) > 0
        r = torch.zeros(lbq.shape[0], D, device=dev, dtype=torch.bool)
        for dd in range(D):
            r[:, dd] = (pos & sel[dd]).any(dim=1)
        return r

    while heap:
        if time.time() > deadline:
            return 'timeout', {'frontier': len(heap), 'bounded': n_bounded}
        rounds += 1
        n_relu_total = sum(net.ops[nm].n for nm in relu_edges)
        widest = max(net.ops[o].n for o in net.order)
        per_dom = (n_relu_total * 10 + q * widest * 12) * 4   # alpha/beta/adam + adjoints
        bs = min(batch, memory.chunk_size(len(heap), per_dom, dev))
        batch_doms = [heapq.heappop(heap) for _ in range(min(bs, len(heap)))]
        B = len(batch_doms)
        blo = lo1.expand(B, -1)
        bhi = hi1.expand(B, -1)
        clamps = {}
        for bi, (_, _, splits) in enumerate(batch_doms):
            for nm, j, sgn in splits:
                if nm not in clamps:
                    clamps[nm] = torch.zeros(B, net.ops[nm].n, device=dev,
                                             dtype=torch.int8)
                clamps[nm][bi, j] = sgn
        # reforward-IBP under the clamps, intersected with the (tighter at
        # the root, clamp-blind) root intermediates: best of both regimes
        ib_state = backward.fwd.interval(net, blo, bhi, return_state=True,
                                         clamps=clamps)
        ib = backward._inter_from_state(net, lambda e: ib_state[e])
        inter = {}
        for k2, v in root_inter.items():
            rv = tuple(t.expand(B, -1) for t in v)
            iv = ib[k2]
            merged = []
            for j2 in range(0, len(rv), 2):
                merged.append(torch.maximum(rv[j2], iv[j2]))
                merged.append(torch.minimum(rv[j2 + 1], iv[j2 + 1]))
            inter[k2] = tuple(merged)
        adj = {}
        lbq = backward.crown(net, blo, bhi, W, inter, clamps=clamps,
                             collect_adjoints=adj)
        lb_ab = backward.alpha_beta_crown(net, blo, bhi, W, inter, clamps,
                                          iters=beta_iters, thresholds=-bias)
        lbq = torch.maximum(lbq, lb_ab)
        n_bounded += B
        refuted = refuted_of(lbq)
        open_mask = ~refuted.all(dim=1)
        if open_mask.any():
            # BaBSR score per relu edge; argmax action per open domain
            best_score = torch.full((B,), -torch.inf, device=dev)
            best_edge = [None] * B
            best_j = torch.zeros(B, dtype=torch.long, device=dev)
            for nm in relu_edges:
                l, h = inter[nm]
                cl = clamps.get(nm)
                if cl is not None:
                    l, h = backward.clamped_bounds((l, h), cl)
                unstable = (l < 0) & (h > 0)
                if not bool(unstable.any()):
                    continue
                intercept = (-h * l / (h - l).clamp_min(1e-30)).clamp_min(0.0)
                a = adj.get(nm)
                s = (a.abs().amax(dim=1) if a is not None else
                     torch.ones_like(l)) * intercept * unstable
                v, j = s.max(dim=1)
                better = v > best_score
                best_score = torch.where(better, v, best_score)
                best_j = torch.where(better, j, best_j)
                for bi in torch.nonzero(better, as_tuple=False).flatten().tolist():
                    best_edge[bi] = nm
            w_dom = (lbq + bias).min(dim=1).values
            for bi in torch.nonzero(open_mask, as_tuple=False).flatten().tolist():
                if best_edge[bi] is None:
                    # no unstable relu left: relaxation exact -> the domain
                    # can only be sat; try to falsify, else give up loudly
                    return 'unknown', {'reason': 'exhausted splits',
                                       'bounded': n_bounded}
                base = batch_doms[bi][2]
                for sgn in (1, -1):
                    heapq.heappush(heap, (float(w_dom[bi]), tick,
                                          base + ((best_edge[bi],
                                                   int(best_j[bi]), sgn),)))
                    tick += 1
            if onnx_path is not None and rounds % attack_every == 1:
                cand, _ = attack.pgd(net, spec, lo=lo1[0], hi=hi1[0],
                                     restarts=128, iters=60, device=device,
                                     time_budget=1.5, seed=rounds)
                if cand is not None:
                    ok, vinfo = attack.validate(onnx_path, spec, cand)
                    if ok:
                        return 'sat', {'witness': np.asarray(
                            vinfo.get('witness_inbox', cand))}
        if rounds % 16 == 0:
            log(f'[vc2/rbab] round={rounds} frontier={len(heap)} '
                f'bounded={n_bounded} t={time.time() - t0:.1f}s')
    return 'unsat', {'bounded': n_bounded, 'rounds': rounds}
