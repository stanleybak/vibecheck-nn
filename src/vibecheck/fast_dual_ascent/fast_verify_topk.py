"""fast_verify_topk.py — the speedup_warm WINNER (see ../plan.md / ../SUMMARY.md).

K=1 warm-start dual node bound with a sort-free line search instead of the full O(n log n)
breakpoint sort (which was 80% of the kernel eager AND blocked torch.compile — InductorError on
`sort` over the full n; note `topk` itself DOES compile). DEFAULT line search = **ls='logbucket'**
(log-spaced histogram, left-edge understep — sort-free AND topk-free, ties topk on compiled wall,
robust to any n). ls='topk' (topk(K) partial sort) is the validated alternative.

Both keep the bound g EXACT (pre-step eval, sound for any λ≥0). The line search only sets children's
warm-start λ → affects node count, not soundness, and (measured) not compiled wall either (the GEMMs
dominate compiled).

INFEASIBILITY CERT — the original slope-drop test (slope0 > Σdecr, "exact in ℝ") turned out UNSOUND
in float32: near the dual optimum both sides are ~1e-3 apart by ~1e-6, and the sign is noise. On 94
Gurobi-labeled joint-infeasibility nodes (metaroom) it false-positive-pruned 42 FEASIBLE nodes
(Gurobi minima as low as −35) — a pruned feasible node can hide a counterexample. The DEFAULT cert
is now the FAR PROBE (see node_bound_logbucket): one aggressive dual step past the kinks, prune only
on a real g(λ_far)>0 witness — sound by weak duality, 0/42 false positives on the labeled cases, and
measured 480,454 vs 549,068 nodes (−12.5%) at equal wall on the saved tinyimagenet q177 state.
VC_FARPROBE=0 restores the legacy slope cert (debug only).

Result on case_1175_unsat: exact-sort 1.03 s → topk eager 0.28 s → **topk compiled 0.134 s**
(unsat, 41,982 nodes vs 41,712 exact, +0.6%). ~53× faster than vibecheck (7.10 s). Bound g
validated against the exact node LP by Gurobi to −1.4e-6.

Why the topk step stays sound and doesn't blow up the frontier (unlike bucket line search):
the line search only sets the children's warm-start λ (the bound g is the pre-step eval, valid
for ANY λ≥0). On the ~90% of nodes whose slope crosses within the K_top smallest breakpoints
the step is EXACT; on the tail it UNDERSTEPS to the K_top-th breakpoint — a shorter, still-
ascending, still-λ≥0 step. Understepping is gentle (node count +0.6%); bucket's ±½-bin
OVERshoot was what blew up.

Interface mirrors fastbab/fast_verify_warm.py.
"""
from __future__ import annotations
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import time
import torch
from .fast_verify_dual import parse_problem, parse_problem_gpu, Problem  # geometry with c_in/z_lo/z_hi

_TOL = 1e-9


def node_bound_topk(F, sides, lam0, lam1, nu=None, K_top=256, use_farprobe=False):
    """K=1 warm-start dual bound. Returns (best [B], lam0', lam1'). The bound `best` = g at the
    inherited λ (sound for any λ≥0); the returned λ is one topk-line-search step ahead for the
    children's warm start. Infeasible nodes (slope never crosses) get best=+inf via the
    order-independent total slope-drop — exact in ℝ but float-fragile near the dual optimum
    (see module docstring); this legacy 'topk' line search has no far-probe port."""
    if use_farprobe:
        raise NotImplementedError("far-probe is implemented for ls='logbucket' only; use the default")
    if nu is not None and nu.shape[1] > 0:
        raise NotImplementedError("sibling halfspaces are for ls='logbucket' only; use the default")
    a_g = F['a_g']; D, n = a_g.shape; B = sides.shape[0]
    el, eh = F['el'], F['eh']; width = eh - el
    sign = (1 - 2 * sides).to(a_g.dtype)
    rho = torch.where(sides == 0, F['ratio_off'], F['ratio_on'])
    c0_path = F['c0'] + torch.where(sides == 0, F['c0_off'], F['c0_on']).sum(1)
    cin, zlo, zhi = F['c_in'], F['z_lo'], F['z_hi']
    b0 = torch.where(sides == 0, (-cin).expand(B, D), cin.expand(B, D))
    b1 = torch.where(sides == 0, zlo.expand(B, D), zhi.expand(B, D))
    rc = F['d_base'].unsqueeze(0) + (rho + sign * (lam0 - lam1)) @ a_g
    x = torch.where(rc < 0, eh.expand(B, n), el.expand(B, n))
    g = c0_path + (rc * x).sum(-1) - (lam0 * b0 + lam1 * b1).sum(-1)
    best = g
    pend = (best <= _TOL).to(a_g.dtype)
    p = x @ a_g.t()
    s0 = sign * (p + cin); s1 = torch.where(sides == 0, -p - zlo, p - zhi)
    sp0 = torch.where((lam0 <= _TOL) & (s0 < 0), torch.zeros_like(s0), s0)
    sp1 = torch.where((lam1 <= _TOL) & (s1 < 0), torch.zeros_like(s1), s1)
    slope0 = (sp0 * s0 + sp1 * s1).sum(-1)
    da = (sign * (sp0 - sp1)) @ a_g
    positive = ((da > 0) & (rc < 0)) | ((da < 0) & (rc > 0))
    decr_all = torch.where(positive, width.unsqueeze(0) * da.abs(), torch.zeros_like(da))
    # slope-drop test (order-independent Σdecr) — exact in ℝ, but fragile in float at the dual opt.
    infeasible = (slope0 > _TOL) & (decr_all.sum(-1) < slope0 - _TOL)
    best = torch.where(infeasible, torch.full_like(best, 1e30), best)
    # topk smallest breakpoints → partial line search (exact on ~90%, sound understep on tail).
    eta_j = torch.where(positive, -rc / da, torch.full_like(rc, float('inf')))
    es, idx = torch.topk(eta_j, min(K_top, n), dim=-1, largest=False)
    es, order = es.sort(-1); idx = torch.gather(idx, -1, order)       # sort just the K (cheap)
    decr_k = torch.gather(decr_all, -1, idx)
    decr_k = torch.where(torch.isfinite(es), decr_k, torch.zeros_like(decr_k))
    below = (slope0.unsqueeze(-1) - decr_k.cumsum(-1)) <= _TOL
    crossed = below.any(-1)
    eta_cross = torch.gather(es, -1, below.float().argmax(-1, keepdim=True)).squeeze(-1)
    finite = torch.isfinite(es); last_fin = (finite.cumsum(-1) * finite).argmax(-1)
    eta_under = torch.gather(es, -1, last_fin.unsqueeze(-1)).squeeze(-1)
    eta = torch.where(crossed, eta_cross, eta_under)
    eta = torch.where(torch.isfinite(eta), eta, torch.zeros_like(eta))
    eta = torch.where((slope0 > _TOL) & ~infeasible, eta, torch.zeros_like(eta)).clamp_min(0.0) * pend
    lam0 = (lam0 + eta.unsqueeze(-1) * sp0).clamp_min(0.0)
    lam1 = (lam1 + eta.unsqueeze(-1) * sp1).clamp_min(0.0)
    return best, lam0, lam1, nu


def node_bound_logbucket(F, sides, lam0, lam1, nu, Kb=256, use_farprobe=False):
    """K=1 warm-start dual bound with a LOG-spaced bucket line search (the default). Same bound g
    as node_bound_topk; differs in the step-η search and carries the sound FAR-PROBE cert (default;
    see module docstring — the legacy slope cert is float-fragile/unsound).

    The line search has to locate where the concave PWL slope crosses zero among the breakpoints
    η_j = -rc_j/da_j. Their VALUE range is dominated by a few huge outliers while the true crossing
    sits at tiny η, so LINEAR bins waste all resolution and overshoot (→ blow-up). LOG bins give
    constant RELATIVE resolution (~e^binw per bin) — fine where the crossing is. We take the
    crossing bin's LEFT edge, so the step UNDERSHOOTS the peak (sound, gentle warm-start; can never
    overshoot). Sort-free AND topk-free: just a scatter_add histogram + cumsum, so it compiles and
    is robust to any n (bins are n-independent; Kb may exceed the number of breakpoints).
    """
    a_g = F['a_g']; D, n = a_g.shape; B = sides.shape[0]
    el, eh = F['el'], F['eh']; width = eh - el
    sign = (1 - 2 * sides).to(a_g.dtype)
    rho = torch.where(sides == 0, F['ratio_off'], F['ratio_on'])
    c0_path = F['c0'] + torch.where(sides == 0, F['c0_off'], F['c0_on']).sum(1)
    cin, zlo, zhi = F['c_in'], F['z_lo'], F['z_hi']
    b0 = torch.where(sides == 0, (-cin).expand(B, D), cin.expand(B, D))
    b1 = torch.where(sides == 0, zlo.expand(B, D), zhi.expand(B, D))
    hs_a, hs_b = F['hs_a'], F['hs_b']                # (M,n), (M,) — M may be 0
    rc = (F['d_base'].unsqueeze(0) + (rho + sign * (lam0 - lam1)) @ a_g
          + nu @ hs_a)
    x = torch.where(rc < 0, eh.expand(B, n), el.expand(B, n))
    g = (c0_path + (rc * x).sum(-1) - (lam0 * b0 + lam1 * b1).sum(-1)
         - (nu * hs_b).sum(-1))
    best = g
    pend = (best <= _TOL).to(a_g.dtype)
    p = x @ a_g.t()
    s0 = sign * (p + cin); s1 = torch.where(sides == 0, -p - zlo, p - zhi)
    sp0 = torch.where((lam0 <= _TOL) & (s0 < 0), torch.zeros_like(s0), s0)
    sp1 = torch.where((lam1 <= _TOL) & (s1 < 0), torch.zeros_like(s1), s1)
    # static-halfspace subgradient (sibling output constraints, INVPROP-style)
    s2 = x @ hs_a.t() - hs_b                          # (B, M)
    sp2 = torch.where((nu <= _TOL) & (s2 < 0), torch.zeros_like(s2), s2)
    slope0 = (sp0 * s0 + sp1 * s1).sum(-1) + (sp2 * s2).sum(-1)
    da = (sign * (sp0 - sp1)) @ a_g + sp2 @ hs_a
    positive = ((da > 0) & (rc < 0)) | ((da < 0) & (rc > 0))
    decr_all = torch.where(positive, width.unsqueeze(0) * da.abs(), torch.zeros_like(da))
    if use_farprobe:
        # No slope extrapolation: infeasibility is certified after the line search by a REAL
        # g(λ_far)>0 witness at a far step (computed below). Keep `best` = pre-step bound here.
        infeasible = torch.zeros(B, dtype=torch.bool, device=a_g.device)
    else:
        # slope-drop test (order-independent Σdecr) — exact in ℝ, but fragile in float at the dual
        # optimum: measured 42/94 FALSE-POSITIVE prunes on Gurobi-labeled joint-infeasibility cases
        # (a false positive prunes a feasible node → can hide a counterexample → UNSOUND). Kept only
        # as the legacy escape hatch (VC_FARPROBE=0).
        infeasible = (slope0 > _TOL) & (decr_all.sum(-1) < slope0 - _TOL)
    best = torch.where(infeasible, torch.full_like(best, 1e30), best)
    # log-spaced histogram of the slope-drops; LEFT edge of the crossing bin = sound understep.
    INF = torch.full_like(rc, float('inf'))
    eta_j = torch.where(positive, -rc / da, INF)
    emin = torch.where(positive, eta_j, INF).min(-1).values.clamp_min(1e-12)
    emax = torch.where(positive, eta_j, torch.full_like(eta_j, -float('inf'))).max(-1).values.clamp_min(1e-12)
    le = torch.log(emin)
    lw = (torch.log(emax) - le).clamp_min(1e-30) / Kb            # safe when emin==emax (1 breakpoint)
    idx = (((torch.log(eta_j.clamp_min(1e-12)) - le.unsqueeze(-1)) / lw.unsqueeze(-1)).floor().long()).clamp(0, Kb - 1)
    idx = torch.where(positive, idx, torch.full_like(idx, Kb))   # park non-breakpoints in a dump bin
    buckets = torch.zeros(B, Kb + 1, device=a_g.device, dtype=a_g.dtype)
    buckets.scatter_add_(1, idx, decr_all)
    crossed = buckets[:, :Kb].cumsum(-1) >= slope0.unsqueeze(-1)
    first = crossed.float().argmax(-1)
    eta = torch.where(crossed.any(-1), torch.exp(le + first.to(a_g.dtype) * lw), emax)
    eta = torch.where(torch.isfinite(eta), eta, torch.zeros_like(eta))   # no-breakpoint nodes → 0
    if use_farprobe:
        # FAR PROBE: instead of extrapolating the slope sign, take ONE aggressive step to the
        # furthest sound point on the ascent ray and evaluate g there. g(λ)≤p* for any λ≥0, so
        # g_far>0 is a real infeasibility/robustness certificate (folded into `best`, pruned by
        # best>0 upstream). Infeasible nodes have an unbounded dual → g climbs fast → one probe
        # usually certifies; feasible nodes crest, so g_far just tightens the bound (never wrong).
        # The furthest valid step is the slope-zero crossing (if the slope turns) else the λ≥0
        # wall, or far past the last kink when no wall binds.
        inf_D = torch.full_like(lam0, float('inf'))
        w0 = torch.where(sp0 < -_TOL, -lam0 / sp0, inf_D)
        w1 = torch.where(sp1 < -_TOL, -lam1 / sp1, inf_D)
        ebnd = torch.minimum(w0.min(-1).values, w1.min(-1).values)        # λ≥0 wall (B,)
        if nu.shape[1] > 0:                                               # ν≥0 wall
            w2 = torch.where(sp2 < -_TOL, -nu / sp2,
                             torch.full_like(nu, float('inf')))
            ebnd = torch.minimum(ebnd, w2.min(-1).values)
        far = torch.where(torch.isfinite(ebnd), ebnd, emax * 1e6)         # past the last kink
        eta_c = torch.where(crossed.any(-1), eta, far)
        eta_c = torch.where((slope0 > _TOL) & torch.isfinite(eta_c), eta_c,
                            torch.zeros_like(eta_c)).clamp_min(0.0)
        rc_f = rc + eta_c.unsqueeze(-1) * da
        x_f = torch.where(rc_f < 0, eh.expand(B, n), el.expand(B, n))
        l0f = (lam0 + eta_c.unsqueeze(-1) * sp0).clamp_min(0.0)
        l1f = (lam1 + eta_c.unsqueeze(-1) * sp1).clamp_min(0.0)
        nuf = (nu + eta_c.unsqueeze(-1) * sp2).clamp_min(0.0)
        g_far = (c0_path + (rc_f * x_f).sum(-1) - (l0f * b0 + l1f * b1).sum(-1)
                 - (nuf * hs_b).sum(-1))
        best = torch.maximum(best, g_far)
    eta = torch.where((slope0 > _TOL) & ~infeasible, eta, torch.zeros_like(eta)).clamp_min(0.0) * pend
    lam0 = (lam0 + eta.unsqueeze(-1) * sp0).clamp_min(0.0)
    lam1 = (lam1 + eta.unsqueeze(-1) * sp1).clamp_min(0.0)
    nu = (nu + eta.unsqueeze(-1) * sp2).clamp_min(0.0)
    return best, lam0, lam1, nu


_KERNELS = {'logbucket': node_bound_logbucket, 'topk': node_bound_topk}


class Verifier:
    def __init__(self, device='cuda', compile=True, ls='logbucket', K=256, chunk=16384,
                 warm_depths=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64), K_top=None):
        torch.backends.cuda.matmul.allow_tf32 = False           # soundness: TF32 flips decisions
        if K_top is not None:                                   # back-compat: K_top= forced topk
            ls, K = 'topk', K_top
        self.device = torch.device(device); self.chunk = chunk; self.ls = ls; self.K = K
        # VC_FARPROBE (DEFAULT ON): sound infeasibility cert. The legacy slope-extrapolation flag is
        # UNSOUND — its float-fragile test produced false-positive prunes (42/94 on Gurobi-labeled
        # joint-infeasibility cases; a wrongly pruned feasible node can hide a counterexample). The
        # far probe replaces it with a real g(λ_far)>0 witness (g(λ)≤p* for any λ≥0, so a positive
        # bound is always a sound certificate). Measured on the saved tinyimagenet q177 Phase-8 state
        # (deterministic, compiled): far 480,454 nodes / ~1.0s vs slope 549,068 / ~1.0s — sound AND
        # −12.5% nodes at equal wall. Set VC_FARPROBE=0 for the legacy slope cert (debug only).
        self.use_farprobe = os.environ.get('VC_FARPROBE', '1') not in ('0', 'false', 'False')
        base = _KERNELS[ls]; use_farprobe = self.use_farprobe
        kern = (lambda F, s, l0, l1, nu: base(F, s, l0, l1, nu, K, use_farprobe))
        self._kernel = torch.compile(kern, dynamic=True) if compile else kern
        self._warm_depths = warm_depths if compile else ()
        self._warmed = set()   # depths already compile-warmed (reused across queries)

    def _upload(self, prob):
        dev = self.device
        G = {k: torch.as_tensor(getattr(prob, k), device=dev, dtype=torch.float32)
             for k in ('a_g', 'ratio_off', 'ratio_on', 'c0_off', 'c0_on', 'c_in', 'z_lo', 'z_hi')}
        G['e_lb'] = torch.as_tensor(prob.e_lb, device=dev, dtype=torch.float32)
        G['e_hi'] = torch.as_tensor(prob.e_hi, device=dev, dtype=torch.float32)
        G['_d_t'] = prob.d_t; G['_e_new'] = prob.e_new_col; G['_c0'] = prob.c0
        n = int(prob.a_g.shape[1])
        if getattr(prob, 'hs_a', None) is not None:
            G['hs_a'] = torch.as_tensor(prob.hs_a, device=dev, dtype=torch.float32)
            G['hs_b'] = torch.as_tensor(prob.hs_b, device=dev, dtype=torch.float32)
        else:
            G['hs_a'] = torch.zeros(0, n, device=dev, dtype=torch.float32)
            G['hs_b'] = torch.zeros(0, device=dev, dtype=torch.float32)
        return G

    def _F(self, G, depth):
        d_base = G['_d_t'].copy(); d_base[G['_e_new'][:depth]] = 0.0
        F = {k: G[k][:depth] for k in ('a_g', 'ratio_off', 'ratio_on', 'c0_off', 'c0_on', 'c_in', 'z_lo', 'z_hi')}
        F['d_base'] = torch.as_tensor(d_base, device=self.device, dtype=torch.float32)
        F['c0'] = G['_c0']; F['el'] = G['e_lb']; F['eh'] = G['e_hi']
        F['hs_a'] = G['hs_a']; F['hs_b'] = G['hs_b']
        return F

    def _bounds(self, F, sides, lam0, lam1, nu):
        B = sides.shape[0]
        best = torch.empty(B, device=self.device)
        o0 = torch.empty_like(lam0); o1 = torch.empty_like(lam1)
        onu = torch.empty_like(nu); i = 0
        while i < B:
            step = min(self.chunk, B - i)
            while True:
                try:
                    bb, l0, l1, nv = self._kernel(F, sides[i:i+step].long().contiguous(),
                                              lam0[i:i+step].contiguous(), lam1[i:i+step].contiguous(),
                                              nu[i:i+step].contiguous())
                    best[i:i+step] = bb; o0[i:i+step] = l0; o1[i:i+step] = l1; onu[i:i+step] = nv
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if step <= 256: raise
                    step //= 2; self.chunk = max(256, min(self.chunk, step))
            i += step
        return best, o0, o1, onu

    def verify_query(self, state, qw, qb, scored_keys, *, time_limit=120.0,
                     extra_hs=()):
        # On CUDA, build the split matrix directly on-device (skips the dense
        # host (n_unstable, n_gens) alloc + upload). Identical Problem otherwise.
        # extra_hs (sibling output constraints for conjunctive disjuncts) needs
        # the host parse path (it projects through the full output map).
        if extra_hs:
            prob = parse_problem(state, qw, qb, scored_keys, extra_hs=extra_hs)
        else:
            prob = (parse_problem_gpu(state, qw, qb, scored_keys, self.device)
                    if self.device.type == 'cuda'
                    else parse_problem(state, qw, qb, scored_keys))
        return self.verify(prob, time_limit=time_limit)

    def verify(self, prob, *, time_limit=120.0):
        dev = self.device; G = self._upload(prob)
        # Compile-warmup pre-specialises the kernel at common depths. The
        # Verifier is reused across queries, so a depth warmed once stays
        # compiled — skip re-warming it (saves redundant per-query launches).
        M = int(G['hs_a'].shape[0])
        for D in self._warm_depths:
            if D <= prob.n_splits and D not in self._warmed:
                z = torch.zeros(8, D, device=dev)
                # warm with DISTINCT lam0/lam1 tensors — passing `z, z` makes torch.compile specialise
                # on `l0 is l1` (aliased), forcing a recompile when verify passes distinct tensors.
                self._kernel(self._F(G, D), torch.zeros(8, D, device=dev, dtype=torch.long), z, z.clone(),
                             torch.zeros(8, M, device=dev))
                self._warmed.add(D)
        if dev.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter(); elapsed = lambda: time.perf_counter() - t0
        if prob.root_bound > 0:
            return 'unsat', dict(nodes=0, depth=0, peak_frontier=0, wall=0.0)
        sides = torch.tensor([[0], [1]], device=dev, dtype=torch.int8)
        lam0 = torch.zeros(2, 1, device=dev); lam1 = torch.zeros(2, 1, device=dev)
        nu = torch.zeros(2, M, device=dev)
        nodes_total = 0; depth = 1; peak = 2

        def _unknown(open_n, reason):
            return 'unknown', dict(nodes=nodes_total, depth=depth, peak_frontier=peak,
                                   open=int(open_n), reason=reason, wall=elapsed())
        while sides.shape[0] > 0:
            if elapsed() > time_limit:
                return _unknown(sides.shape[0], 'time_limit')
            nodes_total += sides.shape[0]; peak = max(peak, sides.shape[0])
            try:
                best, o0, o1, onu = self._bounds(self._F(G, depth), sides, lam0, lam1, nu)
                keep = best <= _TOL
                ss = sides[keep]; l0 = o0[keep]; l1 = o1[keep]; nk = onu[keep]
                if ss.shape[0] == 0:
                    return 'unsat', dict(nodes=nodes_total, depth=depth, peak_frontier=peak, wall=elapsed())
                if depth >= prob.n_splits:
                    return _unknown(ss.shape[0], 'splits_exhausted')
                z8 = torch.zeros(ss.shape[0], 1, device=dev, dtype=torch.int8)
                zf = torch.zeros(ss.shape[0], 1, device=dev)
                sides = torch.cat([torch.cat([ss, z8], 1), torch.cat([ss, z8 + 1], 1)], 0)
                lam0 = torch.cat([torch.cat([l0, zf], 1), torch.cat([l0, zf], 1)], 0)
                lam1 = torch.cat([torch.cat([l1, zf], 1), torch.cat([l1, zf], 1)], 0)
                nu = torch.cat([nk, nk], 0)
            except torch.cuda.OutOfMemoryError:
                open_n = int(sides.shape[0]); sides = lam0 = lam1 = nu = best = o0 = o1 = None
                torch.cuda.empty_cache()
                return 'unknown', dict(nodes=nodes_total, depth=depth, peak_frontier=peak,
                                       open=open_n, reason='oom', wall=elapsed())
            depth += 1
        return 'unsat', dict(nodes=nodes_total, depth=depth, peak_frontier=peak, wall=elapsed())
