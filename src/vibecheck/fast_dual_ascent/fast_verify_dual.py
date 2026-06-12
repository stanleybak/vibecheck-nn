"""fast_verify_dual.py — EXPERIMENTAL variant of fast_verify_clear.py whose per-node bound
is K steps of Lagrangian DUAL ASCENT on the branch cuts (not just the box bound). This is
the factored + bucket-line-search dual ascent from the parent dir, dropped into the BaB.

Why: fast_verify_clear's box bound IGNORES the branch halfspace cuts, so on some robust
disjuncts (e.g. case_7943 disjunct 2) it's too loose — 80% of the blow-up frontier is
actually robust (exact LP > 0) but the box bound can't see it, so the tree explodes. The
dual bound g(λ)=c0+min_box((d+Aᵀλ)·e)−λ·b USES the cuts (λ on the halfspaces) and, with a
few ascent steps, certifies those nodes → collapses the tree. The per-node early-stop
(freeze once best_g>0) also prunes INFEASIBLE nodes for free (their dual is unbounded).

Soundness unchanged: g(λ) ≤ LP_min for ANY λ≥0, so best_g>0 still certifies. TF32 off.

Interface identical to fast_verify_clear, plus Verifier(K=...).
"""
from __future__ import annotations
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import pickle, time
from dataclasses import dataclass
import numpy as np
import torch

_TOL = 1e-9


@dataclass
class Problem:
    n_gens: int
    e_lb: np.ndarray; e_hi: np.ndarray
    d_t: np.ndarray; c0: float
    a_g: np.ndarray              # (S,n) split rows (own e_new col zeroed)
    e_new_col: np.ndarray
    ratio_off: np.ndarray; ratio_on: np.ndarray
    c0_off: np.ndarray; c0_on: np.ndarray
    c_in: np.ndarray             # (S,) z_k center  (cut b-values below)
    z_lo: np.ndarray; z_hi: np.ndarray   # (S,) parallelogram caps
    # Static halfspaces a·e ≤ b in generator space, dualized into every node
    # bound (INVPROP-style sibling output constraints for conjunctive
    # disjuncts; empty for single-conjunct specs = all regular-track).
    hs_a: np.ndarray = None      # (M, n) or None
    hs_b: np.ndarray = None      # (M,) or None

    @property
    def n_splits(self): return self.a_g.shape[0]

    @property
    def root_bound(self):
        return float(self.c0 + np.where(self.d_t > 0, self.d_t * self.e_lb,
                                        self.d_t * self.e_hi).sum())


def parse_problem(state, qw, qb, scored_keys, extra_hs=()) -> Problem:
    """extra_hs: iterable of (w_j, b_j) sibling output constraints. During
    refutation the SAT set may be assumed, and each sibling constraint
    requires w_j·y + b_j ≤ 0; projected through the state's output map this
    is the generator-space halfspace (w_j@G_out)·e ≤ −(w_j@c_out + b_j),
    dualized into every node bound (g stays ≤ p* — sound for any ν ≥ 0)."""
    qw = np.asarray(qw, float); qb = float(qb)
    n = int(state['n_gens']); n_input = int(state['n_input']); ul = state['unstable_list']
    e_lb = np.zeros(n); e_hi = np.zeros(n); e_lb[:n_input] = -1.0; e_hi[:n_input] = 1.0
    nu = len(ul)
    lam = np.empty(nu); mu = np.empty(nu); c_in = np.empty(nu)
    e_new = np.empty(nu, np.int64); a_pu = np.zeros((nu, n)); idx_by_key = {}
    for j, u in enumerate(ul):
        lam[j] = float(u['lam']); mu[j] = float(u['mu']); c_in[j] = float(u['c_in'])
        e_new[j] = int(u['e_new_col']); e_lb[e_new[j]] = -1.0; e_hi[e_new[j]] = 1.0
        a_pu[j, np.asarray(u['row_indices'], np.int64)] = np.asarray(u['row_values'], float)
        a_pu[j, e_new[j]] = 0.0
        idx_by_key[(u['layer_idx'], u['neuron_idx'])] = j
    safe = mu > 1e-12
    # cut b-values: OFF z≤0 (b=-c_in) & z≥-2μ/λ (row -a·e ≤ z_lo); ON z≥0 (b=c_in) & z≤2μ/(1-λ)
    z_lo = np.where(safe & (lam > 1e-12), 2 * mu / np.maximum(lam, 1e-30) + c_in, 1e9)
    z_hi = np.where(safe & (1 - lam > 1e-12), 2 * mu / np.maximum(1 - lam, 1e-30) - c_in, 1e9)
    sidx = np.fromiter((idx_by_key[k] for k in scored_keys), np.int64, len(scored_keys))
    _G_csr = state['obj_G_out_csr']
    d_t = np.asarray(_G_csr.T.dot(qw), float).ravel()   # sparse dot — no (n_out, n_gens) toarray
    c0 = float(qw @ np.asarray(state['obj_c_out'], float) + qb)
    lamS, cinS, enS, safeS, muS = lam[sidx], c_in[sidx], e_new[sidx], safe[sidx], mu[sidx]
    dnew = d_t[enS]; inv = 1.0 / np.where(safeS, muS, 1.0)
    hs_a = hs_b = None
    if extra_hs:
        c_out = np.asarray(state['obj_c_out'], float)
        hs_a = np.stack([np.asarray(_G_csr.T.dot(np.asarray(wj, float)),
                                    float).ravel() for wj, _ in extra_hs])
        hs_b = np.array([-(float(np.asarray(wj, float) @ c_out) + float(bj))
                         for wj, bj in extra_hs])
    return Problem(
        n_gens=n, e_lb=e_lb, e_hi=e_hi, d_t=d_t, c0=c0, a_g=a_pu[sidx], e_new_col=enS,
        ratio_off=np.where(safeS, -(lamS * inv) * dnew, 0.0),
        ratio_on=np.where(safeS, ((1 - lamS) * inv) * dnew, 0.0),
        c0_off=np.where(safeS, -(1 + lamS * cinS * inv) * dnew, -dnew),
        c0_on=np.where(safeS, ((1 - lamS) * cinS * inv - 1) * dnew, -dnew),
        c_in=cinS, z_lo=z_lo[sidx], z_hi=z_hi[sidx],
        hs_a=hs_a, hs_b=hs_b)


def parse_problem_gpu(state, qw, qb, scored_keys, device) -> Problem:
    """GPU-native variant of ``parse_problem``: builds the split matrix ``a_g``
    directly on the device via a flat scatter from the sparse pre-ReLU rows,
    instead of allocating a dense ``(n_unstable, n_gens)`` numpy array on the
    host (≈188 MB on tinyimagenet) and uploading it. Only one ``(S, n_gens)``
    matrix is resident at a time (built per query at BnB time), so this is
    memory-safe on small GPUs. All other fields are small host vectors, kept on
    the CPU (``_F`` masks ``d_t`` per depth there). Bit-identical to
    ``parse_problem`` — validated verdict + node counts match exactly."""
    qw = np.asarray(qw, float); qb = float(qb)
    n = int(state['n_gens']); n_input = int(state['n_input']); ul = state['unstable_list']
    nu = len(ul)
    lam = np.empty(nu); mu = np.empty(nu); c_in = np.empty(nu); e_new = np.empty(nu, np.int64)
    idx_by_key = {}
    for j, u in enumerate(ul):
        lam[j] = float(u['lam']); mu[j] = float(u['mu']); c_in[j] = float(u['c_in'])
        e_new[j] = int(u['e_new_col'])
        idx_by_key[(u['layer_idx'], u['neuron_idx'])] = j
    sidx = np.fromiter((idx_by_key[k] for k in scored_keys), np.int64, len(scored_keys))
    S = int(sidx.size)
    # --- a_g (S, n) built on-device via flat scatter (no S*n host alloc/upload) ---
    a_g = torch.zeros(S, n, device=device, dtype=torch.float32)
    if S:
        lens = np.fromiter((len(ul[j]['row_indices']) for j in sidx), np.int64, S)
        if lens.sum():
            flat_cols = np.concatenate([np.asarray(ul[j]['row_indices'], np.int64) for j in sidx])
            flat_vals = np.concatenate([np.asarray(ul[j]['row_values'], float) for j in sidx])
            row_ids = np.repeat(np.arange(S), lens)
            a_g[torch.as_tensor(row_ids, device=device),
                torch.as_tensor(flat_cols, device=device)] = torch.as_tensor(
                    flat_vals, device=device, dtype=torch.float32)
        enS = e_new[sidx]
        a_g[torch.arange(S, device=device), torch.as_tensor(enS, device=device)] = 0.0
    # --- small host-side scalars (identical to parse_problem) ---
    e_lb = np.zeros(n); e_hi = np.zeros(n); e_lb[:n_input] = -1.0; e_hi[:n_input] = 1.0
    e_lb[e_new] = -1.0; e_hi[e_new] = 1.0
    safe = mu > 1e-12
    z_lo = np.where(safe & (lam > 1e-12), 2 * mu / np.maximum(lam, 1e-30) + c_in, 1e9)
    z_hi = np.where(safe & (1 - lam > 1e-12), 2 * mu / np.maximum(1 - lam, 1e-30) - c_in, 1e9)
    # sparse matvec for d_t — avoids densifying obj_G_out_csr (n_out × n_gens)
    d_t = np.asarray(qw @ state['obj_G_out_csr'], float).ravel()
    c0 = float(qw @ np.asarray(state['obj_c_out'], float) + qb)
    lamS, cinS, enS2, safeS, muS = lam[sidx], c_in[sidx], e_new[sidx], safe[sidx], mu[sidx]
    dnew = d_t[enS2]; inv = 1.0 / np.where(safeS, muS, 1.0)
    return Problem(
        n_gens=n, e_lb=e_lb, e_hi=e_hi, d_t=d_t, c0=c0, a_g=a_g, e_new_col=enS2,
        ratio_off=np.where(safeS, -(lamS * inv) * dnew, 0.0),
        ratio_on=np.where(safeS, ((1 - lamS) * inv) * dnew, 0.0),
        c0_off=np.where(safeS, -(1 + lamS * cinS * inv) * dnew, -dnew),
        c0_on=np.where(safeS, ((1 - lamS) * cinS * inv - 1) * dnew, -dnew),
        c_in=cinS, z_lo=z_lo[sidx], z_hi=z_hi[sidx])


def node_bound_dual(F: dict, sides: torch.Tensor, K: int, Kb: int = 256, use_farprobe: bool = False) -> torch.Tensor:
    """K-step factored dual-ascent bound g (B,). Each split contributes 2 halfspace rows
    (±a_g[j]); λ0,λ1 are their multipliers. rc = d_base + (ρ + sign·(λ0−λ1))·a_g stays a
    GEMM against the shared a_g. Bucket (O(n)) exact line search; per-node early-stop freezes
    a node once best>0 (also prunes infeasible nodes: their dual is unbounded). Sound."""
    a_g = F['a_g']; D, n = a_g.shape; B = sides.shape[0]
    el, eh = F['el'], F['eh']; width = eh - el
    sign = (1 - 2 * sides).to(a_g.dtype)                       # +1 OFF, -1 ON
    rho = torch.where(sides == 0, F['ratio_off'], F['ratio_on'])
    c0_path = F['c0'] + torch.where(sides == 0, F['c0_off'], F['c0_on']).sum(1)
    cin, zlo, zhi = F['c_in'], F['z_lo'], F['z_hi']            # (D,)
    b0 = torch.where(sides == 0, (-cin).expand(B, D), cin.expand(B, D))   # cut rhs, row 0
    b1 = torch.where(sides == 0, zlo.expand(B, D), zhi.expand(B, D))      # cut rhs, row 1
    lam0 = torch.zeros(B, D, device=a_g.device, dtype=a_g.dtype); lam1 = torch.zeros_like(lam0)
    best = torch.full((B,), -float('inf'), device=a_g.device, dtype=a_g.dtype)
    rc = F['d_base'].unsqueeze(0) + rho @ a_g                  # rc at λ=0
    for k in range(K):
        x = torch.where(rc < 0, eh.expand(B, n), el.expand(B, n))
        g = c0_path + (rc * x).sum(-1) - (lam0 * b0 + lam1 * b1).sum(-1)
        best = torch.maximum(best, g)
        last = (k == K - 1)
        if last and not use_farprobe:
            break
        pend = (best <= _TOL).to(a_g.dtype)                    # 1 while uncertified, else 0 (freeze)
        p = x @ a_g.t()                                        # (B,D)  a_g[j]·x*
        s0 = sign * (p + cin)                                  # subgradient, halfspace row 0
        s1 = torch.where(sides == 0, -p - zlo, p - zhi)        #            row 1
        sp0 = torch.where((lam0 <= _TOL) & (s0 < 0), torch.zeros_like(s0), s0)   # mask at λ=0
        sp1 = torch.where((lam1 <= _TOL) & (s1 < 0), torch.zeros_like(s1), s1)
        slope0 = (sp0 * s0 + sp1 * s1).sum(-1)
        da = (sign * (sp0 - sp1)) @ a_g
        w0 = torch.where(sp0 < 0, -lam0 / sp0, torch.full_like(lam0, float('inf')))   # λ≥0 wall
        w1 = torch.where(sp1 < 0, -lam1 / sp1, torch.full_like(lam1, float('inf')))
        ebnd = torch.minimum(w0.min(-1).values, w1.min(-1).values)
        # exact line search via O(n) bucket of generator breakpoints (rc crossing 0)
        positive = ((da > 0) & (rc < 0)) | ((da < 0) & (rc > 0))
        eta_j = torch.where(positive, -rc / da, torch.full_like(rc, float('inf')))
        decr = torch.where(positive, da.abs() * width, torch.zeros_like(da))
        emin = eta_j.min(-1).values
        emax = torch.where(positive, eta_j, torch.full_like(eta_j, -float('inf'))).max(-1).values
        emax = torch.minimum(emax, torch.where(torch.isfinite(ebnd), ebnd, emax))
        binw = (emax - emin).clamp_min(1e-30) / Kb
        bidx = (((eta_j - emin.unsqueeze(-1)) / binw.unsqueeze(-1)).floor().long()).clamp(0, Kb - 1)
        bidx = torch.where(positive, bidx, torch.full_like(bidx, Kb))
        buckets = torch.zeros(B, Kb + 1, device=a_g.device, dtype=a_g.dtype)
        buckets.scatter_add_(1, bidx, decr)
        crossed = buckets[:, :Kb].cumsum(-1) >= slope0.unsqueeze(-1)
        eta = emin + (crossed.float().argmax(-1).to(a_g.dtype) + 0.5) * binw
        eta = torch.where(crossed.any(-1), eta, emax)
        eta = torch.where(slope0 > _TOL, eta, torch.zeros_like(eta))
        eta = torch.minimum(eta, torch.where(torch.isfinite(ebnd), ebnd, eta)).clamp_min(0.0)
        if last:        # FAR PROBE on the last sweep: jump to the furthest sound point, fold g_far into best
            far = torch.where(torch.isfinite(ebnd), ebnd, emax * 1e6)
            eta_c = torch.where((slope0 > _TOL), torch.where(crossed.any(-1), eta, far), torch.zeros_like(eta))
            rc_f = rc + eta_c.unsqueeze(-1) * da
            x_f = torch.where(rc_f < 0, eh.expand(B, n), el.expand(B, n))
            l0f = (lam0 + eta_c.unsqueeze(-1) * sp0).clamp_min(0.0)
            l1f = (lam1 + eta_c.unsqueeze(-1) * sp1).clamp_min(0.0)
            g_far = c0_path + (rc_f * x_f).sum(-1) - (l0f * b0 + l1f * b1).sum(-1)
            best = torch.maximum(best, g_far)
            break
        eta = eta * pend
        lam0 = (lam0 + eta.unsqueeze(-1) * sp0).clamp_min(0.0)
        lam1 = (lam1 + eta.unsqueeze(-1) * sp1).clamp_min(0.0)
        rc = rc + eta.unsqueeze(-1) * da
    return best


class Verifier:
    def __init__(self, device='cuda', compile=True, K=8, chunk=16384,
                 warm_depths=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)):
        # chunk default smaller than the box verifier: the K-step dual kernel holds several
        # more B×n temporaries (rc/x/da/eta_j/decr/...) so fewer nodes fit per call.
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device = torch.device(device); self.chunk = chunk; self.K = K
        # VC_FARPROBE: add a sound g(λ_far)>0 infeasibility/bound cert on the last sweep (default ON).
        use_farprobe = os.environ.get('VC_FARPROBE', '1') not in ('0', 'false', 'False')
        kern = (lambda F, s: node_bound_dual(F, s, K, use_farprobe=use_farprobe))
        self._kernel = torch.compile(kern, dynamic=True) if compile else kern
        self._warm_depths = warm_depths if compile else ()

    def _upload(self, prob):
        dev = self.device
        G = {k: torch.as_tensor(getattr(prob, k), device=dev, dtype=torch.float32)
             for k in ('a_g', 'ratio_off', 'ratio_on', 'c0_off', 'c0_on', 'c_in', 'z_lo', 'z_hi')}
        G['e_lb'] = torch.as_tensor(prob.e_lb, device=dev, dtype=torch.float32)
        G['e_hi'] = torch.as_tensor(prob.e_hi, device=dev, dtype=torch.float32)
        G['_d_t'] = prob.d_t; G['_e_new'] = prob.e_new_col; G['_c0'] = prob.c0
        return G

    def _F(self, G, depth):
        d_base = G['_d_t'].copy(); d_base[G['_e_new'][:depth]] = 0.0      # see fast_verify_clear note (cross-terms)
        F = {k: G[k][:depth] for k in ('a_g', 'ratio_off', 'ratio_on', 'c0_off', 'c0_on', 'c_in', 'z_lo', 'z_hi')}
        F['d_base'] = torch.as_tensor(d_base, device=self.device, dtype=torch.float32)
        F['c0'] = G['_c0']; F['el'] = G['e_lb']; F['eh'] = G['e_hi']
        return F

    def _bounds(self, F, sides):
        """Bound the frontier in chunks; if a chunk OOMs, halve the chunk size and retry
        (auto-fits the GPU). Raises only if even a tiny chunk can't fit."""
        B = sides.shape[0]
        out = torch.empty(B, device=self.device)
        i = 0
        while i < B:
            step = min(self.chunk, B - i)
            while True:
                try:
                    out[i:i + step] = self._kernel(F, sides[i:i + step].long().contiguous())
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if step <= 256:
                        raise
                    step //= 2
                    self.chunk = max(256, min(self.chunk, step))   # remember the smaller safe size
            i += step
        return out

    def verify_query(self, state, qw, qb, scored_keys, *, time_limit=120.0, frontier_cap=None,
                     extra_hs=()):
        if extra_hs:
            raise NotImplementedError(
                'sibling halfspaces: the K-step kernel does not dualize hs rows; '
                "use the default 1-step verifier (ls='logbucket')")
        _dump_dir = os.environ.get('VC_DUMP_BNB_DIR', '')
        if _dump_dir:
            _dump_bnb_instance(state, qw, qb, scored_keys, _dump_dir)
        return self.verify(parse_problem(state, qw, qb, scored_keys),
                           time_limit=time_limit, frontier_cap=frontier_cap)


    def verify(self, prob, *, time_limit=120.0, frontier_cap=None):
        dev = self.device; G = self._upload(prob)
        for D in self._warm_depths:
            if D <= prob.n_splits:
                self._kernel(self._F(G, D), torch.zeros(8, D, device=dev, dtype=torch.long))
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter(); elapsed = lambda: time.perf_counter() - t0
        if prob.root_bound > 0:
            return 'unsat', dict(nodes=0, depth=0, peak_frontier=0, wall=0.0)
        frontier = torch.tensor([[0], [1]], device=dev, dtype=torch.int8)
        nodes_total = 0; depth = 1; peak = 2

        def _unknown(open_n, reason):
            return 'unknown', dict(nodes=nodes_total, depth=depth, peak_frontier=peak,
                                   open=int(open_n), reason=reason, wall=elapsed())

        while frontier.shape[0] > 0:
            if elapsed() > time_limit:
                return _unknown(frontier.shape[0], 'time_limit')
            nodes_total += frontier.shape[0]; peak = max(peak, frontier.shape[0])
            try:
                uncertified = frontier[self._bounds(self._F(G, depth), frontier) <= _TOL]
                if uncertified.shape[0] == 0:
                    return 'unsat', dict(nodes=nodes_total, depth=depth, peak_frontier=peak, wall=elapsed())
                if depth >= prob.n_splits:
                    return _unknown(uncertified.shape[0], 'splits_exhausted')
                if frontier_cap is not None and 2 * uncertified.shape[0] > frontier_cap:
                    return _unknown(2 * uncertified.shape[0], 'frontier_cap')
                z = torch.zeros(uncertified.shape[0], 1, device=dev, dtype=torch.int8)
                next_frontier = torch.cat([torch.cat([uncertified, z], 1), torch.cat([uncertified, z + 1], 1)], 0)
            except torch.cuda.OutOfMemoryError:
                open_n = int(frontier.shape[0]); uncertified = next_frontier = frontier = None
                torch.cuda.empty_cache()
                return 'unknown', dict(nodes=nodes_total, depth=depth, peak_frontier=peak,
                                       open=open_n, reason='oom', wall=elapsed())
            frontier = next_frontier; depth += 1
        return 'unsat', dict(nodes=nodes_total, depth=depth, peak_frontier=peak, wall=elapsed())


# --- debug helper (env-gated via VC_DUMP_BNB_DIR) -------------------------------
_DUMP_BNB_COUNTER = [0]


def _dump_bnb_instance(state, qw, qb, scored_keys, out_dir, query_id=None):
    """Dump the parsed Phase-8 BnB instance to ``out_dir`` as a pickle, for
    offline experimentation (split-order strategies, LP cross-checks). Captures
    the full box+halfspace problem plus the raw spec/relaxation extras so a
    standalone script can rebuild splits exactly. Gated by VC_DUMP_BNB_DIR; never
    runs in production sweeps."""
    os.makedirs(out_dir, exist_ok=True)
    prob = parse_problem(state, qw, qb, scored_keys)
    ul = state['unstable_list']
    idx_by_key = {(u['layer_idx'], u['neuron_idx']): j for j, u in enumerate(ul)}
    sidx = [idx_by_key[k] for k in scored_keys]
    extras = dict(
        spec_qw=np.asarray(qw, float), spec_qb=float(qb), c0=prob.c0,
        d_t=prob.d_t, scored_keys=list(scored_keys),
        lam=np.array([float(ul[j]['lam']) for j in sidx]),
        mu=np.array([float(ul[j]['mu']) for j in sidx]),
        e_new_col=prob.e_new_col,
        layer_idx=np.array([int(ul[j]['layer_idx']) for j in sidx]),
        neuron_idx=np.array([int(ul[j]['neuron_idx']) for j in sidx]),
    )
    n = _DUMP_BNB_COUNTER[0]; _DUMP_BNB_COUNTER[0] += 1
    tag = query_id if query_id is not None else n
    path = os.path.join(out_dir, f'bnb_{tag}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(dict(problem=prob, extras=extras), f)
    return path
