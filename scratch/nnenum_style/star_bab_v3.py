"""v3: star.lp TRIANGLE relaxation (the pruning enabler).

A star = bias + a_mat @ e, e in box [e_lb,e_hi], plus halfspaces C@e<=d. Affine
prop is identical to a zonotope. The ReLU relaxation is the TRIANGLE as DOMAIN
CONSTRAINTS (not a free generator): for unstable neuron i introduce y_i with
  y_i >= 0  (box lb), y_i >= z_i, y_i <= slope*(z_i - lo),  slope=hi/(hi-lo).
Bounding direction w: lift to a_matᵀw and optimize over the domain (LP now; the
box+halfspace closed form is the next speed step). This is the convex hull of the
ReLU -> tight enough to PRUNE most nodes (vs min-area's free generator).

Run: .venv/bin/python scratch/nnenum_style/star_bab_v3.py 1_1 prop_4
"""
import os
# Single-thread BLAS BEFORE numpy import: small acasxu matrices don't benefit from
# multi-threaded OpenBLAS, and under multiprocessing 16 workers x 16 BLAS threads =
# 256 threads on 16 cores -> catastrophic oversubscription. Must precede `import numpy`.
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'
import sys, time
import numpy as np
import gurobipy as gp
sys.path.insert(0, 'scratch/nnenum_style')
from star_bab import load_acasxu, load_spec, build_queries


class EmptyRegion(Exception):
    """Raised when a star's polytope is infeasible (empty -> vacuously safe)."""


class OverapproxCanceled(Exception):
    """Raised when the overapprox star exceeds the generator cap (like nnenum's
    OVERAPPROX_MIN_GEN_LIMIT). Caller falls back to splitting."""


class CexFound(Exception):
    """A concrete counterexample (true SAT witness) was found during the BaB."""
    def __init__(self, x, y, di):
        self.x, self.y, self.di = x, y, di


def _net_forward(layers, x):
    h = x
    for W, b in layers[:-1]:
        h = np.maximum(0.0, W @ h + b)
    return layers[-1][0] @ h + layers[-1][1]


def concrete_unsafe(layers, x, queries):
    """Forward concrete input x (mean-subtracted) through the real network; return
    the disjunct index whose constraints are ALL <= 0 (unsafe), else None. This is
    an EXACT check (no relaxation) -> a hit is a sound counterexample."""
    y = _net_forward(layers, x)
    for di, qs in enumerate(queries):
        if all(float(np.asarray(w, dtype=float) @ y + b) <= 1e-9 for w, b in qs):
            return di, y
    return None


def pgd_refine(layers, xl, xh, x0, qs, steps=60):
    """Gradient-descend from a seed point x0 (e.g. an LP witness) toward disjunct
    qs's unsafe region on the REAL network. Returns a concrete unsafe x or None."""
    import torch
    Ws = [torch.tensor(W, dtype=torch.float64) for W, _ in layers]
    bs = [torch.tensor(b, dtype=torch.float64) for _, b in layers]
    xl_t = torch.tensor(xl, dtype=torch.float64); xh_t = torch.tensor(xh, dtype=torch.float64)
    qw = [torch.tensor(np.asarray(w, dtype=float), dtype=torch.float64) for w, _ in qs]
    qb = [float(b) for _, b in qs]

    def fwd(x):
        h = x
        for i in range(len(Ws) - 1):
            h = torch.relu(h @ Ws[i].T + bs[i])
        return h @ Ws[-1].T + bs[-1]

    x = torch.tensor(np.clip(x0, xl, xh), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([x], lr=0.02)
    for _ in range(steps):
        opt.zero_grad()
        y = fwd(x)
        loss = torch.stack([qw[i] @ y + qb[i] for i in range(len(qs))]).max()
        if loss.item() <= 0:
            break
        loss.backward(); opt.step()
        with torch.no_grad():
            x.clamp_(xl_t, xh_t)
    with torch.no_grad():
        xn = x.detach().numpy()
    return xn if concrete_unsafe(layers, xn, [qs]) is not None else None


def pgd_attack(layers, xl, xh, queries, restarts=30, steps=80, seed_offset=0):
    """Find a concrete counterexample x in [xl,xh] (a point whose output y makes
    SOME disjunct's constraints all <= 0 -> unsafe). Gradient descent (torch) on
    L_d(x)=max_{(w,b) in d}(w.y+b) per disjunct, projected to the box, many random
    restarts. Returns (x, y, disj_idx) if found else None. SOUND: a returned point
    is verified to violate the spec by concrete forward eval (no relaxation)."""
    import torch
    Ws = [torch.tensor(W, dtype=torch.float64) for W, _ in layers]
    bs = [torch.tensor(b, dtype=torch.float64) for _, b in layers]
    xl_t = torch.tensor(xl, dtype=torch.float64)
    xh_t = torch.tensor(xh, dtype=torch.float64)

    def fwd(x):
        h = x
        for i in range(len(Ws) - 1):
            h = torch.relu(h @ Ws[i].T + bs[i])
        return h @ Ws[-1].T + bs[-1]

    def margin(y, qs):  # max constraint value; <=0 means all satisfied (unsafe)
        return torch.stack([torch.tensor(w, dtype=torch.float64) @ y + b
                            for (w, b) in qs]).max()

    g = torch.Generator().manual_seed(1234 + seed_offset)
    span = (xh_t - xl_t)
    for di, qs in enumerate(queries):
        for r in range(restarts):
            x = (xl_t + span * torch.rand(len(xl), generator=g, dtype=torch.float64)
                 ).clone().detach().requires_grad_(True)
            opt = torch.optim.Adam([x], lr=0.05)
            for _ in range(steps):
                opt.zero_grad()
                loss = margin(fwd(x), qs)
                if loss.item() <= 0:
                    break
                loss.backward()
                opt.step()
                with torch.no_grad():
                    x.clamp_(xl_t, xh_t)
            with torch.no_grad():
                y = fwd(x)
                if margin(y, qs).item() <= 1e-9:        # concrete unsafe point
                    return x.detach().numpy(), y.numpy(), di
    return None


def _contract_box_1hs(lo, hi, a, beta):
    """Tighten {e in [lo,hi] : a.e <= beta} per-coordinate (closed form, no LP)."""
    amin = float(np.where(a > 0, a * lo, a * hi).sum())
    if amin > beta + 1e-9:
        return None
    nlo, nhi = lo.copy(), hi.copy()
    for j in np.nonzero(a)[0]:
        jmin = a[j] * lo[j] if a[j] > 0 else a[j] * hi[j]
        bnd = (beta - (amin - jmin)) / a[j]
        if a[j] > 0:
            nhi[j] = min(nhi[j], bnd)
        else:
            nlo[j] = max(nlo[j], bnd)
    return np.minimum(nlo, nhi), nhi


def _witness_contract(env, C, d, n_done, zlo, zhi, wlo, whi):
    """nnenum-style WITNESSED incremental input-box contraction (a faithful port
    of LpStar.update_input_box_bounds + update_input_box_bounds_new). Only the new
    constraint rows C[n_done:], d[n_done:] (one per split) can move a box bound; a
    bound is re-solved only if its stored witness point is cut by a new row. The
    batched 'pre' loop confirms still-at-old-bound dims with one shared LP so the
    count matches nnenum (without it, identical corner-witnesses make the first
    split all-or-nothing). Returns (zlo, zhi, wlo, whi, n_lps) or None if empty.
    SOUND regardless of witness staleness: skipping keeps the parent bound, which
    under/over-estimates the child's true min/max (child region subset parent) ->
    the box only ever over-approximates -> downstream stays an over-approx."""
    k = len(zlo)
    zlo = zlo.copy(); zhi = zhi.copy(); wlo = list(wlo); whi = list(whi)
    tol = 1e-7
    new_rows = range(n_done, C.shape[0])
    skip_lo = np.ones(k, dtype=bool); skip_hi = np.ones(k, dtype=bool)
    for j in range(k):
        for r in new_rows:
            if skip_lo[j] and float(C[r] @ wlo[j]) > d[r] + tol:
                skip_lo[j] = False
            if skip_hi[j] and float(C[r] @ whi[j]) > d[r] + tol:
                skip_hi[j] = False
    n_cut_lo = int((~skip_lo).sum()); n_cut_hi = int((~skip_hi).sum())
    if skip_lo.all() and skip_hi.all():
        V3_CON_TRACE.append((C.shape[0], 0, 0, 0))
        return zlo, zhi, wlo, whi, 0          # new constraint cuts no witness

    m = gp.Model(env=env); m.setParam('OutputFlag', 0); m.setParam('Threads', 1)
    e = m.addMVar(k, lb=-1.0, ub=1.0); m.addConstr(C @ e <= d)
    nlps = 0

    def _solve(sense):  # returns argmin/argmax point or None if infeasible
        nonlocal nlps
        m.ModelSense = sense; m.optimize()
        nlps += 1; LP_SOLVES[0] += 1; CONTRACT_SOLVES[0] += 1
        return e.X if m.status == gp.GRB.OPTIMAL else None

    # ----- lower bounds (minimize) -----
    if not skip_lo.all():
        vec = np.where(skip_lo, 0.0, 1.0)
        while True:
            e.Obj = vec; res = _solve(gp.GRB.MINIMIZE)
            if res is None:
                m.dispose(); return None
            skipped_all = True; skipped_some = False
            for j in range(k):
                if skip_lo[j]:
                    continue
                if abs(res[j] - zlo[j]) < tol:        # bound unchanged
                    wlo[j] = res.copy(); vec[j] = 0.0
                    skip_lo[j] = True; skipped_some = True
                else:
                    skipped_all = False
            if skipped_all or not skipped_some:
                break
        for j in range(k):
            if skip_lo[j]:
                continue
            obj = np.zeros(k); obj[j] = 1.0; e.Obj = obj
            res = _solve(gp.GRB.MINIMIZE)
            if res is None:
                m.dispose(); return None
            zlo[j] = res[j]; wlo[j] = res.copy()

    # ----- upper bounds (maximize) -----
    if not skip_hi.all():
        vec = np.where(skip_hi, 0.0, 1.0)
        while True:
            e.Obj = vec; res = _solve(gp.GRB.MAXIMIZE)
            if res is None:
                m.dispose(); return None
            skipped_all = True; skipped_some = False
            for j in range(k):
                if skip_hi[j]:
                    continue
                if abs(res[j] - zhi[j]) < tol:        # bound unchanged
                    whi[j] = res.copy(); vec[j] = 0.0
                    skip_hi[j] = True; skipped_some = True
                else:
                    skipped_all = False
            if skipped_all or not skipped_some:
                break
        for j in range(k):
            if skip_hi[j]:
                continue
            obj = np.zeros(k); obj[j] = 1.0; e.Obj = obj
            res = _solve(gp.GRB.MAXIMIZE)
            if res is None:
                m.dispose(); return None
            zhi[j] = res[j]; whi[j] = res.copy()

    m.dispose()
    zlo = np.minimum(zlo, zhi)
    V3_CON_TRACE.append((C.shape[0], n_cut_lo, n_cut_hi, nlps))
    return zlo, zhi, wlo, whi, nlps


def _zono_box_bounds(bias, A, lo, hi):
    pos = np.clip(A, 0, None); neg = np.clip(A, None, 0)
    return bias + pos @ lo + neg @ hi, bias + pos @ hi + neg @ lo


def _zono_relu_minarea(bias, A, lo, hi, forced=None):
    """Min-area zonotope ReLU over-approx (new [-1,1] gen per unstable neuron).
    forced: {neuron: +1 active / -1 inactive} -> treated as stable regardless of
    the (loose) zono bound (so a split-decided neuron isn't re-relaxed/re-split)."""
    blo, bhi = _zono_box_bounds(bias, A, lo, hi)
    n = A.shape[0]
    act = blo >= -1e-12; dead = bhi <= 1e-12
    if forced:
        for i, s in forced.items():
            act[i], dead[i] = (s > 0, s < 0)
    unst = ~act & ~dead
    nb = np.where(dead, 0.0, bias).astype(float)
    nA = np.where(dead[:, None], 0.0, A)
    u = np.where(unst)[0]
    if len(u) == 0:
        return nb, nA, lo, hi
    new = np.zeros((n, len(u)))
    for j, i in enumerate(u):
        l, h = blo[i], bhi[i]
        lam = h / (h - l); mu = -h * l / (2 * (h - l))
        nA[i, :] = nA[i, :] * lam; nb[i] = nb[i] * lam + mu; new[i, j] = mu
    nA = np.hstack([nA, new])
    return nb, nA, np.concatenate([lo, -np.ones(len(u))]), np.concatenate([hi, np.ones(len(u))])


def _zono_relu_variant(z, ilo, ihi, kind):
    """ReLU over-approx of zono z=(bias,A,lo,hi) using SHARED bounds ilo,ihi.
    kind in {area, ybloat, interval} (nnenum's three relaxations)."""
    b, A, lo, hi = z
    n = A.shape[0]
    act = ilo >= -1e-12; dead = ihi <= 1e-12
    unst = ~act & ~dead
    nb = np.where(dead, 0.0, b).astype(float)
    nA = np.where(dead[:, None], 0.0, A)
    u = np.where(unst)[0]
    if len(u) == 0:
        return (nb, nA, lo, hi)
    new = np.zeros((n, len(u)))
    for j, i in enumerate(u):
        l, h = ilo[i], ihi[i]
        if kind == 'area':
            lam = h / (h - l); mu = -h * l / (2 * (h - l))
            nA[i, :] = nA[i, :] * lam; nb[i] = nb[i] * lam + mu; new[i, j] = mu
        elif kind == 'ybloat':
            yo = -l / 2.0; nb[i] = nb[i] + yo; new[i, j] = yo
        else:  # interval
            nA[i, :] = 0.0; nb[i] = h / 2.0; new[i, j] = h / 2.0
    nA = np.hstack([nA, new])
    return (nb, nA, np.concatenate([lo, -np.ones(len(u))]),
            np.concatenate([hi, np.ones(len(u))]))


V3_SPLITS = []   # trace: (li, ni, n_unstable_at_li, n_constraints) for first splits
V3_TRACE = []    # (path, action) per node visited; path = tuple of (li,ni,sign)
FAST_BOUNDS = False   # LP only box-unstable neurons (skip box-stable) — set in verify()
SINGLE_BOUND = False  # LP only the binding side of box-unstable neurons
ZONO_PREFILTER = False  # min-area zono prefilter: LP only zono-unstable neurons
MULTIZONO = False       # multi-type (area/ybloat/interval) zono prefilter + LP triangle
WITNESS_BOUNDS = False  # use concrete sim as witness -> 1 LP/neuron (nnenum's trick)
LP_SOLVES = [0]       # count of actual simplex solves (optimize calls)
SPEC_SOLVES = [0]     # of those, how many are spec-check (conjunct) solves
BOUND_NEURONS = [0]   # neurons LP-tightened (bound solves / sides)
CONTRACT_SOLVES = [0] # of LP_SOLVES, how many are input-box LP-contraction solves
MZ_CON_CALLS = []     # per overapprox_multizono call: input-box contraction LPs
MZ_BND_CALLS = []     # per overapprox_multizono call: neuron-bounding LPs
WITNESS_CONTRACT = False  # nnenum-style witnessed incremental input-box contraction
NNENUM_SINGLE = False  # nnenum-style branching: LP one side, keep ZONO for the other
ATTACK = False         # extract LP-witness at spec check -> concrete cex (fast SAT)
BEST_BD_STOP = False   # Gurobi BestBdStop: stop the stability-proving LP at the sign
DEADLINE = None        # absolute time.time() deadline (shared across WS workers)
PHASE_T = {}           # phase microbenchmark accumulators (s); enable with PHASE_ON
PHASE_ON = False
def _pt(key, dt):
    if PHASE_ON:
        PHASE_T[key] = PHASE_T.get(key, 0.0) + dt
FRONTIER = []          # un-pruned nodes collected at FRONTIER_DEPTH (parallel seeding)
FRONTIER_DEPTH = None  # when set, process() defers nodes at this split-depth to FRONTIER
V3_CON_TRACE = []     # per witnessed-contraction call: (depth, n_cut_lo, n_cut_hi, nlps)
V3_ZONO_LP_DUMP = []  # at depth-4 L0 node: (neuron, zono_lo, lp_lo, zono_hi, lp_hi)
V3_CAPTURE = {}  # capture the first star about to split at a target layer
V3_CAPTURE_LI = -1  # set to a layer index to capture that layer's first split-node
MZ_TRACE = []    # (layer, n_LPd, n_zono_unstable) for the ROOT multizono overapprox


class TStar:
    def __init__(self, bias, a_mat, e_lb, e_hi, C, d):
        self.bias, self.a_mat = bias, a_mat
        self.e_lb, self.e_hi, self.C, self.d = e_lb, e_hi, C, d

    @classmethod
    def from_box(cls, lo, hi):
        c = (lo + hi) / 2; r = (hi - lo) / 2; k = len(lo)
        return cls(c, np.diag(r), -np.ones(k), np.ones(k), np.zeros((0, k)), np.zeros(0))

    def linear(self, W, b):
        return TStar(W @ self.bias + b, W @ self.a_mat, self.e_lb, self.e_hi, self.C, self.d)

    def _model(self, env):
        m = gp.Model(env=env); m.setParam('OutputFlag', 0); m.setParam('Threads', 1)
        e = m.addMVar(self.a_mat.shape[1], lb=self.e_lb, ub=self.e_hi)
        if self.C.shape[0]:
            m.addConstr(self.C @ e <= self.d)
        return m, e

    def neuron_bounds(self, env):
        """LP per-neuron pre-activation [lo,hi] (lift each row, optimize).
        Raises EmptyRegion if the polytope is infeasible (empty = vacuously safe).

        FAST_BOUNDS: skip the LP for neurons the cheap box bound (ignoring C,d)
        already proves stable — box-stable => LP-stable, and a stable neuron's
        ReLU is exact regardless of how tight its bound is. Only box-UNSTABLE
        neurons get the LP (the ones whose triangle slope actually matters).
        SINGLE_BOUND: for those, LP only the nearer-to-zero side, keep the box
        bound on the other (nnenum's OVERAPPROX_BOTH_BOUNDS=False)."""
        a = self.a_mat
        elo, ehi = self.e_lb, self.e_hi
        if FAST_BOUNDS and self.C.shape[0]:
            # Contract the generator box with the constraints (closed-form box+
            # halfspace, no LP) -> tighter prefilter -> fewer neurons need an LP.
            cl, ch = elo.copy(), ehi.copy()
            for r in range(self.C.shape[0]):
                res = _contract_box_1hs(cl, ch, self.C[r], self.d[r])
                if res is None:
                    raise EmptyRegion()
                cl, ch = res
            elo, ehi = cl, ch
        pos = np.clip(a, 0, None); neg = np.clip(a, None, 0)
        box_lo = self.bias + pos @ elo + neg @ ehi
        box_hi = self.bias + pos @ ehi + neg @ elo
        n = len(self.bias)
        if FAST_BOUNDS:
            need = np.where((box_lo < -1e-9) & (box_hi > 1e-9))[0]
        else:
            need = np.arange(n)
        lo = box_lo.copy(); hi = box_hi.copy()
        if len(need) == 0:
            return lo, hi
        m, e = self._model(env)
        BOUND_NEURONS[0] += len(need)
        for i in need:
            i = int(i)
            # Set the linear objective via coefficients directly (no Gurobi
            # expression build — ~45% faster); the bias constant is added back.
            e.Obj = self.a_mat[i]
            do_lo = True; do_hi = True
            if SINGLE_BOUND:   # LP only the binding side (nearer 0), keep box other
                if -box_lo[i] <= box_hi[i]:
                    do_hi = False
                else:
                    do_lo = False
            if do_lo:
                m.ModelSense = gp.GRB.MINIMIZE; m.optimize(); LP_SOLVES[0] += 1
                if m.status != gp.GRB.OPTIMAL:
                    m.dispose(); raise EmptyRegion()
                lo[i] = m.ObjVal + self.bias[i]
            if do_hi:
                m.ModelSense = gp.GRB.MAXIMIZE; m.optimize(); LP_SOLVES[0] += 1
                if m.status != gp.GRB.OPTIMAL:
                    m.dispose(); raise EmptyRegion()
                hi[i] = m.ObjVal + self.bias[i]
        m.dispose()
        return lo, hi

    def dir_lb(self, w, b, env):
        """min of w.y+b over the polytope; +inf if the region is empty (safe)."""
        m, e = self._model(env)
        m.setObjective(w @ self.bias + b + (w @ self.a_mat) @ e, gp.GRB.MINIMIZE)
        m.optimize()
        v = m.ObjVal if m.status == gp.GRB.OPTIMAL else float('inf')
        m.dispose()
        return v

    def conjunct_safe(self, qs, env):
        """True if the UNSAFE conjunct {w.y+b <= 0 for ALL (w,b) in qs} is
        infeasible over the star (-> property proven safe for this disjunct).
        This is the EXACT joint check (nnenum's get_violation_star); strictly
        tighter than requiring some single query > 0, and cheaper (1 LP)."""
        m, e = self._model(env)
        for w, b in qs:
            m.addConstr((w @ self.a_mat) @ e <= -(float(w @ self.bias) + b))
        m.setObjective(0)
        m.optimize(); LP_SOLVES[0] += 1; SPEC_SOLVES[0] += 1
        safe = m.status != gp.GRB.OPTIMAL   # infeasible unsafe-conjunct -> safe
        m.dispose()
        return safe

    def conjunct_witness(self, qs, env, kin):
        """If the unsafe conjunct is feasible, return the input-space witness
        (first `kin` gen coords of the DEEPEST-violation LP point), else None.
        Minimizes t s.t. w.y+b <= t for all q (push as far into the unsafe region
        as the relaxation allows) -> the witness is the most likely real cex."""
        m, e = self._model(env)
        t = m.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
        for w, b in qs:
            m.addConstr((w @ self.a_mat) @ e - t <= -(float(w @ self.bias) + b))
        m.setObjective(t, gp.GRB.MINIMIZE)
        m.optimize(); LP_SOLVES[0] += 1; SPEC_SOLVES[0] += 1
        # feasible unsafe region iff min-max t <= 0 (all constraints satisfiable <=0)
        ev = e.X[:kin].copy() if (m.status == gp.GRB.OPTIMAL and t.X <= 1e-9) else None
        m.dispose()
        return ev

    def relu_triangle(self, lo, hi):
        """Build post-ReLU star with the triangle relaxation for unstable neurons."""
        n, k = self.a_mat.shape
        active = lo >= -1e-12
        dead = hi <= 1e-12
        unst = ~active & ~dead
        u_idx = np.where(unst)[0]
        nu = len(u_idx)
        k2 = k + nu
        bias2 = np.where(active, self.bias, 0.0).astype(float)
        a2 = np.zeros((n, k2))
        a2[:, :k] = np.where(active[:, None], self.a_mat, 0.0)
        # new vars (one per unstable) carry the post-relu value directly
        for j, i in enumerate(u_idx):
            a2[i, :k] = 0.0
            a2[i, k + j] = 1.0
            bias2[i] = 0.0
        e_lb2 = np.concatenate([self.e_lb, np.zeros(nu)])
        e_hi2 = np.concatenate([self.e_hi, hi[u_idx]])
        # constraints: pad old, add 2 per unstable
        Cold = np.zeros((self.C.shape[0], k2))
        if self.C.shape[0]:
            Cold[:, :k] = self.C
        rows = [Cold] if self.C.shape[0] else []
        ds = [self.d] if self.C.shape[0] else []
        for j, i in enumerate(u_idx):
            ai = self.a_mat[i]; bi = self.bias[i]
            slope = hi[i] / (hi[i] - lo[i])
            # y >= z : -y + ai·e_old <= -bi   (over old vars; y is col k+j)
            r1 = np.zeros(k2); r1[:k] = ai; r1[k + j] = -1.0
            rows.append(r1[None, :]); ds.append(np.array([-bi]))
            # y <= slope(z-lo) : y - slope*ai·e_old <= slope*(bi - lo)
            r2 = np.zeros(k2); r2[:k] = -slope * ai; r2[k + j] = 1.0
            rows.append(r2[None, :]); ds.append(np.array([slope * (bi - lo[i])]))
        C2 = np.concatenate(rows, 0) if rows else np.zeros((0, k2))
        d2 = np.concatenate(ds) if ds else np.zeros(0)
        return TStar(bias2, a2, e_lb2, e_hi2, C2, d2)

    def add_split(self, ni, lo_i, hi_i, active):
        """Branch neuron ni's sign (exact): add z_ni>=0 or z_ni<=0 to the domain."""
        ai = self.a_mat[ni]; bi = self.bias[ni]; k = self.a_mat.shape[1]
        if active:   # z>=0 : -ai·e <= bi
            row, rhs = -ai, bi
        else:        # z<=0 :  ai·e <= -bi
            row, rhs = ai, -bi
        C = np.concatenate([self.C, row[None, :]], 0) if self.C.shape[0] else row[None, :]
        d = np.concatenate([self.d, [rhs]])
        return TStar(self.bias, self.a_mat, self.e_lb, self.e_hi, C, d)


def verify(net, prop, max_splits=2_000_000, timeout=300, gen_cap=None,
           fast_bounds=False, single_bound=False, zono_prefilter=False,
           witness=False, multizono=False, witness_contract=False,
           nnenum_single=False, start_node=None, n_workers=1, frontier_depth=None,
           pgd=False, best_bd_stop=False, ws=None, chunk=0):
    global FAST_BOUNDS, SINGLE_BOUND, ZONO_PREFILTER, WITNESS_BOUNDS, MULTIZONO
    global WITNESS_CONTRACT, NNENUM_SINGLE, FRONTIER_DEPTH, ATTACK, BEST_BD_STOP
    global DEADLINE
    FAST_BOUNDS = fast_bounds; SINGLE_BOUND = single_bound
    ZONO_PREFILTER = zono_prefilter; WITNESS_BOUNDS = witness; MULTIZONO = multizono
    WITNESS_CONTRACT = witness_contract; NNENUM_SINGLE = nnenum_single; ATTACK = pgd
    BEST_BD_STOP = best_bd_stop
    LP_SOLVES[0] = 0; SPEC_SOLVES[0] = 0; BOUND_NEURONS[0] = 0; CONTRACT_SOLVES[0] = 0
    MZ_CON_CALLS.clear(); MZ_BND_CALLS.clear(); V3_CON_TRACE.clear()
    V3_ZONO_LP_DUMP.clear()
    layers, sub = load_acasxu(net)
    spec = load_spec(net, prop)
    xl = np.asarray(spec.x_lo).flatten() - sub
    xh = np.asarray(spec.x_hi).flatten() - sub
    c0 = (xl + xh) / 2; r0 = (xh - xl) / 2
    queries = build_queries(spec, len(layers[-1][1]))
    env = gp.Env(empty=True); env.setParam('OutputFlag', 0); env.start()
    # gen_cap: None -> no cap (uncapped, like before). 'widest' -> widest layer
    # width (nnenum-style: ~50 for acasxu). int -> that value.
    if gen_cap == 'widest':
        gen_cap = max(W.shape[0] for W, _b in layers)
    cap = float('inf') if gen_cap is None else float(gen_cap)
    stats = {'splits': 0, 'leaves': 0, 'lp': 0, 'active_branches': 0,
             'inactive_branches': 0, 'overapprox_canceled': 0}
    t0 = time.perf_counter(); sys.setrecursionlimit(1_000_000)
    DEADLINE = ws[4] if ws is not None else time.time() + timeout

    # SAT is found via the LP-witness + PGD-refine inside spec_safe (ATTACK), which
    # already fires at the root overapprox -- no separate pre-attack needed.

    kin = len(xl)

    def spec_safe(star):
        # EXACT joint check: safe iff every unsafe conjunct is infeasible over the
        # star (nnenum's get_violation_star). Strictly tighter than per-query.
        # ATTACK: when a conjunct is feasible, pull the LP point's input coords and
        # test them on the REAL network -> a hit is a sound concrete cex (raise).
        for qs in queries:
            stats['lp'] += 1
            if ATTACK:
                ev = star.conjunct_witness(qs, env, kin)
                if ev is not None:                     # relaxed unsafe region feasible
                    x_in = c0 + r0 * ev
                    hit = concrete_unsafe(layers, x_in, queries)
                    if hit is not None:
                        raise CexFound(x_in, hit[1], hit[0])
                    xr = pgd_refine(layers, xl, xh, x_in, qs)   # refine seed -> real cex
                    if xr is not None:
                        raise CexFound(xr, _net_forward(layers, xr), 0)
                    return False                       # not a real cex here -> split
            elif not star.conjunct_safe(qs, env):
                return False
        return True

    def overapprox_from(s, lj):
        """s = INPUT star to layer lj; propagate with triangle relaxations to out.
        Raises OverapproxCanceled if the star's generator count exceeds `cap`
        (nnenum-style: bail and split rather than pay a huge LP)."""
        if s.a_mat.shape[1] > cap:
            raise OverapproxCanceled
        for l in range(lj, len(layers)):
            s = s.linear(*layers[l])
            if l == len(layers) - 1:
                break
            lo, hi = s.neuron_bounds(env); stats['lp'] += 2 * len(lo)
            s = s.relu_triangle(lo, hi)
            if s.a_mat.shape[1] > cap:
                raise OverapproxCanceled
        return s

    def compute_sim(s):
        """A concrete feasible input point propagated through the EXACT network ->
        per-layer pre-activations. A valid WITNESS: sim[l][i] is achievable, so it
        proves the sign of one bound side (nnenum's both_bounds=False trick)."""
        m, e = s._model(env)
        m.setObjective(0); m.optimize(); LP_SOLVES[0] += 1
        if m.status != gp.GRB.OPTIMAL:
            m.dispose(); return 'empty'
        ev = np.array(e.X); m.dispose()
        x = c0 + r0 * ev[:len(c0)]
        sims = []; xx = x
        for W, bb in layers:
            z = W @ xx + bb; sims.append(z); xx = np.clip(z, 0, None)
        return sims

    def conjunct_safe_zono(zb, za, zlo, zhi, qs):
        """Joint spec check over a ZONOTOPE (1 LP, not per-neuron): is the unsafe
        conjunct {w_j.y+b_j<=0 for all j} infeasible over {zb+za.e, e in box}?"""
        m = gp.Model(env=env); m.setParam('OutputFlag', 0); m.setParam('Threads', 1)
        e = m.addMVar(len(zlo), lb=zlo, ub=zhi)
        for w, b in qs:
            m.addConstr((w @ za) @ e <= -(float(w @ zb) + b))
        m.setObjective(0); m.optimize(); LP_SOLVES[0] += 1
        safe = m.status != gp.GRB.OPTIMAL
        m.dispose()
        return safe

    def overapprox_zono(bias0, amat0, C, d, li, sim):
        """PURE-ZONO overapprox (like nnenum's shallow phase): propagate a min-area
        zono with NO per-neuron LP, then a joint zono spec check (1 LP/disjunct).
        Returns (safe, li_lo, li_hi); li_lo=None if region empty (=> safe)."""
        k = amat0.shape[1]
        zlo, zhi = -np.ones(k), np.ones(k)
        for r in range(C.shape[0]):
            res = _contract_box_1hs(zlo, zhi, C[r], d[r])
            if res is None:
                return True, None, None
            zlo, zhi = res
        zb, za = bias0.copy(), amat0.copy()
        li_lo = li_hi = None
        for l in range(li, len(layers)):
            if l > li:
                W, bb = layers[l]; zb = W @ zb + bb; za = W @ za
            if l == len(layers) - 1:
                break
            nlo, nhi = _zono_box_bounds(zb, za, zlo, zhi)
            if l == li:
                li_lo, li_hi = nlo.copy(), nhi.copy()
            zb, za, zlo, zhi = _zono_relu_minarea(zb, za, zlo, zhi)
        safe = all(conjunct_safe_zono(zb, za, zlo, zhi, qs) for qs in queries)
        return safe, li_lo, li_hi

    KINDS = ('area', 'ybloat', 'interval')

    def overapprox_multizono(bias0, amat0, C, d, li, pbounds, cstate=None):
        """LP-triangle overapprox with a MULTI-TYPE zono prefilter: propagate 3
        zonos (area/ybloat/interval) over the contracted box, INTERSECT their
        bounds, and LP only the intersected-zono-unstable neurons. INCREMENTAL:
        intersect with the parent's carried LP-tightened bounds `pbounds`
        (sound — child region ⊆ parent), so split-stabilized neurons are not
        re-LP'd. `cstate` carries the parent's input-box contraction state
        (n_done, zlo, zhi, wlo, whi) for nnenum-style witnessed incremental
        contraction. Returns (safe, li_lo, li_hi, new_bounds, new_cstate)."""
        k = amat0.shape[1]
        _con0 = LP_SOLVES[0]            # trace: contraction LPs this call
        if cstate is None:             # root: full box, box corners as witnesses
            n_done = 0
            zlo, zhi = -np.ones(k), np.ones(k)
            wlo = [-np.ones(k)] * k; whi = [np.ones(k)] * k
        else:
            n_done, zlo, zhi, wlo, whi = cstate
            zlo, zhi = zlo.copy(), zhi.copy()
        if C.shape[0]:
            _c0 = time.perf_counter()
            if WITNESS_CONTRACT:
                rv = _witness_contract(env, C, d, n_done, zlo, zhi, wlo, whi)
                if rv is None:         # region empty -> vacuously safe
                    MZ_CON_CALLS.append(LP_SOLVES[0] - _con0)
                    return True, None, None, {}, None
                zlo, zhi, wlo, whi, _ = rv
                _pt('contract_lp', time.perf_counter() - _c0)
            else:
                # EXACT LP contraction of the input-generator box vs ALL split
                # constraints from scratch (the pre-witness path). 2k LPs/node.
                mc = gp.Model(env=env); mc.setParam('OutputFlag', 0); mc.setParam('Threads', 1)
                ec = mc.addMVar(k, lb=-1.0, ub=1.0); mc.addConstr(C @ ec <= d)
                zlo, zhi = -np.ones(k), np.ones(k)
                for j in range(k):
                    obj = np.zeros(k); obj[j] = 1.0; ec.Obj = obj
                    mc.ModelSense = gp.GRB.MINIMIZE; mc.optimize()
                    LP_SOLVES[0] += 1; CONTRACT_SOLVES[0] += 1
                    if mc.status != gp.GRB.OPTIMAL:
                        mc.dispose()
                        MZ_CON_CALLS.append(LP_SOLVES[0] - _con0)
                        return True, None, None, {}, None
                    zlo[j] = mc.ObjVal
                    mc.ModelSense = gp.GRB.MAXIMIZE; mc.optimize()
                    LP_SOLVES[0] += 1; CONTRACT_SOLVES[0] += 1
                    zhi[j] = mc.ObjVal
                mc.dispose()
                zlo = np.minimum(zlo, zhi)
        new_cstate = (C.shape[0], zlo, zhi, wlo, whi)
        MZ_CON_CALLS.append(LP_SOLVES[0] - _con0)
        _bnd0 = LP_SOLVES[0]            # trace: bounding LPs this call
        s = TStar(bias0, amat0, -np.ones(k), np.ones(k), C, d)
        zs = [(bias0.copy(), amat0.copy(), zlo.copy(), zhi.copy()) for _ in KINDS]
        li_lo = li_hi = None; new_bounds = {}
        for l in range(li, len(layers)):
            if l > li:
                W, bb = layers[l]; s = s.linear(W, bb)
                zs = [(W @ b + bb, W @ A, lo, hi) for (b, A, lo, hi) in zs]
            if l == len(layers) - 1:
                break
            _z0 = time.perf_counter()
            bnds = [_zono_box_bounds(*z) for z in zs]
            ilo = np.maximum.reduce([x[0] for x in bnds])   # intersect (max lo)
            ihi = np.minimum.reduce([x[1] for x in bnds])   # intersect (min hi)
            if l in pbounds:                                # INCREMENTAL: ∩ parent
                plo, phi = pbounds[l]
                ilo = np.maximum(ilo, plo); ihi = np.minimum(ihi, phi)
            need = np.where((ilo < -1e-9) & (ihi > 1e-9))[0]   # still-unstable
            if li == 0 and len(pbounds) == 0 and len(MZ_TRACE) < 6:
                MZ_TRACE.append((2 * l + 1, len(need)))   # root, per-layer LP'd
            lo, hi = ilo.copy(), ihi.copy()
            _pt('zono_bounds', time.perf_counter() - _z0)
            if len(need):
                _m0 = time.perf_counter()
                m, e = s._model(env)
                _pt('gurobi_model_build', time.perf_counter() - _m0)
                _lp0 = time.perf_counter()
                BOUND_NEURONS[0] += len(need)
                for i in need:
                    i = int(i); e.Obj = s.a_mat[i]
                    bbthr = -float(s.bias[i])   # value-cross-0 threshold on the obj
                    if SINGLE_BOUND and -ilo[i] >= ihi[i]:
                        # leans negative -> prove dead first (ub<=0); EARLY-REJECT:
                        # if dead, 1 LP and the other side (zono lo) is unused.
                        if BEST_BD_STOP:
                            m.setParam('BestBdStop', bbthr)   # stop once max value<=0
                        m.ModelSense = gp.GRB.MAXIMIZE; m.optimize(); LP_SOLVES[0] += 1
                        if BEST_BD_STOP and m.status == gp.GRB.USER_OBJ_LIMIT:
                            hi[i] = 0.0; continue              # proven dead, sign only
                        if m.status != gp.GRB.OPTIMAL:
                            m.dispose(); return True, None, None, {}, new_cstate
                        hi[i] = m.ObjVal + s.bias[i]
                        if hi[i] <= 1e-9:
                            continue                       # dead -> 1 LP
                        if NNENUM_SINGLE and l == li:
                            continue   # branch layer only: keep lo[i]=ilo[i] (ZONO)
                        if BEST_BD_STOP:
                            m.setParam('BestBdStop', gp.GRB.INFINITY)  # exact for relax
                        m.ModelSense = gp.GRB.MINIMIZE; m.optimize(); LP_SOLVES[0] += 1
                        lo[i] = m.ObjVal + s.bias[i]
                    elif SINGLE_BOUND:
                        # leans positive -> prove active first (lb>=0)
                        if BEST_BD_STOP:
                            m.setParam('BestBdStop', bbthr)   # stop once min value>=0
                        m.ModelSense = gp.GRB.MINIMIZE; m.optimize(); LP_SOLVES[0] += 1
                        if BEST_BD_STOP and m.status == gp.GRB.USER_OBJ_LIMIT:
                            lo[i] = 0.0; continue              # proven active, sign only
                        if m.status != gp.GRB.OPTIMAL:
                            m.dispose(); return True, None, None, {}, new_cstate
                        lo[i] = m.ObjVal + s.bias[i]
                        if lo[i] >= -1e-9:
                            continue                       # active -> 1 LP
                        if NNENUM_SINGLE and l == li:
                            continue   # branch layer only: keep hi[i]=ihi[i] (ZONO)
                        if BEST_BD_STOP:
                            m.setParam('BestBdStop', gp.GRB.INFINITY)  # exact for relax
                        m.ModelSense = gp.GRB.MAXIMIZE; m.optimize(); LP_SOLVES[0] += 1
                        hi[i] = m.ObjVal + s.bias[i]
                    else:                                  # both bounds
                        m.ModelSense = gp.GRB.MINIMIZE; m.optimize(); LP_SOLVES[0] += 1
                        if m.status != gp.GRB.OPTIMAL:
                            m.dispose(); return True, None, None, {}, new_cstate
                        lo[i] = m.ObjVal + s.bias[i]
                        m.ModelSense = gp.GRB.MAXIMIZE; m.optimize(); LP_SOLVES[0] += 1
                        hi[i] = m.ObjVal + s.bias[i]
                m.dispose()
                _pt('gurobi_lp_solve', time.perf_counter() - _lp0)
            if li == 0 and l == 0 and C.shape[0] == 4 and not V3_ZONO_LP_DUMP:
                # capture zono (pre-LP) vs LP (post) bounds at the depth-4 L0 node
                for i in need:
                    i = int(i)
                    V3_ZONO_LP_DUMP.append(
                        (i, round(float(ilo[i]), 4), round(float(lo[i]), 4),
                         round(float(ihi[i]), 4), round(float(hi[i]), 4)))
            new_bounds[l] = (lo.copy(), hi.copy())
            if l == li:
                li_lo, li_hi = lo.copy(), hi.copy()
            _r0 = time.perf_counter()
            s = s.relu_triangle(lo, hi)
            if s.a_mat.shape[1] > cap:
                stats['overapprox_canceled'] += 1
                return False, li_lo, li_hi, new_bounds, new_cstate
            zs = [_zono_relu_variant(zs[i], lo, hi, KINDS[i]) for i in range(len(KINDS))]
            _pt('star_relu_triangle', time.perf_counter() - _r0)
        MZ_BND_CALLS.append(LP_SOLVES[0] - _bnd0)
        _s0 = time.perf_counter()
        rv_safe = spec_safe(s)
        _pt('spec_check', time.perf_counter() - _s0)
        return rv_safe, li_lo, li_hi, new_bounds, new_cstate

    def process(s, li, path=(), pbounds=None, cstate=None):
        """s = PRE-activation star at layer li (linear already applied).
        pbounds: parent's carried per-layer LP-tightened bounds (incremental).
        cstate: parent's input-box contraction state (witnessed contraction)."""
        if pbounds is None:
            pbounds = {}
        if DEADLINE is not None and time.time() > DEADLINE:
            return None
        if li == len(layers) - 1:           # output layer
            stats['leaves'] += 1
            ok = spec_safe(s)
            V3_TRACE.append((path, 'leaf_safe' if ok else 'leaf_UNSAFE'))
            return ok
        out = None; nb = {}; ncs = cstate
        if MULTIZONO:
            safe_z, lo, hi, nb, ncs = overapprox_multizono(
                s.bias, s.a_mat, s.C, s.d, li, pbounds, cstate)
            if lo is None:
                return True                      # empty region -> vacuously safe
            if safe_z:
                # SPLIT_IF_IDLE measured to NEVER trigger here (v3 prunes at deep,
                # already-stable nodes -> no unstable neuron to split on, unlike
                # nnenum whose expensive overapprox is what it distributes). Removed.
                stats['leaves'] += 1
                V3_TRACE.append((path, f'PRUNE@L{2*li+1}'))
                return True
        elif ZONO_PREFILTER:
            sim = compute_sim(s) if WITNESS_BOUNDS else None
            if sim == 'empty':
                return True                      # infeasible region -> safe
            safe_z, lo, hi = overapprox_zono(s.bias, s.a_mat, s.C, s.d, li, sim)
            if lo is None:
                return True                      # empty region -> vacuously safe
            if safe_z:
                stats['leaves'] += 1
                V3_TRACE.append((path, f'PRUNE@L{2*li+1}'))
                return True
        else:
            lo, hi = s.neuron_bounds(env); stats['lp'] += 2 * len(lo)
            # PRUNE attempt: relax this layer + all remaining with triangles.
            # If the overapprox exceeds the gen cap, bail (-> split), like nnenum.
            try:
                out = overapprox_from(s.relu_triangle(lo, hi), li + 1)
                if spec_safe(out):
                    stats['leaves'] += 1
                    V3_TRACE.append((path, f'PRUNE@L{2*li+1}'))
                    return True
            except OverapproxCanceled:
                stats['overapprox_canceled'] += 1  # too many gens -> fall through to split
        unst = (lo < -1e-9) & (hi > 1e-9)
        if not unst.any():
            # no unstable HERE (relu is exact) but a LATER layer needs splitting:
            # descend with the exact post-relu, carrying the tightened bounds.
            # cstate carries forward unchanged (same C,d -> 0 contraction LPs).
            return process(s.relu_triangle(lo, hi).linear(*layers[li + 1]),
                           li + 1, path, nb, ncs)
        if stats['splits'] > max_splits:
            return False
        scores = np.minimum(hi, -lo); scores[~unst] = -np.inf
        ni = int(np.argmax(scores)); stats['splits'] += 1
        sa = s.add_split(ni, lo[ni], hi[ni], True)   # active branch z>=0 (identity)
        si = s.add_split(ni, lo[ni], hi[ni], False)  # inactive branch z<=0 (-> dead/0)
        if FRONTIER_DEPTH is not None and len(path) >= FRONTIER_DEPTH:
            # work-stealing seeding: defer BOTH children WITH this node's already-
            # computed bounds (nb, ncs) so the worker does NOT re-run this node's
            # overapprox -- only the child's own overapprox (which is unavoidable).
            FRONTIER.append((sa, li, path + ((2 * li + 1, ni, True),), nb, ncs))
            FRONTIER.append((si, li, path + ((2 * li + 1, ni, False),), nb, ncs))
            return True
        V3_SPLITS.append((li, ni, round(float(scores[ni]), 4), int(unst.sum()),
                          round(float(lo[ni]), 4), round(float(hi[ni]), 4)))
        stats['active_branches'] += 1
        ra = process(sa, li, path + ((2 * li + 1, ni, True),), nb, ncs)
        if ra is not True:
            return ra                                 # active unsafe/timeout -> propagate
        stats['inactive_branches'] += 1
        return process(si, li, path + ((2 * li + 1, ni, False),), nb, ncs)

    global FRONTIER_DEPTH

    if ws is not None:                          # WORK-STEALING worker (LOCAL stack)
        import queue as _pyq
        q, idle, found, res_q, _deadline, _chunk, nw = ws
        local = []; local_cex = None; reason = 'done'; marked_idle = False
        t_work = 0.0; t_idle = 0.0           # instrumentation: busy vs waiting
        while True:
            if found.value:
                reason = 'sat'; break
            if time.time() > DEADLINE:
                reason = 'deadline'; break
            if not local:                       # out of local work -> steal/terminate
                _ti = time.time()
                try:
                    local.append(q.get_nowait())
                    if marked_idle:
                        with idle.get_lock():
                            idle.value -= 1
                        marked_idle = False
                    t_idle += time.time() - _ti
                except _pyq.Empty:
                    if not marked_idle:
                        with idle.get_lock():
                            idle.value += 1
                        marked_idle = True
                    if idle.value >= nw and q.empty():
                        t_idle += time.time() - _ti
                        break                   # all idle + queue empty -> done
                    time.sleep(0.0005)
                    t_idle += time.time() - _ti
                    continue
            node = local.pop()                  # DFS on the local stack
            _tw = time.time()
            FRONTIER.clear(); FRONTIER_DEPTH = len(node[2]) + _chunk
            try:
                verdict = process(*node)
            except CexFound as c:
                verdict = False; local_cex = (c.x, c.y, c.di)
            FRONTIER_DEPTH = None
            local.extend(FRONTIER); FRONTIER.clear()
            t_work += time.time() - _tw
            if verdict is False:                # SAT -> stop everyone
                with found.get_lock():
                    found.value = 1
                reason = 'sat'; break
            # donate surplus the moment ANY worker is idle (responsive rebalancing);
            # give away the OLDEST half (shallowest -> biggest subtrees).
            if len(local) > 1 and idle.value > 0:
                give = local[: max(1, len(local) // 2)]
                del local[: len(give)]
                for d in give:
                    q.put(d)
        res_q.put({'lp': LP_SOLVES[0], 'splits': stats['splits'],
                   'cex': local_cex, 'reason': reason,
                   'work': t_work, 'idle': t_idle})
        env.dispose()
        return None, 0.0, stats

    if start_node is not None:                  # (legacy single-subtree worker)
        try:
            safe = process(*start_node)
        except CexFound as c:
            safe = False; stats['cex'] = (c.x, c.y, c.di)
        dt = time.perf_counter() - t0; env.dispose()
        return safe, dt, stats

    if n_workers <= 1:                          # serial (default)
        try:
            safe = process(TStar.from_box(xl, xh).linear(*layers[0]), 0)
        except CexFound as c:
            safe = False; stats['cex'] = (c.x, c.y, c.di)
        dt = time.perf_counter() - t0; env.dispose()
        return safe, dt, stats

    # ---- WORK-STEALING parallel: root seed + local-stack steal/donate ----
    import multiprocessing as mp
    env.dispose()                               # main holds no env; workers make their own
    root = (TStar.from_box(xl, xh).linear(*layers[0]), 0, (), {}, None)
    q = mp.Queue(); q.put(root)
    idle = mp.Value('i', 0)
    found = mp.Value('i', 0)
    res_q = mp.Queue()
    deadline = time.time() + timeout
    shared = (q, idle, found, res_q, deadline, chunk, n_workers)
    settings = dict(max_splits=max_splits, timeout=timeout, gen_cap=gen_cap,
                    fast_bounds=fast_bounds, single_bound=single_bound,
                    zono_prefilter=zono_prefilter, witness=witness, multizono=multizono,
                    witness_contract=witness_contract, nnenum_single=nnenum_single,
                    pgd=pgd, best_bd_stop=best_bd_stop, chunk=chunk)
    procs = [mp.Process(target=_ws_worker, args=(net, prop, settings, shared))
             for _ in range(n_workers)]
    for p in procs:
        p.start()
    # drain results while workers run (avoid Queue deadlock on join)
    collected = [res_q.get() for _ in range(n_workers)]
    for p in procs:
        p.join()
    reasons = set(); tot_work = 0.0; tot_idle = 0.0
    for r in collected:
        LP_SOLVES[0] += r['lp']; stats['splits'] += r['splits']
        reasons.add(r['reason'])
        tot_work += r.get('work', 0.0); tot_idle += r.get('idle', 0.0)
        if r['cex']:
            stats['cex'] = r['cex']
    stats['work_s'] = tot_work; stats['idle_s'] = tot_idle
    stats['idle_pct'] = 100 * tot_idle / max(1e-9, tot_work + tot_idle)
    if found.value:
        safe = False                            # SAT (concrete counterexample)
    elif 'deadline' in reasons:
        safe = None                             # timed out with work remaining
    else:
        safe = True                             # tree exhausted -> verified
    dt = time.perf_counter() - t0
    return safe, dt, stats


def _ws_worker(net, prop, settings, shared):
    """Long-lived work-stealing worker: builds context once, loops on the queue."""
    verify(net, prop, ws=shared, **settings)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('net'); ap.add_argument('prop')
    ap.add_argument('--timeout', type=float, default=300)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--results-file', default=None)
    a = ap.parse_args()
    safe, dt, stats = verify(a.net, a.prop, gen_cap=None, multizono=True,
                             witness_contract=True, single_bound=True, pgd=True,
                             n_workers=a.workers, timeout=a.timeout)
    # VNNCOMP-style verdict: unsat=verified safe, sat=counterexample, timeout/unknown
    verdict = 'unsat' if safe is True else ('sat' if safe is False else 'timeout')
    if a.results_file:
        with open(a.results_file, 'w') as f:
            f.write(verdict + '\n')
    print(f'V3 {a.net} {a.prop}: {verdict} {dt:.2f}s splits={stats["splits"]} '
          f'LP={LP_SOLVES[0]}', flush=True)
