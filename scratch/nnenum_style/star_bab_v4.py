"""v4: nnenum-style multi-round escalation BaB (what nnenum ACTUALLY does).

Per BaB node, in increasing cost:
  1. CHEAP rung: zonotope (min-area) overapprox to output, spec via closed-form
     projection. ZERO LPs. Closes most nodes (nnenum did prop_4 with 0 LPs).
  2. HARD rung: star.lp TRIANGLE overapprox (Gurobi LP bounds + LP spec). Only on
     nodes the cheap zono can't close (nnenum used 123 LPs on prop_3, 0 on prop_4).
  3. SPLIT: pick an unstable ReLU, branch its sign. After each split, CONTRACT the
     e-box with the closed-form single-halfspace tightening (exact for 1 halfspace,
     O(n), no LP) — this is the box+halfspace primitive, the cheap version of the
     contraction prefilter that the ablation showed was decisive.

Soundness: min-area zono and triangle-LP are both sound over-approximations;
single-halfspace box contraction only removes box points that violate the split,
so it stays an outer box of the true constrained set. Leaves (all ReLUs resolved)
are exact.

Run: .venv/bin/python scratch/nnenum_style/star_bab_v4.py 1_1 prop_4
"""
import sys, time
import numpy as np
import gurobipy as gp
sys.path.insert(0, 'scratch/nnenum_style')
from star_bab import load_acasxu, load_spec, build_queries
from star_bab_v3 import TStar, EmptyRegion

FIRST_SPLITS = []   # (li, ni, blo, bhi, n_unstable_at_li) for the first few splits


# ---------- closed-form single-halfspace box contraction (no LP) ----------
def contract_box_1hs(lo, hi, a, beta):
    """Tighten {e in [lo,hi] : a.e <= beta} per-coordinate (exact for 1 halfspace,
    O(nnz)). Returns (lo,hi) or None if the halfspace is infeasible wrt the box."""
    amin = float(np.where(a > 0, a * lo, a * hi).sum())   # box-min of a.e
    if amin > beta + 1e-9:
        return None
    nlo, nhi = lo.copy(), hi.copy()
    for j in np.nonzero(a)[0]:
        jmin = a[j] * lo[j] if a[j] > 0 else a[j] * hi[j]
        bnd = (beta - (amin - jmin)) / a[j]               # a_j e_j <= beta - rest
        if a[j] > 0:
            nhi[j] = min(nhi[j], bnd)
        else:
            nlo[j] = max(nlo[j], bnd)
    nlo = np.minimum(nlo, nhi)                            # guard numeric crossing
    return nlo, nhi


def contract_joint(lo, hi, C, d, env):
    """Tightest box containing {e in [lo,hi] : C.e<=d} via per-dim LP (JOINT, the
    CONTRACT_ZONOTOPE_LP prefilter). Returns (lo,hi) or None if infeasible."""
    if C.shape[0] == 0:
        return lo, hi
    m = gp.Model(env=env); m.setParam('OutputFlag', 0); m.setParam('Threads', 1)
    e = m.addMVar(len(lo), lb=lo, ub=hi)
    m.addConstr(C @ e <= d)
    nlo, nhi = lo.copy(), hi.copy()
    for j in range(len(lo)):
        m.setObjective(e[j], gp.GRB.MINIMIZE); m.optimize()
        if m.status != gp.GRB.OPTIMAL:
            m.dispose(); return None
        nlo[j] = max(lo[j], m.ObjVal)
        m.setObjective(e[j], gp.GRB.MAXIMIZE); m.optimize(); nhi[j] = min(hi[j], m.ObjVal)
    m.dispose()
    return np.minimum(nlo, nhi), nhi


# ---------- cheap zono rung (box domain, min-area relu, 0 LP) ----------
def zono_box_bounds(bias, A, lo, hi):
    pos = np.clip(A, 0, None); neg = np.clip(A, None, 0)
    return bias + pos @ lo + neg @ hi, bias + pos @ hi + neg @ lo


def zono_relu_minarea(bias, A, lo, hi, forced):
    """Apply ReLU as a min-area zonotope over-approx (new [-1,1] gen per unstable).
    forced: {neuron: +1 active / -1 inactive} treated as stable."""
    blo, bhi = zono_box_bounds(bias, A, lo, hi)
    n, k = A.shape
    act = blo >= -1e-12; dead = bhi <= 1e-12
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
    lo2 = np.concatenate([lo, -np.ones(len(u))])
    hi2 = np.concatenate([hi, np.ones(len(u))])
    return nb, nA, lo2, hi2


def relu_variant(z, ilb, iub, kind, forced):
    """Apply ReLU to a zono z=(bias,A,lo,hi) using the SHARED (intersected) layer
    bounds ilb,iub. kind in {area,ybloat,interval} (nnenum's three relaxations)."""
    b, A, lo, hi = z
    n, k = A.shape
    act = ilb >= -1e-12; dead = iub <= 1e-12
    for i, s in forced.items():
        act[i], dead[i] = (s > 0, s < 0)
    unst = ~act & ~dead
    nb = np.where(dead, 0.0, b).astype(float)
    nA = np.where(dead[:, None], 0.0, A)
    u = np.where(unst)[0]
    if len(u) == 0:
        return (nb, nA, lo, hi)
    new = np.zeros((n, len(u)))
    for j, i in enumerate(u):
        l_, h_ = ilb[i], iub[i]
        if kind == 'area':
            lam = h_ / (h_ - l_); mu = -h_ * l_ / (2 * (h_ - l_))
            nA[i, :] = nA[i, :] * lam; nb[i] = nb[i] * lam + mu; new[i, j] = mu
        elif kind == 'ybloat':
            yo = -l_ / 2.0; nb[i] = nb[i] + yo; new[i, j] = yo
        else:  # interval
            nA[i, :] = 0.0; nb[i] = h_ / 2.0; new[i, j] = h_ / 2.0
    nA = np.hstack([nA, new])
    return (nb, nA, np.concatenate([lo, -np.ones(len(u))]),
            np.concatenate([hi, np.ones(len(u))]))


def zono_spec_safe(bias, A, lo, hi, queries):
    for qs in queries:
        ok = False
        for w, b in qs:
            wa = w @ A
            lb = float(w @ bias + b + np.where(wa > 0, wa * lo, wa * hi).sum())
            if lb > 1e-7:
                ok = True; break
        if not ok:
            return False
    return True


def verify(net, prop, max_splits=2_000_000, timeout=300, use_lp_rung=True, joint=False,
          spec_aware_split=False, multi=False):
    layers, sub = load_acasxu(net)
    spec = load_spec(net, prop)
    xl = np.asarray(spec.x_lo).flatten() - sub
    xh = np.asarray(spec.x_hi).flatten() - sub
    queries = build_queries(spec, len(layers[-1][1]))
    env = gp.Env(empty=True); env.setParam('OutputFlag', 0); env.start()
    st = {'splits': 0, 'leaves': 0, 'lp': 0, 'zono_close': 0, 'lp_close': 0}
    t0 = time.perf_counter(); sys.setrecursionlimit(1_000_000)
    L = len(layers)

    def zono_overapprox(bias, A, lo, hi, li, forced):
        b, Am, l, h, f = bias, A, lo, hi, forced
        for layer in range(li, L - 1):
            b, Am, l, h = zono_relu_minarea(b, Am, l, h, f); f = {}
            W, bb = layers[layer + 1]; b = W @ b + bb; Am = W @ Am
        return zono_spec_safe(b, Am, l, h, queries)

    KINDS = ('area', 'ybloat', 'interval')

    def multizono_overapprox(bias, A, lo, hi, li, forced):
        """nnenum's cheap rung: 3 zonos with SHARED progressively-tightened layer
        bounds. Safe if ANY proves the spec (each is a valid over-approx)."""
        zs = [(bias.copy(), A.copy(), lo.copy(), hi.copy()) for _ in KINDS]
        f = forced
        for layer in range(li, L - 1):
            bnds = [zono_box_bounds(*z) for z in zs]
            ilb = np.maximum.reduce([x[0] for x in bnds])
            iub = np.minimum.reduce([x[1] for x in bnds])
            zs = [relu_variant(zs[i], ilb, iub, KINDS[i], f) for i in range(len(KINDS))]
            f = {}
            W, bb = layers[layer + 1]
            zs = [(W @ z[0] + bb, W @ z[1], z[2], z[3]) for z in zs]
        return any(zono_spec_safe(*z, queries) for z in zs)

    def split_scores(bias, A, lo, hi, li, forced, unst):
        """Spec-aware split ranking (nnenum SPLIT_ONE_NORM idea): propagate the
        cheap zono to output, tracking the error-generator column each li-unstable
        neuron creates; score a neuron by how much its generator contributes to the
        (unclosed) spec directions. Returns dict neuron->score."""
        k0 = A.shape[1]
        u = np.where(unst)[0]
        col_of = {int(u[j]): k0 + j for j in range(len(u))}
        b, Am, l, h = zono_relu_minarea(bias, A, lo, hi, forced)
        for layer in range(li + 1, L):
            W, bb = layers[layer]; b = W @ b + bb; Am = W @ Am
            if layer == L - 1:
                break
            b, Am, l, h = zono_relu_minarea(b, Am, l, h, {})
        # for each disjunct, take its least-satisfied (max) direction; sum |w·col|
        scores = {i: 0.0 for i in col_of}
        for qs in queries:
            # pick the direction whose lb is smallest (the binding one)
            best_w, best_lb = None, np.inf
            for w, bq in qs:
                wa = w @ Am
                lb = float(w @ b + bq + np.where(wa > 0, wa * l, wa * h).sum())
                if lb < best_lb:
                    best_lb, best_w = lb, w
            if best_lb > 1e-7:
                continue                                   # this disjunct already safe
            wa = best_w @ Am
            for i, col in col_of.items():
                scores[i] += abs(wa[col])                  # gen box is [-1,1]
        return scores

    def lp_spec_safe(s):
        """exact LP spec check on a star (box+C,d). Empty region -> safe."""
        for qs in queries:
            st['lp'] += len(qs)
            if not any(s.dir_lb(w, b, env) > 1e-7 for w, b in qs):
                return False
        return True

    def lp_overapprox(bias, A, lo, hi, C, d, li, forced):
        try:
            s = TStar(bias, A, lo, hi, C, d)
            if li == L - 1:                               # exact output: no relu
                return lp_spec_safe(s)
            blo, bhi = s.neuron_bounds(env); st['lp'] += 2 * len(blo)
            for i, sg in forced.items():
                if sg > 0: blo[i] = max(blo[i], 0.0)
                else: bhi[i] = min(bhi[i], 0.0)
            s = s.relu_triangle(blo, bhi)
            for layer in range(li + 1, L):
                s = s.linear(*layers[layer])
                if layer == L - 1:
                    break
                blo, bhi = s.neuron_bounds(env); st['lp'] += 2 * len(blo)
                s = s.relu_triangle(blo, bhi)
            return lp_spec_safe(s)
        except EmptyRegion:
            return True                                   # empty polytope = safe

    def process(bias, A, lo, hi, C, d, li, forced):
        if time.perf_counter() - t0 > timeout:
            return None
        if li == L - 1:                                   # output: no relu left
            st['leaves'] += 1
            if zono_spec_safe(bias, A, lo, hi, queries):
                st['zono_close'] += 1; return True
            r = lp_overapprox(bias, A, lo, hi, C, d, li, forced)  # exact LP
            if r is True: st['lp_close'] += 1
            return r
        # rung 1: cheap zono (single min-area, or nnenum's 3-type intersection)
        cheap = multizono_overapprox if multi else zono_overapprox
        if cheap(bias, A, lo, hi, li, forced):
            st['leaves'] += 1; st['zono_close'] += 1; return True
        # rung 2: triangle-LP (only on nodes the zono can't close)
        if use_lp_rung and lp_overapprox(bias, A, lo, hi, C, d, li, forced):
            st['leaves'] += 1; st['lp_close'] += 1; return True
        # rung 3: split an unstable ReLU at this layer
        blo, bhi = zono_box_bounds(bias, A, lo, hi)
        unst = (blo < -1e-9) & (bhi > 1e-9)
        for i in forced:
            unst[i] = False
        if not unst.any():                                # advance to next layer
            nb, nA, nlo, nhi = zono_relu_minarea(bias, A, lo, hi, forced)
            W, bb = layers[li + 1]; nb = W @ nb + bb; nA = W @ nA
            return process(nb, nA, nlo, nhi, C, d, li + 1, {})
        if st['splits'] > max_splits:
            return False
        if spec_aware_split:
            sc = split_scores(bias, A, lo, hi, li, forced, unst)
            ni = max(sc, key=sc.get) if sc else int(np.argmax(
                np.where(unst, np.minimum(bhi, -blo), -np.inf)))
        else:
            scores = np.minimum(bhi, -blo); scores[~unst] = -np.inf
            ni = int(np.argmax(scores))
        st['splits'] += 1
        if len(FIRST_SPLITS) < 8:
            FIRST_SPLITS.append((li, ni, float(blo[ni]), float(bhi[ni]),
                                 int(unst.sum())))
        a_ni = A[ni]; b_ni = bias[ni]
        for sign, (a_hs, beta) in ((+1, (-a_ni, b_ni)), (-1, (a_ni, -b_ni))):
            C2 = np.vstack([C, a_hs]) if C.shape[0] else a_hs[None, :]
            d2 = np.concatenate([d, [beta]])
            ct = (contract_joint(lo, hi, C2, d2, env) if joint
                  else contract_box_1hs(lo, hi, a_hs, beta))
            if ct is None:                                # branch infeasible -> safe
                continue
            nlo, nhi = ct
            r = process(bias, A, nlo, nhi, C2, d2, li, {**forced, ni: sign})
            if r is not True:
                return r
        return True

    k = len(xl)
    c0 = (xl + xh) / 2; r0 = (xh - xl) / 2
    W0, b0 = layers[0]
    bias = W0 @ c0 + b0
    A = W0 @ np.diag(r0)
    lo = -np.ones(k); hi = np.ones(k)
    C = np.zeros((0, k)); d = np.zeros(0)
    safe = process(bias, A, lo, hi, C, d, 0, {})
    dt = time.perf_counter() - t0; env.dispose()
    return safe, dt, st


if __name__ == '__main__':
    net = sys.argv[1] if len(sys.argv) > 1 else '1_1'
    prop = sys.argv[2] if len(sys.argv) > 2 else 'prop_4'
    use_lp = '--nolp' not in sys.argv
    joint = '--joint' in sys.argv
    saware = '--saware' in sys.argv
    multi = '--multi' in sys.argv
    safe, dt, st = verify(net, prop, use_lp_rung=use_lp, joint=joint,
                          spec_aware_split=saware, multi=multi)
    r = 'SAFE' if safe else ('TIMEOUT' if safe is None else 'not-proven')
    print(f'V4 {net} {prop}: {r} {dt:.2f}s splits={st["splits"]} leaves={st["leaves"]} '
          f'zono_close={st["zono_close"]} lp_close={st["lp_close"]} lp={st["lp"]}')
