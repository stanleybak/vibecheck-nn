"""explore01: does K-step Lagrangian dual ascent match Gurobi on the star
contraction/bound problem, as the user claims (default k=1, converge by k=10/100)?

The star bound problem at a BaB node is an LP:
    min  const + w_a · e      over   e in [lo,hi],  C·e <= d
where w_a = w@a_mat, const = w@bias + b, and C·e<=d are the accumulated split
constraints (the triangle constraints + sign splits). This is EXACTLY the
box+halfspaces formulation. Dualize the M halfspaces (λ>=0):
    g(λ) = const - λ·d + Σ_i min_{e_i∈[lo_i,hi_i]} (w_a + Cᵀλ)_i · e_i
g is concave; every g(λ) is a valid lower bound (weak duality); LP strong duality
=> sup_λ g = Gurobi optimum. Projected subgradient ascent, K steps, best-so-far.

We pull REAL star nodes out of v3's search on 1_1 prop_4 and compare the spec
direction lower bound: Gurobi LP vs dual-ascent at K = 1,2,5,10,20,50,100,200.
"""
import sys, time
import numpy as np
import gurobipy as gp
sys.path.insert(0, 'scratch/nnenum_style')
from star_bab import load_acasxu, load_spec, build_queries
from star_bab_v3 import TStar


def dual_ascent_lb(star, w, b, K, tol=1e-12):
    """Lower bound of (w·bias+b) + (w·a_mat)·e over {e∈[lo,hi], C·e<=d} via K
    EXACT-LINE-SEARCH dual-ascent steps (replicates _batched_dual_ascent's step:
    project the subgradient, then walk the rc-flip breakpoints to the optimal η
    along that direction). Each g(λ) is a valid lower bound; best-so-far returned."""
    lo, hi = star.e_lb, star.e_hi
    width = hi - lo
    w_a = w @ star.a_mat                           # 'd' in the dual-ascent code
    const = float(w @ star.bias + b)               # 'c0'
    C, d = star.C, star.d                          # 'A', 'b'
    M = C.shape[0]
    if M == 0:                                     # pure box — closed form
        return const + float(np.where(w_a > 0, w_a * lo, w_a * hi).sum())
    lam = np.zeros(M)
    best = -np.inf
    for _k in range(K):
        # Recompute rc from lam each iter so g is ALWAYS a valid dual value:
        # the clamp(lam>=0) can otherwise desync an incrementally-updated rc and
        # let g exceed the primal min (a weak-duality / soundness violation).
        rc = w_a + C.T @ lam
        x_star = np.where(rc < 0, hi, lo)          # box minimizer
        g = const + float(rc @ x_star) - float(lam @ d)
        if g > best:
            best = g
        s = C @ x_star - d                         # subgradient (violation)
        s_proj = np.where((lam <= tol) & (s < 0), 0.0, s)
        da = C.T @ s_proj                          # induced direction in rc-space
        slope0 = float(s_proj @ s)
        if slope0 <= tol:
            break
        # breakpoints η_i = -rc_i/da_i where x_i flips; walk until slope ≤ 0
        positive = ((da > 0) & (rc < 0)) | ((da < 0) & (rc > 0))
        with np.errstate(divide='ignore', invalid='ignore'):
            etas = np.where(positive, -rc / da, np.inf)
        order = np.argsort(etas)
        etas_s = etas[order]; rc_s = rc[order]; da_s = da[order]; w_s = width[order]
        decr = np.where(rc_s < 0, w_s * da_s, -w_s * da_s)
        decr = np.where(np.isfinite(etas_s), decr, 0.0)
        slope = slope0; eta_star = 0.0; chosen = False
        for i in range(len(etas_s)):
            if not np.isfinite(etas_s[i]):
                break
            slope -= decr[i]
            if slope <= tol:
                eta_star = etas_s[i]; chosen = True
                break
        if not chosen:                             # use last finite breakpoint
            fin = np.isfinite(etas_s)
            eta_star = etas_s[fin][-1] if fin.any() else 0.0
        if not np.isfinite(eta_star):
            eta_star = 0.0
        lam = np.maximum(0.0, lam + eta_star * s_proj)
    return best


def collect_nodes(net, prop, want=6, max_depth=12):
    """Run a partial v3 search, snapshot a few PRE-output stars with nonempty C."""
    layers, sub = load_acasxu(net)
    spec = load_spec(net, prop)
    xl = np.asarray(spec.x_lo).flatten() - sub
    xh = np.asarray(spec.x_hi).flatten() - sub
    queries = build_queries(spec, len(layers[-1][1]))
    env = gp.Env(empty=True); env.setParam('OutputFlag', 0); env.start()
    nodes = []

    def overapprox_from(s, lj):
        for l in range(lj, len(layers)):
            s = s.linear(*layers[l])
            if l == len(layers) - 1:
                break
            lo, hi = s.neuron_bounds(env)
            s = s.relu_triangle(lo, hi)
        return s

    def walk(s, li, depth):
        if len(nodes) >= want or depth > max_depth or li == len(layers) - 1:
            return
        lo, hi = s.neuron_bounds(env)
        unst = (lo < -1e-9) & (hi > 1e-9)
        out = overapprox_from(s.relu_triangle(lo, hi), li + 1)
        if out.C.shape[0] > 0:
            nodes.append((out, queries))   # snapshot a real output-space star
        if not unst.any():
            walk(s.relu_triangle(lo, hi).linear(*layers[li + 1]), li + 1, depth)
            return
        scores = np.minimum(hi, -lo); scores[~unst] = -np.inf
        ni = int(np.argmax(scores))
        walk(s.add_split(ni, lo[ni], hi[ni], True), li, depth + 1)
        walk(s.add_split(ni, lo[ni], hi[ni], False), li, depth + 1)

    walk(TStar.from_box(xl, xh).linear(*layers[0]), 0, 0)
    return nodes, env


if __name__ == '__main__':
    net = sys.argv[1] if len(sys.argv) > 1 else '1_1'
    prop = sys.argv[2] if len(sys.argv) > 2 else 'prop_4'
    nodes, env = collect_nodes(net, prop)
    print(f'collected {len(nodes)} real star nodes from {net} {prop}\n')
    Ks = [1, 2, 5, 10, 20, 50, 100, 200]
    for idx, (star, queries) in enumerate(nodes):
        w, b = queries[0][0]                       # first spec direction
        t = time.perf_counter(); gur = star.dir_lb(w, b, env)
        gur_ms = (time.perf_counter() - t) * 1000
        M = star.C.shape[0]; n = star.a_mat.shape[1]
        print(f'node {idx}: n={n} dims, M={M} halfspaces | Gurobi lb={gur:+.5f} ({gur_ms:.1f} ms)')
        for K in Ks:
            t = time.perf_counter(); da = dual_ascent_lb(star, w, b, K)
            da_ms = (time.perf_counter() - t) * 1000
            print(f'    K={K:>4}: dual lb={da:+.5f}  gap={gur-da:+.2e}  ({da_ms:.2f} ms)')
        print()
    env.dispose()
