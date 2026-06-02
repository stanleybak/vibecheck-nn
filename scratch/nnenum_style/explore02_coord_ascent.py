"""explore02: does COORDINATE (Gauss-Seidel) dual ascent close the gap to Gurobi
where steepest-subgradient + exact line search STALLED (explore01)?

For each halfspace m, hold the other λ fixed and solve the EXACT 1D dual max over
λ_m >= 0 (box_halfspace's single-halfspace concave dual: walk the breakpoints
t_i = -r_i/C_{m,i} where box coords flip, stop when the directional slope <= 0).
Cycle all m, K times. Every g(λ) is a valid lower bound -> SOUND. Compare to
Gurobi on the same real prop_4 star nodes used in explore01.
"""
import sys, time
import numpy as np
import gurobipy as gp
sys.path.insert(0, 'scratch/nnenum_style')
from explore01_dualascent_vs_gurobi import collect_nodes, dual_ascent_lb


def coord_1d_max(r, Cm, dm, lo, hi):
    """argmax_{t>=0} of  -t*dm + Σ_i min_{e∈[lo,hi]}((r_i + t*Cm_i) e_i).
    Concave PWL in t; returns optimal t (>=0)."""
    # slope(t) = -dm + Σ_i Cm_i * x_i(t),  x_i = lo if (r_i+t*Cm_i)>0 else hi
    x0 = np.where(r > 0, lo, hi)
    slope = -dm + float(Cm @ x0)
    if slope <= 1e-15:
        return 0.0                       # already at/over the max at t=0
    # breakpoints where r_i + t*Cm_i = 0 with t>0
    with np.errstate(divide='ignore', invalid='ignore'):
        t_i = np.where(Cm != 0, -r / Cm, np.inf)
    valid = (t_i > 0) & np.isfinite(t_i)
    t_v = t_i[valid]; Cm_v = Cm[valid]
    order = np.argsort(t_v)
    t_s = t_v[order]; Cm_s = Cm_v[order]
    w = (hi - lo)
    w_v = w[valid][order]
    # at each breakpoint the coord flips lo<->hi; slope changes by -|Cm_i|*width_i
    decr = np.abs(Cm_s) * w_v
    t_prev = 0.0
    for k in range(t_s.size):
        new_slope = slope - decr[k]
        if new_slope <= 1e-15:
            return float(t_s[k])         # slope crosses 0 here
        slope = new_slope
    return float(t_s[-1]) if t_s.size else 0.0


def coord_ascent_lb(star, w, b, K):
    lo, hi = star.e_lb, star.e_hi
    w_a = w @ star.a_mat
    const = float(w @ star.bias + b)
    C, d = star.C, star.d
    M = C.shape[0]
    if M == 0:
        return const + float(np.where(w_a > 0, w_a * lo, w_a * hi).sum())
    lam = np.zeros(M)
    best = -np.inf
    for _k in range(K):
        for m in range(M):
            rc = w_a + C.T @ lam
            r = rc - lam[m] * C[m]                 # rc without m's contribution
            lam[m] = coord_1d_max(r, C[m], d[m], lo, hi)
            # record the sound dual value at the new lam
            rc2 = r + lam[m] * C[m]
            x = np.where(rc2 < 0, hi, lo)
            g = const + float(rc2 @ x) - float(lam @ d)
            if g > best:
                best = g
    return best


if __name__ == '__main__':
    net = sys.argv[1] if len(sys.argv) > 1 else '1_1'
    prop = sys.argv[2] if len(sys.argv) > 2 else 'prop_4'
    nodes, env = collect_nodes(net, prop)
    print(f'collected {len(nodes)} nodes from {net} {prop}  '
          f'(steepest=explore01, coord=this)\n')
    for idx, (star, queries) in enumerate(nodes[:4]):
        w, b = queries[0][0]
        gur = star.dir_lb(w, b, env)
        M = star.C.shape[0]; n = star.a_mat.shape[1]
        print(f'node {idx}: n={n}, M={M} | Gurobi={gur:+.5f}')
        for K in [1, 2, 5, 10, 20]:
            t = time.perf_counter(); cd = coord_ascent_lb(star, w, b, K)
            cd_ms = (time.perf_counter() - t) * 1000
            st = dual_ascent_lb(star, w, b, K * M)   # match #coord-solves budget
            print(f'    cycles={K:>3} ({K*M} solves): coord={cd:+.5f} gap={gur-cd:+.2e}'
                  f' ({cd_ms:.1f}ms) | steepest(={K*M}it)={st:+.5f} gap={gur-st:+.2e}')
        print()
    env.dispose()
