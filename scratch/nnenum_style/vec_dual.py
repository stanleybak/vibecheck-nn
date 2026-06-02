"""Vectorized dual ascent: bound ALL neurons in a layer at once (one batched
numpy pass of dual-ascent + per-row sort), vs the per-neuron Gurobi LP loop."""
import sys, time, pickle
sys.path.insert(0, 'scratch/nnenum_style')
import numpy as np, gurobipy as gp
from star_bab_v3 import TStar

d = pickle.load(open('/tmp/bench_star.pkl', 'rb'))
bias, A, e_lb, e_hi, C, dd = (d['bias'], d['a_mat'], d['e_lb'], d['e_hi'],
                               d['C'], d['d'])
n, ngen = A.shape; M = C.shape[0]
env = gp.Env(empty=True); env.setParam('OutputFlag', 0); env.start()
s = TStar(bias, A, e_lb, e_hi, C, dd)

# exact LP reference
t = time.perf_counter(); lo_lp, hi_lp = s.neuron_bounds(env)
lp_t = time.perf_counter() - t


def vec_dual_lb(A_obj, c0, K):
    """Sound lower bound of A_obj[i]·e + c0[i] over {e in box, C e<=d}, for ALL
    rows i at once. Vectorized exact-line-search dual ascent (resynced rc)."""
    nn = A_obj.shape[0]; width = e_hi - e_lb
    lam = np.zeros((nn, M)); best = np.full(nn, -np.inf)
    for _ in range(K):
        rc = A_obj + lam @ C
        xs = np.where(rc > 0, e_lb, e_hi)               # box minimizer (broadcast)
        g = c0 + (rc * xs).sum(1) - (lam * dd).sum(1)
        best = np.maximum(best, g)
        s_ = xs @ C.T - dd
        sp = np.where((lam <= 1e-12) & (s_ < 0), 0.0, s_)
        da = sp @ C
        slope0 = (sp * s_).sum(1)
        with np.errstate(divide='ignore', invalid='ignore'):
            eta = np.where(((da > 0) & (rc < 0)) | ((da < 0) & (rc > 0)),
                           -rc / da, np.inf)
        o = np.argsort(eta, 1)
        eta_s = np.take_along_axis(eta, o, 1); rc_s = np.take_along_axis(rc, o, 1)
        da_s = np.take_along_axis(da, o, 1)
        w_s = np.take_along_axis(np.broadcast_to(width, (nn, ngen)), o, 1)
        decr = np.where(rc_s < 0, w_s * da_s, -w_s * da_s)
        decr = np.where(np.isfinite(eta_s), decr, 0.0)
        slope_after = slope0[:, None] - np.cumsum(decr, 1)
        below = slope_after <= 1e-12
        first = below.argmax(1)
        es = np.take_along_axis(eta_s, first[:, None], 1)[:, 0]
        es = np.where(below.any(1) & np.isfinite(es), es, 0.0)
        es = np.maximum(es, 0.0)
        lam = np.maximum(0.0, lam + es[:, None] * sp)
    return best


for K in (1, 5, 20, 100):
    t = time.perf_counter()
    lda = vec_dual_lb(A, bias.copy(), K)
    hda = -vec_dual_lb(-A, -bias.copy(), K)
    vt = time.perf_counter() - t
    stable = int(((lda >= 0) | (hda <= 0)).sum())
    print(f'VEC dual k={K:>3}: {vt*1000:6.2f} ms (both bounds, all {n}) | '
          f'proves {stable}/{n} stable | gap lo {np.mean(lo_lp-lda):.3f}')
lp_stable = int(((lo_lp >= 0) | (hi_lp <= 0)).sum())
print(f'Gurobi LP loop:   {lp_t*1000:6.2f} ms | proves {lp_stable}/{n} stable (exact)')
env.dispose()
