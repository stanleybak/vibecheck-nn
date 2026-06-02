"""Benchmark neuron-bound backends on a representative acasxu overapprox star:
  - Gurobi LP (exact) — raw warm solve time
  - box bound (k=0 dual, ignore C,d)
  - dual ascent k=1,2,5,20,100 (box+halfspace, sound outer bound)
How many neurons does each PROVE stable (vs the LP), and how fast?
"""
import sys, time, pickle
sys.path.insert(0, 'scratch/nnenum_style')
import numpy as np, gurobipy as gp
from star_bab_v3 import TStar
from star_bab import load_acasxu, load_spec
from explore01_dualascent_vs_gurobi import dual_ascent_lb

layers, sub = load_acasxu('1_1'); spec = load_spec('1_1', 'prop_3')
xl = np.asarray(spec.x_lo).flatten() - sub; xh = np.asarray(spec.x_hi).flatten() - sub
k = len(xl); c0 = (xl + xh) / 2; r0 = (xh - xl) / 2
env = gp.Env(empty=True); env.setParam('OutputFlag', 0); env.start()
W0, b0 = layers[0]
s = TStar(W0 @ c0 + b0, W0 @ np.diag(r0), -np.ones(k), np.ones(k),
          np.zeros((0, k)), np.zeros(0))
# propagate the overapprox to layer 3 (a representative mid-deep star)
for l in range(0, 3):
    lo, hi = s.neuron_bounds(env); s = s.relu_triangle(lo, hi)
    s = s.linear(*layers[l + 1])
n, ngen = s.a_mat.shape; M = s.C.shape[0]
pickle.dump(dict(bias=s.bias, a_mat=s.a_mat, e_lb=s.e_lb, e_hi=s.e_hi,
                 C=s.C, d=s.d), open('/tmp/bench_star.pkl', 'wb'))
print(f'star: {n} neurons, {ngen} generators, {M} halfspace constraints')

# --- Gurobi exact LP, raw warm solve time ---
m, e = s._model(env)
t = time.perf_counter()
lo = np.empty(n); hi = np.empty(n)
for i in range(n):
    obj = s.bias[i] + s.a_mat[i] @ e
    m.setObjective(obj, gp.GRB.MINIMIZE); m.optimize(); lo[i] = m.ObjVal
    m.setObjective(obj, gp.GRB.MAXIMIZE); m.optimize(); hi[i] = m.ObjVal
gur_t = time.perf_counter() - t
lp_stable = int(((lo >= 0) | (hi <= 0)).sum())
print(f'\nGurobi LP (2n={2*n} warm solves): {gur_t*1000:.1f} ms total, '
      f'{gur_t*1e6/(2*n):.1f} us/solve | proves {lp_stable}/{n} stable')

# --- box bound (k=0) ---
a = s.a_mat; pos = np.clip(a, 0, None); neg = np.clip(a, None, 0)
box_lo = s.bias + pos @ s.e_lb + neg @ s.e_hi
box_hi = s.bias + pos @ s.e_hi + neg @ s.e_lb
box_stable = int(((box_lo >= 0) | (box_hi <= 0)).sum())
print(f'box (k=0, ignore C,d):            proves {box_stable}/{n} stable')

# --- dual ascent k iterations (loop per neuron; both bounds) ---
unit = np.eye(n)
for K in (1, 2, 5, 20, 100):
    t = time.perf_counter()
    lda = np.array([dual_ascent_lb(s, unit[i], 0.0, K) for i in range(n)])
    hda = np.array([-dual_ascent_lb(s, -unit[i], 0.0, K) for i in range(n)])
    da_t = time.perf_counter() - t
    da_stable = int(((lda >= 0) | (hda <= 0)).sum())
    gap_lo = np.mean(lo - lda); gap_hi = np.mean(hda - hi)   # dual looser by this
    print(f'dual k={K:>3}: {da_t*1000:6.1f} ms | proves {da_stable}/{n} stable '
          f'| mean bound gap vs LP: lo {gap_lo:.4f}, hi {gap_hi:.4f}')
env.dispose()
