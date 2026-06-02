"""nnenum-style star/zonotope BaB prototype for ACAS Xu (vibecheck-flavored).

Copies nnenum's core: propagate a zonotope (overapprox ALL unstable ReLUs), check
the spec on the output; if proven safe -> PRUNE (no split). Only when the
overapprox can't prove safety do we case-SPLIT one ReLU (sign constraint) and
recurse. This is the opposite of input-splitting: we split the actual nonlinearity
and let the cheap overapprox prune the rest.

Single-threaded. Gurobi (Threads=1) only for the CONSTRAINED bound LPs (a star
with split-constraints); unconstrained stars use the closed-form zonotope bound
(no LP). Start on easy cases (prop_4: 0 splits; prop_3: 1) to validate.

Run: .venv/bin/python scratch/nnenum_style/star_bab.py 1_1 prop_4
"""
import sys, time
import numpy as np
import onnx, onnx.numpy_helper as onh

BENCH = '/home/stan/repositories/vnncomp2025_benchmarks/benchmarks/acasxu_2023'


def load_acasxu(net):
    """Return (layers, sub) where layers = [(W (out,in), b)], applied as
    a = relu(W@prev + b) for all but the last (last is linear). `sub` is the
    input normalization constant (x -> x - sub)."""
    m = onnx.load(f'{BENCH}/onnx/ACASXU_run2a_{net}_batch_2000.onnx')
    inits = {i.name: onh.to_array(i).astype(np.float64) for i in m.graph.initializer}
    sub = inits.get('input_AvgImg')
    sub = sub.flatten() if sub is not None else None
    # MatMul W are (in,out) -> transpose to (out,in); Add b are (out,)
    Ws = [inits[k].T for k in sorted(inits) if k.endswith('_MatMul_W')]
    bs = [inits[k] for k in sorted(inits) if k.endswith('_Add_B')]
    return list(zip(Ws, bs)), sub


def load_spec(net, prop):
    from vibecheck.vnnlib_loader import load_vnnlib
    p = f'{BENCH}/vnnlib/{prop}.vnnlib'
    import os
    if not os.path.exists(p):
        import gzip, shutil
        with gzip.open(p + '.gz', 'rb') as fi, open(p, 'wb') as fo:
            shutil.copyfileobj(fi, fo)
    return load_vnnlib(p)


class Star:
    """x = bias + a_mat @ e, e in [-1,1]^k, with C @ e <= d (split constraints)."""
    __slots__ = ('bias', 'a_mat', 'C', 'd')

    def __init__(self, bias, a_mat, C, d):
        self.bias = bias; self.a_mat = a_mat; self.C = C; self.d = d

    @classmethod
    def from_box(cls, lo, hi):
        c = (lo + hi) / 2.0
        rad = (hi - lo) / 2.0
        a = np.diag(rad)
        return cls(c, a, np.zeros((0, len(lo))), np.zeros(0))

    def linear(self, W, b):
        return Star(W @ self.bias + b, W @ self.a_mat, self.C, self.d)

    def bounds(self, gp_env=None):
        """Per-neuron [lo, hi]. No constraints -> closed-form zono bound.
        With constraints -> Gurobi LP per active direction (only the unstable
        ones the caller asks about, but here we do all via LP for simplicity)."""
        if self.C.shape[0] == 0:
            r = np.abs(self.a_mat).sum(axis=1)
            return self.bias - r, self.bias + r
        return self._lp_bounds(gp_env)

    def _lp_bounds(self, gp_env):
        import gurobipy as gp
        k = self.a_mat.shape[1]
        m = gp.Model(env=gp_env)
        m.setParam('OutputFlag', 0); m.setParam('Threads', 1)
        e = m.addMVar(k, lb=-1.0, ub=1.0)
        if self.C.shape[0]:
            m.addConstr(self.C @ e <= self.d)
        lo = np.empty(len(self.bias)); hi = np.empty(len(self.bias))
        for i in range(len(self.bias)):
            obj = self.bias[i] + self.a_mat[i] @ e
            m.setObjective(obj, gp.GRB.MINIMIZE); m.optimize(); lo[i] = m.objVal
            m.setObjective(obj, gp.GRB.MAXIMIZE); m.optimize(); hi[i] = m.objVal
        m.dispose()
        return lo, hi


def overapprox_relu(star, lo, hi):
    """Zonotope min-area ReLU overapprox: stable neurons exact, unstable get a
    new generator. Returns a new Star (k grows by #unstable)."""
    n = len(star.bias)
    active = lo >= 0
    dead = hi <= 0
    unst = ~active & ~dead
    lam = np.where(unst, hi / (hi - lo + 1e-30), np.where(active, 1.0, 0.0))
    mu = np.where(unst, -hi * lo / (2 * (hi - lo + 1e-30)), 0.0)
    new_bias = lam * star.bias + mu
    new_a = lam[:, None] * star.a_mat
    # add one generator per unstable neuron (coeff mu on its row)
    u_idx = np.where(unst)[0]
    if len(u_idx):
        G = np.zeros((n, len(u_idx)))
        G[u_idx, np.arange(len(u_idx))] = mu[u_idx]
        new_a = np.concatenate([new_a, G], axis=1)
        C = np.concatenate([star.C, np.zeros((star.C.shape[0], len(u_idx)))], axis=1) \
            if star.C.shape[0] else np.zeros((0, new_a.shape[1]))
    else:
        C = star.C
    return Star(new_bias, new_a, C, star.d)


def build_queries(spec, n_out):
    """Group spec into disjuncts of linear queries (w, b): verified iff EVERY
    disjunct has SOME query with (w·y + b) > 0 for the whole set."""
    from collections import defaultdict
    disj = defaultdict(list)
    for qi, (di, w, b) in enumerate(spec.as_linear_queries(n_out)):
        disj[di].append((np.asarray(w, dtype=np.float64).flatten(), float(b)))
    return list(disj.values())


def spec_safe(star, queries, env):
    """SAFE iff every disjunct has a query whose LOWER bound over the star is >0.
    Lower bound of w·(bias + a_mat@e) + b: exact zono range if unconstrained,
    else a Gurobi LP (the tight, direction-projected bound — NOT per-output box)."""
    import gurobipy as gp
    for qs in queries:                       # each disjunct
        ok = False
        for w, b in qs:
            c = w @ star.bias + b
            row = w @ star.a_mat
            if star.C.shape[0] == 0:
                lb = c - np.abs(row).sum()
            else:
                k = star.a_mat.shape[1]
                m = gp.Model(env=env); m.setParam('OutputFlag', 0); m.setParam('Threads', 1)
                e = m.addMVar(k, lb=-1.0, ub=1.0)
                m.addConstr(star.C @ e <= star.d)
                m.setObjective(c + row @ e, gp.GRB.MINIMIZE); m.optimize()
                lb = m.objVal; m.dispose()
            if lb > 1e-7:
                ok = True; break
        if not ok:
            return False
    return True


# ---- the BaB ----
def verify(net, prop, max_splits=200000):
    layers, sub = load_acasxu(net)
    spec = load_spec(net, prop)
    x_lo = np.asarray(spec.x_lo, dtype=np.float64).flatten()
    x_hi = np.asarray(spec.x_hi, dtype=np.float64).flatten()
    if sub is not None:
        x_lo = x_lo - sub; x_hi = x_hi - sub
    import gurobipy as gp
    env = gp.Env(empty=True); env.setParam('OutputFlag', 0); env.start()
    n_relu = len(layers) - 1
    n_out = len(layers[-1][1])
    queries = build_queries(spec, n_out)
    stats = {'leaves': 0, 'splits': 0, 'lp': 0}

    def split_constraint(st, ni, active):
        a_row = st.a_mat[ni]
        if active:  # z_ni >= 0  <=>  -a_row @ e <= bias[ni]
            row, rhs = -a_row[None, :], st.bias[ni]
        else:       # z_ni <= 0  <=>   a_row @ e <= -bias[ni]
            row, rhs = a_row[None, :], -st.bias[ni]
        C = np.concatenate([st.C, row], axis=0) if st.C.shape[0] else row
        d = np.concatenate([st.d, [rhs]])
        return Star(st.bias, st.a_mat, C, d)

    def overapprox_to_output(st_preact, li):
        """Overapprox every remaining ReLU from layer li onward; return the
        OUTPUT star (so the spec is checked direction-projected, not box)."""
        ast = overapprox_relu(st_preact, *st_preact.bounds(env))
        if st_preact.C.shape[0]:
            stats['lp'] += 1
        for lj in range(li + 1, len(layers)):
            ast = ast.linear(*layers[lj])
            if lj == len(layers) - 1:
                break
            alo, ahi = ast.bounds(env)
            if ast.C.shape[0]:
                stats['lp'] += 1
            ast = overapprox_relu(ast, alo, ahi)
        return ast

    import sys as _sys
    _sys.setrecursionlimit(100000)

    def process(st_preact, li):
        """st_preact: PRE-activation star at layer li (linear already applied)."""
        if li == len(layers) - 1:   # output layer pre-activation = output
            stats['leaves'] += 1
            return spec_safe(st_preact, queries, env)
        lo, hi = st_preact.bounds(env)
        if st_preact.C.shape[0]:
            stats['lp'] += 1
        unst = (lo < 0) & (hi > 0)
        if not unst.any():
            relu_st = overapprox_relu(st_preact, lo, hi)  # exact (lam in {0,1})
            return process(relu_st.linear(*layers[li + 1]), li + 1)
        # try to PRUNE with a full overapprox to the output
        if spec_safe(overapprox_to_output(st_preact, li), queries, env):
            stats['leaves'] += 1
            return True
        if stats['splits'] > max_splits:
            return False
        # split the most-ambiguous unstable neuron at THIS layer
        scores = np.minimum(hi, -lo); scores[~unst] = -np.inf
        ni = int(np.argmax(scores))
        stats['splits'] += 1
        return (process(split_constraint(st_preact, ni, True), li)
                and process(split_constraint(st_preact, ni, False), li))

    t = time.perf_counter()
    root = Star.from_box(x_lo, x_hi).linear(*layers[0])
    safe = process(root, 0)
    dt = time.perf_counter() - t
    env.dispose()
    return safe, dt, stats


if __name__ == '__main__':
    net = sys.argv[1] if len(sys.argv) > 1 else '1_1'
    prop = sys.argv[2] if len(sys.argv) > 2 else 'prop_4'
    safe, dt, stats = verify(net, prop)
    print(f'STARBAB {net} {prop}: {"SAFE(holds)" if safe else "not-proven"} '
          f'{dt:.2f}s  splits={stats["splits"]} leaves={stats["leaves"]} lp={stats["lp"]}')
