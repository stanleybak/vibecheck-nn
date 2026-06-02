"""Micro-measurement of the LP-contraction at the FIRST split — to check the
LOGIC (does contracting the e-box recover the LP-tight bound?) and SPEED (cost of
the per-dim contraction LPs vs per-neuron LPs vs the cheap box bound).

nnenum's contract = per-dim min/max LP of e over {box, C·e<=d}, tightening the
e-box; then bounds use the CHEAP box (bias + signed a·e over the box), no per-
neuron LP. Witness-tracking skips dims the new constraint doesn't touch (the speed
win) — measured here as 'dims actually tightened'.
"""
import sys, time
import numpy as np
import gurobipy as gp
sys.path.insert(0, 'scratch/nnenum_style')
from star_bab import load_acasxu, load_spec, overapprox_relu


class BStar:
    """bias + a_mat@e ; e in [e_lb, e_hi] (a contractible box) ; C@e <= d."""
    def __init__(self, bias, a_mat, e_lb, e_hi, C, d):
        self.bias, self.a_mat, self.e_lb, self.e_hi, self.C, self.d = bias, a_mat, e_lb, e_hi, C, d

    @classmethod
    def from_box(cls, lo, hi):
        c = (lo + hi) / 2; r = (hi - lo) / 2
        k = len(lo)
        return cls(c, np.diag(r), -np.ones(k), np.ones(k), np.zeros((0, k)), np.zeros(0))

    def linear(self, W, b):
        return BStar(W @ self.bias + b, W @ self.a_mat, self.e_lb, self.e_hi, self.C, self.d)

    def box_bounds(self):
        """CHEAP zono bound over the (contracted) box — no LP."""
        a = self.a_mat
        lo = self.bias + (np.where(a > 0, a * self.e_lb, a * self.e_hi)).sum(1)
        hi = self.bias + (np.where(a > 0, a * self.e_hi, a * self.e_lb)).sum(1)
        return lo, hi

    def lp_bound_neuron(self, i, env, lower=True):
        k = self.a_mat.shape[1]
        m = gp.Model(env=env); m.setParam('OutputFlag', 0); m.setParam('Threads', 1)
        e = m.addMVar(k, lb=self.e_lb, ub=self.e_hi)
        if self.C.shape[0]:
            m.addConstr(self.C @ e <= self.d)
        m.setObjective(self.bias[i] + self.a_mat[i] @ e,
                       gp.GRB.MINIMIZE if lower else gp.GRB.MAXIMIZE)
        m.optimize(); v = m.objVal; m.dispose()
        return v

    def add_split(self, ni, active):
        a = self.a_mat[ni]
        row, rhs = (-a, self.bias[ni]) if active else (a, -self.bias[ni])
        C = np.concatenate([self.C, row[None, :]], 0) if self.C.shape[0] else row[None, :]
        d = np.concatenate([self.d, [rhs]])
        return BStar(self.bias, self.a_mat, self.e_lb.copy(), self.e_hi.copy(), C, d)

    def contract(self, env, only_dims=None):
        """Tighten e-box per dim via LP. only_dims (witness opt) limits which dims.
        Returns (#dims tightened, #LPs, seconds)."""
        k = self.a_mat.shape[1]
        dims = range(k) if only_dims is None else only_dims
        t = time.perf_counter(); nlp = 0; tightened = 0
        m = gp.Model(env=env); m.setParam('OutputFlag', 0); m.setParam('Threads', 1)
        e = m.addMVar(k, lb=self.e_lb, ub=self.e_hi)
        if self.C.shape[0]:
            m.addConstr(self.C @ e <= self.d)
        for j in dims:
            m.setObjective(e[j], gp.GRB.MINIMIZE); m.optimize(); nlp += 1
            nl = m.objVal
            m.setObjective(e[j], gp.GRB.MAXIMIZE); m.optimize(); nlp += 1
            nu = m.objVal
            if nl > self.e_lb[j] + 1e-9 or nu < self.e_hi[j] - 1e-9:
                tightened += 1
            self.e_lb[j] = max(self.e_lb[j], nl); self.e_hi[j] = min(self.e_hi[j], nu)
        m.dispose()
        return tightened, nlp, time.perf_counter() - t


def run(net, prop):
    layers, sub = load_acasxu(net)
    spec = load_spec(net, prop)
    xl = np.asarray(spec.x_lo).flatten() - sub
    xh = np.asarray(spec.x_hi).flatten() - sub
    env = gp.Env(empty=True); env.setParam('OutputFlag', 0); env.start()
    # propagate (overapprox) to the first layer with an unstable neuron
    st = BStar.from_box(xl, xh).linear(*layers[0])
    li = 0
    while True:
        lo, hi = st.box_bounds()
        unst = (lo < 0) & (hi > 0)
        if unst.any():
            break
        # need a relu-applied star to continue; reuse star_bab.overapprox_relu on a
        # plain (bias,a_mat) view — keep box untouched (still [-1,1] there)
        from star_bab import Star
        z = Star(st.bias, st.a_mat, st.C, st.d)
        z = overapprox_relu(z, lo, hi)
        # grow e-box for the new generators
        ng = z.a_mat.shape[1] - st.a_mat.shape[1]
        st = BStar(z.bias, z.a_mat,
                   np.concatenate([st.e_lb, -np.ones(ng)]),
                   np.concatenate([st.e_hi, np.ones(ng)]), z.C, z.d).linear(*layers[li + 1])
        li += 1
    # FIRST split: most-ambiguous unstable neuron
    scores = np.minimum(hi, -lo); scores[~unst] = -np.inf
    ni = int(np.argmax(scores))
    print(f'first split at layer {li}, neuron {ni}: pre-act bound [{lo[ni]:.3f},{hi[ni]:.3f}], '
          f'k(gens)={st.a_mat.shape[1]}, n_unstable_this_layer={int(unst.sum())}')
    child = st.add_split(ni, active=True)   # z_ni >= 0 branch
    # --- micro-measure on this child ---
    # contract
    import copy
    c2 = copy.deepcopy(child)
    nt, nlp, secs = c2.contract(env)
    print(f'CONTRACT (full, k={child.a_mat.shape[1]} dims): tightened {nt} dims, '
          f'{nlp} LPs, {secs*1000:.1f} ms')
    # bound of a sample neuron at the NEXT relu layer, 3 ways
    from star_bab import Star as PStar
    def next_layer_star(bs):
        z = PStar(bs.bias, bs.a_mat, bs.C, bs.d)
        zlo, zhi = bs.box_bounds()
        z = overapprox_relu(z, zlo, zhi)
        z = z.linear(*layers[li + 1])
        return z, zlo, zhi
    # uncontracted vs contracted: compare the relu pre-act bound at next layer
    for tag, bs in [('uncontracted', child), ('contracted', c2)]:
        zlo, zhi = bs.box_bounds()
        # widest unstable neuron's box width as a proxy for tightness
        w = (zhi - zlo)
        print(f'  {tag}: next-relu input box-width sum={w.sum():.3f} '
              f'(mean {w.mean():.3f})')
    # LP-exact bound of one neuron (the split one's layer is done; pick neuron 0 next layer)
    nxt = child.linear(*layers[li + 1]) if li + 1 < len(layers) else None
    if nxt is not None:
        t = time.perf_counter()
        lp_lo = child.lp_bound_neuron(0, env, lower=True)
        lp_ms = (time.perf_counter() - t) * 1000
        blo, _ = child.box_bounds(); clo, _ = c2.box_bounds()
        print(f'  neuron-0 (this layer) lower bound: box(uncontr)={blo[0]:.4f} '
              f'box(contr)={clo[0]:.4f} LP-exact={lp_lo:.4f} (LP {lp_ms:.1f} ms)')
    env.dispose()


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else '3_3',
        sys.argv[2] if len(sys.argv) > 2 else 'prop_2')
