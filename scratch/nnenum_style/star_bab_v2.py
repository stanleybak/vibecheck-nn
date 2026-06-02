"""v2: contracted-box BaB (nnenum's actual fast path). Cheap box bounds + e-box
CONTRACTION after each split (the measured #1 lever). Contraction backend is
pluggable: 'gurobi' (joint LP, matches nnenum's accuracy) or 'boxhs' (vibecheck's
closed-form box+halfspace — exact for ONE halfspace, incremental per split, ~free).

Step 1 (this file, gurobi): match nnenum's VERDICT + roughly its split count.
Step 2: swap backend='boxhs' and compare speed at equal accuracy.

Run: .venv/bin/python scratch/nnenum_style/star_bab_v2.py 3_3 prop_2 gurobi
"""
import sys, time
import numpy as np
import gurobipy as gp
sys.path.insert(0, 'scratch/nnenum_style')
from star_bab import load_acasxu, load_spec, build_queries, overapprox_relu, Star
from micro_contract import BStar


def box_dir_lb(bstar, w):
    """Lower bound of w·(bias + a@e) over the contracted box (cheap, no LP)."""
    wa = w @ bstar.a_mat
    return float(w @ bstar.bias
                 + np.where(wa > 0, wa * bstar.e_lb, wa * bstar.e_hi).sum())


def box_spec_safe(bstar, queries):
    for qs in queries:
        if not any(box_dir_lb(bstar, w) + b > 1e-7 for w, b in qs):
            return False
    return True


def relu_grow(bstar, lo, hi):
    """overapprox a relu layer on a BStar, growing the e-box for new gens."""
    z = overapprox_relu(Star(bstar.bias, bstar.a_mat, bstar.C, bstar.d), lo, hi)
    ng = z.a_mat.shape[1] - bstar.a_mat.shape[1]
    return BStar(z.bias, z.a_mat,
                 np.concatenate([bstar.e_lb, -np.ones(ng)]) if ng else bstar.e_lb,
                 np.concatenate([bstar.e_hi, np.ones(ng)]) if ng else bstar.e_hi,
                 z.C, z.d)


def verify(net, prop, backend='gurobi', max_splits=2_000_000, timeout=300):
    layers, sub = load_acasxu(net)
    spec = load_spec(net, prop)
    xl = np.asarray(spec.x_lo).flatten() - sub
    xh = np.asarray(spec.x_hi).flatten() - sub
    queries = build_queries(spec, len(layers[-1][1]))
    env = gp.Env(empty=True); env.setParam('OutputFlag', 0); env.start()
    stats = {'splits': 0, 'leaves': 0, 'contr_lp': 0, 'contr_s': 0.0}
    t0 = time.perf_counter()
    sys.setrecursionlimit(1_000_000)

    def contract(bstar):
        if bstar.C.shape[0] == 0:
            return
        if backend == 'gurobi':
            _, nlp, secs = bstar.contract(env)     # full per-dim joint LP
            stats['contr_lp'] += nlp; stats['contr_s'] += secs
        # 'boxhs' backend wired in step 2

    def overapprox_to_output(bstar, li):
        b = relu_grow(bstar, *bstar.box_bounds())
        for lj in range(li + 1, len(layers)):
            b = b.linear(*layers[lj])
            if lj == len(layers) - 1:
                break
            b = relu_grow(b, *b.box_bounds())
        return b

    def process(b, li):
        if time.perf_counter() - t0 > timeout:
            return None
        if li == len(layers) - 1:
            stats['leaves'] += 1
            return box_spec_safe(b, queries)
        lo, hi = b.box_bounds()
        unst = (lo < 0) & (hi > 0)
        if not unst.any():
            return process(relu_grow(b, lo, hi).linear(*layers[li + 1]), li + 1)
        if box_spec_safe(overapprox_to_output(b, li), queries):
            stats['leaves'] += 1
            return True
        if stats['splits'] > max_splits:
            return False
        scores = np.minimum(hi, -lo); scores[~unst] = -np.inf
        ni = int(np.argmax(scores)); stats['splits'] += 1
        ca = b.add_split(ni, True); contract(ca)
        ci = b.add_split(ni, False); contract(ci)
        ra = process(ca, li)
        if ra is None or ra is False:
            return ra
        return process(ci, li)

    root = BStar.from_box(xl, xh).linear(*layers[0])
    safe = process(root, 0)
    dt = time.perf_counter() - t0
    env.dispose()
    return safe, dt, stats


if __name__ == '__main__':
    net = sys.argv[1] if len(sys.argv) > 1 else '3_3'
    prop = sys.argv[2] if len(sys.argv) > 2 else 'prop_2'
    backend = sys.argv[3] if len(sys.argv) > 3 else 'gurobi'
    safe, dt, stats = verify(net, prop, backend=backend)
    print(f'V2[{backend}] {net} {prop}: {"SAFE" if safe else ("TIMEOUT" if safe is None else "not-proven")} '
          f'{dt:.2f}s splits={stats["splits"]} leaves={stats["leaves"]} '
          f'contr_lps={stats["contr_lp"]} contr_time={stats["contr_s"]:.1f}s')
