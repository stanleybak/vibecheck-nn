"""acasxu sweep with PER-CASE RACING between two configs:
  v1: verify_graph with α-CROWN at leaves + multi-α PGD (good for prop_2/3/4)
  v2: verify_graph with batched + clipping (good for prop_1 UNSAT)

For each case, launch both configs in parallel processes. First to
return a definitive verdict (sat/verified) wins; the other is killed.
If both finish as unknown, report unknown. Sweep total wall time ≈
1x (parallel) but per-case compute = 2x.

Usage:
    .venv/bin/python tests/sweep_acasxu_race.py [TIMEOUT_SECS]
"""
import sys, os, time, csv, signal
sys.stdout.reconfigure(line_buffering=True)
import multiprocessing as mp
import numpy as np

ROOT = os.path.expanduser(
    '~/repositories/vnncomp2025_benchmarks/benchmarks/acasxu_2023')
REF_CSV = os.path.expanduser(
    '~/repositories/vnncomp2025_results/alpha_beta_crown/'
    '2025_acasxu_2023/results.csv')
TIMEOUT = int(sys.argv[1]) if len(sys.argv) > 1 else 60


V1_OVERRIDES = dict(
    pgd_alpha_multi=True, pgd_init_mode='osi', pgd_iter=200,
    pgd_restarts=200,
    input_split_leaf_pgd_enabled=True, input_split_leaf_pgd_time=0.1,
    input_split_batched_enabled=False,
)
V2_OVERRIDES = dict(
    pgd_alpha_multi=True, pgd_init_mode='osi', pgd_iter=200,
    pgd_restarts=200,
    input_split_batched_enabled=True,
    input_split_batched_clip_enabled=True,
    input_split_batch_size=4096,
    input_split_batched_max_worklist=2_000_000,
)


def _worker(args):
    """One verification config in a subprocess. Returns (label, verdict)."""
    label, net_path, vnn_path, overrides, timeout = args
    import numpy as np
    from vibecheck.network import ComputeGraph
    from vibecheck.vnnlib_loader import load_vnnlib
    from vibecheck.config_profiles import default_settings_for
    from vibecheck.verify_graph import verify_graph
    g = ComputeGraph.from_onnx(net_path, dtype=np.float32)
    spec = load_vnnlib(vnn_path)
    s = default_settings_for(g, spec, device='gpu', bits=32,
                              total_timeout=timeout, pgd_restarts=200)
    for k, v in overrides.items():
        s[k] = v
    s.print_progress = False
    g.optimize(s)
    t0 = time.time()
    try:
        r, _ = verify_graph(g, spec, s)
    except Exception as e:
        r = f'err:{type(e).__name__}'
    return label, r, time.time() - t0


def race_case(net_path, vnn_path, timeout):
    """Launch both configs in parallel; return first definitive verdict."""
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(2)
    jobs = [
        pool.apply_async(_worker, [('v1', net_path, vnn_path,
                                      V1_OVERRIDES, timeout)]),
        pool.apply_async(_worker, [('v2', net_path, vnn_path,
                                      V2_OVERRIDES, timeout)]),
    ]
    t_start = time.time()
    winner = None
    winner_dt = None
    other = []
    while True:
        for j in jobs:
            if j.ready():
                label, r, dt = j.get()
                if winner is None and r in ('sat', 'verified'):
                    winner = (label, r)
                    winner_dt = dt
                    pool.terminate(); pool.join()
                    return winner, winner_dt, other
                other.append((label, r, dt))
                jobs.remove(j)
                break
        else:
            if time.time() - t_start > timeout + 5:
                pool.terminate(); pool.join()
                if other and winner is None:
                    return (other[0][0], other[0][1]), other[0][2], other[1:]
                return ('timeout', 'unknown'), timeout, other
            time.sleep(0.05)
            continue
        if not jobs:
            break
    # Both finished without verifying; report first
    if other:
        if winner is None:
            winner = (other[0][0], other[0][1])
            winner_dt = other[0][2]
            other = other[1:]
    return winner or ('?', 'unknown'), winner_dt or 0.0, other


def main():
    ref = []
    with open(REF_CSV) as f:
        for row in csv.reader(f):
            net = row[1].split('/')[-1]
            vnn = row[2].split('/')[-1]
            if 'ACASXU' not in net:
                continue
            ref.append((net, vnn, row[4], float(row[5])))

    print(f'acasxu race sweep — timeout={TIMEOUT}s/case, {len(ref)} cases')
    print(f'{"case":<55} {"ref":<6} {"vc":<10} {"sec":<6} {"abc":<6} '
          f'{"winner":<7} {"match"}')
    print('-' * 100)
    ok = 0; wrong = 0; unk = 0
    for net, vnn, ref_v, ref_t in ref:
        net_path = (f'{ROOT}/onnx/{net}' if os.path.exists(f'{ROOT}/onnx/{net}')
                     else f'{ROOT}/onnx/{net}.gz')
        vnn_path = (f'{ROOT}/vnnlib/{vnn}' if os.path.exists(f'{ROOT}/vnnlib/{vnn}')
                     else f'{ROOT}/vnnlib/{vnn}.gz')
        (winner, r), dt, other = race_case(net_path, vnn_path, TIMEOUT)
        aligned = (r == ref_v
                    or (ref_v == 'unsat' and r == 'verified')
                    or (ref_v == 'sat' and r == 'sat'))
        if aligned: ok += 1
        elif r == 'unknown': unk += 1
        else: wrong += 1
        match = 'OK' if aligned else ('UNK' if r == 'unknown' else 'WRONG')
        case = f'{net[:35]} {vnn}'[:55]
        print(f'{case:<55} {ref_v:<6} {r:<10} {dt:5.1f} {ref_t:5.1f} '
              f'{winner:<7} {match}')
    print('-' * 100)
    print(f'TOTAL: ok={ok}/{len(ref)}  unknown={unk}  wrong={wrong}')


if __name__ == '__main__':
    main()
