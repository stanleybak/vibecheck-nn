"""acasxu_2023 sweep using configs/acasxu_2023.yaml. Mirrors AB-CROWN-
style table output for side-by-side comparison.

Usage:
    .venv/bin/python tests/sweep_acasxu.py [TIMEOUT_SECS]
"""
import sys, time, csv, numpy as np
sys.stdout.reconfigure(line_buffering=True)

from vibecheck.network import ComputeGraph
from vibecheck.vnnlib_loader import load_vnnlib
from vibecheck.settings import default_settings
from vibecheck.config_loader import load_config
from vibecheck.verify_graph import verify_graph

import os
ROOT = os.path.expanduser(
    '~/repositories/vnncomp2025_benchmarks/benchmarks/acasxu_2023')
REF_CSV = os.path.expanduser(
    '~/repositories/vnncomp2025_results/alpha_beta_crown/'
    '2025_acasxu_2023/results.csv')
CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'configs', 'acasxu_2023.yaml')
TIMEOUT = int(sys.argv[1]) if len(sys.argv) > 1 else 60


def main():
    # Reference verdicts from AB-CROWN published results.
    ref = []
    with open(REF_CSV) as f:
        for row in csv.reader(f):
            net = row[1].split('/')[-1]
            vnn = row[2].split('/')[-1]
            if 'ACASXU' not in net:
                continue
            ref.append((net, vnn, row[4], float(row[5])))

    yaml_overrides = load_config(CFG_PATH)
    print(f'acasxu sweep — timeout={TIMEOUT}s/case, config={CFG_PATH}, '
          f'{len(ref)} cases')
    print(f'{"case":<55} {"ref":<6} {"vc":<10} {"sec":<6} {"abc_sec":<7} '
          f'{"match"}')
    print('-' * 95)
    ok = 0; wrong = 0; unk = 0
    for net, vnn, ref_v, ref_t in ref:
        import os
        net_path = (f'{ROOT}/onnx/{net}'
                     if os.path.exists(f'{ROOT}/onnx/{net}')
                     else f'{ROOT}/onnx/{net}.gz')
        vnn_path = (f'{ROOT}/vnnlib/{vnn}'
                     if os.path.exists(f'{ROOT}/vnnlib/{vnn}')
                     else f'{ROOT}/vnnlib/{vnn}.gz')
        try:
            g = ComputeGraph.from_onnx(net_path, dtype=np.float32)
            spec = load_vnnlib(vnn_path)
            overrides = dict(device='gpu', bits=32, total_timeout=TIMEOUT,
                              pgd_restarts=200)
            overrides.update(yaml_overrides)
            s = default_settings(**overrides)
            s.print_progress = False
            g.optimize(s)
            t0 = time.time()
            r, det = verify_graph(g, spec, s)
            dt = time.time() - t0
        except Exception as e:
            r = f'err:{type(e).__name__}'
            dt = 0.0
        aligned = (r == ref_v
                    or (ref_v == 'unsat' and r == 'verified')
                    or (ref_v == 'sat' and r == 'sat'))
        if aligned: ok += 1
        elif r == 'unknown': unk += 1
        else: wrong += 1
        match = 'OK' if aligned else ('UNK' if r == 'unknown' else 'WRONG')
        case = f'{net[:35]} {vnn}'[:55]
        print(f'{case:<55} {ref_v:<6} {r:<10} {dt:5.1f} {ref_t:6.1f} {match}')
    print('-' * 95)
    print(f'TOTAL: ok={ok}/{len(ref)}  unknown={unk}  wrong={wrong}')


if __name__ == '__main__':
    main()
