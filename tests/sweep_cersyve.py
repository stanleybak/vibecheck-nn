"""Cersyve 12-case sweep via direct API. Mirrors AB-CROWN-style table
output for quick side-by-side checks during benchmark optimization.

Usage:
    .venv/bin/python tests/sweep_cersyve.py [TIMEOUT_SECS]

Output columns:
    case · ref · vc · sec · nodes · match
where match=OK iff vc result aligns with ref (ref='unsat'→'verified',
ref='sat'→'sat').
"""
import sys, time, numpy as np
sys.stdout.reconfigure(line_buffering=True)

from vibecheck.network import ComputeGraph
from vibecheck.vnnlib_loader import load_vnnlib
from vibecheck.settings import default_settings
from vibecheck.config_loader import load_config
from vibecheck.verify_graph import verify_graph

ROOT = '/home/stan/repositories/vnncomp2025_benchmarks/benchmarks/cersyve'
CFG_PATH = '/home/stan/repositories/vibecheck/configs/cersyve.yaml'

INSTANCES = [
    ('lane_keep_pretrain_con',   'prop_lane_keep',  'sat'),
    ('lane_keep_pretrain_inv',   'prop_lane_keep',  'sat'),
    ('lane_keep_finetune_con',   'prop_lane_keep',  'unsat'),
    ('lane_keep_finetune_inv',   'prop_lane_keep',  'unsat'),
    ('pendulum_pretrain_con',    'prop_pendulum',   'sat'),
    ('pendulum_pretrain_inv',    'prop_pendulum',   'sat'),
    ('pendulum_finetune_con',    'prop_pendulum',   'unsat'),
    ('pendulum_finetune_inv',    'prop_pendulum',   'unsat'),
    ('point_mass_pretrain_con',  'prop_point_mass', 'sat'),
    ('point_mass_pretrain_inv',  'prop_point_mass', 'sat'),
    ('point_mass_finetune_con',  'prop_point_mass', 'unsat'),
    ('point_mass_finetune_inv',  'prop_point_mass', 'unsat'),
]

TIMEOUT = int(sys.argv[1]) if len(sys.argv) > 1 else 30


def main():
    yaml_overrides = load_config(CFG_PATH)
    print(f'cersyve sweep — timeout={TIMEOUT}s per case, config={CFG_PATH}')
    print(f'{"case":<32} {"ref":<6} {"vc":<10} {"sec":<6} {"nodes":<7} '
          f'{"match"}')
    print('-' * 80)
    ok = 0
    for net, vnn, ref in INSTANCES:
        g = ComputeGraph.from_onnx(f'{ROOT}/onnx/{net}.onnx.gz',
                                     dtype=np.float32)
        spec = load_vnnlib(f'{ROOT}/vnnlib/{vnn}.vnnlib.gz')
        overrides = dict(device='gpu', bits=32, total_timeout=TIMEOUT,
                          pgd_restarts=100)
        overrides.update(yaml_overrides)
        s = default_settings(**overrides)
        s.print_progress = False
        g.optimize(s)
        t0 = time.time()
        result, det = verify_graph(g, spec, s)
        dt = time.time() - t0
        aligned = (result == ref
                    or (ref == 'unsat' and result == 'verified')
                    or (ref == 'sat' and result == 'sat'))
        match = 'OK' if aligned else ('UNK' if result == 'unknown' else 'WRONG')
        if aligned:
            ok += 1
        print(f'{net:<32} {ref:<6} {result:<10} {dt:5.1f} '
              f'{str(det.get("input_split_n_nodes", "-")):<7} {match}')
    print('-' * 80)
    print(f'TOTAL: {ok}/{len(INSTANCES)}')


if __name__ == '__main__':
    main()
