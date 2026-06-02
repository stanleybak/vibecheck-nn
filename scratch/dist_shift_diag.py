"""Diagnose a dist_shift UNSAT miss phase-by-phase: why does verify_graph return
unknown in ~6s (well under the 60s budget) instead of verifying like ABC (8s)?
Compares a MISS case vs a PASSING case. Prints phase timing + final spec margin.
"""
import sys, time
import numpy as np
from pathlib import Path

from vibecheck.network import ComputeGraph
from vibecheck.vnnlib_loader import load_vnnlib
from vibecheck.settings import default_settings
from vibecheck.config_loader import load_config
from vibecheck.verify_graph import verify_graph

BENCH = '/home/stan/repositories/vnncomp2025_benchmarks/benchmarks/dist_shift_2023'
CFG = load_config('configs/dist_shift_2023.yaml')

# arg: vnnlib index name (default the first miss)
cases = sys.argv[1:] or ['index7027_delta0.13', 'index7901_delta0.13']  # miss, pass

for name in cases:
    graph = ComputeGraph.from_onnx(f'{BENCH}/onnx/mnist_concat.onnx', dtype=np.float32)
    spec = load_vnnlib(f'{BENCH}/vnnlib/{name}.vnnlib')
    ov = dict(device='gpu', bits=32, total_timeout=60, pgd_restarts=128)
    ov.update(CFG)
    settings = default_settings(**ov)
    settings.print_progress = True
    graph.optimize(settings)
    print(f'\n########## {name} ##########', flush=True)
    t0 = time.perf_counter()
    result, det = verify_graph(graph, spec, settings)
    wall = time.perf_counter() - t0
    print(f'>>> {name}: {result} in {wall:.2f}s  phase={det.get("phase","?")}', flush=True)
    # dump any margin / timing keys in details
    for k in sorted(det):
        if any(s in k.lower() for s in ('margin', 'lb', 'phase', 'time', 'timing', 'leaves', 'unknown', 'budget')):
            v = det[k]
            if isinstance(v, (int, float, str)):
                print(f'    det[{k}] = {v}', flush=True)
