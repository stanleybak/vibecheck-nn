"""Measure the freeze-replay hybrid on the hard cases (the Option-B fallback
target) + confirm it stays sound. If hybrid(3_3)+~20s stall < 116s, the simple
adaptive fallback (input-split primary, time/leaf-stall -> hybrid) closes it.
"""
import time
import numpy as np
from vibecheck.network import ComputeGraph
from vibecheck.vnnlib_loader import load_vnnlib
from vibecheck.settings import default_settings
from vibecheck.config_loader import load_config
from vibecheck.verify_graph import verify_graph

B = '/home/stan/repositories/vnncomp2025_benchmarks/benchmarks/acasxu_2023'
CFG = load_config('configs/acasxu_2023.yaml')
for net, prop in [('3_3', 'prop_2'), ('4_2', 'prop_2')]:
    g = ComputeGraph.from_onnx(f'{B}/onnx/ACASXU_run2a_{net}_batch_2000.onnx', dtype=np.float32)
    sp = load_vnnlib(f'{B}/vnnlib/{prop}.vnnlib')
    ov = dict(device='gpu', bits=32, total_timeout=200, pgd_restarts=100); ov.update(CFG)
    ov['use_hybrid_acasxu'] = True   # freeze-replay hybrid
    s = default_settings(**ov); s.print_progress = False
    g.optimize(s)
    t = time.perf_counter(); r, d = verify_graph(g, sp, s); w = time.perf_counter() - t
    print(f'HYBRID {net} {prop}: {r} in {w:.1f}s phase={d.get("phase")}', flush=True)
