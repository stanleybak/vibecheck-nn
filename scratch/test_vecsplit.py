"""Validate the vectorized on-GPU 2-way split: correctness (SAT cases still caught,
UNSAT verified) + speed (3_3 prop_2 should drop well below the old 112s). The
explore02 regression was 1_5 SAT breaking on the reordered split — leaf-PGD
(worst-margin) should now catch it regardless of order.
"""
import time
import numpy as np
from pathlib import Path

from vibecheck.network import ComputeGraph
from vibecheck.vnnlib_loader import load_vnnlib
from vibecheck.settings import default_settings
from vibecheck.config_loader import load_config
from vibecheck.verify_graph import verify_graph

B = '/home/stan/repositories/vnncomp2025_benchmarks/benchmarks/acasxu_2023'
CFG = load_config('configs/acasxu_2023.yaml')
CASES = [
    ('1_5 prop_2 SAT', '1_5', 'prop_2', 'sat'),
    ('1_9 prop_7 SAT', '1_9', 'prop_7', 'sat'),
    ('1_1 prop_3 UNSAT', '1_1', 'prop_3', 'verified'),
    ('1_1 prop_1 UNSAT', '1_1', 'prop_1', 'verified'),
    ('3_3 prop_2 UNSAT (SPEED)', '3_3', 'prop_2', 'verified'),
]
for tag, net, prop, exp in CASES:
    g = ComputeGraph.from_onnx(f'{B}/onnx/ACASXU_run2a_{net}_batch_2000.onnx', dtype=np.float32)
    sp = load_vnnlib(f'{B}/vnnlib/{prop}.vnnlib')
    ov = dict(device='gpu', bits=32, total_timeout=180, pgd_restarts=100); ov.update(CFG)
    s = default_settings(**ov); s.print_progress = False
    g.optimize(s)
    t = time.perf_counter(); r, d = verify_graph(g, sp, s); w = time.perf_counter() - t
    ok = 'OK ' if r == exp else 'XX '
    print(f'  {ok}{tag:26s} -> {r:9s} {w:7.2f}s (exp {exp}) phase={d.get("phase")}', flush=True)
