"""explore16: validate leaf-PGD fix on the 2 narrow-SAT acasxu cases that the
input-split+CROWN path missed (1_5 prop_2, 1_9 prop_7). Expect both -> sat now.
Also re-checks 1_1 prop_3 UNSAT didn't regress (leaf-PGD must not false-sat).
"""
import time
import numpy as np
from pathlib import Path

from vibecheck.network import ComputeGraph
from vibecheck.vnnlib_loader import load_vnnlib
from vibecheck.settings import default_settings
from vibecheck.config_loader import load_config
from vibecheck.verify_graph import verify_graph

BENCH = '/home/stan/repositories/vnncomp2025_benchmarks/benchmarks/acasxu_2023'
CFG = load_config(str(Path('configs/acasxu_2023.yaml')))
CASES = [
    ('1_5 prop_2 SAT', 'onnx/ACASXU_run2a_1_5_batch_2000.onnx', 'vnnlib/prop_2.vnnlib', 'sat'),
    ('1_9 prop_7 SAT', 'onnx/ACASXU_run2a_1_9_batch_2000.onnx', 'vnnlib/prop_7.vnnlib', 'sat'),
    ('1_1 prop_3 UNSAT', 'onnx/ACASXU_run2a_1_1_batch_2000.onnx', 'vnnlib/prop_3.vnnlib', 'verified'),
]

for tag, net, vnn, expected in CASES:
    graph = ComputeGraph.from_onnx(f'{BENCH}/{net}', dtype=np.float32)
    spec = load_vnnlib(f'{BENCH}/{vnn}')
    ov = dict(device='gpu', bits=32, total_timeout=116, pgd_restarts=100)
    ov.update(CFG)
    settings = default_settings(**ov)
    settings.print_progress = False
    graph.optimize(settings)
    t0 = time.perf_counter()
    result, det = verify_graph(graph, spec, settings)
    wall = time.perf_counter() - t0
    ok = 'OK ' if result == expected else 'XX '
    print(f'  {ok}{tag:18s} -> {result:9s} {wall:7.2f}s (exp {expected}, '
          f'phase={det.get("phase","?")})', flush=True)
