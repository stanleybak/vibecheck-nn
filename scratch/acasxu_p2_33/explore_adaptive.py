"""explore_adaptive: input-split primary + leaf-cap stall escalation to the
freeze-replay hybrid. Expect: easy cases stay fast (input-split), 3_3 escalates
to the hybrid (~10s churn + 46s = ~56s), SAT still caught by leaf-PGD. All <116s.
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
CASES = [
    ('3_3', 'prop_2', 'verified'), ('4_2', 'prop_2', 'verified'),
    ('1_1', 'prop_3', 'verified'), ('1_1', 'prop_1', 'verified'),
    ('1_5', 'prop_2', 'sat'), ('1_9', 'prop_7', 'sat'),
]
for net, prop, exp in CASES:
    g = ComputeGraph.from_onnx(f'{B}/onnx/ACASXU_run2a_{net}_batch_2000.onnx', dtype=np.float32)
    sp = load_vnnlib(f'{B}/vnnlib/{prop}.vnnlib')
    ov = dict(device='gpu', bits=32, total_timeout=200, pgd_restarts=100); ov.update(CFG)
    ov['input_split_batched_stall_leaf_cap'] = 300000
    ov['acasxu_hybrid_on_stall'] = True
    s = default_settings(**ov); s.print_progress = False
    g.optimize(s)
    t = time.perf_counter(); r, d = verify_graph(g, sp, s); w = time.perf_counter() - t
    ok = 'OK ' if r == exp else 'XX '
    print(f'ADAPT {ok}{net} {prop}: {r} in {w:.1f}s (exp {exp}) phase={d.get("phase")}', flush=True)
