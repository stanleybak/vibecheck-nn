"""explore12: end-to-end input-split BaB with mutual zono<->CROWN intermediates.

Compares wall time + verdict for the input-split-CROWN path (no hybrid) with
crown_intermediate_sweeps = 1 (CROWN-only) vs 3 (mutual, converged per
explore11). Does the tighter per-leaf bound cut leaves enough to beat the ~3x
bound cost? prop_2 3_3 was 62-111s on this path; prop_1 ~1.2s.
"""
import sys
import time
import numpy as np

from vibecheck.network import ComputeGraph
from vibecheck.vnnlib_loader import load_vnnlib
from vibecheck.settings import default_settings
from vibecheck.verify_graph import verify_graph

BENCH = '/home/stan/repositories/vnncomp2025_benchmarks/benchmarks/acasxu_2023'
CASES = [
    ('prop_2 3_3', 'onnx/ACASXU_run2a_3_3_batch_2000.onnx', 'vnnlib/prop_2.vnnlib', 'verified'),
    ('prop_1 1_1', 'onnx/ACASXU_run2a_1_1_batch_2000.onnx', 'vnnlib/prop_1.vnnlib', 'verified'),
    ('prop_3 1_1', 'onnx/ACASXU_run2a_1_1_batch_2000.onnx', 'vnnlib/prop_3.vnnlib', 'verified'),
]

sweeps = int(sys.argv[1]) if len(sys.argv) > 1 else 3

for tag, net, vnn, expected in CASES:
    graph = ComputeGraph.from_onnx(f'{BENCH}/{net}', dtype=np.float32)
    spec = load_vnnlib(f'{BENCH}/{vnn}')
    settings = default_settings(
        device='gpu', bits=32, total_timeout=120,
        use_hybrid_acasxu=False,
        input_split_batched_enabled=True,
        input_split_crown_intermediate=True,
        input_split_crown_intermediate_sweeps=sweeps,
        disable_sat_finding=True,   # UNSAT cases: skip PGD, pure bound race
    )
    settings.print_progress = False
    graph.optimize(settings)
    t0 = time.perf_counter()
    result, det = verify_graph(graph, spec, settings)
    wall = time.perf_counter() - t0
    ok = 'OK ' if result == expected else 'XX '
    print(f'  {ok}{tag:14s} sweeps={sweeps} -> {result:9s} {wall:7.2f}s '
          f'(exp {expected}, phase={det.get("phase","?")})')
