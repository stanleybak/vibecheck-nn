"""explore13: prop_2 3_3 throughput — batch_size sweep (the ABC gap hypothesis).

explore12 showed the mutual tightening doesn't help end-to-end: prop_2 3_3 root
barely tightens (4%), and the BaB "no divergence" -> leaves DO close, just slowly.
ABC solves it in 18s with plain `bound_prop_method: crown` + batch_size 16384;
we default to 4096. Hypothesis: on a tiny FC net the GPU is under-utilised at
4096, so 4x the batch = ~4x throughput. Sweep input_split_batch_size.
"""
import sys
import time
import numpy as np

from vibecheck.network import ComputeGraph
from vibecheck.vnnlib_loader import load_vnnlib
from vibecheck.settings import default_settings
from vibecheck.verify_graph import verify_graph

BENCH = '/home/stan/repositories/vnncomp2025_benchmarks/benchmarks/acasxu_2023'
NET = 'onnx/ACASXU_run2a_3_3_batch_2000.onnx'
VNN = 'vnnlib/prop_2.vnnlib'

for bs in (4096, 16384, 32768, 65536):
    graph = ComputeGraph.from_onnx(f'{BENCH}/{NET}', dtype=np.float32)
    spec = load_vnnlib(f'{BENCH}/{VNN}')
    settings = default_settings(
        device='gpu', bits=32, total_timeout=120,
        use_hybrid_acasxu=False,
        input_split_batched_enabled=True,
        input_split_crown_intermediate=True,
        input_split_crown_intermediate_sweeps=1,
        input_split_batch_size=bs,
        disable_sat_finding=True,
    )
    settings.print_progress = False
    graph.optimize(settings)
    t0 = time.perf_counter()
    result, det = verify_graph(graph, spec, settings)
    wall = time.perf_counter() - t0
    ok = 'OK ' if result == 'verified' else 'XX '
    print(f'  {ok}batch={bs:6d} -> {result:9s} {wall:7.2f}s '
          f'(phase={det.get("phase","?")})', flush=True)
