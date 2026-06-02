"""explore17: diagnose why 1_9 prop_7 SAT is missed. Is the witness findable on
the FULL root box (=> strengthen Phase-0 PGD) or only on a shrunk leaf (=>
strengthen leaf-PGD)? Run _simple_pgd on the full box at increasing restarts, and
_pgd_attack_general, and report.
"""
import time
import numpy as np
import torch

from vibecheck.network import ComputeGraph
from vibecheck.vnnlib_loader import load_vnnlib
from vibecheck.settings import default_settings
from vibecheck.verify_hybrid_acasxu import _simple_pgd

BENCH = '/home/stan/repositories/vnncomp2025_benchmarks/benchmarks/acasxu_2023'
NET, VNN = 'onnx/ACASXU_run2a_1_9_batch_2000.onnx', 'vnnlib/prop_7.vnnlib'
device, dtype = torch.device('cuda'), torch.float32

graph = ComputeGraph.from_onnx(f'{BENCH}/{NET}', dtype=np.float32)
spec = load_vnnlib(f'{BENCH}/{VNN}')
settings = default_settings(device='gpu', bits=32)
graph.optimize(settings)
gg = graph.gpu_graph(device, dtype)
n_out = next(int(op['W'].shape[0]) for op in reversed(gg['ops']) if op.get('type') == 'fc')
print(f'prop_7 1_9: {len(spec.disjuncts)} disjuncts, n_out={n_out}')

xl = torch.tensor(np.asarray(spec.x_lo).flatten(), dtype=dtype, device=device)
xh = torch.tensor(np.asarray(spec.x_hi).flatten(), dtype=dtype, device=device)
print('input widths:', (xh - xl).cpu().numpy())

for R in (1000, 10000, 50000, 200000):
    t = time.perf_counter()
    sat, w = _simple_pgd(xl, xh, spec, gg, n_out, device, dtype,
                         n_restarts=R, n_iter=100, seed=0)
    print(f'  _simple_pgd full box R={R:7d}: sat={sat}  ({time.perf_counter()-t:.2f}s)', flush=True)
    if sat:
        break
