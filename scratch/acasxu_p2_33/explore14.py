"""explore14: does the mutual zono<->CROWN tightening help on LEAVES, not just root?

User: "not just on root but also on leaves." The mutual is already applied per
leaf-batch (verify_graph.py:9857), but explore11 only measured the ROOT margin.
Here: bisect prop_2 3_3 along its widest input dim to increasing depth (narrower
leaves), and at each depth measure the WORST-query backward-CROWN margin with
CROWN-only vs mutual(x4). If the mutual's relative gain collapses as the box
shrinks, leaves genuinely don't benefit (zono ~= CROWN on narrow boxes) and the
per-leaf mutual cost is pure waste; if it persists, leaves DO benefit.
"""
import numpy as np
import torch

from vibecheck.network import ComputeGraph
from vibecheck.vnnlib_loader import load_vnnlib
from vibecheck.verify_zono_bnb import (
    _spec_backward_graph_batched, _crown_intermediate_batched)

BENCH = '/home/stan/repositories/vnncomp2025_benchmarks/benchmarks/acasxu_2023'
NET, VNN = 'onnx/ACASXU_run2a_3_3_batch_2000.onnx', 'vnnlib/prop_2.vnnlib'
device, dtype = torch.device('cpu'), torch.float64

graph = ComputeGraph.from_onnx(f'{BENCH}/{NET}', dtype=np.float32)
spec = load_vnnlib(f'{BENCH}/{VNN}')
gg = graph.gpu_graph(device, dtype)
n_out = next(int(op['W'].shape[0]) for op in reversed(gg['ops'])
             if op.get('type') == 'fc')
queries = spec.as_linear_queries(n_out)
spec_ew = {qi: (torch.as_tensor(w, dtype=dtype, device=device).flatten(), float(b))
           for qi, (di, w, b) in enumerate(queries)}

xl = torch.tensor(np.asarray(spec.x_lo).flatten(), dtype=dtype, device=device)
xh = torch.tensor(np.asarray(spec.x_hi).flatten(), dtype=dtype, device=device)


def margin(method, xl1, xh1):
    xl1 = xl1.unsqueeze(0); xh1 = xh1.unsqueeze(0)
    if method == 'crown':
        tight = _crown_intermediate_batched(gg, xl1, xh1, device, dtype, n_sweeps=1)
    else:
        tight = _crown_intermediate_batched(gg, xl1, xh1, device, dtype, n_sweeps=4)
    sl = _spec_backward_graph_batched(tight, xl1, xh1, gg, spec_ew, device, dtype)
    return float(sl.min().item())


print(f'{"depth":>5} {"width(max)":>11} {"crown":>10} {"mutual":>10} {"gain%":>7}')
cl, ch = xl.clone(), xh.clone()
for depth in range(0, 13, 2):
    w = ch - cl
    mc = margin('crown', cl, ch)
    mm = margin('mutual', cl, ch)
    gain = 100.0 * (mm - mc) / abs(mc) if mc != 0 else 0.0
    print(f'{depth:5d} {float(w.max()):11.5f} {mc:10.3f} {mm:10.3f} {gain:7.2f}')
    # bisect the widest dim twice (depth+2): keep the lower half each time.
    for _ in range(2):
        d = int(torch.argmax(ch - cl))
        ch = ch.clone(); ch[d] = (cl[d] + ch[d]) / 2
