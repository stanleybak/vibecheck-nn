"""explore11: mutual zono<->CROWN intermediate-bound tightening — root spec margin.

Hypothesis (user's idea): at each layer compute BOTH the forward-zono and the
backward-CROWN pre-ReLU bound, intersect, and feed the intersection back into
BOTH the next zono layer's relaxation AND the next CROWN. The zono tracks
input-symbol correlations interval-CROWN drops, so the intersection can beat
either alone and compound across layers.

Measures the ROOT (un-split) spec margin = min over queries of the backward-CROWN
spec lower bound, using `tight` from:
  - forward zono only            (baseline loose)
  - CROWN-only  (n_sweeps=1)     (explore10: -722 on 3_3 prop_2, -2148 on prop_1)
  - mutual      (n_sweeps=2,3,5) (this exploration)
Less negative = tighter. Target: beat the CROWN-only numbers.
"""
import time
import numpy as np
import torch

from vibecheck.network import ComputeGraph
from vibecheck.vnnlib_loader import load_vnnlib
from vibecheck.verify_zono_bnb import (
    _forward_zonotope_graph_batched, _spec_backward_graph_batched,
    _crown_intermediate_batched)

BENCH = '/home/stan/repositories/vnncomp2025_benchmarks/benchmarks/acasxu_2023'
CASES = [
    ('prop_2 3_3', 'onnx/ACASXU_run2a_3_3_batch_2000.onnx', 'vnnlib/prop_2.vnnlib'),
    ('prop_1 1_1', 'onnx/ACASXU_run2a_1_1_batch_2000.onnx', 'vnnlib/prop_1.vnnlib'),
]

device = torch.device('cpu')
dtype = torch.float64


def root_margin(tight, xl, xh, gg, spec_ew):
    sl = _spec_backward_graph_batched(tight, xl, xh, gg, spec_ew, device, dtype)
    return float(sl.min().item())  # worst query margin


for tag, net, vnn in CASES:
    graph = ComputeGraph.from_onnx(f'{BENCH}/{net}', dtype=np.float32)
    spec = load_vnnlib(f'{BENCH}/{vnn}')
    gg = graph.gpu_graph(device, dtype)

    n_out = None
    for op in reversed(gg['ops']):
        if op.get('type') == 'fc':
            n_out = int(op['W'].shape[0]); break
    queries = spec.as_linear_queries(n_out)
    spec_ew = {qi: (torch.as_tensor(w, dtype=dtype, device=device).flatten(),
                    float(b)) for qi, (di, w, b) in enumerate(queries)}

    xl = torch.tensor(np.asarray(spec.x_lo).flatten(), dtype=dtype,
                      device=device).unsqueeze(0)
    xh = torch.tensor(np.asarray(spec.x_hi).flatten(), dtype=dtype,
                      device=device).unsqueeze(0)

    print(f'\n=== {tag} ({n_out} outputs, {len(queries)} queries) ===')

    t = time.perf_counter()
    sb, _ = _forward_zonotope_graph_batched(xl, xh, gg, device, dtype)
    m = root_margin(sb, xl, xh, gg, spec_ew)
    print(f'  fwd-zono only      margin={m:10.3f}   ({1e3*(time.perf_counter()-t):6.1f} ms)')

    for ns in (1, 2, 3, 5):
        t = time.perf_counter()
        tight = _crown_intermediate_batched(gg, xl, xh, device, dtype, n_sweeps=ns)
        ms = 1e3 * (time.perf_counter() - t)
        m = root_margin(tight, xl, xh, gg, spec_ew)
        label = 'CROWN-only' if ns == 1 else f'mutual x{ns}'
        print(f'  {label:18s} margin={m:10.3f}   ({ms:6.1f} ms)')
