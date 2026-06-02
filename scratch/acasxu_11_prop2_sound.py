"""CRITICAL soundness check: ACASXU 1_1 prop_2. ABC says SAT; our audit said
verified. Confirm ground truth: is there a REAL counterexample? Run strong PGD +
onnxruntime forward. If a real violating point exists, our 'verified' is UNSOUND.
"""
import numpy as np
import torch

from vibecheck.network import ComputeGraph
from vibecheck.vnnlib_loader import load_vnnlib
from vibecheck.settings import default_settings
from vibecheck.config_loader import load_config
from vibecheck.verify_graph import verify_graph
from vibecheck.verify_hybrid_acasxu import _simple_pgd

B = '/home/stan/repositories/vnncomp2025_benchmarks/benchmarks/acasxu_2023'
NET = f'{B}/onnx/ACASXU_run2a_1_1_batch_2000.onnx'
VNN = f'{B}/vnnlib/prop_2.vnnlib'
dev, dt = torch.device('cuda'), torch.float32

graph = ComputeGraph.from_onnx(NET, dtype=np.float32)
spec = load_vnnlib(VNN)
s = default_settings(device='gpu', bits=32)
graph.optimize(s)
gg = graph.gpu_graph(dev, dt)
n_out = next(int(op['W'].shape[0]) for op in reversed(gg['ops']) if op.get('type') == 'fc')
print('disjuncts:', len(spec.disjuncts), 'n_out:', n_out)
xl = torch.tensor(np.asarray(spec.x_lo).flatten(), dtype=dt, device=dev)
xh = torch.tensor(np.asarray(spec.x_hi).flatten(), dtype=dt, device=dev)

# Strong PGD to find a counterexample (ground truth).
sat, w = _simple_pgd(xl, xh, spec, gg, n_out, dev, dt, n_restarts=200000, n_iter=200)
print(f'>>> _simple_pgd 200k restarts: sat={sat}')
if sat:
    # verify the witness via onnxruntime
    import onnxruntime as ort
    sess = ort.InferenceSession(NET)
    iname = sess.get_inputs()[0].name
    ishape = sess.get_inputs()[0].shape
    xw = np.asarray(w, dtype=np.float32).reshape([d if isinstance(d, int) else 1 for d in ishape])
    y = sess.run(None, {iname: xw})[0].flatten()
    print('  witness onnx output:', y)
    _, ck = spec.check(y, y)
    print('  spec.check worst_margin:', ck.get('worst_margin'))
    print('  REAL COUNTEREXAMPLE' if ck.get('worst_margin', 0) < 0 else '  not actually violating')

# Now run our production verify with the acasxu config.
graph2 = ComputeGraph.from_onnx(NET, dtype=np.float32)
spec2 = load_vnnlib(VNN)
CFG = load_config('configs/acasxu_2023.yaml')
ov = dict(device='gpu', bits=32, total_timeout=120, pgd_restarts=100); ov.update(CFG)
s2 = default_settings(**ov); s2.print_progress = False
graph2.optimize(s2)
r, d = verify_graph(graph2, spec2, s2)
print(f'>>> OUR verdict (acasxu config): {r}  phase={d.get("phase")}')
