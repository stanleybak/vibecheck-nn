"""explore_optionb: adaptive escalation = input-split primary + hybrid fallback.

Hypothesis: hard prop_2-unsat cases blow the input-split worklist FAST (3_3:
1.88M leaves, ~26k/s), so a worklist cap detects the stall in a few seconds;
falling back to the freeze-replay hybrid (which does 3_3 in ~47s) then verifies
within budget. Easy cases (prop_1) verify before the cap -> no fallback, stay
<1s. This makes the hybrid's tightness available WITHOUT its prop_1 freeze cost.

Measures, for 3_3/4_2 prop_2 (hard) and 1_1 prop_3/prop_1 (easy):
  - input-split with max_worklist=CAP: verdict + wall (= stall-detect time on hard)
  - verify_hybrid: verdict + wall (the fallback)
  - => combined estimate for the hard cases.
"""
import sys, time
import numpy as np
from vibecheck.network import ComputeGraph
from vibecheck.vnnlib_loader import load_vnnlib
from vibecheck.settings import default_settings
from vibecheck.config_loader import load_config
from vibecheck.verify_graph import verify_graph

B = '/home/stan/repositories/vnncomp2025_benchmarks/benchmarks/acasxu_2023'
CFG = load_config('configs/acasxu_2023.yaml')
CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
CASES = [
    ('3_3', 'prop_2', 'hard'), ('4_2', 'prop_2', 'hard'),
    ('1_1', 'prop_3', 'easy'), ('1_1', 'prop_1', 'easy'),
]


def run(net, prop, max_worklist=None, hybrid=False):
    g = ComputeGraph.from_onnx(f'{B}/onnx/ACASXU_run2a_{net}_batch_2000.onnx', dtype=np.float32)
    sp = load_vnnlib(f'{B}/vnnlib/{prop}.vnnlib')
    ov = dict(device='gpu', bits=32, total_timeout=200, pgd_restarts=100); ov.update(CFG)
    if hybrid:
        ov['use_hybrid_acasxu'] = True
    elif max_worklist is not None:
        ov['input_split_batched_max_worklist'] = max_worklist
    s = default_settings(**ov); s.print_progress = False
    g.optimize(s)
    t = time.perf_counter(); r, d = verify_graph(g, sp, s); w = time.perf_counter() - t
    return r, w, d.get('phase'), d.get('batched_n_leaves')


for net, prop, kind in CASES:
    r1, w1, ph1, lv1 = run(net, prop, max_worklist=CAP)
    line = f'{net} {prop} ({kind}): input-split(cap={CAP}) -> {r1} {w1:.1f}s phase={ph1} leaves={lv1}'
    if kind == 'hard' and r1 != 'verified':
        r2, w2, ph2, _ = run(net, prop, hybrid=True)
        line += f'  || hybrid -> {r2} {w2:.1f}s  || COMBINED ~{w1 + w2:.1f}s'
    print('  ' + line, flush=True)
