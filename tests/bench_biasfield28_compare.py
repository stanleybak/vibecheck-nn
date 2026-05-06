"""Bench: verify_graph end-to-end on biasfield_28 with vs without sparse_alpha.

Captures total wall time, verdict, phase timings, and per-iter alpha-CROWN
spec_lb history (via monkey-patching the alpha_crown module).
"""
import os
import sys
import time
import pickle
import numpy as np
from pathlib import Path

ONNX = "/home/stan/repositories/vnncomp2025_benchmarks/benchmarks/relusplitter/onnx/cifar_biasfield_vnncomp2022_cifar_bias_field_28.onnx"
VNN = "/home/stan/repositories/vnncomp2025_benchmarks/benchmarks/relusplitter/vnnlib/cifar_biasfield_vnncomp2022_prop_28.vnnlib"


def make_settings(sparse_alpha, n_iters=100):
    from vibecheck.settings import default_settings
    s = default_settings()
    s.alpha_crown_sparse_alpha = sparse_alpha
    s.zono_lift_alpha_iters = n_iters
    s.zono_lift_alpha_lr = 0.1   # match AB lr_alpha
    s.alpha_crown_lr_decay = 0.98  # match AB
    return s


def install_iter_capture():
    """Monkey-patch alpha_crown to capture iter histories."""
    import vibecheck.alpha_crown as ac
    captured = {'batched': [], 'fixed': []}

    orig_batched = ac.run_alpha_crown_batched
    orig_fixed = ac.run_alpha_crown_fixed_intermediate_batched

    def wrap_batched(*args, **kw):
        t0 = time.time()
        ret = orig_batched(*args, **kw)
        elapsed = time.time() - t0
        best_lbs, alpha_params, best_bounds, histories = ret
        captured['batched'].append({
            'n_q': len(histories),
            'n_iters_actual': max((len(h) for h in histories), default=0),
            'best_lbs': best_lbs.tolist(),
            'histories': [h[:] for h in histories],
            'wall_s': elapsed,
            'sparse_alpha': kw.get('sparse_alpha', False),
            'lr': kw.get('lr', None),
        })
        return ret

    def wrap_fixed(*args, **kw):
        t0 = time.time()
        ret = orig_fixed(*args, **kw)
        elapsed = time.time() - t0
        best_lbs, alpha_params, best_bounds, histories = ret
        captured['fixed'].append({
            'n_q': len(histories),
            'n_iters_actual': max((len(h) for h in histories), default=0),
            'best_lbs': best_lbs.tolist(),
            'histories': [h[:] for h in histories],
            'wall_s': elapsed,
            'sparse_alpha': kw.get('sparse_alpha', False),
            'lr': kw.get('lr', None),
        })
        return ret

    ac.run_alpha_crown_batched = wrap_batched
    ac.run_alpha_crown_fixed_intermediate_batched = wrap_fixed
    return captured


def run_one(sparse_alpha, label, n_iters=100):
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    from vibecheck.onnx_loader import load_onnx
    from vibecheck.vnnlib_loader import load_vnnlib
    from vibecheck.verify_graph import verify_graph

    captured = install_iter_capture()
    settings = make_settings(sparse_alpha, n_iters=n_iters)
    print(f"\n=== {label}: sparse_alpha={sparse_alpha} n_iters={n_iters} ===", flush=True)

    g = load_onnx(ONNX)
    g.optimize(settings)
    spec = load_vnnlib(VNN)

    t0 = time.time()
    result, details = verify_graph(g, spec, settings)
    wall = time.time() - t0

    out = {
        'label': label,
        'sparse_alpha': sparse_alpha,
        'verdict': result,
        'wall_s': wall,
        'details_timing': details.get('timing', {}),
        'details_phase': details.get('phase', None),
        'remaining': details.get('remaining', None),
        'iter_capture': captured,
    }
    print(f"verdict={result} wall={wall:.2f}s phase={out['details_phase']}",
          flush=True)
    print("timing:", out['details_timing'], flush=True)
    print("batched calls:", len(captured['batched']),
          "fixed calls:", len(captured['fixed']), flush=True)
    for i, c in enumerate(captured['batched']):
        print(f"  batched[{i}] n_q={c['n_q']} n_iters={c['n_iters_actual']} wall={c['wall_s']:.2f}s sparse={c['sparse_alpha']} lr={c['lr']}",
              flush=True)
    for i, c in enumerate(captured['fixed']):
        print(f"  fixed[{i}] n_q={c['n_q']} n_iters={c['n_iters_actual']} wall={c['wall_s']:.2f}s sparse={c['sparse_alpha']} lr={c['lr']}",
              flush=True)
    return out


if __name__ == '__main__':
    Path('/tmp/abcrown_runs').mkdir(parents=True, exist_ok=True)
    out_path = '/tmp/abcrown_runs/biasfield28_vibecheck_compare.pkl'
    results = []
    # Sweep: (sparse_alpha, n_iters)
    grid = [
        (True, 10),   # default n_iters with sparse
        (False, 10),  # default n_iters dense
        (True, 100),  # AB-matching iters with sparse
        (False, 100), # AB-matching iters dense (likely OOM)
    ]
    for sa, ni in grid:
        try:
            r = run_one(sa, f"sparse={sa}_iters={ni}", n_iters=ni)
            results.append(r)
        except Exception as e:
            print(f"FAILED sparse={sa} iters={ni}: {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()
            results.append({'label': f'sparse={sa}_iters={ni}', 'error': str(e)})
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nWrote {out_path}", flush=True)
