"""End-to-end verify_graph timing on biasfield_28 with vs without sparse_alpha."""
import os, sys, time, pickle
from pathlib import Path

ONNX = "/home/stan/repositories/vnncomp2025_benchmarks/benchmarks/relusplitter/onnx/cifar_biasfield_vnncomp2022_cifar_bias_field_28.onnx"
VNN = "/home/stan/repositories/vnncomp2025_benchmarks/benchmarks/relusplitter/vnnlib/cifar_biasfield_vnncomp2022_prop_28.vnnlib"


def run(sparse_alpha):
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    from vibecheck.settings import default_settings
    from vibecheck.onnx_loader import load_onnx
    from vibecheck.vnnlib_loader import load_vnnlib
    from vibecheck.verify_graph import verify_graph

    s = default_settings()
    s.alpha_crown_sparse_alpha = sparse_alpha
    print(f"\n=== sparse_alpha={sparse_alpha} ===", flush=True)

    g = load_onnx(ONNX)
    g.optimize(s)
    spec = load_vnnlib(VNN)

    t0 = time.time()
    try:
        result, details = verify_graph(g, spec, s)
        wall = time.time() - t0
        print(f"verdict={result} wall={wall:.2f}s", flush=True)
        print(f"timing: {details.get('timing', {})}", flush=True)
        print(f"phase: {details.get('phase')}", flush=True)
        print(f"remaining: {details.get('remaining')}", flush=True)
        return {'sparse': sparse_alpha, 'verdict': result, 'wall': wall,
                'timing': dict(details.get('timing', {})),
                'phase': details.get('phase'),
                'remaining': details.get('remaining')}
    except Exception as e:
        wall = time.time() - t0
        print(f"FAILED: {type(e).__name__}: {e} (wall={wall:.2f}s)", flush=True)
        return {'sparse': sparse_alpha, 'error': f"{type(e).__name__}: {e}",
                'wall': wall}


if __name__ == '__main__':
    Path('/tmp/abcrown_runs').mkdir(parents=True, exist_ok=True)
    res = []
    for sa in [True, False]:
        r = run(sa)
        res.append(r)
    with open('/tmp/abcrown_runs/biasfield28_endtoend.pkl', 'wb') as f:
        pickle.dump(res, f)
    print("\nSummary:")
    for r in res:
        print(f"  sparse={r.get('sparse')} verdict={r.get('verdict','?')} wall={r.get('wall',0):.2f}s err={r.get('error','-')}")
