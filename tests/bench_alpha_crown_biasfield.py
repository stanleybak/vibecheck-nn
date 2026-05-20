"""Bench alpha_crown_batched on cifar_biasfield_28 (relusplitter onnx).

Reports per-layer unstable counts and the wall time / iteration count of
the alpha-CROWN inner loop. Used to verify the sparse-α and fp16 changes
match AB-CROWN's behaviour.

Run on the GPU server:
    .venv/bin/python tests/bench_alpha_crown_biasfield.py [--sparse] [--dtype fp32|fp16]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from vibecheck.onnx_loader import load_onnx
from vibecheck.vnnlib_loader import load_vnnlib
from vibecheck.settings import default_settings
from vibecheck.verify_zono_bnb import _forward_zonotope_graph
from vibecheck import alpha_crown as ac


BENCH_ROOT = Path.home() / "repositories" / "vnncomp2025_benchmarks" / "benchmarks" / "relusplitter"
ONNX = BENCH_ROOT / "onnx" / "cifar_biasfield_vnncomp2022_cifar_bias_field_28_RSPLITTER_cifar_biasfield_vnncomp2022_prop_28.onnx"
SPEC = BENCH_ROOT / "vnnlib" / "cifar_biasfield_vnncomp2022_prop_28.vnnlib"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse", action="store_true", help="enable sparse alpha")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--n-iters", type=int, default=20)
    ap.add_argument("--n-q", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dtype = {"fp32": torch.float32, "fp16": torch.float16,
             "bf16": torch.bfloat16}[args.dtype]
    device = torch.device(args.device)

    print(f"Loading {ONNX.name}")
    graph = load_onnx(str(ONNX))
    settings = default_settings()
    graph.optimize(settings)
    spec = load_vnnlib(str(SPEC))

    in_shape = graph.input_shape
    in_dim = int(np.prod(in_shape))
    xl_np = spec.x_lo.flatten().astype(np.float32)
    xh_np = spec.x_hi.flatten().astype(np.float32)
    xl = torch.from_numpy(xl_np).to(device)
    xh = torch.from_numpy(xh_np).to(device)
    print(f"input dim = {in_dim}")

    # Build gg via gpu_graph and then run forward zonotope to get bbr_init.
    gg = graph.gpu_graph(device, dtype)
    sb, _z_final = _forward_zonotope_graph(xl, xh, gg, device, dtype)

    bbr_init = {L: (lo.detach().cpu().numpy().astype(np.float64),
                    hi.detach().cpu().numpy().astype(np.float64))
                for L, (lo, hi) in sb.items()}

    print(f"#relu layers = {len(bbr_init)}")
    total_n = total_un = 0
    for L in sorted(bbr_init):
        lo, hi = bbr_init[L]
        n = lo.size
        un = int(((lo < 0) & (hi > 0)).sum())
        total_n += n
        total_un += un
        print(f"  L={L:>3}  n={n:>5}  unstable={un:>5}  ({100.0*un/n:.1f}%)")
    print(f"  total n={total_n} unstable={total_un} ({100.0*total_un/total_n:.1f}%)")

    # Build first-disjunct query (w_q, b_q).
    # cifar = 10 output classes
    n_out = 10
    queries = spec.as_linear_queries(n_out)
    rows = [w for _, w, _ in queries]
    rhs = [b for _, _, b in queries]
    if not rows:
        raise RuntimeError("no constraints")
    n_q = min(args.n_q, len(rows))
    w_qs = np.stack(rows[:n_q]).astype(np.float32)
    b_qs = np.array(rhs[:n_q], dtype=np.float32)

    relu_layers = sorted(bbr_init.keys())
    intermediate_start_nodes = relu_layers[1:]  # all but the first ReLU
    unstable_indices = {}
    for L in intermediate_start_nodes:
        lo, hi = bbr_init[L]
        un = np.where((lo < 0) & (hi > 0))[0].tolist()
        unstable_indices[L] = un

    print(f"\n=== Run alpha_crown_batched n_q={n_q} n_iters={args.n_iters}"
          f" sparse={args.sparse} dtype={args.dtype} ===")
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    best_lbs, alpha_params, _, histories = ac.run_alpha_crown_batched(
        gg, xl, xh, bbr_init, w_qs, b_qs,
        intermediate_start_nodes, unstable_indices,
        device, dtype,
        n_iters=args.n_iters, lr=0.25, lr_decay=0.98,
        early_stop_on_positive=False,
        sparse_alpha=args.sparse,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    # Param count.
    total_params = 0
    for S, dct in alpha_params.items():
        for L, t in dct.items():
            total_params += t.numel()
    print(f"  elapsed       = {elapsed:.3f} s")
    print(f"  per-iter      = {elapsed/args.n_iters*1000:.1f} ms")
    print(f"  total alpha params = {total_params}")
    print(f"  best_lbs[:8]  = {[f'{x:+.4f}' for x in best_lbs[:8]]}")
    print(f"  iter0 lbs     = {[f'{h[0]:+.4f}' for h in histories[:4]]}")
    print(f"  iter-1 lbs    = {[f'{h[-1]:+.4f}' for h in histories[:4]]}")


if __name__ == "__main__":
    main()
