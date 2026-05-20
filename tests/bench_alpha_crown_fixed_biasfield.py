"""Bench `run_alpha_crown_fixed_intermediate_batched` (the AB-CROWN-equivalent
config: fix_intermediate_bounds=True, sparse_alpha) on cifar_biasfield_28.

Run on the GPU server:
    .venv/bin/python tests/bench_alpha_crown_fixed_biasfield.py [--sparse]
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
    ap.add_argument("--sparse", action="store_true")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--n-iters", type=int, default=20)
    ap.add_argument("--n-q", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = {"fp32": torch.float32, "fp16": torch.float16,
             "bf16": torch.bfloat16}[args.dtype]

    graph = load_onnx(str(ONNX))
    settings = default_settings()
    graph.optimize(settings)
    spec = load_vnnlib(str(SPEC))

    gg = graph.gpu_graph(device, dtype)
    xl = torch.from_numpy(spec.x_lo.flatten().astype(np.float32)).to(
        device).to(dtype)
    xh = torch.from_numpy(spec.x_hi.flatten().astype(np.float32)).to(
        device).to(dtype)
    sb, _ = _forward_zonotope_graph(xl, xh, gg, device, dtype)
    bbr_init = {L: (lo.cpu().float().numpy().astype(np.float64),
                    hi.cpu().float().numpy().astype(np.float64))
                for L, (lo, hi) in sb.items()}

    queries = spec.as_linear_queries(10)
    n_q = min(args.n_q, len(queries))
    w_qs = np.stack([q[1] for q in queries[:n_q]]).astype(np.float32)
    b_qs = np.array([q[2] for q in queries[:n_q]], dtype=np.float32)

    print(f"=== fixed_intermediate_batched n_q={n_q} n_iters={args.n_iters}"
          f" sparse={args.sparse} ===")
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    best_lbs, alpha_params, _, histories = ac.run_alpha_crown_fixed_intermediate_batched(
        gg, xl, xh, bbr_init, w_qs, b_qs,
        device, dtype,
        n_iters=args.n_iters, lr=0.25, lr_decay=0.98,
        early_stop_on_positive=False,
        sparse_alpha=args.sparse,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    total_params = sum(t.numel() for d in alpha_params.values() for t in d.values())
    print(f"  elapsed       = {elapsed:.4f} s")
    print(f"  per-iter      = {elapsed/args.n_iters*1000:.2f} ms")
    print(f"  total alpha params = {total_params}")
    print(f"  best_lbs[:8]  = {[f'{x:+.4f}' for x in best_lbs[:8]]}")
    print(f"  iter0 lbs     = {[f'{h[0]:+.4f}' for h in histories[:4]]}")
    print(f"  iter-1 lbs    = {[f'{h[-1]:+.4f}' for h in histories[:4]]}")


if __name__ == "__main__":
    main()
