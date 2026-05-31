# soundnessbench — PARKED (GPU-blocked)

Soundness-stress benchmark: 50 instances, one large conv net
(`128 → Gemm(12288) → ReLU → Reshape → Conv×6 → Gemm`, ~240K ReLUs). Every
instance is **SAT** with an adversarially-hidden counterexample; the benchmark
exists to catch *unsound* verifiers (a `verified`/`unsat` here = unsound).

## Status

- **vibecheck is SOUND on it**: every probed instance returns `unknown`, never a
  false `verified`. (model_0: PGD-500 sat=False, then the zonotope forward OOMs
  → `unknown`.) This is the property the benchmark checks, and we pass it.
- **NOT yet solved** (matching ABC's `sat`). Two blockers:
  1. **Memory**: the dense zonotope forward allocates ~15 GB at the wide conv
     layers (`apply_relu`); the local GPU is 7.5 GB → CUDA OOM. Needs the 24 GB
     AWS GPU or a chunked / generator-capped `apply_relu`.
  2. **Hidden CEX**: PGD (500 restarts) does not find the witness — these CEXs
     are designed to defeat incomplete attacks (ABC spends ~100 s of BaB). A
     stronger attack / BaB is needed, which itself needs the bounds (→ memory).

## Config (`configs/soundnessbench.yaml`)

`auto_route_milp_for_conv: false` — keep the net in verify_graph so Phase-0 PGD
attacks BEFORE the heavy bound propagation (the witness is what we need; a
verified bound is impossible since every case is SAT).

## Reproduce

```bash
B=$VNNCOMP/benchmarks/soundnessbench
.venv/bin/python -m vibecheck.main --net $B/onnx/model.onnx \
  --spec $B/vnnlib/model_0.vnnlib --config configs/soundnessbench.yaml \
  --pgd-restarts 500 --timeout 90 --results-file /tmp/r.txt   # -> unknown (sound)
```

## Next steps when unparked

Run on the 24 GB AWS GPU; add a memory-aware (chunked) `apply_relu` so the dense
zono fits smaller GPUs; strengthen the attack (BaB-guided) to find the CEX.
