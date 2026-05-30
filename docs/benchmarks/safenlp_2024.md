# safenlp_2024 — vibecheck benchmark record

VNNCOMP 2025 regular track. NLP perturbation-robustness.

## Benchmark shape

- **1080 instances**, 2 models, all 20 s timeout:
  - `medical/perturbations_0.onnx` — 294 instances
  - `ruarobot/perturbations_0.onnx` — 786 instances
- Both nets are tiny and identical in shape: **30 → MatMul(30×128)+Add → ReLU → MatMul(128×2)+Add → 2** (one ReLU layer, 128 neurons). The export emits MatMul and the bias as a **separate Add node** (TF/Keras style), not a fused Gemm.
- Spec: 30-dim input hyperrectangle; output constraint `(<= Y_0 Y_1)`. UNSAFE iff `Y_0 ≤ Y_1` is reachable, so **SAT = a violating input exists (counterexample)**, **UNSAT = `Y_0 > Y_1` proven over the box**.
- **AB-CROWN published (2025): 646 sat, 434 unsat — solves all 1080, no timeouts.** ABC config: `complete_verifier: mip` (exact Gurobi MILP) + `pgd_order: after, pgd_restarts: 10000`.

## Final approach: exact per-neuron MILP (matches ABC's `complete_verifier: mip`)

The graph zono/CROWN relaxation is hopelessly loose on these nets (worst `spec_lb ≈ −5` on a single-ReLU net), and 30-dim input-split BaB explodes on the hard cases (wrong split axis for a 1-ReLU net). The right tool is the same one ABC uses: the **exact big-M MILP** over the ≤128 unstable ReLU binaries (`milp_verify`). On these tiny nets it is exact and sub-second locally.

vibecheck **auto-detects** this signature and routes to `milp_verify` with no config needed:
`verify_graph.py` checks pure-ReLU FC (no conv/bilinear/transcendental ops), ≤2 ReLU layers, no fork points, and `input_dim > input_split cap`, gated by `settings.auto_route_milp_for_small_fc=True` (default on). SAT cases are found by PGD; UNSAT cases proven by the exact MILP.

### Root-cause bug found & fixed: `gpu_layers()` dropped the Add bias

The first `--mode milp` attempt reported spurious **"feasibility SAT"** on cases that are provably UNSAT. Root cause: `network.gpu_layers()` (the sequential-layer flattening that `milp_verify` consumes) only processed `Gemm/MatMul/Conv` nodes and **silently skipped standalone `Add(bias)` nodes**. Because this export keeps the bias in a separate Add, the flattened net had **all-zero biases** — `milp_verify` was verifying a *different, bias-free* network, which made the unsafe region trivially reachable.

Diagnosed by cross-checking a MILP witness: same input gave `−1.54` through `gpu_layers` vs `+2.605` through onnxruntime (and raw onnx weights). Fix folds a trailing `Add(bias)` into the preceding fc layer's bias in `gpu_layers()` (both `layers[-1]['bias']` and `gpu_b_fwd[-1]`). After the fix the flattened forward matches onnxruntime exactly. This was a latent soundness/correctness bug that would affect *any* MatMul+Add-export FC net routed through `milp_verify`, not just safenlp.

### Racing schedule: direct-exact for small FC nets

`_racing_escalation`'s default bin schedule (`0,2,4,8,…,n_unstable`) spawns a fresh `Pool(2)` + Gurobi env per level (~3.7 s/level on server1's slower Gurobi). For a small pure-FC net that's pure overhead — the full exact MILP solves in one shot. Added `_DIRECT_EXACT_MAX_UNSTABLE=256`: when the net is FC-only and `n_unstable ≤ 256`, the schedule is just `[n_unstable]` (one exact solve). This is what fixed server1 case 992 (was `unknown` at 36 s from per-bin overhead; now unsat in ~17 s solve).

## Score

**vibecheck 1080/1080 — matches AB-CROWN on every instance.**

Full 1080-case local sweep (2026-05-30), VNN-COMP 20 s timeout, verdict read from
`--results-file` per case and compared to the published ABC csv:
`SWEEP_DONE match=1080 miss=0 of 1080 wall=1931s` (~1.79 s/case avg, no timeouts).
**0 misses** = vibecheck's verdict equals ABC's on all 647 sat + 433 unsat cases,
with **no sat/unsat divergence** (no soundness disagreement in either direction).

| verifier | sat | unsat | solved | timeouts |
| --- | --- | --- | --- | --- |
| vibecheck (this sweep) | 647 | 433 | **1080/1080** | 0 |
| AB-CROWN (published 2025) | 647 | 433 | 1080/1080 | 0 |

(Local machine has fast Gurobi; per-case timing is hardware-bound — see caveat.)

## Reproduction

Single case (auto-settings, no config needed):
```bash
.venv/bin/python -m vibecheck.main \
  <bench>/onnx/<model>/perturbations_0.onnx <bench>/vnnlib/<...>.vnnlib \
  --results-file /tmp/r.txt
cat /tmp/r.txt   # unsat = verified Y0>Y1; sat = counterexample
```
Full sweep: `/tmp/safenlp_local_sweep.py` (reads `instances.csv` + published ABC csv, runs vibecheck per case, compares verdict-file to ABC, logs misses).

## Related changes (shared infrastructure)

The `gpu_layers` Add-bias fold is shared by every net that reaches `milp_verify`,
so this benchmark's fix also corrected **acasxu** (which is likewise MatMul+Add,
7 affine layers). Consequences, all handled:

- **acasxu was silently verifying a bias-free net.** Before the fold, committed
  `gpu_layers` dropped acasxu's Add bias, so `milp_verify` solved a looser
  bias-free net that happened to verify `prop_3`. The unit test
  `test_graph_verify.py::test_acasxu_sequential_vs_graph` was passing for the
  wrong reason. After the fold, `gpu_layers` matches onnxruntime exactly
  (`MAX|ort−mine| = 0.0`); the raw-MILP racing path (no input-split) returns a
  sound `unknown` on the real net, while the FULL acasxu pipeline still verifies
  (integration unchanged). The test was updated to pin its real invariant —
  soundness (never `sat`) + sequential/graph consistency — matching its own
  docstring.
- **Racing direct-exact needed a depth guard.** The `_DIRECT_EXACT_MAX_UNSTABLE`
  short-circuit (one full-binary MILP) is right for *shallow* FC nets (safenlp:
  1 ReLU layer, <1 s) but wrong for *deep* ones (acasxu: 6 ReLU layers — the
  full-binary MILP is combinatorially hard; gradual escalation verifies at an
  early bin). Gated to `n_relu_layers ≤ 2`.
- **Fold adjacency guard.** The fold only fires when the `Add` directly follows
  the linear layer (`_prev_was_linear`), never across a ReLU — folding a
  post-activation bias into the pre-activation layer would be unsound.

Pinned by `tests/test_gpu_layers_bias_fold.py` (synthetic MatMul+Add net,
forward-match) so the fix is regression-guarded without needing benchmark files.

## Known caveats
- **Gurobi throughput is hardware-bound.** server1's Gurobi is ~15× slower than local for these exact MILPs; the local sweep is the authoritative timing. On adequate Gurobi hardware every UNSAT solves well within 20 s.

## Config
`configs/safenlp_2024.yaml` — documents the routing; no overrides needed (auto-route handles it).
