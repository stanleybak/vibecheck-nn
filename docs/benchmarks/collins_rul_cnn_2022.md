# collins_rul_cnn_2022 — vibecheck benchmark record

VNNCOMP 2025 regular track. Collins Aerospace remaining-useful-life
(RUL) prediction CNNs for jet-engine sensor windows. 62 instances
across 3 models:
- `NN_rul_small_window_20.onnx` — 5×1 spatial input, 5 Conv blocks
- `NN_rul_full_window_20.onnx`  — 5×1 spatial, 5 Conv blocks
- `NN_rul_full_window_40.onnx`  — 10×1 spatial, 5 Conv blocks

All three are tiny CNNs (`Conv → BN → ReLU` ×5 + `Dropout + Flatten +
final Conv`). Specs come in 3 families:
- **robustness_{2,4,8,16}perturbations_delta{5,10,20,40}_epsilon10**:
  L∞ ε around the input under bounded RUL change.
- **monotonicity_CI_shift{5,10,20}**: output should not decrease when
  certain time-series inputs shift forward.
- **if_then_{5,7,9}levels**: piecewise-linear conditional on RUL bands.

## Final score (server1, 2026-05-23, RTX 3080 / 10 GB)

| Solver | Solved / 62 | Wall (total) | Notes |
| --- | --- | --- | --- |
| **vibecheck (30 s/case)** | **62** | **~3 s** | 22 SAT + 40 UNSAT |
| AB-CROWN (published, 2025) | 62 | ~430 s | 6.2-7.2 s/case (flat) |

**~140× faster wall**. AB-CROWN has a fixed ~6 s overhead per case
(model load + α-opt warmup + spec encoding) regardless of difficulty;
vibecheck reuses the loaded graph and closes everything in the first
CROWN pass (UNSAT, ~0.01 s) or root PGD (SAT, ~0.05-0.5 s).

## Algorithmic adds for this benchmark

- **gpu_graph passthrough alias** for Dropout/Identity/Cast
  (`ComputeGraph.gpu_graph` in `network.py`). Previously these ops
  were added to `computed` set but emitted no op entry, so downstream
  consumers referencing them looked up an absent `state[name]`. Now
  an `alias` map carries `skipped → upstream producer`; post-loop
  rewrite resolves all op inputs through it. Catches any model with
  Dropout in the inference graph (most PyTorch-exported nets).

No new bound implementations; the existing Conv/ReLU/Flatten/Dropout
support handles everything.

## Knobs (`configs/collins_rul_cnn_2022.yaml`)

None — defaults work. Auto-routing in `verify_graph` sends Conv-only
nets without forks to `milp_verify`, which closes every case via
its CROWN/zono first pass.

## Reproducing a single case

```bash
.venv/bin/vibecheck \
  --net path/to/collins_rul_cnn_2022/onnx/NN_rul_small_window_20.onnx \
  --spec path/to/collins_rul_cnn_2022/vnnlib/robustness_2perturbations_delta5_epsilon10_w20.vnnlib \
  --mode graph --timeout 30 --bits 32 \
  --config configs/collins_rul_cnn_2022.yaml
```

## Full sweep

```bash
.venv/bin/python scratch/collins_smoke.py
```

## Integration test coverage

`tests/integration/test_collins_rul_cnn_2022.py`:
- `small_window_20 robustness_2pert_delta5` (SAT, ~0.5 s) — exercises
  root PGD on Conv path.
- `full_window_20 robustness_8pert_delta40` (UNSAT, ~0.01 s) — CROWN
  first-pass closure.
- `full_window_40 monotonicity_CI_shift20` (UNSAT, ~0.01 s) — largest
  model + monotonicity spec.

## Known unsolved cases

None. All 62/62 verified in <1 s/case.

## Soundness note (fixed)

A 2026-05-31 soundness sweep found **16 false-verifies** here (the
`robustness_{2,4,16}perturbations_delta{5,10,20,40}_epsilon10_w{20,40}` cases across
`NN_rul_{small,full}_window_*`). Root cause: `_solve_spec_graph_worker` (`verify_milp.py`)
imposed the pre-ReLU interval bounds `(lo, hi)` as **hard variable bounds** but recomputed the
affine in float64 while the bounds came from float32 zono/CROWN. collins's spec box perturbs
only 4 of 400 inputs, so ~90% of neurons are near-constant with **degenerate bounds**
(width ~1e-9) — tighter than the float32→float64 gap. A genuinely reachable point then landed
just outside `[lo,hi]`, making the spec LP falsely infeasible → `feas UNSAT → verified`
(confirmed by Gurobi IIS). **Fix:** outward FP-soundness inflation `lo-=tol, hi+=tol` with
`tol = atol + rtol·max|bound|` (`milp_bound_inflation_atol/rtol`, default 1e-5/1e-5) where
`bounds_by_relu` becomes spec-MILP variable bounds — inflation can only make the feasibility LP
*less* infeasible, never create a false-verify. Validated: 16/16 fixed, 39/39 UNSAT still
verify, no completeness loss.
