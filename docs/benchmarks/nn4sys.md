# nn4sys — vibecheck benchmark record (partial)

VNNCOMP 2025 regular track. 194 instances across 12 ML-for-systems
models: index learning (lindex, lindex_deep), cardinality estimation
(mscn_128d/2048d, with dual variants), video bitrate selection
(pensieve_small/mid/big × simple/parallel). All 194 ABC verdicts are
UNSAT (the benchmark is a stress test for soundness of learned
database/networking predictors).

## Final score (server1, 2026-05-24, RTX 3080 / 10 GB, 30 s budget)

| Solver | Solved / 194 | Wall (total) | Notes |
| --- | --- | --- | --- |
| **vibecheck** | **46** | **~1407 s** | 24 lindex + 22 pensieve_simple |
| AB-CROWN (published, 2025) | 194 | ~2559 s | all UNSAT |

**0 WRONG** (was 22 pre-fix — see soundness fixes below).

## Two soundness bugs caught here (fixed in main, not nn4sys-specific)

1. **milp_verify Gurobi `FeasibilityTol=1e-6` vs float32-zono ulp gap**
   (`verify_milp.py:_solve_spec_worker`). Float32 zono forward produces
   active-neuron bounds tight to ~1 ulp; the float64 LP arithmetic
   computes `expr + b_j` 1-3 ulp outside those bounds → Gurobi
   reports INFEASIBLE → caller returns `verified` on a real SAT
   case. Caught on metaroom 4cnn_ry_99_16 / spec_43 (1/10 runs
   `verified` on ABC-witnessed SAT). Fix: `setParam('FeasibilityTol',
   1e-5)` — the gap is at the precision floor, no algorithmic
   loosening. Regression test in `tests/test_milp.py`.

2. **vnnlib parser dropped per-disjunct X constraints + overwrote X
   bounds across (and ...) blocks** (`vnnlib_loader.py:_parse_or_and`).
   Mixed X/Y conjuncts like
   `(and (>= X_0 a) (<= X_0 b) (<= Y_0 c))` had:
     - X bounds overwritten across blocks → global x_lo/x_hi was
       the LAST block's range, not the UNION
     - X constraints stripped from Conjunct, leaving Y-only — so
       a PGD witness with X outside the conjunct's subrange but Y
       satisfying the Y-constraint was falsely flagged as SAT.
   Caught on nn4sys lindex_10000 (22 false-SAT verdicts). Fix:
   per-block X bounds → UNION for global x_lo/x_hi + store
   per-disjunct subbox in `Conjunct.input_lo`/`input_hi`;
   `VNNSpec.check_witness(x, y)` evaluates the full X-AND-Y unsafe
   region. 4 regression tests in `tests/test_spec.py`.

## Algorithmic adds for this benchmark

- **Per-disjunct sub-verification** (`verify_graph.py:
  _verify_per_disjunct_subboxes`): when conjuncts carry X subboxes,
  decompose into one sub-verification per unique subbox, with
  batched-CROWN fast path for many small subs. Lindex_10000 (10000
  unique X subboxes, each 1-input ReLU MLP) verifies in 0.49 s via
  batched zono forward + per-sub bounds-check.
- **Gather op support** (`network.py:gg-ops`, dispatches reuse Slice
  handlers): unlocks pensieve_simple models. Gather is semantically a
  flat-index selection — same handler shape as Slice.

## Knobs (`configs/nn4sys.yaml`)

No tuning; defaults verify all currently-supported model types.

## Known unsolved

- **83 pensieve_*_parallel cases**: ONNX model expects input shape
  `[12, 8]` but vibecheck's onnx_loader picks `[1, 8]` (drops the
  first non-batch dim incorrectly). Reshape semantics for input dim 0
  also need work. Separate workstream from this benchmark.
- **65 mscn_* cases**: models use Div + ReduceSum + Sigmoid (latter
  supported); Div and ReduceSum dispatchers not implemented for
  zonotope forward. Separate op-support workstream.

## Reproducing

```bash
.venv/bin/vibecheck \
  --net path/to/nn4sys/onnx/lindex.onnx \
  --spec path/to/nn4sys/vnnlib/lindex_10000.vnnlib \
  --mode graph --timeout 30 --bits 32
```

## Integration test coverage

`tests/integration/test_nn4sys.py`:
- `lindex / lindex_10000` (UNSAT, ~0.5 s) — per-disjunct fast batched
  CROWN over 10000 subboxes. Regression for both soundness fixes
  (parser unsoundness was fired by this case).
- `pensieve_small_simple / pensieve_simple_0` (UNSAT, ~2 s) — Gather
  op support + per-disjunct.
