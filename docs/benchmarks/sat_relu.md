# sat_relu

VNNCOMP benchmark: 100 random 3-SAT instances compiled into integer-weighted
ReLU networks. Each net is `n_vars → Gemm → ReLU → Gemm → 2` (n_vars ∈ 6..100,
H ∈ ~30..400 ReLUs). The unsafe spec is a **conjunction** `Y_0 >= 1 AND Y_1 <= 0`
that is reachable iff the encoded SAT formula is satisfiable. `sat_*` instances
are SAT (a witness exists → verdict `sat`); `unsat_*` are UNSAT (verified).

## Score

- **vibecheck: 100/100** (50 sat + 50 unsat), ~0.2 s per MILP, whole sweep < ~7 min.
- AB-CROWN published: 100/100 (~6 s/case).
- (vc sweep id: `scratch/sweep_sat_relu.sh`, 2026-05-31; ABC ref
  `~/repositories/vnncomp2025_results/alpha_beta_crown/2025_sat_relu/results.csv`.)

## How vibecheck solves it

- **`sat_*`** — PGD root attack finds the satisfying assignment (~1 s). The
  conjunction is handled jointly by PGD, no MILP needed.
- **`unsat_*`** — the exact per-neuron MILP proves the joint unsafe region
  empty. Four pieces, all on `bench/sat_relu`:

  1. **Conjunctive-disjunct feasibility** (`verify_milp._milp_verify_graph`).
     A disjunct with >1 conjunct is unsafe only where ALL its halfspaces hold
     *simultaneously* (their intersection). The per-query escalation could only
     verify it if some single halfspace were empty; here each halfspace alone
     is reachable but the intersection is empty. Fixed by a per-disjunct
     feasibility MILP that adds **every** conjunct as a constraint and proves
     it infeasible (`_solve_spec_graph_worker` feasibility mode now accepts a
     2-D `query_w`). This is a **general completeness bug**, not sat_relu-only.

  2. **Racing premature-termination fix** (`_racing_escalation_graph`). feas
     (joint MILP infeasible → safe) and opt (ObjBound>0 → safe) race per bin
     count. The opt worker minimises only the FIRST halfspace's margin, which
     for a conjunction is reachable → "escalate"; the old code terminated the
     pool the instant opt said escalate, **killing the still-running feas
     worker** about to prove the joint MILP infeasible. Now it returns verified
     as soon as either proves safe and escalates only once BOTH have finished.
     Also general.

  3. **Classification-preserving inflation clamp** (`_inflate_milp_bounds`).
     The FP bound inflation (collins_rul soundness fix) widened `lo` below 0 for
     always-active neurons; on these integer nets ~all active neurons have
     `lo == 0` exactly, so every one was reclassified unstable and binarised
     (252 → 317 degenerate binaries). The inflation now clamps so an active
     (lo≥0) neuron keeps lo≥0 and a dead (hi≤0) neuron keeps hi≤0 (still a sound
     over-approximation).

  4. **`configs/sat_relu.yaml`** (see below).

## Config knobs (`configs/sat_relu.yaml`) and why

- `milp_bound_inflation_atol: 0` / `milp_bound_inflation_rtol: 0` — the nets are
  **integer-weighted**, so pre-ReLU bounds are exact integers and the
  float32→float64 gap is zero; the inflation is unnecessary *and harmful* here:
  non-integer big-M coefficients (e.g. 4.00001) defeat Gurobi's integer-aware
  presolve/cuts, turning a 0.1 s infeasibility proof into a 90 s timeout.
  Disabling it is sound (the exact integer var bounds contain the reachable
  range exactly).
- `input_split_max_dims: 0` — route **every** case to the exact MILP. Small
  instances (≤20 inputs) otherwise take the input-split graph pipeline, which
  cannot prove the conjunctive unsafe region and times out. sat_relu gains
  nothing from input-splitting.

## Reproduce

```bash
B=$VNNCOMP/benchmarks/sat_relu
# single case
.venv/bin/python -m vibecheck.main --net $B/onnx/unsat_v100_c373.onnx \
  --spec $B/vnnlib/unsat_v100_c373.vnnlib --config configs/sat_relu.yaml \
  --timeout 60 --results-file /tmp/r.txt   # -> unsat
# full sweep (verdict from --results-file; expected from filename prefix)
bash scratch/sweep_sat_relu.sh configs/sat_relu.yaml 60
```

## Integration tests (`tests/integration/test_sat_relu.py`)

- `sat_v30_c38` (SAT, PGD root attack).
- `unsat_v30_c38` (UNSAT, small conjunctive MILP).
- `unsat_v100_c373` (UNSAT, large — exercises the integer-bound + routing fixes).
- `sat_v30_c38` SOUNDNESS probe (`disable_sat_finding=True` → must be `unknown`,
  never `verified` — guards the joint-feasibility encoding's soundness).

## Known unsolved

None at the competition timeout.
