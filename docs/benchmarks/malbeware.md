# malbeware — vibecheck benchmark record

## UPDATE 2026-06-02 — re-swept on AWS A10G, 150/150 vs ABC (regression stays closed)

Re-confirmed the Phase-8 racing regression (fixed by `7da043b`, defaults
`phase8_race_all_bins` + `phase8_high_bin_bestbdstop`) is closed on current main.
Full 150-case sweep on the A10G (config `malbeware.yaml`, verdict from
`--results-file`): **150/150 match ABC — 19 sat + 131 unsat, 0 unsound /
0 mismatch / 0 incomplete.** The previously-regressed `4-25` eps-3 cases verify
in ~2.5 s (warm GPU); the all-neurons `bins=4917` racer wins with a `BestBdStop`
margin certificate `lb > 0`.

**Sweep-harness lesson (re-learned):** malbeware reuses each vnnlib basename
across all 3 onnx models, and ABC gives **different verdicts per model** (e.g.
`Alueron.gen!J idx-21`: linear-25→unsat, 4-25→**sat**, 16-25→unsat). A first pass
that matched ABC by vnnlib *basename* produced two spurious `UNSOUND-sat` flags;
matching by the **(onnx, vnnlib) pair** clears them — vc's `sat` verdicts are
onnxruntime-validated witnesses (`_validate_sat_witness`, no false sat) and agree
with ABC's per-model verdict. See [[project_audit_abc_key_collision]].

VNNCOMP 2025 regular track. 150 instances across 3 malware-image
classifier models (64×64 grayscale → 25 classes):

- `malware_malimg_family_scaled_linear-25.onnx` — pure linear (Flatten
  + Gemm), no ReLU. 50 instances.
- `malware_malimg_family_scaled_4-25.onnx` — 1 Conv (4 filters) +
  ReLU + Flatten + Gemm. 50 instances.
- `malware_malimg_family_scaled_16-25.onnx` — 1 Conv (16 filters) +
  ReLU + Flatten + Gemm. 50 instances.

Specs are pairwise-AND disjuncts over the 24 non-true classes
(`Y_i ≥ Y_true` unsafe).

## Final score (server1, 2026-05-24, RTX 3080 / 10 GB, 30 s budget)

| Solver | Solved / 150 | Wall (total) | Notes |
| --- | --- | --- | --- |
| **vibecheck** | **150** | **~66 s** | 19 SAT + 131 UNSAT |
| AB-CROWN (published, 2025) | 150 | ~1026 s | all verified / sat |

**0 misses.** Wall: 66 s vs AB-CROWN's 1026 s — **~15× faster** at full
coverage.

## Algorithmic adds for this benchmark

- **Pure-linear (no ReLU) fast path** in `_phase1_bab_refine`
  (`verify_graph.py`) — bail early when `bounds_by_relu` is empty,
  since there's nothing to BaB-tighten on an affine net. Pre-fix this
  crashed with `ValueError: max() arg is an empty sequence` on all
  50 linear-25 cases.
- **Match AB-CROWN's full-MIP fallback**: ABC's malbeware config uses
  `complete_verifier: mip` (single full-MIP solve on α-CROWN-tightened
  bounds; no β-CROWN). Our Phase 8 fallback defaulted to 200 binarized
  neurons, plateauing at LB ≈ {-0.59, -1.19} on the 4-25 eps-3 cases.
  Raising `phase8_high_bin_count` to encode ALL unstable neurons
  (~5k-8k binaries) lets the fallback return INFEASIBLE → verified in
  ~5-18 s, matching ABC's wall.

## Knobs (`configs/malbeware.yaml`)

- `auto_route_milp_for_conv: false` — the historical `milp_verify`
  pipeline (auto-selected for conv nets > 20 input dims) racing-
  escalates Gurobi bins without converging on these eps-3 cases;
  the graph pipeline (α-CROWN + spec MILP) closes everything.
- `input_split_enabled: false` — 4096-input image is too high-dim
  for input split BaB; disabled to keep the graph pipeline as the
  default routing path.
- `phase8_high_bin_count: all` / `phase8_high_bin_time_limit: 25.0`
  — full-unstable MILP encoding (the `all` sentinel = every unstable
  neuron; default 200 is far too small for these cases) used by the
  post-racing fallback backup. The 4-25 eps-3 cases need ALL ~4.9k
  unstable neurons binarized to prove the margin bound > 0.

## Regression + fix (2026-05-30)

A later merge (`c521691`, the Phase-8 rewrite) regressed 4-25 from
~6 s → 30 s timeout. Cause (measured): the parallel racing escalates
small bin counts `[8..40]` and **waits on its slowest level** since
none close (`bins=40` ran 48 s on the 4917-unstable case), consuming
the whole budget before the all-neurons fallback could run. Two
default-on fixes (no malbeware-specific flag needed beyond
`high_bin_count: all`):

- **`phase8_race_all_bins: true`** — queues an all-neurons **cuts-ON**
  task at the FRONT of the racing pool, racing it concurrently from
  t=0. It wins the race on these cases (`bins=4917: UNSAT lb=+0.0137,
  2.2 s`) while the cheap small bins race alongside for easy cases.
- **`phase8_high_bin_bestbdstop: true`** — the high-bin proof minimizes
  the spec margin and stops via `BestBdStop` once its lower bound
  exceeds tol > 0, yielding an explicit margin certificate instead of
  an opaque Gurobi INFEASIBLE (more robust to numeric fragility).

Soundness re-confirmed: `BestBdStop` proves `min(qw·y+qb) ≥ +0.015 > 0`
in 1.8 s, and AB-CROWN independently returns `unsat` (16 s) on 4-25.

## Reproducing a single case

```bash
.venv/bin/vibecheck \
  --net path/to/malbeware/onnx/malware_malimg_family_scaled_16-25.onnx \
  --spec path/to/malbeware/vnnlib/malbeware_family-Allaple.A_label-2_eps-3_idx-11.vnnlib \
  --mode graph --timeout 30 --bits 32 \
  --config configs/malbeware.yaml
```

## Full sweep

```bash
.venv/bin/python scratch/malbeware_smoke.py
```

## Integration test coverage

`tests/integration/test_malbeware.py`:
- `linear-25 Adialer.C idx-0` (UNSAT, ~0.5 s) — regression for the
  pure-linear no-ReLU fast path.
- `16-25 Allaple.A idx-11` (SAT, ~3 s) — root-PGD finds the witness.
- `16-25 Wintrim.BX idx-119` (SAT, ~3 s) — root-PGD finds the witness.

## Known unsolved cases

None.
