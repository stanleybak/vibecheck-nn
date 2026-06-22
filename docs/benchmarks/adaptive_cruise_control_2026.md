# adaptive_cruise_control_non_linear_2026

Extended-track, 50 instances, **v2** specs, 600s... (per-instance timeout 100s in
`instances.csv`). 2-input adaptive-cruise-control MLPs (`acc-2000000-64-64-64-64-…`,
ReLU, 1 output) with **NONLINEAR v2 specs**: the DNF atoms are polynomials in the
input `X` and the output `Y` up to degree 2 — `X*Y` coupling, `Y²`, plus a nonlinear
**input** constraint `200·X0 ≥ X1²`.

## How vibecheck handles it (network augmentation)

vibecheck verifies a net against a LINEAR spec, so each nonlinear-v2 instance is
**transpiled** (`nonlinear_augment.py`, wired at `main._maybe_nonlinear_augment`):
build an augmented ONNX that runs the original net `f` to get `Y`, computes every
distinct monomial of `(X, Y)` (via `Gather`/`Mul`), and emits each constraint
polynomial `p_c(X,Y) = W_c·v + b_c` as an extra output (`Gemm`). The spec becomes a
linear DNF over those outputs: `clause = AND of (p_c ≤ 0)`. A nonlinear input
constraint `g(X) {<,≤} 0` is folded in as another atom. Correctness is gated by an
onnxruntime oracle (augmented output == polynomial(X, f(X)) within 5e-3). The
augmented net's bilinear `Mul` routes it through the acopf/trig verifier
(`verify_graph._verify_nonlinear_graph`).

The emitted counterexample carries the **original** net's `X`/`Y` (the augmented
outputs are the constraint polynomials — wrong shape for the scorer);
`main._counterexample_sexpr_orig` recomputes `Y` from the original net in float32
CPU ORT (the scorer's arithmetic).

## Soundness: strict `<0` → non-strict `≤0` bound (NOT `≤ -margin`)

A strict atom `p_c < 0` is encoded as its **closure** `p_c ≤ 0` (a SUPERSET of the
true unsafe set), so proving it empty soundly proves `unsat`. Encoding it as
`≤ -MARGIN` (a strict SUBSET, missing the band `(-MARGIN, 0)`) would **false-unsat**
a shallow real CE — fixed here and in `network_pair` (iso/mono). The strict
semantics are enforced downstream by `_sat_disposition`: a CLEAR CE (margin
`≤ -atol`) commits `sat`; the measure-zero boundary (`p_c = 0`) validates only
within tolerance, so it is persisted as a within-tol sat while the search keeps
looking for a clear CE or an `unsat` proof. Pinned by
`tests/test_nonlinear_augment_strict.py`.

## CE search: GPU PGD, multi-α + per-disjunct, loop-until-clear

These are tolerance-BOUNDARY instances — the true f64 min of `p_c` is exactly 0
(verified by exhaustive 2-D grid), so a sound `unsat` is unavailable on the SAT
ones; the scorer runs the net in float32 where rounding gives a clear CE
(`p_c ≈ -2.4e-4`). The needle CE sits where the curved input-constraint boundary
meets the output boundary, across a 62-disjunct DNF. The default PGD (single
step-size + one joint loss over all 62 disjuncts) misses it even at 2000 restarts
(measured). What works:

- **`pgd_per_restart_disjunct`** — each restart descends ONE disjunct's AND-group
  (mirrors α,β-CROWN's per-OR-clause attack pools), so every disjunct gets dedicated
  gradient instead of a diffuse joint sum.
- **`pgd_alpha_multi`** — partition restarts across log-spaced step sizes.
- **budget-fill loop** (`pgd_sat_min_time`) — re-run fresh deterministic-seeded
  batches (`pgd_seed + loop_i`) and **keep looping past a within-tol hit until a
  CLEAR CE is found** (a fresh batch deepens a near-boundary hit past the tolerance).
- runs on **GPU** (the acopf path now follows `settings.device`; `_resolve_device`).

Tuned to a **10/10 hit rate across seeds 0..9** on the hard instances (3/5/8/33/40/
46/47), ~10–14s each. Config: `configs/adaptive_cruise_control_non_linear_2026.yaml`.

## Results vs α,β-CROWN

Clean A10G sweep (official 100s timeout): **47/50 — 17 sat / 30 unsat / 3 timeout**,
**matching ABC's 47/50**. All sat are clear or within-tol CEs (ORT-validated, sound);
all unsat are bound proofs. Timeouts: instances 27, 34, 42 (neither a CE nor an
unsat proof within 100s). Two sat run to ~100s (within-tol, kept searching) — verdict
correct, ~3% over the deadline (scorer-accepted).

## Repro

```
# one instance (auto-augments + GPU PGD)
.venv/bin/python -m vibecheck.main \
  --net <bench>/2.0/onnx/acc-…-0.9.onnx \
  --spec <bench>/2.0/vnnlib/instance_3.vnnlib \
  --config configs/adaptive_cruise_control_non_linear_2026.yaml \
  --device gpu --bits 32 --timeout 100 --results-file out.txt
# seed-0..9 tuning harness: scratch/adaptive_seed_tune.py gpu 512 200 1 1 14
```

Integration pins: `tests/integration/test_adaptive_cruise_control_2026.py`
(instances 3/8/33, GPU, seed 0).

## Known unresolved

- 3 timeouts (27/34/42) — neither attack nor proof closes them in 100s.
- The trig UNSAT input-split BaB can overshoot the deadline on a degenerate
  synthetic (pre-existing; real instances respect it within ~3%).
