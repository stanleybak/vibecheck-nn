# Soundness + completeness sweep — 2026-05-31

Full subprocess-isolated sweep of the 12 completed regular-track benchmarks
(tinyimagenet excluded — its timeout fix is WIP) on the AWS g5 (A10G 24 GB).
Each case is its own `vibecheck.main` run; the verdict is read from
`--results-file` only. Driver: `scripts/aws_overnight_sweep.py`. Raw results:
AWS `~/persistent_runs/aws_sweep/results.csv` (also pulled to
`/tmp/aws_sweep_results.csv`).

Three phases:
- **A — soundness**: every case AB-CROWN labels `sat`, run with
  `--disable-sat-finding`. A `verified` (unsat) here means the bounds/MILP
  proved unsat on a case that *has* a counterexample → **unsound**.
- **B — completeness**: every case AB-CROWN labels `unsat`, full competition
  timeout. Did vibecheck also verify within budget? (vs ABC.)
- **C — stretch**: cases ABC did NOT solve. (Not reached this run.)

## Phase A — soundness (COMPLETE, 927 cases) — **BOTH ROOT CAUSES FIXED**

**17 false-verifies, confined to two benchmarks. The other 10 are clean.**

| benchmark | false-verify (unsound) | status |
| --- | --- | --- |
| collins_rul_cnn_2022 | **16** | **FIXED** (16/16 now `unknown` w/ PGD off; 39/39 UNSAT still verify) |
| dist_shift_2023 | **1** (index4312) | **FIXED** (7/7 Phase-A now sound on AWS) |
| acasxu, cersyve, cgan, cifar100, cora, linearizenn, malbeware, metaroom, nn4sys, safenlp | 0 | clean |

Both turned out to be **two distinct bugs in two different pipelines** — the
original "same gen-LP class" hypothesis was wrong (verified by isolating each).

### Root cause 1 — collins_rul (verify_milp.py spec MILP)

`_solve_spec_graph_worker` imposes the pre-ReLU interval bounds `(lo, hi)` as
**hard variable bounds**, but recomputes the affine in float64 while the bounds
come from float32 zono/CROWN. collins's spec box perturbs only 4 of 400 inputs,
so ~90 % of neurons are near-constant and their bounds are **degenerate
(width ~1e-9)** — *tighter than the float32→float64 gap*. A genuinely reachable
point then lands just outside `[lo,hi]` → the spec LP is falsely infeasible →
`feas UNSAT → verified`. Confirmed by Gurobi IIS: the conflict is active-ReLU
variable upper bounds vs. the affine row, and inflating the bounds by ≥1e-4
flips it to feasible (correct). **Fix:** outward FP-soundness inflation
`lo-=tol, hi+=tol`, `tol = atol + rtol·max|bound|` (`milp_bound_inflation_atol/
rtol`, default 1e-5/1e-5), applied where `bounds_by_relu` becomes spec-MILP
variable bounds. Inflation only ever makes the feasibility LP *less* infeasible
— it can never create a false-verify. Validated: 16/16 fixed, 39/39 UNSAT still
verify, no completeness loss.

### Root cause 2 — dist_shift index4312 (verify_graph.py α-zono fallback)

`_build_alpha_zono_lp` defaulted every z_alpha generator column to **[0,0]** and
only opened input symbols + this query's *listed* unstable `e_new` columns to
[-1,1]. But `state_from_alpha_zono` **reserves** `e_new` columns for unstable
neurons it skips (no pre-ReLU snapshot — mnist_concat's encoder/generator ReLUs,
~740 of them), which never enter `unstable_list`. Those "orphan" columns carry
real objective weight (Σ|coef|≈0.17); fixing them at 0 collapses each neuron's
parallelogram `λ·z + μ·(1+e_new)` to its **center line** — an unsound ReLU
enclosure that excludes reachable outputs. The bin-0 LP then disagreed with
α-CROWN's spec LB (−1.19 vs −1.36), and binarising the classifier ReLUs cut off
a real CEX → ObjBound +0.044 ≥ tol → false `verified` (only on AWS, where the
GPU-float-dependent scoring picked 58 vs 56 bins; the relaxation was unsound
*everywhere* — local stayed negative at −0.056 by luck). The CEX has true margin
−0.249; a sound relaxation min must be ≤ that. **Fix:** every generator column
is a noise symbol ∈ [-1,1] (zero columns contribute nothing) — open them all.
Now bin-0 LP = −1.36 (matches α-CROWN) and fallback lb = −0.40 ≤ −0.25 (sound).
This is the **same `[0,0]`-padding class** already fixed in `_build_phase1_lp`
and the dense/sparse builder; `_build_alpha_zono_lp` was the missed third
builder. Validated on AWS: 5/5 runs `unknown`.

**These are LATENT**: PGD finds the CEX in production (verdict `sat`), so the
competition verdicts are still correct — but the bounds themselves are unsound,
and would false-verify if PGD ever missed a CEX.

The 16 collins_rul cases are `robustness_{2,4,16}perturbations_delta{5,10,20,40}_
epsilon10_w{20,40}` across the `NN_rul_{small,full}_window_*` nets.

## Phase B — completeness (1253 / 1291, vs ABC)

| benchmark | vc verified / ABC unsat |
| --- | --- |
| safenlp_2024 | 433/433 |
| nn4sys | 156/156 (partial — 39 not reached) |
| malbeware | 131/131 |
| cifar100_2024 | 100/100 |
| metaroom_2023 | 94/94 |
| linearizenn_2024 | 59/59 |
| collins_rul_cnn_2022 | 39/39 |
| cora_2024 | 22/22 |
| cgan_2023 | 9/9 |
| cersyve | 6/6 |
| **acasxu_2023** | **136/139** |
| **dist_shift_2023** | **56/65** |

vibecheck matches ABC's completeness on **10 of 12 benchmarks (100%)**. Gaps:
- acasxu: 3 — `prop_2`, `prop_3` (`unknown`), and `prop_6` → **vc=`sat` while
  ABC=`unsat`** (a disagreement: vibecheck false-sat, or ABC false-unsat — needs
  a witness check).
- dist_shift: 9 (`index7027, 9733, 9154, 5291, 2481, 2750, 112, 3805, 6107`),
  all `unknown` — likely fallout of the dist_shift soundness fix making cases
  fail-loud/unknown rather than (un)soundly verify.

## What remains to measure

1. **Re-run the full Phase A** after the soundness fixes → confirm all 17 come
   back non-`verified`.
2. **Phase B tail**: nn4sys 39 cases not reached; re-run dist_shift's 9 misses
   after the soundness fix (they may flip).
3. **Phase C** (99 ABC-unsolved): can vibecheck prove any?
4. **acasxu prop_6 disagreement**: extract a witness, decide vc-false-sat vs
   abc-false-unsat.
5. tinyimagenet completeness (Phase 1+2 ~50s starves the dual-ascent) — separate
   from the timeout fix already landed.

## AWS env notes

The g5 venv was missing **PyYAML** and **onnxruntime** (each silently turned
every `--config` case into `unknown`); installed. A real `onnx_forward` bug
(input registered under an initializer name for acasxu) is fixed in this branch.
Benchmark `.gz` files are auto-gunzipped by the sweep.
