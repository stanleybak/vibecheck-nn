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

## Phase A — soundness (COMPLETE, 927 cases)

**17 false-verifies, confined to two benchmarks. The other 10 are clean.**

| benchmark | false-verify (unsound) |
| --- | --- |
| collins_rul_cnn_2022 | **16** |
| dist_shift_2023 | **1** (index4312) |
| acasxu, cersyve, cgan, cifar100, cora, linearizenn, malbeware, metaroom, nn4sys, safenlp | 0 |

**Root cause (collins_rul, confirmed by reproducing a PGD CEX):** the gen-LP
**LP relaxation (`Racing bins=0`) falsely proves infeasible** despite a real
counterexample — Phase 1 CROWN leaves worst≈−21, then bins=0 closes it as
`feas UNSAT → verified`. The LP relaxation is supposed to be a *sound
over-approximation* (it must contain every reachable output, incl. the CEX); it
excludes the CEX, so the gen-LP construction is unsound for this net. **Same
class as the dist_shift gen-LP bug** fixed earlier for index1285 (commit
93699b0); index4312 routes through a still-unsound path, so that fix is
incomplete.

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
