# cersyve — vibecheck benchmark record

VNNCOMP 2025 regular track. Small (2-D/4-D input) MLP/ResNet controllers
(pendulum, lane_keep, point_mass) × pretrain/finetune × con/inv. 12
instances total. Every instance has a single AND-conjunct unsafe
condition with 2 threshold constraints on different outputs (e.g.
`Y_0 ≤ 0 AND Y_1 ≥ 0`).

## Final score (local, 2026-05-20, RTX 3080 / 10 GB)

| Solver | Solved / 12 | Notes |
| --- | --- | --- |
| **vibecheck (60 s/case)** | **12** | 6 SAT + 6 UNSAT — match AB-CROWN's set |
| AB-CROWN (published) | 12 | Each in 6-9 s; vibecheck per-case time 0.1-17 s |

The hardest 4-D UNSAT cases (`lane_keep_finetune_inv`,
`point_mass_finetune_inv`) take ~15-17 s for us vs ~8 s for AB-CROWN —
gap is per-leaf throughput (no GPU batching of the per-halfspace clip),
not algorithmic. All other cases finish in <1 s.

## Algorithmic wins vs published AB-CROWN

- **Batched input-split BaB with domain clipping** (`_input_split_batched`
  in `verify_graph.py`). The closer of the gap:
  - Worklist-based BaB: pop up to `input_split_batch_size` (4096) boxes
    per iteration, stack into `(B, n_in)` tensors, run ONE batched
    forward zonotope + spec backward CROWN. ~33 K leaves/sec
    throughput (vs ~30 leaves/sec sequential — 1000× speedup).
  - **Domain clipping** after CROWN backward: use the per-query input-
    space linearization `spec_q(x) ≥ A_q·x + b_q` to shrink each leaf
    to the bounding box of the polytope `∩_q {A_q·x + b_q ≤ 0}`. If
    the polytope is empty, the leaf is verified directly. Cheap per-
    halfspace approximation (per-dim projection) — `O(B × n_in ×
    n_queries)` vector ops per iteration. Closes the 4-D
    `_finetune_inv` boundary cases that pure per-query CROWN can't.
    Mirrors AB-CROWN's `clip_input_domain: complete`.

- Per-leaf JOINT-AND infeasibility LPs (`_joint_and_infeasible_in_box`,
  `_joint_and_infeasible_zono`, `_joint_and_infeasible_triangle_lp`).
  Used by the sequential path (still in tree); kept because they
  catch some 2-D UNSAT cases faster than batched without help.

## Pre-merge bug audit (uncovered during this benchmark)

Cersyve forced exercise of code paths that had silent bugs masked by
cifar/tinyimagenet's pairwise-only constraints. All five are fixed:

1. **`Conjunct.margin` used `min` over constraints** — wrong AND
   semantics. A conjunct is safe iff ANY constraint is provably
   violated, not all. Caused spurious "verified" verdicts on multi-
   constraint conjuncts. Fixed to `max`. (silent on pairwise:
   `min(single) == max(single)`)
2. **PGD threshold-constraint margins had FLIPPED SIGN**. Pairwise
   margins were correct, so cifar/tinyimagenet were unaffected. On
   cersyve all 6 SAT cases returned `unknown` (PGD gradient-descended
   AWAY from witnesses that brute-force sampling finds in <100k
   samples). Fixed in `_build_constraint_matrices`. Test:
   `tests/test_pgd_margin_signs.py`.
3. **Pre-cascade PGD hook `spec_lbs` index** was keyed by disjunct id,
   not query index. For multi-query disjuncts (cersyve: 2 per disj)
   one query silently overwrote the other under the shared disjunct
   key; downstream sort produced garbage order. Refactored into
   `_zono_spec_lbs_and_open_qis` helper. Test:
   `tests/test_zono_spec_lbs_indexing.py`.
4. **ORT-based SAT validator** rejected EVERY PGD witness on cersyve
   as "spurious" because `ort.InferenceSession(...)` can't read the
   shipped `.onnx.gz` files. Fixed via in-memory gzip decompress
   before session load.
5. **CROWN-input-space spec bias sign** in
   `_zono_spec_lbs_and_open_qis` used `wc - bq` instead of `wc + bq`.
   Silent on pairwise (bias=0) and on cersyve pendulum (threshold
   value=0), but would have produced wrong-signed lbs on any
   threshold spec with `value ≠ 0`. Caught + fixed; test in
   `tests/test_zono_spec_lbs_indexing.py`.

## Knobs (`configs/cersyve.yaml`)

- `input_split_batched_enabled: true` — switches input-split dispatch
  from sequential per-leaf processing to the new worklist + batched-
  bound driver. The throughput jump is what makes the 4-D UNSAT cases
  tractable.
- `input_split_batched_clip_enabled: true` — domain clipping inside
  the batched BaB. **The actual game-changer**: without it we top out
  at 10/12; with it we get 12/12 within budget.
- `input_split_batch_size: 4096` — leaves per batched CROWN call.
  Tuned for the 10 GB GPU; the largest intermediate (`(B, n_out,
  K_max)`) is ~70 MB at 4096 batch, ~280 MB at 16384. Higher values
  would help for the few iterations that exceed the worklist, but
  4096 already hits diminishing returns on these tiny networks.

## Reproducing a single case

```bash
.venv/bin/vibecheck \
  --net path/to/cersyve/onnx/point_mass_finetune_inv.onnx.gz \
  --spec path/to/cersyve/vnnlib/prop_point_mass.vnnlib.gz \
  --mode graph --timeout 60 --bits 32 \
  --config configs/cersyve.yaml
```

## Full sweep

```bash
.venv/bin/python tests/sweep_cersyve.py 60   # 60 s/case
```

## Integration test coverage

`tests/integration/test_cersyve.py`:
- `pendulum_pretrain_inv` (SAT, ~0.3 s) — exercises PGD on AND-
  conjunct (regression for the PGD sign-flip + conjunct margin bugs).
- `point_mass_finetune_inv` (UNSAT, ~17 s, 4-D input) — regression
  for batched + clipping (was unverified at 600 s without it).
- `lane_keep_finetune_inv` (UNSAT, ~15 s, 4-D input) — same path,
  second case.

## Known unsolved cases

None at 60 s/case. AB-CROWN is still ~2× faster on the hardest two
cases — could be closed by (a) batching the per-halfspace clip itself
on GPU instead of CPU-side per-leaf loop, (b) smart-branching axis
selection (SB heuristic) instead of widest-axis, (c) tightening the
clipping via LP-per-leaf instead of per-halfspace approximation. Not
priorities given 12/12 already.
