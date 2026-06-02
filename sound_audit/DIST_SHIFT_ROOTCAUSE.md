# dist_shift unsoundness — isolated root cause (2026-05-30)

Method: validated CEX for index1285 (vibecheck PGD; onnxruntime confirms Y_9-Y_5=+0.79
→ real SAT). Measured, not speculated, at every stage:

1. INPUT box: correct. spec.x_lo/x_hi match the vnnlib; CEX inside box on all 792 dims. ✓
2. Per-neuron PRE-activation bounds (bounds_by_relu, L0 relu / L1 sigmoid / L2 / L3):
   ALL contain the CEX value at every neuron (stable+unstable). SOUND. ✓
3. Base forward-zono OUTPUT (832 gens): range of (Y_5-Y_9)=[-978,1184] contains the
   CEX margin -0.79. SOUND. ✓
4. Verdict 'verified' comes from phase = **spec_milp** (gen-LP/MILP infeasibility),
   spec_lbs={0: 1.0} (1.0 = "MILP proved INFEASIBLE" sentinel).
5. alpha-zono OUTPUT zonotope (obj_G_out, 812 gens): range of (Y_5-Y_9)=[-7.23,23.61]
   contains -0.79. SOUND.
6. ROOT CAUSE — gen-LP column misalignment in the alpha_zono spec MILP:
   - precompute_alpha_zono_state advances `cur_n_gens` by `n_cells` (784) for the
     sigmoid layer (verify_gen_lp.py ~line 871), assuming z_alpha keeps one γ column
     per sigmoid cell. But z_alpha is COMPRESSED — its real column count n_gens=812,
     while the gen-LP's model reaches cur_n_gens=**1596** (overshoot = 784 = the sigmoid
     n_cells). The "tolerate mismatch" comment (line 959) masks this.
   - Consequence: every `e_new_col` index is computed against the WRONG (assumed) layout.
     L0's ReLU constraints get `e_new_col` 792-797, but in the real obj_G_out cols
     792-796 are ZERO (1/6 nonzero) — NOT L0's relu-slack columns. And cols 0-783
     (fixed-image input dims, should be zero generators) are 763/784 NONZERO. So the
     assumed layout `[input 0..791][L0 slack]...` does not match z_alpha's actual layout.
   - The ReLU triangle/binary constraints are therefore bound to the wrong generator
     columns → over-restrict the LP feasible region → false INFEASIBLE → false 'verified'.

WHY dist_shift specifically: it's the only regular-track sigmoid benchmark with SAT cases;
the misalignment is triggered by the sigmoid's compressed-generator layout. cgan/nn4sys
sigmoid nets are all-UNSAT (no wrong verdict possible).

FIX OPTIONS:
 (A) Make the gen-LP derive e_new_col from z_alpha's ACTUAL generator layout (track the
     real column index each relu/sigmoid slack occupies through alpha-CROWN compression),
     instead of assuming n_cells per sigmoid. Correct + keeps verification power.
 (B) Conservative/safe: when cur_n_gens != n_gens (layout inconsistent), REFUSE to conclude
     'verified' from the alpha_zono MILP (return unknown / fall back to a sound method).
     Turns silent unsoundness into a sound 'unknown'. dist_shift would then need a
     different sound path to actually verify its UNSAT cases.
 (C) Route nets containing sigmoid/tanh away from the alpha_zono gen-LP infeasibility
     path entirely until (A) lands.

## FIX IMPLEMENTED (B, safe-first via LOUD RAISE — per user: raise, don't mask)

Per the user's direction ("raise errors if sizes are not as expected, not
silently ignoring or masking"), fix B is implemented as loud raises, not a
silent fallback:

- `verify_gen_lp.state_from_phase1`: raise NotImplementedError when
  `n_gens != n_input + n_unstable_relu` (the documented column-model invariant).
  The mismatch == unaccounted sigmoid/tanh γ columns (index1285: n_gens=804 vs
  n_input(8)+unstable(26)=34, i.e. 770 sigmoid columns) → ReLU e_new_col indices
  would be misaligned → could certify a false UNSAT. Raise instead.
- `verify_gen_lp.state_from_alpha_zono`: raise NotImplementedError when
  `cur_n_gens != n_gens` (same class, compressed-z_alpha positional model).
- `verify_graph._per_neuron_adaptive_bounds` (x2, the bab_refine 1380/2045
  asserts): bare `assert target_op_name is not None` → descriptive
  NotImplementedError explaining the ReLU-adaptive path was invoked on a
  non-ReLU (sigmoid) layer.

Validation:
- tests/test_dist_shift_soundness.py (the SAT case index1285, sat-finding off):
  was returning a false 'verified'; now the gen-LP path raises loudly → test
  passes (no false verified). Pins the soundness invariant.
- No false positives on pure-ReLU gen-LP nets: acasxu + nn4sys integration
  (12 cases) pass; the invariant n_gens == n_input + n_unstable holds for them.
- nn4sys mscn (sigmoid) verifies WITHOUT raising — it uses a different path, so
  no regression.

CONSEQUENCE: dist_shift now FAILS LOUD (sound) on its sigmoid nets instead of
either crashing with a bare assert or (on the adapt-off path) emitting a false
'verified'. It still does NOT verify its UNSAT cases — that needs fix A (track
the real sigmoid/tanh generator-column indices through alpha-CROWN compression
so e_new_col is always correct). The dist_shift integration tests stay red until
A; that is the honest "cannot soundly verify yet" state.

## MEASURED root cause (after user insisted: measure, don't speculate)

The earlier "column misalignment / e_new_col ignores sigmoid" hypothesis was
WRONG (disproved by a toy + direct measurement). Measured on index1285 with the
validated CEX, via LP feasibility + Gurobi IIS on the dumped gen-LP state:

- bare output zonotope G: SOUND — `c + G·e = CEX_output` is LP-feasible.
- ReLU bounds lo/hi: SOUND — all IIS neurons' CEX pre-activations within [lo,hi].
- ReLU TRIANGLE constraints (y>=0, y>=z): CONSISTENT (LP output=CEX+triangles feasible).
- ReLU big-M BINARY constraints (y<=hi·s, y<=z-lo·(1-s), s in {0,1}): INCONSISTENT —
  MILP (output=CEX + big-M) is INFEASIBLE. Gurobi IIS = {output rows} ∪ {ReLU
  big-M rows at L0 n11, L2 n19, L3 n16/18/20/24/30}.
- e_new_col PLACEMENT is correct: L0 input row reproduces the actual pre-act
  (0.9153); every L3 e_new_col column equals μ·W_out[:,neuron] exactly.

CONCLUSION: no single `e` both reproduces the CEX output via G AND satisfies the
per-neuron ReLU relation via the recorded z_k row. The rec_zono pre-act row
parametrization (row_indices/row_values) for POST-SIGMOID ReLUs is not consistent
with how z_k contributes to G, so the big-M exact-ReLU encoding contradicts a
point the zonotope certifies reachable. Remaining for real fix A: construct the
full physical e (needs sigmoid α/β/γ) to name the exact inconsistent column, find
the code divergence (post-sigmoid relu recording vs z_final), + a minimal toy
that reaches state_from_phase1.

## DEFINITIVE root cause (via the ACTUAL builder + IIS, not hand-reconstruction)

Using `build_gen_lp_from_state` (the real float32 constraints) + adding the CEX
output and computing the Gurobi IIS, there are TWO independent bugs:

1. phase1 builder `_build_phase1_lp`: the `_unused` gap columns (the sigmoid γ
   slack between ReLU e_new columns) are padded with vars FIXED to [0,0].
   MEASURED: with [0,0] the relaxation EXCLUDES the CEX (status INFEASIBLE);
   freeing those 770 columns to [-1,1] makes it CONTAIN the CEX (FEASIBLE).
   The pure-LP (no binaries) already excludes it — IIS = {tri_up_L2_*, tri_*_L3_*}
   + output rows. FIX: free `_unused` generator columns to [-1,1].

2. alpha_zono builder `_build_alpha_zono_lp` (the path that ACTUALLY verifies
   dist_shift, via state_by_qi): cur_n_gens=1596 != n_gens=812 — z_alpha is
   COMPRESSED but the positional column model advances by n_cells per sigmoid
   cell, so ReLU e_new_col indices are misaligned with z_alpha's real columns.
   MEASURED: here freeing the [0,0] padding does NOT help (CEX still excluded) —
   the misalignment is the defect. This is the original "track the columns"
   hypothesis, CONFIRMED for this path. FIX: advance cur_n_gens by z_alpha's
   ACTUAL per-layer column count (track compression), or don't compress z_alpha.

dist_shift's false-verify comes from bug #2 (alpha_zono). Fix B (raise on
cur_n_gens != n_gens) correctly guards it. Bug #1 is a second, real latent
unsoundness in the phase1 path (flagged by tests/test_gen_lp_soundness_invariants).

## RESOLVED (2026-05-30) — both bugs fixed, all measured

CORRECTION to the "DEFINITIVE root cause" section above: there is NO compression
of z_alpha. Instrumenting the α-forward's per-op generator count for index1285
showed it grows EXACTLY by the positional model's increments:
  input 8 → relu L0 +6 → sigmoid L1 +784 → relu L2 +10 → relu L3 +4 = 812.
The positional model SHOULD also give 812. It gave 1596 because
`state_from_alpha_zono` seeded `cur_n_gens = n_input = len(x_lo) = 792` — the
input DIMENSION — when z_alpha has only 8 INPUT GENERATORS (one per perturbed
dim; TorchZonotope.from_input_bounds allocates a column only where radii != 0).
Overshoot = 792 − 8 = 784, exactly. Every post-input ReLU e_new_col was then 784
too high, landing on zero columns of obj_G_out.

FIXES (both in verify_gen_lp.py):
  BUG 1 — `_unused` gap/trailing padding freed lb,ub=0 → [-1,1] in BOTH
    `_build_phase1_lp` and `_build_alpha_zono_lp` (4 sites). Sigmoid γ-slack
    columns are real noise symbols; pinning to 0 under-approximated the zonotope.
    Guard: tests/test_gen_lp_soundness_invariants.py::test_genlp_relaxation_equals_zonotope.
  BUG 2 — `state_from_alpha_zono` now sets n_input = count_nonzero(x_hi > x_lo)
    (input-generator count), matching state_from_phase1's rec_zono['n_input'].
    The cur_n_gens == n_gens raise is kept as a now-satisfied invariant guard.
  The state_from_phase1 "refuse sigmoid" raise was replaced by an in-range /
    distinct e_new_col check (sigmoid nets are sound now that BUG 1 is fixed).

VALIDATION (clean code, no bypass):
  index4739 (SAT)  sat-finding on  → sat       (PGD)        ✓
  index7901 (UNSAT) sat-finding on  → verified  (spec_milp 5s) ✓
  index2204 (UNSAT) sat-finding on  → verified  (spec_milp 4s) ✓
  index1285 (SAT)  sat-finding OFF → unknown   (no false verified) ✓
