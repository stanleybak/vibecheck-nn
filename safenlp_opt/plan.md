# safenlp_2024 optimization — living log

## Consolidations
(none yet)

## Explorations
### explore01 — ground truth + where the racing lives
ABC published runtimes: hard ruarobot UNSAT cases (992/3558/1889) take ABC ~7-8s each (non-trivial unsat, not
  instant). So vibecheck has a 20s budget for cases ABC needs 7-8s on. (NOTE: my display sed mislabeled "unsat"
  as "sat" — greedy regex grabbed the 'sat' inside 'unsat'; verdicts ARE unsat per col-5 of the csv.)
Racing "feasibility SAT → escalate" logic = verify_milp.py:1559 `_racing_escalation`; feasibility MILP builder at
  verify_milp.py:1378-1522 (`mode=='feasibility'` at 1522). ABC safenlp config: exp_configs/vnncomp24/safenlp.yaml
  (on server1). NEXT: read racing code (why unknown at max bins=118?) + ABC config (their approach).
### explore02 — GROUND TRUTH: hard cases ARE unsat; clean exact big-M MILP proves them in <1s ✅
Independent exact MILP (onnx weights + VALID interval pre-act bounds + big-M binaries, explore02_exact_milp.py):
  ruarobot 992 min(Y0-Y1)=+0.627 unsat; 3558 +0.429 unsat; 1889 +0.316 unsat; 132 -1.70 sat (ctrl). ALL match ABC.
  Status=2 (optimal) in <1s each. So an exact MILP is the right tool (= ABC's `complete_verifier: mip`) and it's
  FAST. ⇒ vibecheck's racing "feasibility SAT @ bins=118" is SPURIOUS (the exact MILP is actually INFEASIBLE for
  the unsafe region). My big-M encoding == vibecheck's (verify_milp.py:1464-1472) — so the bug is the BOUNDS (lo,hi)
  vibecheck passes to the worker, OR the scored_keys/milp_set wiring (binaries not applied to the right neurons).
  Note: my interval bounds gave 103 unstable for 992 vs vibecheck's 118 — different bound source.
ABC safenlp approach (exp_configs/vnncomp24/safenlp.yaml): complete_verifier: mip + pgd_order:after restarts:10000.
  Dead simple — exact MIP for unsat, heavy PGD for sat. ABC ~7-8s/case.
NEXT: instrument vibecheck's feasibility worker — on SAT, eval the witness through the real net; if Y0>Y1 it's
  spurious (bounds/wiring bug). Then fix (clean exact MILP path for these small nets, or fix the racing bounds).
### explore03 — ROOT CAUSE: milp_verify's `layers_np` ≠ onnx (net-loading bug) ✅✅
The bins=118 MILP "feasibility SAT" witness, evaluated on the SAME input:
  onnxruntime = +2.605 (safe), raw-onnx-weights = +2.605 (safe), vibecheck layers_np = -1.542 (claims unsafe).
  Two independent evaluators agree the real net is SAFE there → vibecheck's `layers_np` (milp_verify's net repr)
  is WRONG. milp_verify is verifying a DIFFERENT network → phantom counterexamples → can never prove unsat.
  Likely a weight-orientation/structure bug in layers_np extraction for the 2-MatMul/2-Add/1-ReLU FC structure.
  (The MAIN graph net is fine: PGD matched all 12 SAT, graph-mode verified 6/12 UNSAT.) Δ is +2.6 vs -1.5 = large
  ⇒ structural (transpose/bias/op), not numerical. NEXT: find layers_np builder, diff its W/bias vs onnx, fix.
### explore04 — THE BUG: gpu_layers() drops the Add(bias) → fc bias = ALL ZEROS ✅✅✅
onnx = MatMul -> Add(b1) -> Relu -> MatMul -> Add(b2). Real biases (b1=[-0.127,0.054,...], b2=[0.196,-0.196]).
gpu_layers() fc layers: W correct (orientation right, ==onnx W.T) but **bias = [0,0,0...]** for BOTH layers — the
  separate Add(MatMul_out, bias) node is NOT folded into the fc bias. So milp_verify (which builds layers_np from
  gpu_layers) verifies a BIAS-FREE net → wrong outputs (-9.95 vs onnxruntime -5.99) → phantom counterexamples →
  "feasibility SAT" → can never prove unsat. Main graph keeps Add as its own node ⇒ zono/CROWN/PGD CORRECT (that's
  why only the MILP path fails). Affects any TF/Keras-exported FC net (separate MatMul+Add, not a fused Gemm).
FIX: make gpu_layers() fold a following Add-with-constant into the fc/conv bias. Then milp_verify is exact and
  safenlp UNSAT cases verify in <1s (like ABC's complete_verifier:mip). NEXT: find+fix gpu_layers in network.py.
### explore05 — FIX: gpu_layers folds Add(bias) → net matches onnx ✅
network.py gpu_layers(): added `elif node.op_type=='Add' and 'bias' in node.params and layers:` → folds the
  Add's constant bias into the preceding linear layer's bias (+ gpu_b_fwd). Verified: gpu_layers biases now ==
  onnx (b1,b2); forward Y0-Y1 = onnxruntime exactly (-3.74275 == -3.74275). NEXT: end-to-end --mode milp on the
  3 hard UNSAT (should verify now) + SAT control; then full graph-mode + sweep.
### explore06 — fix works END-TO-END (--mode milp): 992/3558/1889 → unsat (2.4-3.4s), 132 → sat ✓
NOTE: default graph mode uses gpu_graph (bias was ALREADY correct there) — its unknowns were loose-CROWN/no-input-
  split, a SEPARATE issue. The bias bug only broke --mode milp (gpu_layers). Now milp_verify is exact (= ABC's
  complete_verifier:mip). Plan: route safenlp to milp (it does PGD for sat + exact MILP for unsat). Test --mode
  milp on full 24-sample; then figure routing (config/auto-detect) so the sweep uses it.
### explore07 — auto-route small FC → milp_verify (verify_graph.py) + setting
Added: verify_graph auto-routes nets with (no bilinear, n_in>split_cap, ≤2 ReLU layers, no conv, no forks) to
  milp_verify — mirrors auto_route_milp_for_conv. Setting auto_route_milp_for_small_fc=True (settings.py). So the
  DEFAULT mode (no --mode/--config, what the harness uses) now routes safenlp to the exact MILP. Gate is narrow:
  mnist_fc (4-6 ReLU)/acasxu (6 ReLU) NOT caught (n_relu>2); pensieve/mscn have bilinear → excluded. Testing
  default-mode 24-sample + regression (mnist_fc/acasxu must NOT route) + unit/integration + full sweep next.
### explore08 — DEFAULT auto-route: 24/24 ✅ (safenlp solved on the sample, no --mode/--config needed)
NEXT: sync to server1, launch full 1080 sweep (investigate any miss case-by-case per user). Local regression:
  unit + integration (milp_verify users: oval21/cifar/nn4sys + the gpu_layers bias fix) + confirm mnist_fc/acasxu
  don't mis-route. Clean up the VC_MILP_WITNESS diagnostic after.
### explore09 — server1 sweep MISS on 992 (unknown 36s) → INVESTIGATE (user rule): racing too slow on server1
992 on server1 --mode milp: Phase1 worst=-4.75 (bias fix tightened from -10.7), then racing reaches only bins=16
  in 21s → timeout → unknown. Local reaches bins=118 in ~3s. CAUSE: _racing_escalation spawns a fresh
  multiprocessing.Pool(2)+Gurobi env PER bin level (8 levels: 0,2,4,8,16,32,64,118); per-level overhead ~3.7s on
  server1 (vs ~0.4s local) → can't reach the exact level (118) in the 20s budget. For tiny nets the exact MILP
  (118 binaries) is <1s in ONE solve → gradual escalation is counterproductive here. FIX: for small nets
  (n_unstable ≤ threshold) skip the schedule, go straight to bins=n_unstable (exact). Stopped the slow-code sweep.
### explore10/11 — server1 verifies (slow Gurobi); the 5 integration fails are PRE-EXISTING (stash-proven)
- 992 on IDLE server1 (racing fix): unsat, racing→bins=103, optimize lb=0.0056 in 15.9s → verified (17.38s solve
  < 20s budget, tight). server1's Gurobi ~15x slower than local for the exact MILP. (Earlier 40s was transient
  contention from the dying sweep.) Local: 24/24 fast (~2-3s).
- INTEGRATION 5 failed (dist_shift x2 sigmoid, malbeware full-MIP, tinyimagenet x2 hard-UNSAT): STASH TEST — same
  5 fail on COMMITTED code WITHOUT my changes (5 failed/4 passed). ⇒ PRE-EXISTING (not from bias/auto-route/racing).
  Flag to user (they're in committed main, possibly from the c521691 bundle; I never ran the FULL integration suite
  before committing — only test_nn4sys). My safenlp changes are integration-NEUTRAL (identical failure set).
- DEFENSIVE NARROWING (so my changes provably can't touch those benchmarks): auto-route → pure-ReLU only (exclude
  Sigmoid/Tanh/etc; fixes any dist_shift mis-route); racing direct-exact → FC-only (exclude conv; protects
  tinyimagenet/malbeware timing). Then re-verify safenlp 24/24 + integration unchanged.
### explore12 — narrowing OK (24/24 still); full 1080 sweep launched LOCALLY (server1 Gurobi too slow)
After defensive narrowing (auto-route pure-ReLU, racing FC-only): safenlp default 24/24 unchanged. server1 Gurobi
  ~15x slower (15.9s exact MILP) → full sweep would take hours; safenlp tiny (no OOM) so running full 1080 LOCALLY
  (fast Gurobi). Monitoring /tmp/safenlp_local_misses.txt; investigating each miss per user rule.
PENDING for user: 5 PRE-EXISTING integration failures in committed main (dist_shift x2, malbeware, tinyimagenet x2)
  — stash-proven not from my safenlp work; flag for separate fix. server1 Gurobi slowness = hardware caveat.

### explore13 — FINALIZATION (sweep in flight, finalize artifacts in parallel)
While the local 1080 sweep runs (clean so far: 281/281 match, 0 miss @ ~1.75s/case), finalized the mergeable
  artifacts (none touch the running subprocesses' behaviour — each case re-imports fresh):
- configs/safenlp_2024.yaml: rewritten as a documenting comment-only config (loads to {} → auto-route handles it;
  no --config needed, like nn4sys). Removed the superseded input_split_enabled/max_dims=35 (that was the WORSE
  recon approach; auto-route to exact MILP supersedes it and catches safenlp BEFORE the input-split dispatch).
- docs/benchmarks/safenlp_2024.md: rewritten with the root-cause (gpu_layers dropped Add bias → all-zero biases →
  spurious "feasibility SAT"), the fix (fold trailing Add(bias) into preceding fc), the FC-only direct-exact
  racing schedule, reproduction cmds. Score line marked PENDING (fill from sweep).
- tests/integration/test_safenlp_2024.py: 1 SAT (ruarobot 132, PGD) + 2 hard UNSAT (ruarobot 992 + 3558 — the
  input-split-explode cases the exact MILP cracks). max_wall_s placeholders (10/15/15) to be tightened from a
  clean in-process timing run AFTER the sweep (no contention).
- verify_milp.py: removed the VC_MILP_WITNESS investigation diagnostic (np.save + forward-witness print). Parses
  OK; xv still used. Behaviour-neutral (was gated off).
- safenlp_opt/ stays UNTRACKED (matches precedent — no prior benchmark committed its _opt workspace; not in
  .gitignore but won't be staged).
NEXT: sweep completion → fill score + investigate any miss → clean in-process timing for the 3 integration cases
  → finalize max_wall_s → run safenlp integration (3 cases must pass) → re-confirm the 5 pre-existing fails
  unchanged → pre-merge gap report to user (do NOT merge without explicit approval).

### explore14 — CRITICAL: unit suite caught a regression I INTRODUCED → fixed two ways
Ran full unit suite (cov) post-finalization: 1 FAILED — test_acasxu_sequential_vs_graph
  (test_graph_verify.py): result_seq 'verified'→'unknown'. STASH-BISECT: committed PASSES, mine FAILS
  ⇒ MY regression. Further bisect (stash one file at a time):
  - stash verify_milp only (keep network) → STILL fails ⇒ network.py (gpu_layers Add-fold) is the culprit.
  - stash network only (keep verify_milp depth-guard) → PASSES ⇒ my racing fix is fine.
ROOT CAUSE (subtle + important): acasxu IS MatMul:7+Add:7 (separate bias). Committed gpu_layers DROPPED the
  Add bias → milp_verify was solving a BIAS-FREE acasxu net (a different, looser net) that happened to verify
  prop_3 as 'verified'. The test was passing FOR THE WRONG REASON. My Add-fold makes gpu_layers match
  onnxruntime EXACTLY (MAX|ort-mine|=0.0, proven), so milp_verify now solves the REAL net — which the raw-MILP
  racing path (no input-split) can't crack (exhausts schedule at ~13s, 'unknown' even at 60s; SOUND, incomplete).
  Confirmed real acasxu capability intact: FULL pipeline integration (test_acasxu_2023) still 3/3 verified.
TWO FIXES:
  (1) verify_milp racing: my _DIRECT_EXACT_MAX_UNSTABLE short-circuit was FC-only but depth-blind. Direct-exact
      (one full-binary MILP) is great for SHALLOW nets (safenlp 1-relu, <1s) but TERRIBLE for DEEP ones
      (acasxu 6-relu: full-binary MILP combinatorially hard; gradual escalation verifies at an early bin).
      Added depth guard: direct-exact only when _n_relu_layers<=2 (= #affine - output). safenlp(1)✓ acasxu(6)→gradual.
      [This alone did NOT fix the test — the test failure was the bias fix, not racing — but it's a correct, needed guard.]
  (2) test_acasxu_sequential_vs_graph: the hard `assert result_seq=='verified'` CONTRADICTED the test's own
      docstring ("'verified' and 'unknown' both acceptable"). It was over-fit to the bias-free net. Updated to pin
      the REAL invariant: both paths SOUND (in {verified,unknown}, never sat) AND CONSISTENT (seq==graph; both
      'unknown' now). NOT masking — gpu_layers proven correct vs onnxruntime + full pipeline still verifies.
HARDENING: gpu_layers Add-fold could mis-fold a post-ReLU Add(bias) into the pre-ReLU linear (unsound; bias can't
  commute through ReLU). Real nets are all adjacent (MatMul→Add), but added `_prev_was_linear` adjacency guard:
  fold ONLY when the Add directly follows the linear (no activation between). Re-verified acasxu exact (0.0).
LESSON: a unit test that passes can be passing for the wrong reason (verifying a buggy looser net). A correctness
  fix can EXPOSE such a test. Always stash-bisect "is this mine?" and forward-cross-check vs onnxruntime to decide
  fix-code-vs-fix-test. NEVER weaken a test without proving the new behavior is the correct one.
NEXT: re-run FULL unit suite (post all 3 edits) → 0 fail; confirm 5 pre-existing integration fails unchanged;
  gap report (safenlp 1080/1080 + the acasxu near-miss I caught + the 5 pre-existing).

### explore15 — CONSOLIDATION: all gates green; safenlp DONE; 5 pre-existing flagged
FINAL verification (all edits in tree):
- safenlp full local sweep: SWEEP_DONE match=1080 miss=0 of 1080 (647 sat + 433 unsat, 0 soundness divergence).
- Unit suite: 773 passed, 0 failed (incl. new test_gpu_layers_bias_fold + updated acasxu equivalence).
- Integration: 39 passed (safenlp 3 + acasxu 3 + all prior), 5 PRE-EXISTING failed (unchanged), 1 skipped.
  ⇒ my change set is integration-NEUTRAL on the 5; adds 3 passing safenlp cases.
Change set (mergeable, contained): network.py (Add-fold+adjacency guard), settings.py (auto_route flag),
  verify_graph.py (auto-route shallow FC→milp), verify_milp.py (direct-exact depth guard; witness removed),
  tests/test_graph_verify.py (acasxu invariant), +configs/safenlp_2024.yaml +docs +2 tests. 129/-20 src lines.
5 PRE-EXISTING failures (NOT mine, stash-proven; from c521691/1fea218 lineage; surfaced because full integration
  was never run before those merges — only test_nn4sys):
  - dist_shift x2: REAL code regression. settings.py:397 made phase1_method='bab_refine' the GLOBAL default;
    bab_refine→_per_neuron_adaptive_bounds asserts (verify_graph.py:1380) on SIGMOID nets (no relu op at the
    target layer). Fix options: guard bab_refine to require relu / fall back to legacy on non-relu nets, OR set
    configs/dist_shift_2023.yaml phase1_method: legacy.
  - malbeware x1: 'unknown' at 30.3s (full-MIP didn't converge; hardware-suspect — local Gurobi slower than the
    tuning box). Needs a remote re-run to classify timing-vs-regression.
  - tinyimagenet x2: CUDA OOM on the local 8GB GPU (wrong-machine; needs server1 10GB / AWS 24GB).
DECISION: do NOT fix the 5 on the safenlp branch (branch hygiene + out of scope). Flag to user with root cause +
  proposed fixes. safenlp branch is merge-ready pending user's pre-merge approval (CLAUDE.md step 7 + ask-before-merge).
STATUS: safenlp_2024 = SOLVED, all parts covered. Awaiting user.
