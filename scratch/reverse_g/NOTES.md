# Reverse-mode G construction — progress

## Goal
Build the per-query α-zono state's G (the row_indices/row_values per unstable
neuron) by REVERSE-mode (backward from the ~2053 unstable neurons) instead of
FORWARD-mode (propagate ~11461 generators). Since #unstable < #generators,
reverse should be ~5.6× cheaper in compute AND memory, and avoids the 1.7GB
dense forward — the wall that killed batched-#1.

## Algorithm (validated on toys, all machine-epsilon PASS)
For each unstable neuron k at ReLU layer L, backward from its pre-activation:
- linear^T (fc: W^T; conv: conv_transpose) hop by hop toward input
- at each EARLIER unstable ReLU L' passed through: slack-col(L',j) = mu_{L'j} *
  (backward signal at post-ReLU output neuron j), THEN scale signal by lam_{L'}
- at input: input-col(i) = (backward signal at input i) * radius_i
- at ADD (skip): fan signal to both branches, sum contributions at shared input
Centers c_in: cheap point-forward applying (lam,mu) at ReLUs (no generators).
lam,mu,unstable: from bbr + alpha directly (no forward).

## Toys passed (scratch/reverse_g/)
- toy_dense.py     : 2-relu fc chain                  PASS (0, 4e-16)
- toy_dense_n.py   : 4-relu chain, multi-hop slacks   PASS (<2e-15), no spurious cols
- toy_conv.py      : conv1-relu-conv2-relu-fc         PASS (0, 9e-16)
- toy_skip.py      : ResNet add (main+skip)           PASS (9e-16)

## Next
- gg-DAG reverse walker validated vs forward_zono_dir_adaptive + state_from_alpha_zono
  on a small ONNX (conv/relu/add/fc), then tinyimagenet ResNet on AWS.
- Measure: reverse build time vs forward 1.26s; peak mem vs 1.7GB.

## gg-DAG reverse walker — VALIDATED vs real forward (scratch/reverse_g/reverse_build.py)
- gg_ground_truth.py + compare.py: conv-relu-conv-relu-fc through gg vs
  forward_zono_dir_adaptive+state_from_alpha_zono: PASS
  (Δlam=0 Δmu=0 Δc_in=9e-16 e_new_col exact Δrow=2e-16)
- test_skip_gg.py: residual block (conv-relu-conv-Add-relu-fc): PASS
  (Δlam=0 Δmu=0 Δc_in=0 e_new_col exact Δrow=3e-16)
Covers all tinyimagenet ResNet ops: conv(+pad), relu, add-merge(skip), reshape, fc.
reverse_build is CPU/numpy prototype; needs GPU port for timing.

## Next: tinyimagenet ResNet
1. Validate G_backward==G_forward on the REAL net (in-pipeline hook, one query).
2. GPU port + measure: build time vs forward 1.26s; peak mem vs 1.7GB.

## tinyimagenet ResNet — VALIDATED + MEASURED (A10G, 6546, all 13 phase-8 queries)
Reverse state vs forward (forward_zono_dir_adaptive + state_from_alpha_zono):
- correctness: fwd_nu=rev_nu=2051, ALL columns match (e_new_col enc_bad=0),
  max|Δrow| = 2.2e-08 .. 3.7e-08 across the 13 queries (fp32 accumulation;
  toys were fp64 at machine-eps). => reverse == forward to fp32 precision.
- timing: reverse build = 0.31-0.37 s / query
  vs forward_zono_dir_adaptive (1.26s) + state_from_alpha_zono (0.18s) = ~1.44s
  => ~4.6x faster on the part it replaces, and avoids the 1.7GB dense forward.

Bugs fixed scaling toy->real: (1) .detach() on grad tensors; (2) strided-conv
conv_transpose needs output_padding (compute from in/out shapes); (3) index
tensors must be dtype=long.

## Projected end-to-end (per-query state build)
current 2.16s = aCROWN .13 + ew .01 + build_dir .007 + FWD 1.26 + STATE .18 + score .57
reverse 1.03s = aCROWN .13 + ew .01 + build_dir .007 + REVERSE 0.31 + score .57
=> ~2.1x per query; Phase-8 state build 28s -> ~13s; Phase 8 42.7s -> ~28s on 6546.
Reverse is low-memory (no 1.7GB), so it ALSO unblocks batching #1 and composes
with the min-area-tiered idea + the box-halfspace-scoring-skip idea.

## Status / next
- ALGORITHM + gg implementation: DONE, validated toys(machine-eps) + real net(fp32 ~3e-8).
- reverse_build.py is a working GPU prototype (scratch/reverse_g/).
- NEXT (integration, not yet done): wire build_state_reverse into phase-8 as the
  state builder behind a flag (replace forward_zono+state), re-validate verdicts
  match on a sweep, measure end-to-end wall. Then it's a real production speedup.
- reverse build (0.31s) still has headroom: the per-neuron CPU nonzero extraction
  loop (745 neurons) is unvectorized — could push lower.

## END-TO-END INTEGRATED + VALIDATED (phase8_reverse_g flag) — 6546, A10G
build_state_reverse now also produces obj_c_out/obj_G_out_csr (seed from output
neurons) + lo/hi/form fields -> COMPLETE drop-in state. Wired in verify_graph
per-query loop behind setting phase8_reverse_g (default False, CUDA-only).
6546 with phase8_reverse_g=true:
- verdict=unsat (matches forward)
- per-query BnB nodes match forward: q9=110,q25=8,q31=0,q66=14,q76=4,q135=14,
  q139=2,q150=2,q160=14,q179=292,q182=6 identical; q32=2252(fwd2272),
  q53=136286(fwd~138k) within fp32/run-to-run noise. all unsat.
- Phase 8: 21.5s vs forward 42.7s  => HALVED.
- total wall 97.8s -> 88.0s (budget margin 2s -> 12s).

Files: src/vibecheck/reverse_g.py (module), setting phase8_reverse_g in settings.py,
wired in verify_graph.py per-query state-build loop. scratch/reverse_g/ has the
toys + ground-truth + compare harnesses (all PASS).

## Remaining / next
- regression sweep: confirm verdicts unchanged across SAT + UNSAT case types.
- output objective validated machine-eps on toy; on real net it rode along in the
  end-to-end verdict match (BnB used it). Could add an explicit obj match check.
- further speedup headroom: vectorize the per-neuron CPU nonzero extraction;
  the ~12s 'other' phase-8 cost (shared gen_lp precompute) is now the largest chunk.

## REGRESSION SWEEP (phase8_reverse_g=true, every 8th tinyimagenet case, vs AB-CROWN)
25 cases | UNSOUND: 0 | CRASH: 0 | miss_unsat: 0 | miss_sat: 0
verdict agreement: unsat/unsat 21, sat/sat 3, unknown/timeout 1 (both undecided).
=> 24/25 decided cases match AB-CROWN exactly; 0 regressions, 0 unsoundness.
Results: /home/stan/Desktop/temp/reverse_g_sweep/results.csv
CONCLUSION: reverse-mode G state build is correct + sound across case types and
halves Phase 8. Ready to flip phase8_reverse_g default on after a broader sweep.

## FULL ABLATION (measured A10G, 6546 robust, additive; --timeout 200 so all complete)
config                          verdict  p1       p8       total
0 legacy BnB, no cap            unsat    78.4s    80.0s    165.0s
1 +fast verifier (no cap)       unsat    77.2s    42.5s    126.2s   (p8: 80->42.5, BnB ~2x)
2 +pre-cascade cap 5s           unsat    61.9s    42.4s    110.9s   (p1: 77->62, pre-cascade 22->5s)
3 +reverse_g state build        unsat    59.8s    21.4s     87.7s   (p8: 42->21, state build ~2x)
Cumulative legacy->all: total 165->88s (1.9x), p8 80->21.4s (3.7x).
+ batched-over-directions reverse (isolated): reverse build 5.44s->2.87s (1.89x), peak 0.75->8.75GB.
Note: at the real 100s VNNCOMP budget the pre-cascade cap FLIPS 6546 unknown->unsat
(uncapped 22s pre-cascade starves phase8); at 200s here all complete so it shows as p1 saving.

## SAT-schedule ideas (separate, SAT cases)
- pre-cascade cap 5s + Phase-9 restrict_disj: 38/38 SAT caught, 0 regressions (earlier sweep).
- reverse_g regression sweep (25 cases): 0 unsound/crash, verdicts match AB-CROWN.

## DEAD ENDS (measured negatives, do NOT pursue)
- batched FORWARD (#1): 2.6x SLOWER on ResNet (memory-bound: 1.7GB/dir, OOM at full batch).
- shared-state reuse (#2): shared-alpha state 6/13 close (catastrophic, spec unsolved);
  min-area 11/13 but case still unsolved. The hard specs need per-direction tightening.
- iterative shared-alpha refit: 0/13 close at root (per-spec-alpha ceiling 1/13). BnB-bound.

## OOM-SAFE batched reverse (build_states_reverse_batched_safe)
Chunks the D directions; on torch.cuda.OutOfMemoryError halves the chunk and
retries (down to 1 = sequential), remembering the smaller safe size. Safe on the
10GB RTX 3080 (batched peak was 8.75GB at D=13 on the A10G).
Fallback TEST (scratch/reverse_g/test_batched_oom.py): inject fake OOM for chunk>3,
D=8 -> wrapper halved 8->4->2, 4 chunks of 2, results identical to unconstrained
batched (Δ=0). PASS. Module now at src/vibecheck/reverse_batched.py.

## PHASE-8 PROFILE (reverse_g ON, 6546, A10G) — corrects the "12s precompute" guess
p8=21.46s: scoring 6.95s | state_build(reverse) 5.73s | unaccounted 5.04s |
bnb 1.97s | alpha_crown 1.56s | capture_ew 0.12 build_dir 0.09 unstable_idx 0.01 |
gen_lp precompute = 0.07s (NOT the bottleneck — reuses phase1 zono).
Real targets ranked: (1) box-halfspace SCORING 6.95s, mostly wasted (11/13 queries
close in <=14 nodes regardless of order) -> skip for trivial closers; (2) state build
5.73s -> batched reverse ~2.9s; (3) unaccounted ~5s, likely per-query empty_cache
(13 GPU syncs) -> drop it (reverse is low-mem); (4) alpha_crown 1.56s (batchable).
Stacking scoring-skip + drop-empty_cache + batched could push p8 ~21.5 -> ~8-10s.

## PHASE-8 PROFILE — fully accounted (reverse_g, 6546, p8=21.56s)
scoring 6.99s (CPU numpy lagrangian_min, per-neuron, 2052x2x13 -- NOT GPU logbucket)
state_build(reverse) 5.78s (GPU; batched -> ~2.9s)
bnb_parse+upload 3.61s (the "unaccounted": state-dict -> parse_problem dense a_g -> GPU upload; GPU->CPU->GPU roundtrip)
bnb_node 1.99s | alpha_crown 1.55s | fast_verifier_build(compile) 0.83s | empty_cache 0.58s | ew+build+unstable 0.22s
Levers ranked: scoring (skip-trivial or GPU-batch ~5-6s) | parse+upload (build Problem directly on GPU ~3.6s) |
state_build (batched reverse ~2.9s). gen_lp precompute is 0.07s (NOT a factor).

## BATCHED STATE BUILD WIRED + MEASURED (6546, A10G, --timeout 200)
Flags: phase8_reverse_g=true + phase8_reverse_g_batched=true (verify_graph.py _rev_batched branch).
Collects per-query alphas, one build_states_reverse_batched_safe call, distributes states, scores each.
verdict=unsat; all 13 BnB node counts IDENTICAL to sequential reverse_g (q9=110,q32=2554,q53=139478,...).
state_build 5.78 -> 2.95s (1.96x). p8 total 21.56 -> 18.36s.

## SCORING OPTIMIZED — SPARSE-SUPPORT (CPU), NOT GPU. MEASURED, not guessed.
Q: "is scoring using the fast GPU log bucket sort?" A: NO. score_box_halfspace_delta_lb is CPU numpy.
The GPU logbucket is a SEPARATE path (fast_dual_ascent BnB node line-search), unrelated to branching scores.
Profiled the 7.34s: NOT the python breakpoint loop (vectorizing lagrangian_min: 3.48->3.32s, negligible).
Real cost = dense O(n_gens=11461) elementwise ops inside lagrangian_min, x53378 calls (26689 neurons x2 sides).
FIX (landed): the halfspace a=row_full is SPARSE (only row_indices nonzero), so every breakpoint + the whole
dual live on the neuron's support; the sum(|d|) term folds in unchanged off-support mass via a precomputed
base_abs_sum. New _lmin_sparse in verify_gen_lp.py. Bit-exact (<=4e-15) vs dense over all 26689 neurons.
  dump timing all-13: dense 3.32s -> sparse 1.87s (1.8x).  in-pipeline scoring 7.34 -> 4.11s.
GPU NOT A WIN (measured, A10G): padded-batch 2.79s (host ragged->pad python loop is the bottleneck);
dense-batch 1.85s (== CPU sparse; dominated by the (N,n_gens) mostly-inf torch.sort) AND fp32 degrades
branch ranking to spearman 0.92 (dense) / 0.97 (padded). So kept CPU sparse: faster-or-equal, fp64-exact,
no ranking loss, works on server1 (no extra GPU mem). Unit test: tests/test_score_box_halfspace.py (7 tests,
sparse==dense incl. degenerate-mu fallback + infeasible +inf branch; scoring fn now 100% line-covered).

## PHASE-8 PROFILE — batched state + sparse scoring (6546, A10G, p8=15.02s)
scoring 4.11 | bnb_parse+upload 2.97 | state_build 2.92 | bnb_node 1.92 | alpha_crown 1.69 | compile 0.82 | unacc 0.59
Cumulative this session: 42.7s (forward) -> 21.56 (reverse_g) -> 18.36 (batched state) -> 15.02 (sparse scoring).
Next lever: bnb_parse+upload 2.97s (build fast-verifier Problem directly on GPU, skip state-dict->CPU->GPU roundtrip).

## PARSE+UPLOAD BUILT ON GPU + WARMUP-SKIP (6546, A10G, p8=14.46s). MEASURED.
Decomposed the 2.97s "bnb_parse+upload" first: parse_problem 1.11s (dense (2053,11461)=188MB host alloc +
per-split scatter loop) | a_g upload 0.34s | per-query compile-warmup ~1.5s.
FIX 1 (build on GPU): parse_problem_gpu (fast_verify_dual.py) builds a_g directly on-device via a FLAT scatter
  from the sparse rows (only the nnz upload, not 188MB), and d_t via sparse matvec (qw @ csr, no toarray()).
  Only one (S,n) matrix resident at a time (built per-query at BnB time) -> memory-safe on server1.
FIX 2 (warmup-skip): the Verifier is reused across queries; a depth warmed once stays compiled. Added
  self._warmed set in fast_verify_topk.Verifier (the PRODUCTION one; NOT fast_verify_dual.Verifier which is
  unused) -> skip re-warming -> kills ~1.4s of redundant per-query kernel launches.
CORRECTNESS: dump test (production topk Verifier) verify(parse_problem) vs verify(parse_problem_gpu) =
  0 mismatches, a_g bit-identical (max|Δ|=0), identical nodes on all 13 disjuncts.
  Unit test tests/test_parse_problem_gpu.py (3 tests, CPU device: a_g fp32-exact, all scalar fields to 1e-12,
  incl. reordered-subset scored_keys + empty scored_keys).
  In-pipeline: small/stable trees IDENTICAL between CPU-parse(15.02) and GPU-parse(14.46) runs (q9=108,q25=8,
  q31=0,q66=14,q179=292...). Only the 2 huge trees jitter (q53 139478->143600->144152 across 3 runs; q32 ~2500)
  = PRE-EXISTING fp32 GPU state-build nondeterminism (varies run-to-run regardless of parse), verdict unsat throughout.
bnb_parse+upload 2.97 -> 2.31s. p8 15.02 -> 14.46s.

## PHASE-8 PROFILE — + GPU parse + warmup-skip (6546, A10G, p8=14.46s)
scoring 4.07 | state_build 2.94 | bnb_parse+upload 2.31 | bnb_node 1.93 | alpha_crown 1.78 | compile 0.82 | unacc 0.60
Cumulative this session: 42.7s (fwd) -> 21.56 (reverse_g) -> 18.36 (batched) -> 15.02 (sparse scoring) -> 14.46 (GPU parse).
Remaining levers: scoring 4.07 (CPU sparse; GPU not a win — see above) | state_build 2.94 (reuse rowG for a_g to
skip CSR roundtrip, but couples builder<->verifier + holds D matrices) | alpha_crown 1.78 (batchable).
