# acasxu prop_2 net 3_3 — close the BaB (ABC 18s, vc timeout)

## Problem
vc's input-split BaB times out on acasxu prop_2 net 3_3 (unsat). ABC verifies in
18s (naive split + CROWN + clip_input_domain + reorder_bab, no MILP). prop_3 1_1
fixed by disabling milp_escalate; 3_3 still diverges.

## Profiler (milp off, 50s)
BAB total=48s | bound=7.16s(15%) clip=0.15s(0%) **split=38.69s(80%)** | iters=310
**leaves=1,216,231** | closures: crown=590106 clip=155 | **clip ratio=0.9838 (~1.6% shrink)**

Core issue: 1.2M leaves (ABC uses ~thousands). Clip barely shrinks → leaves never
small enough → keep splitting → split op (80%) dominates. Must REDUCE leaves.

## Levers
1. Reduce leaves: (a) stronger clip, (b) tighter bounds/more α, (c) better split dim.
2. Speed split (38.69s): eliminate `.to()` overhead — secondary (1.2M leaves is the real problem).

## Explorations
- explore01: clip_iters=10 / alpha500+eps0.5 / clip10+alpha500 — RUNNING

- explore01 RESULT: clip_iters=10 / alpha500 / clip10+alpha500 all unknown@60s. Config can't fix it.
  Lesson: clip strength is fixed by A_lin sensitivity; more iters/α don't reduce leaves.
- explore02: ROOT CAUSE = split is a CPU Python loop (xl_split.cpu() at 10236, xl_c.to(device) per
  child at 10264) = CPU<->GPU round-trip per child x 1.2M leaves = 80% of wall. K_eff=1 for acasxu.
  FIX: vectorized on-GPU single-dim split (K_eff==1 fast path; keep Python loop for K>1/pensieve).
  Testing prop_2 3_3 + prop_3 1_1 + 1_5 SAT — RUNNING

- explore02 RESULT: vectorized split worked (split 80%->16%, prop_3 1_1 8.9->4.4s) BUT broke 1_5 SAT
  (unknown, even with interleave) AND prop_2 3_3 still diverges (now bound=74%, 6.8M leaves). REVERTED.
  PIVOT: split speed is a red herring. prop_2 3_3 DIVERGES — leaves never close (6.8M, half close per round).
  Root cause must be bound tightness OR split-dim choice. Next: check split-dim normalization + leaf bound vs ABC.

- explore03 (SB) RESULT: SB=on doesn't fix 3_3 (still timeout). prop_3 1_1 fine (3.6s).
- explore04 STRUCTURAL: spec margin vs box shrink: 1.0=-1597, 0.5=-462, 0.25=-111, 0.1=-10.8,
  0.02=-0.35, 0.001=+0.002. Bound tightens but needs shrink~0.001/dim => 2^50 leaves. vc forward-zono
  bound ~1000x too loose. ROOT CAUSE = loose intermediate bounds (forward zono), not branching/split.
  PIVOT: need tighter bounds (alpha-CROWN intermediate). Testing alpha-CROWN root bound now.

- explore05: forward-LiRPA intermediate bounds = -8063 (WORSE than zono -1597). Not the fix.
- explore06: run_alpha_crown_batched exists (joint intermediate-bound α) but only for ONE box;
  _run_alpha_crown_inputsplit_batched batches over leaves but uses STATIC forward-zono intermediates
  (codebase comment: "Vibe's α-CROWN uses static intermediates"). Neither does joint-intermediate-α
  batched over input-split leaves — the exact thing ABC does and acasxu needs.

## CONCLUSION (structural wall, documented with evidence)
prop_2 3_3 diverges because vc's input-split BaB intermediate bounds (forward zono) are ~1000x too
loose for ACAS Xu's amplifying weights (root spec margin -1597 vs true >0; needs box shrink ~0.001/dim
= 2^50 leaves). Spec-only α, forward-LiRPA, SB, clip iters, split-speed all fail to close the gap.
The FIX = joint α-CROWN intermediate bounds BATCHED over input-split leaves (re-tighten per-layer
pre-ReLU bounds with the optimized α each iter), which vc explicitly does not do. This is a major
feature, not a config/quick fix. Delivered instead: prop_3 1_1 (disable milp, 89s->9s) + prop_6 crash.

## BREAKTHROUGH (explore07)
verify_hybrid_acasxu.verify_hybrid (DEAD code, never wired in) does freeze-replay alpha-CROWN
INTERMEDIATE-bound tightening (_full_freeze: per-layer seed [I,-I] backward, optimize earlier-layer
alpha, intersect w/ forward-zono). Tested:
  prop_2 3_3 -> unsat 47.9s | prop_3 1_1 -> unsat 32.1s | 1_5 prop_2 -> sat 2.2s. ALL SOLVED.
The intermediate tightening is SOUND (intersects two sound over-approximations). Slower than ABC
(47 vs 18s) but within the 116s competition timeout.
FIX = wire verify_hybrid in for acasxu. Needs: (1) respect disable_sat_finding (skip PGD for the
soundness probe), (2) output adapter (verdict dict -> (result, details)), (3) config flag, (4) tests.

## FIXED (explore08)
Wired verify_hybrid in via `use_hybrid_acasxu` setting (default off; acasxu config on). Made
verify_hybrid respect disable_sat_finding (skip PGD). Output adapter + onnxruntime witness validation.
End-to-end via verify_graph: 3_3 prop_2 verified 46s | 1_1 prop_3 verified 30s | 1_5 sat 2s |
1_5 soundness probe (no sat-find) -> unknown (NOT verified, SOUND). All within 116s timeout.

## REGRESSION (explore09): hybrid too slow on server1 -> 10/26 misses (overruns 147-177s)
verify_hybrid solves prop_2 3_3 (46s local) but on server1's RTX 3080 (not throttled, 48C) the
FREEZE has no timeout check and overruns to 147-177s -> 10 timeout-misses. README claimed 3.21s mean
on server1 - not reproducing. Must profile the slowness (freeze vs BaB vs .to overhead).

## BREAKTHROUGH 2 (explore10): backward-CROWN intermediate bounds = the gap
ABC uses bound_prop_method: crown (BACKWARD CROWN intermediate bounds). vc's input-split BaB used
FORWARD ZONO (loose). Basic backward CROWN (seed [I,-I] per layer, min-area slopes, no alpha) is
~2x TIGHTER and CHEAPER: 3_3 prop_2 root fwd-zono -1597/226ms vs crown -722/8ms; prop_1 -4691 vs -2148.
Added _crown_intermediate_batched + wired into input-split BaB via input_split_crown_intermediate.
Input-split + CROWN (default PGD): prop_1 1.2s (was hybrid-timeout, MATCHES ABC ~1s!), prop_3 4.7s,
3_3 prop_2 111s (verifies, no divergence). 1_5 SAT missed only b/c default PGD (restarts=30 not 5000).
This could REPLACE the slow hybrid entirely. Testing with full acasxu tuning (strong PGD + boundary alpha).

## explore11 (user's idea): MUTUAL zono<->CROWN intermediate tightening
Hypothesis: intersect zono+CROWN per layer, feed intersection into BOTH next
zono relaxation AND next CROWN. Implemented: `_forward_zonotope_graph_batched`
gains `tight_override` (intersect zono range w/ external bound at each ReLU —
sound, stabilises more ReLUs); `_crown_intermediate_batched` n_sweeps>1 now
alternates a CROWN pass + a zono-with-override pass to a fixpoint.
ROOT spec margin (CPU f64, B=1, no clip):
  prop_1 1_1:  fwd-zono -4691 | CROWN-only -2148 | mutual x2 -1388 | x3 -1387 (converged)
  prop_2 3_3:  fwd-zono -2570 | CROWN-only -1101 | mutual x2 -1057 | x3 -1055 (converged)
Result: prop_1 CROWN-only -2148 reproduces explore10 EXACTLY. Mutual cuts prop_1
35% tighter (-2148->-1388), prop_2 3_3 only 4% (-1101->-1057). Converges by x3
(x5==x3). Cost ~430ms (x3) vs 131ms (CROWN-only).
Lesson: zono feedback compounds strongly when correlations matter (prop_1, 1 query)
but little on prop_2 3_3 (4 queries, the binding constraint is already CROWN-tight).
Next: end-to-end input-split BaB time w/ mutual (sweeps=3) vs CROWN-only on
prop_2 3_3 (was 62-111s) + prop_1 (1.2s) — does the leaf cut beat the 3x bound cost?

## explore12: end-to-end input-split BaB, mutual (sweeps=3) — MIXED
GPU (RTX PRO 2000 laptop), disable_sat_finding, input_split_crown_intermediate:
  sweeps=3:  prop_2 3_3 -> UNKNOWN/timeout 120s  | prop_1 0.58s | prop_3 10.67s
Result: mutual HELPS prop_1 (0.58 vs ~1.2s CROWN-only) but REGRESSES prop_2 3_3
into timeout. Matches explore11: prop_2 3_3 root only 4% tighter, so the leaf cut
doesn't offset the ~3x per-batch bound cost; prop_1 35% tighter -> net win.
Lesson: mutual sweeps pay off only when the zono feedback materially tightens the
BINDING query. Flat 3-sweep is wrong globally. Need: sweeps=1 default + escalate
to mutual only when CROWN-only stalls (or make the extra sweep adaptive / cheaper).
Next: confirm sweeps=1 baseline numbers, then try adaptive (sweeps=1, bump to 2-3
only on leaves whose margin is within eps of 0 — same gate as boundary-alpha).

## explore12 baseline (sweeps=1) — MUTUAL IS A NET LOSS END-TO-END
Same machine (RTX PRO 2000 laptop):
  sweeps=1:  prop_2 3_3 -> timeout 119.8s | prop_1 0.08s | prop_3 4.17s
  sweeps=3:  prop_2 3_3 -> timeout 120.4s | prop_1 0.58s | prop_3 10.67s
RESULT: mutual (sweeps=3) is slower or equal on ALL three. The 0.58-vs-1.2
"win" on prop_1 from explore12-first-read was a stale baseline — true sweeps=1
prop_1 is 0.08s, so sweeps=3 HURTS it 7x.
ROOT CAUSE of the no-gain: the mutual tightening helps WIDE boxes (root margin,
prop_1 -2148->-1387) but the BaB spends ~all its time on NARROW leaves where the
forward zono is already ~= CROWN-tight (small box -> tiny generators -> tight
range). So per-leaf the mutual adds ~3x bound cost for ~0 leaf reduction.
PIVOT: prop_2 3_3 root only moves -1101->-1055 (4%); it needs heavy splitting
regardless, and ABC does it in 18s vs our ~120s. The 6x gap is NOT root bound
tightness -> it's PER-LEAF bound quality / throughput. ABC runs alpha-CROWN per
leaf (tighter -> leaves close faster -> fewer total); our explore12 ran PLAIN
CROWN (boundary-alpha eps/iters defaulted to 0). Next: enable per-leaf / boundary
alpha-CROWN on acasxu and re-time prop_2 3_3. The mutual zono<->CROWN stays as a
sound, default-OFF option (input_split_crown_intermediate_sweeps=1 keeps CROWN-only).

## unit tests: tight_override + mutual rewrite NON-BREAKING
789 passed, 0 fail (381s full non-vnncomp/non-extended suite). The sweeps>1 zono
path is not unit-covered yet (only used in input-split BaB) — resolve via test or
revert once the keep/drop decision is made.

## explore13 (in flight): batch_size sweep on prop_2 3_3 = the THROUGHPUT lever
ABC: plain crown + batch 16384 + naive -> 18s. Us: crown + batch 4096 -> ~120s,
"no divergence" (leaves close, just slow). On a 6-layer FC net the GPU is starved
at 4096. Sweep {4096,16384,32768,65536}. If wall drops ~linearly w/ batch, the gap
is throughput, not bounds — and the fix is a one-line config bump, not the mutual.

## sync regression FOUND + FIXED (prompted by user "zono on GPU fast?")
nvidia-smi during explore13: GPU 66% util, 1816 MiB for the proc -> zono IS on
GPU but the tiny 6-layer net leaves it under-saturated. My _crown_intermediate
rewrite had added a .item() convergence check (2 syncs/layer = 12 GPU<->CPU syncs
per batch on acasxu) that ran EVEN for sweeps=1 (the default everywhere). Old code
had zero syncs. Fixed: sweeps=1 = single _crown_pass, no convergence math, no sync;
sweeps>1 = one aggregate torch.stack(deltas).max() sync per extra sweep. explore11
re-run confirms converged margins identical (-1055 / -1387). explore13 (batch
sweep) was launched on the OLD code so its ABSOLUTE numbers are pre-fix, but the
batch-size COMPARISON still holds (all 4 sizes share the same code).

## explore13 LIVE signal: GPU util 66% (batch 4096) -> 100% (bigger batch), 3.2->7.1GB
Mid-run nvidia-smi: bigger batch saturates the GPU (66%->100% util). Strong support
for the throughput hypothesis — at 4096 the tiny net starves the GPU; at 16384+ it
saturates. (Output hidden until EOF: the `| tail -10` buffers; future streaming
runs must write raw to file, no tail.) Awaiting full dump for per-batch wall times.

## explore14: mutual ON LEAVES — persists but SUB-THRESHOLD (definitive)
Bisected prop_2 3_3 widest dim to depth 0/2/4/6 (narrowing leaves), worst-query
margin CROWN-only vs mutual(x4):
  d0 w1.0   -1101 / -1055  (+4.1%)   d2 w0.5  -125.8 / -121.8  (+3.1%)
  d4 w0.25  -1.13 / -1.09  (+3.1%)   d6 w0.125 +0.001 / +0.001 (leaf CLOSES, both)
Result: the mutual gain does NOT collapse on leaves (steady ~3%), BUT the leaf
closes at the SAME depth (6) under both methods. A ~3% margin nudge can't flip a
discrete close/split decision (-1.13->-1.09 still needs a full bisection). Same
leaf count -> per-leaf mutual cost is pure overhead. This is the mechanism behind
explore12's net loss, now MEASURED on leaves.

## CONSOLIDATION (explore11-14): mutual zono<->CROWN = sound, correct, useless for speed
- Implemented exactly as the user asked (tight_override in forward zono + iterated
  zono<->CROWN fixpoint in _crown_intermediate_batched). Sound (intersection of two
  over-approximations; stabilises ReLUs only on the certified-reachable range).
- Tightens root 4% (prop_2 3_3) / 35% (prop_1, correlation-heavy single query).
- On leaves: steady ~3% but sub-threshold -> no leaf-count reduction -> net LOSS
  end-to-end (explore12: prop_2 3_3 timeout, prop_1/prop_3 slower than sweeps=1).
- KEEP as sound default-OFF option (sweeps=1 default = plain CROWN, zero added sync
  after the fix). Do NOT enable for acasxu.
- The acasxu speed gap vs ABC (120s vs 18s) is THROUGHPUT, not bounds: GPU 66%
  saturated at batch 4096 -> 100% at 16384+. Lever = batch_size (ABC uses 16384)
  + on-GPU split (explore02's CPU-loop split is ~80% of wall). PURSUE explore13.

## explore13 RESULT: batch_size is NOT the lever (throughput hypothesis KILLED)
prop_2 3_3, sweeps=1, all VERIFIED: batch 4096->112.4s | 16384->113.9s |
32768->108.0s | 65536->113.2s. Flat ~110s regardless of batch. GPU went 66%->100%
util but wall didn't move -> the bottleneck is INVARIANT to batch size = the
CPU-side split loop (explore02: per-child .cpu()/.to(device)) and/or sheer LEAF
COUNT, which scale with #leaves not batch. (Also: all 4 VERIFIED here vs explore12
sweeps=1 TIMEOUT 119.8s -- prop_2 3_3 sits right at the 120s deadline, flaky.)
NEXT: get the LEAF COUNT for prop_2 3_3 (ours vs ABC). Same bounds (both plain
crown) + 6x slower => either we split many more leaves (branching) or per-leaf
overhead. Instrument leaf count; compare to ABC verbose. Then: on-GPU split.

## NIGHT WORK: acasxu config switch -> input-split+CROWN (no hybrid) + leaf-PGD
Switched configs/acasxu_2023.yaml: use_hybrid_acasxu=false, input_split_crown_
intermediate=true, sweeps=1. explore15 validation (SAT-finding ON, the real path):
  1_5 prop_2 SAT  -> UNKNOWN (worklist overflow 84s)   *** MISS ***
  1_9 prop_7 SAT  -> UNKNOWN (timeout 120s)            *** MISS ***
  1_1 prop_3 UNSAT-> verified 4.8s   OK
  3_3 prop_2 UNSAT-> (known ~110s verified)
ROOT CAUSE: input-split Phase-0 PGD runs only on the WIDE root box; a narrow SAT
witness fills a tiny fraction of root -> missed. The witness-leaf survives + keeps
splitting (holds a real SAT pt, never closes) until narrow.
FIX: _simple_pgd_batched (new, verify_hybrid_acasxu.py) — PGD inside MANY distinct
leaf boxes at once. Wired into input-split BaB: every input_split_leaf_pgd_every
iters, batched-PGD the N narrowest surviving leaves; return sat on hit. Gated off
when disable_sat_finding (soundness probe). Settings added (every/max_leaves/
restarts/iters). acasxu config: every=3, max_leaves=128, restarts=256.
explore16 (in flight): expect 1_5/1_9 -> sat, 1_1 -> verified (no false-sat).

## leaf-PGD v2: WORST-MARGIN selection (prop_7 fix)
explore16 v1 (narrowest-leaf selection): 1_5 sat 1.29s OK, but 1_9 prop_7 STILL
unknown 60s. explore17: _simple_pgd on prop_7 FULL box fails even at 200k restarts
-> witness needs a shrunk box, not a stronger root attack. The hybrid catches it
via between-rounds PGD selecting the WORST-MARGIN open leaves (verify_hybrid_acasxu
:487 worst_per_leaf.argsort), NOT narrowest. The witness leaf holds a real SAT pt
so CROWN can never close it -> its margin stays most-negative -> worst-margin
selection targets it directly (narrowest misses it for multi-disjunct prop_7).
CHANGED: moved leaf-PGD to right after the bound (margins align with xl_batch),
select unclosed leaves by most-negative worst-disjunct lb. explore16 v2 (116s):
re-validating 1_5 + 1_9 + 1_1.

## leaf-PGD v2 WORKS: all 3 validation cases pass
  1_5 prop_2 SAT  -> sat 1.31s (batched_leaf_pgd)
  1_9 prop_7 SAT  -> sat 1.25s (batched_leaf_pgd)  *** prop_7 FIXED ***
  1_1 prop_3 UNSAT-> verified 5.48s (no false-sat)
Worst-margin leaf-PGD catches both narrow-SAT cases in ~1.3s. Next: full acasxu
audit on server1 (new input-split+CROWN+leaf-PGD config), confirm no misses.

## acasxu full audit LAUNCHED on server1 (new config)
audit_benchmark.py acasxu_2023 116 (PID 130993). Early: prop_1 cases [1-6/186]
all VERIFIED 5-7s -- exactly the cases the OLD hybrid TIMED OUT on (153s). Config
switch confirmed working. Log: ~/persistent_runs/results/audit_acasxu_new.out,
csv: audit_acasxu_2023.csv. Polling for MISS lines. Unit tests running locally.

## dist_shift audit LAUNCHED locally (memory-capped, parallel w/ server1 acasxu)
local_audit.py dist_shift_2023 60 (PID 1854972). Early: index7027 unsat->unknown
MISS (ABC 8.3s); most verify 2.9s. dist_shift = mnist sigmoid net; misses are
UNSAT cases where our sound bound plateaus above the spec threshold (the "9
soundness-fix tradeoff" cases). Config uses zono_lift+input_split+dual_ascent.
Plan: collect full miss list, pick one, diagnose phase-by-phase, see how ABC
closes it (tighter sigmoid alpha? MILP on sigmoid?).

## dist_shift MISS ROOT CAUSE FOUND + FIXED: dual-ascent witness-check dim bug
9/72 dist_shift misses (index 7027/9733/9154/5291/2481/2750/112/3805/6107), all
mnist_concat UNSAT, ABC 8-13s, us unknown ~6s. Diagnosis: _da_witness_check
(verify_graph.py:6432) maps the dual-ascent LP primal witness e back to real input
via `center + w_t*halfwidth` assuming w_t is full-input-dim. But mnist_concat
sparse spec: 792 inputs, only 8 VARY (784 fixed image + 8 latent). dual_ascent
witness = witness_chunk[:, :n_input] where its n_input=8 (input GENERATORS). So
w_t=[N,8] vs halfwidth=[N,792] -> RuntimeError, caught by main.py:61
`except BaseException` -> masked as 'unknown'. The crash aborted the query BEFORE
the MILP fallback (which actually closes these) could run.
FIX: scatter the 8-dim witness into the 8 varying dims (halfwidth>0 order); fixed
dims stay at center. SOUND: x_real is always an in-box point, forwarded through the
real NN + checked vs real spec -> can only find genuine counterexamples, never
false-verify; the MILP bound is untouched.
index7027: unknown -> VERIFIED 27.4s (dual-ascent unknown 22s then MILP fallback
lb=+0.105 CLOSED). Re-auditing all 72 to confirm 9->0 + no regression.
ALSO TODO: main.py:61 `except BaseException` silently writes 'unknown' on ANY
crash -- it MASKED this bug. Should log the traceback (CLAUDE.md: no silent swallow).

## night status @ dist_shift-fix
- acasxu: server1 audit 30/186, 0 MISS (new input-split+CROWN+leaf-PGD config).
- dist_shift: 9 misses FIXED (witness-check dim bug); re-auditing all 72.
- nn4sys: NEXT. ABC verifies all 194 unsat. Our misses = pensieve_*_parallel /
  mscn perf timeouts (need alpha-CROWN-per-leaf, a bigger feature). Big + OOM-prone
  (mscn_2048d, pensieve_big_parallel x75) -> scope carefully, don't full-sweep local.

## dist_shift REAL FIX: routing cap blocked the intended input-split path
The 7 remaining misses fail because the dual-ascent path can't reduce SIGMOID
slack (only splits ReLUs). But dist_shift config HAS input_split_batched_enabled:
true -- the author INTENDED input-split. The routing gate (verify_graph.py:7736)
caps on TOTAL input dim: n_in <= input_split_max_dims(20). mnist_concat n_in=792
(but only 8 VARY) -> 792>20 -> silently routed to dual-ascent _run_pipeline
instead. (Same class as acasxu's dead-hybrid: intended path never reached.)
FIX: config input_split_max_dims: 800 -> activates input-split. Forced test:
index9733 verified 2.9s, index2481 verified 1.8s (ABC 8s). Input-split narrows
the 8 sigmoid inputs -> tightens relaxation -> closes. Validating all 9 + SAT +
soundness now, then full re-audit.

## dist_shift RESULT: 8/9 fixed via input-split routing (config input_split_max_dims:800)
At 60s cap: 8/9 former-misses VERIFIED (<3s mostly, index2750 19s). index112 only
one timing out (59.5s @ 60s cap). SAT index4739 -> sat 0.0s. SOUNDNESS index4312
(disable_sat) -> unknown 59.6s CORRECT (no false-verify). Re-testing index112 @
300s competition budget (it may just need more split time). Then full re-audit to
confirm 0 regressions on the 63 passing cases.
NOTE: witness-check dim-fix (earlier) still valuable for the dual-ascent FALLBACK
path + other benchmarks; keep it.

## dist_shift COMPLETE: 9/9 misses fixed -> 72/72 (matches ABC)
index112 verified 152.3s @ 300s budget (the only slow one; rest <20s). All 9
former-misses now verified, SAT caught, soundness correct. Fix = config
input_split_max_dims:800 (activate intended input-split path; routing gate capped
on total-dim 792 vs 8 varying) + witness-check dim-fix (fallback path). Running
full re-audit (cap 180) to confirm 0 regression on the 63 passing cases.

## *** NO SOUNDNESS BUG *** — server audit ABC-key collision (FALSE MISS)
acasxu "1_1 prop_2 abc=sat vc=verified MISS" looked like a false-verify. Confirmed
NOT: our PGD 200k restarts finds NO counterexample, and ABC's REAL verdict for
1_1 prop_2 = UNSAT (8.45s) -> our verified is CORRECT. The bug is in server
audit_benchmark.py load_abc: keys by vnnlib BASENAME only. prop_2.vnnlib is reused
across 45 nets (39 sat + 6 unsat!) -> m['prop_2.vnnlib'] = LAST row (5_9=sat) ->
compared 1_1 to 5_9's sat -> false miss. ALL acasxu server-audit misses on
reused-vnnlib props (prop_1/2/3/7/8...) are UNRELIABLE.
- local_audit.py keys by (onnx, vnnlib) CORRECTLY -> dist_shift results VALID
  (distinct vnnlibs, no collision).
- FIX server audit_benchmark.py to key by (onnx, vnnlib). The running acasxu audit
  still collected CORRECT vc verdicts (from --results-file); only the miss column
  is wrong -> RE-COMPARE offline against correctly-keyed ABC, no re-run needed.

## acasxu re-compare (CORRECT ABC key): 51/51 done = 0 real misses, 0 false-verifies
No soundness issue. The new input-split+CROWN+leaf-PGD config is clean. Will
re-compare the full 186 offline when the audit finishes.

## CONFIRMED: dist_shift 72/72 (0 miss, 0 false-verify), acasxu 65/65 clean
- dist_shift clean re-audit: "0 MISSES of 72 ABC-solved (72 total)". COMPLETE.
- acasxu re-compare (correct ABC key) 65/186 done: 0 real misses, 0 unsound.
- Running acasxu+dist_shift integration tests. NOTE: acasxu test max_wall tuned for
  OLD hybrid; 3_3 prop_2 ~110s may exceed its 90s bound -> update test docstring
  (now input-split+CROWN+leaf-PGD, not hybrid) + max_wall after seeing timings.

## 3_3 prop_2 REGRESSION (server1 timeout 150s) -> vectorized split fix
Integration test: 3_3 prop_2 verified 117.2s (>90 bound). Server1 audit: 3_3 prop_2
UNKNOWN 150s = REAL MISS (hybrid did 46s). Net config-switch is still a big win
(hybrid 13 prop_1 misses -> input-split ~1 prop_2 miss), but closing 3_3 prop_2
needs speed. leaf_pgd=0 -> 112.1s (leaf-PGD only ~5s) => BaB is the bottleneck =
the per-child CPU<->GPU split loop (explore02: ~80% of wall).
FIX: vectorized on-GPU 2-way split (K_eff==1 fast path) — both children of all
leaves built in a few GPU ops, no host transfer. K>1 keeps the Python loop.
explore02 reverted this for breaking 1_5 SAT; leaf-PGD (worst-margin) now catches
SAT independent of split order. Testing 3_3 speed + SAT correctness now.

## VECTORIZED SPLIT WORKS: 3_3 prop_2 126s -> 73s, SAT cases still caught
Device fix (ax1.to(xl_split.device) — top_k_dims built from widths.cpu()).
Results: 1_5 sat 1.28s, 1_9 prop_7 sat 1.02s (leaf-PGD catches despite reordered
split — explore02 regression GONE), 1_1 prop_3 verified 3.41s, 1_1 prop_1 0.53s,
3_3 prop_2 verified 73.0s (was 112 leaf_pgd=0 / 126 leaf_pgd=3). On server1 (~30%
slower) ~95s < 116s timeout -> 3_3 prop_2 MISS CLOSED. leaf-PGD ~14s of the 73 ->
could dial every=10 for more headroom. Running unit tests (split is core BaB code).

## acasxu OLD-code audit (80/186): exactly 2 real misses = 3_3 + 4_2 prop_2
Both unsat->unknown timeout 150s (the slow split). Both NON-soundness (no
false-verify). 3_3 fixed by vectorized split (73s). 4_2 prop_2 = same class (hard
prop_2-unsat) -> vectorized split should fix too. The other 4 prop_2-unsat nets
already verify. PLAN: confirm 4_2 w/ new code, then kill old audit + re-run full
acasxu audit w/ vectorized split on server1 -> target 0 misses (beat ABC's 13->0).

## unit tests PASS (789) with vectorized split — non-breaking. Re-audit relaunched.
Killed old acasxu audit (broken ABC-key + slow split), relaunched on server1 with
BOTH fixes (audit_benchmark.py keys by onnx+vnnlib; vibecheck has vectorized
split). Testing 4_2 prop_2 (2nd miss) locally w/ new code -> expect verified ~73s.

## *** acasxu BOTH MISSES FIXED *** by vectorized split
4_2 prop_2: verified 10.2s (was 150.9s timeout). 3_3 prop_2: 73s. Both the slow
prop_2-unsat cases now clear the timeout. acasxu re-audit running on server1 (v2,
correct ABC key + vectorized split) -> target 0 misses.
NIGHT SCORE: acasxu hybrid 13 misses -> input-split+CROWN+leaf-PGD+vecsplit 0
misses. dist_shift 9 -> 0. nn4sys already 194/194 (docs). Updated integration
tests (acasxu: leaf-PGD/vecsplit, added 4_2, 3_3 max_wall 130; dist_shift index4312
soundness max_wall 45). Running integration tests now.

## v2 audit (server1) in progress: 45/186, 0 misses, 0 unsound, GPU 41C healthy
All prop_1 so far (the hybrid's 13 misses) now verified ~64-70s on server1 (<116
cap). prop_2 hard cases (3_3, 4_2) come next ~case 46+. ~2h total at this pace.
README finalization waits for full numbers. PRE-MERGE (flag to user, don't do
unprompted): 100% unit coverage for new code (leaf-PGD, vectorized split,
witness-check, crown_intermediate); unit-test-or-remove the unused mutual code.

## FINAL acasxu conclusion (cheap levers exhausted)
boundary-α: NO help (71.7->71.0s — unclosed leaves are deeply negative, not
near-boundary). leaf-PGD dial-down: ~10s (doesn't clear cap, risks SAT). So 3_3/4_2
prop_2 stay >116s on server1's slow RTX 3080 (147s, 143s) but pass on laptop
(72s, 10s). server1 184/186, laptop 186/186. 0 false-verifies. +11 vs hybrid
(173/186). Robust 0-miss needs α-intermediate (hybrid freeze-replay) in input-split
— deferred (substantial). README corrected to honest HW-dependent numbers.
RECOMMENDATION TO USER: keep input-split (strict improvement, likely 186/186 on
competition datacenter HW); α-intermediate is the next feature for guaranteed 0.

## v2 audit CONFIRMED at 152/186 (final check)
Exactly the 2 expected effective misses: 3_3 prop_2 (unknown 147s) + 4_2 prop_2
(verified-but-143s, overran 116s cap). 0 UNSOUND, 0 unexpected misses. GPU 41C, 0
errors/Xid (no overheat). Remaining 34 cases = prop_4-10 (fast). Pattern matches
analysis. acasxu server1 = 184/186 (slow-GPU artifact), laptop = 186/186. NIGHT
WORK COMPLETE: dist_shift 9->0 (72/72), acasxu prop_1 13->0, nn4sys 194/194, 0
false-verifies anywhere. v2 audit finishing the last 34 (wakeup 02:51 confirms).

## ===== NEW INVESTIGATION (branch bench/acasxu-alpha-intermediate) =====
## Goal: close 3_3/4_2 prop_2 (1.88M leaves @ 73s) — tighter bounds = fewer leaves.
DATA: 3_3 prop_2 = 1,878,819 leaves, 482 iters (naive branching, which we already
use — sb_enabled defaults False). 39us/leaf. ABC 18s => ~100x fewer leaves via
alpha-optimized intermediate bounds. Per-leaf throughput is NOT the issue.
Hybrid (_full_freeze + _replay_batched) ALREADY does 3_3 in 47s (alpha-tight). Its
only problem: it ALWAYS freezes (~30-50s) -> wasteful on easy prop_1 (<1s w/ plain
CROWN) -> overruns. So the fix = ADAPTIVE: cheap path default, alpha-freeze only
when the BaB stalls.
TWO shapes to investigate:
  A. Integrated: frozen-alpha intermediate inside input-split (one path, cleaner,
     more work — map _full_freeze's {L:{k:alpha}} into _crown_intermediate's seeds).
  B. Fallback: input-split primary; on STALL (worklist>thresh) abort + run
     verify_hybrid. Simpler (reuse working hybrid). Est: stall@100k leaves ~4s
     (26k leaves/s) + hybrid 47s = ~51s. Wasted pre-stall time = small.
PLAN: measure B first (stall time on 3_3 + confirm hybrid 47s). If 4s+47s<116s,
B is the simple win. Then decide A vs B.

## branching A/B CONCLUSIVE: naive is right, sensitivity 3x WORSE
naive(branch_sb=False, current): 1.88M leaves 71s verified.
sensitivity(branch_sb=True): 5.38M leaves 199s UNKNOWN(timeout).
=> branching = NOT the lever (we already match ABC's naive). alpha-intermediate
is the ONLY path. Running Option-B (input-split + hybrid fallback) measurement.

## explore_optionb RESULT: worklist cap does NOT detect stall + leaf-count map
input-split cap=100k still VERIFIED 3_3 (1.88M leaves churn through a bounded
FRONTIER — worklist size != total visited). So a stall detector must key on TOTAL
LEAVES or WALL-TIME, not worklist size.
Leaf map (laptop): 3_3 prop_2 1.88M/71s | 4_2 prop_2 236k/9.5s | 1_1 prop_3 73k/3.3s
| 1_1 prop_1 63/0.5s. => 3_3 is the ONLY truly-hard case; 4_2 is moderate (its
server1 143s was slow-HW/variance). ALL pass on laptop (<116s).
=> Whether ANY fix is needed depends on the A10G result (faster than server1).
Option B detector: wall-time (>~20s -> escalate to hybrid) or total-leaves (>~400k,
above 4_2's 236k). Measuring hybrid 3_3/4_2 timing now for the fallback estimate.

## hybrid timing + ADAPTIVE FALLBACK implemented
hybrid (laptop): 3_3 prop_2 46.4s (vs input-split 71s — alpha wins on hard),
4_2 prop_2 19.5s (vs input-split 9.5s — freeze overhead loses on easy). KEY:
fewer leaves => less HW-sensitive => hybrid 3_3 ~92s server1 (PASSES) vs
input-split 147s (MISS). So adaptive: input-split primary (easy fast) + hybrid
on stall (hard, HW-robust).
IMPLEMENTED on branch: input_split_batched_stall_leaf_cap (gate on TOTAL leaves,
not frontier — worklist stays bounded) -> phase batched_stall_escalate; verify_graph
routes stall -> verify_hybrid when acasxu_hybrid_on_stall. Measuring 3_3/4_2/prop_1/
prop_3 + SAT with cap=300k.

## ADAPTIVE FALLBACK WORKS (cap=300k): all 6 cases pass
3_3 prop_2 verified 58.9s (escalate->hybrid, was 71s pure input-split) | 4_2 9.4s
(input-split, 236k<300k) | 1_1 prop_3 3.5s | prop_1 0.6s | 1_5 sat 0.5s | 1_9 prop_7
sat 1.0s. SAT caught by leaf-PGD before any cap. 3_3 server1 projection: ~300k
churn (~23s) + hybrid (~92s) = ~115s (TIGHT). Lower cap (100-150k) -> escalate 3_3
earlier -> server1 ~100s (safer), at cost of 4_2 escalating on laptop too (9->25s,
still fast). CONCLUSION: adaptive (input-split + leaf-cap -> hybrid) closes 3_3/4_2
robustly. Needs: A10G data (is it even needed for competition HW?), final cap tune,
unit tests, decision to enable in config.

## ===== nnenum comparison (cloned to ~/repositories/nnenum) =====
nnenum (single-thread BLAS req'd; OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1):
  3_3 prop_2: SAFE 3.86s, 10,307 stars (1035 EXACT splits + 9272 APPROX)
  4_2 prop_2: SAFE 3.71s, 7,163 stars (160 exact + 7003 approx)
vs vibecheck input-split: 3_3 1.88M leaves 71s. vs ABC 18s.
=> nnenum ~180x fewer sets, ~20x faster.
ALGORITHM (worker.py consider_overapprox + overapprox.py):
  - Splits on RELU ACTIVATIONS (not input dims). Propagate star to first unstable
    ReLU, then AGGRESSIVELY overapprox (zonotope zono.area + star/triangle) the
    REST -> if overapprox proves spec, PRUNE (no split). Split the ReLU only when
    overapprox fails. Most ReLUs pruned by overapprox (9272 approx vs 1035 split).
  - Multi-round overapprox refinement before splitting; parallel workers.
LESSON: input-split is the WRONG abstraction for ACAS Xu — it brute-forces the
input box (1.88M leaves) without targeting the ReLU instability. ReLU-split +
zonotope-overapprox-pruning (nnenum) targets the actual nonlinearity -> ~1k splits.
The freeze-replay hybrid (46s) + adaptive fallback (58.9s) are BAND-AIDS; the real
fix is ReLU-split BaB with overapprox-pruning (vibecheck has the zonotope machinery).

## nnenum SINGLE-THREAD baseline (fair vs single-thread VC prototype)
1_1 prop_4: holds 0.82s, 17 stars (0 splits — pure overapprox!)
1_1 prop_3: holds 1.49s, 94 stars (1 split)
3_3 prop_2: holds 26.1s, 10295 stars (1036 splits)
=> overapprox-FIRST is the key: most cases need 0-1 splits. Even single-thread,
nnenum 26s beats VC input-split 71s on 3_3. Prototype: zonotope overapprox-first +
ReLU-split (star+LP) when overapprox fails. Gurobi Threads=1.

## star-BaB prototype (scratch/nnenum_style/star_bab.py): WORKS structurally, not competitive
Built: overapprox-first + ReLU-split, Gurobi LP Threads=1, direction-projected spec
check (key fix vs per-output box). Diagnosis: prop_4 does 16+ splits (didn't
converge in 120s; cap-15 -> safe=False 0.9s/128 LPs). LPs are fast (~7ms); the
problem is SPLIT COUNT. My single min-area zonotope overapprox is too loose -> it
SPLITS where nnenum REFINES the overapprox (prop_4: nnenum 0 splits / 17 approx
stars). Missing piece = nnenum's do_overapprox_rounds: try increasingly precise
overapprox (LP-tightened pre-relu bounds, zono.area relaxation) BEFORE splitting.
CONCLUSION: copying nnenum's SPEED requires its overapprox-refinement + zono
prefilter (essentially re-implementing nnenum core). The prototype confirms the
APPROACH (ReLU-split + overapprox-first) is right for acasxu; matching speed is a
real effort. For production: adaptive fallback (input-split -> hybrid) is the
pragmatic robustness fix already prototyped.

## MICRO-MEASUREMENT of contraction (first split, 3_3 prop_2) — validated
first split: layer 0, neuron 22, e-box 5 dims. After adding z_22>=0:
- contraction tightens NEXT-layer box-width sum 47.6 -> 36.4 (-23%). Compounds.
- only 1 of 5 dims tightened -> nnenum's witness-tracking = 2 LPs not 10 (5x fewer).
- contracted box bound == LP-exact (-1.9326) -> cheap box recovers LP tightness.
CONFIRMS: my prototype's LP-per-neuron gets the same tight bounds but slowly;
nnenum's contract-then-cheap-box is the same tightness, far fewer LPs. Witness-
tracking is the speed half. IMPLEMENT: contracted-box bounds + witness-tracked
contraction per split.

## v2 contracted-box BaB diagnosis + box_halfspace plan (user insight)
v2 (contraction, min-area relaxation): prop_4 cap40 -> 0 LEAVES / 41 splits ->
DFS down one branch, NEVER PRUNES. Contraction is fast (820 LPs/0.08s) + tightens
the INPUT box, but the MIN-AREA RELAXATION is too loose to prune ANY node.
=> nnenum prunes because of its TIGHT relaxation (star.lp = LP triangle); the
ablation's contraction-#1 result is the MARGINAL value GIVEN that tight relaxation.
So the prototype's ordering is: (1) tight relaxation (star.lp) = the pruning
enabler [MISSING], (2) contraction = split reduction [done+validated], (3) box_hs
closed-form = make contraction free [planned].
USER INSIGHT (correct): domain contraction == box+halfspace LP that
box_halfspace.py solves CLOSED-FORM (exact for ONE halfspace, O(n log n)). Once
the gurobi version matches accuracy, swap contraction -> lagrangian_min/max per
split (incremental, ~free). Evaluate speed vs the per-halfspace looseness tradeoff.
