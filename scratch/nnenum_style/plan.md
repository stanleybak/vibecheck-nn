# nnenum-style star BaB for acasxu long timeouts — plan.md

Goal: beat vibecheck's input-split BaB on the acasxu long-timeout cases
(3_3 / 4_2 prop_2, currently ~73 s vectorized). Reference: nnenum single-thread
42.7 s / 11863 stars on 3_3 prop_2.

## Frozen targets (nnenum single-thread, measured)
- 1_1 prop_4: SAFE 0.82 s, 17 stars, **0 exact splits**
- 1_1 prop_3: SAFE 1.49 s, 94 stars, 1 split
- 3_3 prop_2: SAFE ~26–42 s, ~10k stars

## Measured nnenum ablation (3_3 prop_2, single-thread)
- baseline 42.73 s / 11863 stars
- init-overapprox OFF (no_quick): 43.01 s / 11863 — **ZERO effect** (irrelevant)
- no_starlp (triangle off): 58.10 s / 37473 — refinement modest (~1.4×)
- area_only: 60.14 s / 73805 — 6× stars, 1.4× time
- **no_contract_lp: TIMEOUT >150 s / 104239 — DECISIVE.** LP-contraction
  prefilter is the #1 lever.

## Explorations
- **v1 (star_bab.py)**: star + min-area free-generator relaxation, LP bounds.
  Too loose to prune → timed out. Lesson: free-generator relaxation insufficient.
- **v2 (star_bab_v2.py)**: contracted-box BaB (cheap box bounds + per-dim LP
  contraction of e-box after each split). 0 leaves / 41 splits — box bounds even
  after contraction too loose to prune. Lesson: need a tighter *bound*, not just
  a tighter box.
- **micro_contract.py**: first-split contraction micro-measure on 3_3 prop_2.
  23% downstream tightening, 1/5 dims tightened, contracted box == LP-exact.
  Lesson: contraction recovers LP-exact box but the box bound itself is the limit.
- **v3 (star_bab_v3.py)**: star.lp TRIANGLE relaxation (convex hull as domain
  constraints) + branch-most-ambiguous. **CORRECT verdicts** (prop_4/prop_3 SAFE
  after the descend fix) but **310 splits on prop_4 vs nnenum's 0**, 25.7 s.

## DECISIVE finding (2026-06-01)
Micro-check of v3's **root** triangle-LP spec lower bounds for 1_1 prop_4:
`[-0.070, -0.080, -0.080, -0.135]` — **all negative**. The standard triangle
relaxation, even with LP-exact intermediate bounds (the tightest single-pass
convex relaxation), **cannot prove prop_4 at the root**.

→ nnenum's "0 exact splits" is NOT the triangle relaxation being tighter. It is
**counterexample-guided overapprox refinement** (the 17 "stars" = a handful of
guided neuron refinements that go beyond the convex relaxation). My v3 does the
same kind of case-work but **un-guided** (splits the most-ambiguous neuron
blindly) → 310 splits instead of ~a dozen guided ones.

**Two independent levers, ranked by measured impact:**
1. **Split SELECTION** (counterexample-guided, like nnenum) — closes the 310-vs-0
   gap. This is the accuracy/split-count lever.
2. **Bound SPEED** (replace Gurobi LP with box_halfspace dual ascent) — the user's
   thread: same box+halfspace formulation as dual ascent, K-step tunable, should
   match Gurobi with enough iterations. This is the wall-clock lever per split.

## Dual-ascent-vs-Gurobi (explore01 / explore02, 2026-06-01)
User hypothesis: dual ascent (default K=1) matches Gurobi with enough iters.
Tested on 6 REAL prop_4 star-LP nodes (n=29–72 dims, M=52–134 halfspaces),
comparing the spec-direction lower bound.

- **explore01 (steepest subgradient + exact line search** — replicates
  `_batched_dual_ascent`'s η-breakpoint step):
  - **Incremental rc (production scheme): OVERSHOOTS Gurobi** — negative gaps
    (dual lb > LP optimum) by ~0.005 at K=200. That is a weak-duality / SOUNDNESS
    violation: the clamp(λ≥0) desyncs the incrementally-updated rc from w_a+Cᵀλ,
    so g is no longer a valid dual value. **The apparent "matches Gurobi" is this
    unsound overshoot, not convergence.**
  - **rc resynced from λ each iter (sound): STALLS** at gap 0.005–0.014; does NOT
    reach Gurobi.
- **explore02 (coordinate / Gauss-Seidel, exact 1D per halfspace — the
  "one-halfspace-exact" primitive):** also STALLS (gap 0.001–0.015), even at
  2680 solves. Sometimes better than steepest (node 2), sometimes worse (node 1).
- **Cost:** sound dual ascent needs 10–25 ms for many cycles; **Gurobi solves
  these exactly in 0.8–1.8 ms.** These acasxu LPs are TINY (5 inputs) — Gurobi is
  already optimal AND faster. Dual ascent only pays off batched on BIG problems
  (cifar100: thousands of neurons on GPU), where it is already used.

**CONCLUSION: dual ascent is the WRONG speed lever for acasxu.** Sound first-order
ascent stalls short of Gurobi here, and even at break-even accuracy it is slower
because the LPs are tiny. Not a win for this benchmark.

⚠ **Production flag:** the measured overshoot means `_batched_dual_ascent`'s
incremental-rc + clamp scheme can return `best_g` ABOVE the true LP optimum. If
`best_g>0` is used to certify "safe", a spurious overshoot could false-verify.
Production splits on no-progress so it may be masked — but this needs a dedicated
check (construct a node where the true optimum is small-negative and confirm
best_g never crosses 0 spuriously). Logged for a separate soundness audit.

## Consolidated picture for acasxu 3_3/4_2 (the long timeouts)
1. nnenum's edge is **counterexample-guided overapprox refinement**, NOT the
   triangle relaxation's tightness (root triangle-LP can't prove prop_4: lb<0)
   and NOT contraction speed.
2. Dual ascent won't speed the small acasxu LPs (above).
3. → The only real lever to match nnenum's split count is **counterexample-guided
   split selection** on top of v3's (correct) triangle relaxation. Whether that
   beats vibecheck's existing vectorized input-split (73 s on 3_3 prop_2) is the
   open question. The current production path already verifies these (just slow);
   the nnenum port is a research bet, not a clear win.

## TRACED nnenum directly (2026-06-01) — corrects two earlier WRONG claims
Ran nnenum single-thread with a `do_first_relu_split` counter (its own venv
~/repositories/nnenum/.venv, needs OPENBLAS/OMP=1):
- **1_1 prop_4: SAFE, 41 stars, 40 ReLU splits, 0 LPs, 0.46 s.**
- **1_1 prop_3: SAFE, 169 stars, 168 ReLU splits, 123 LPs, 1.52 s.**

**CORRECTION 1 — "0 splits" was a mis-recollection.** nnenum DOES branch-and-bound
with ReLU case-splitting (40 / 168 splits). The verbose "Total Stars: N
(0 exact, N approx)" means: every BnB node was closed by OVERAPPROXIMATION
("approx"), 0 needed full EXACT enumeration to a concrete leaf. `finished_stars`
= BnB nodes; `finished_approx_stars` (worker.py:259) increments when overapprox
proves a node's subtree safe. Splits still happen to build the tree.

**CORRECTION 2 — no counterexample-guided refinement for proving SAFE.** The CE
machinery (`test_abstract_violation`, overapprox.py:183) minimizes the violation
star, runs it through the REAL network, checks for a real violation — it is a
SAT-FALSIFICATION step (find a concrete counterexample), NOT a refinement that
helps prove UNSAT. Proving safe = the multi-round overapprox escalation only.

**The rounds (Settings.OVERAPPROX_TYPES):**
1. `[zono.area]`  2. `[zono.area, ybloat, interval]`  3. `[…, star.lp]`.
Cheap zono first; `star.lp` (the triangle, = my v3) only on the last round.
prop_4 closed with ZERO LPs (pure zonotope rounds, never reached star.lp).
prop_3 used 123 LPs (star.lp kicked in for harder nodes). Default
CONTRACT_ZONOTOPE_LP=True (the decisive ablation lever on the hard 3_3 case).

## Why gurobi-v3 doesn't match nnenum (the REAL gap, measured)
v3 vs nnenum split counts: prop_4 **310 vs 40** (~8×); prop_3 **544 vs 168** (~3×).
Two compounding causes — NOT the LP solver, NOT CE refinement:
1. **Split SELECTION.** nnenum uses SPLIT_ONE_NORM ordering; v3 splits the
   most-ambiguous neuron (min(hi,-lo)) and descends every unstable neuron. ~3–8×
   more splits.
2. **Overapprox COST per node.** nnenum closes most nodes with the CHEAP zonotope
   (0 LPs for prop_4), escalating to star.lp only when zono fails. v3 pays a
   Gurobi LP at every node. So v3 is both more splits AND more $/split.

→ To match nnenum: (a) cheap zono overapprox FIRST, LP only on hard nodes
(multi-round escalation), and (b) better split ordering. The triangle-LP alone
(v3) is the most expensive rung used least often by nnenum.

## v4 (zono-first / LP-on-hard escalation, 2026-06-01)
Built the multi-round structure: cheap min-area zono rung -> triangle-LP rung ->
split with closed-form single-halfspace box contraction (contract_box_1hs, O(n),
0 LP). CORRECT (prop_4/prop_3 both SAFE). But SLOWER than v3:
- prop_4: 583 splits, zono_close=80, lp_close=504, 36 s  (nnenum 40 splits / 0 LP)
- prop_3: 827 splits, zono_close=92, lp_close=736, 64 s  (nnenum 168 / 123 LP)

DIAGNOSIS: the cheap zono closes only ~14% of nodes (80/584); nnenum closes ~100%
of prop_4 with pure zono. Worse, split selection uses the LOOSE zono box bounds
(more neurons look unstable -> more splits than v3's LP-bound selection). So the
escalation backfires unless the cheap rung is strong.

The measured fix (from the earlier ablation): nnenum's cheap rung is NOT a single
min-area zono — it INTERSECTS zono.area + zono.ybloat + zono.interval (shared,
progressively tightened layer bounds). Ablation `area_only` was 6x MORE stars on
3_3 prop_2 -> the multi-type intersection is a ~6x split lever. v4 uses area only.
NEXT: add ybloat+interval with shared intersected bounds to v4's cheap rung.

## CONSOLIDATION (2026-06-01) — three levers tested, gap NOT closed
v4 (zono-first / LP-on-hard, the requested rebuild) is CORRECT (SAFE) but does
~10x nnenum's splits and is slower than v3:

| variant            | prop_4 splits / time | prop_3 splits / time |
|--------------------|----------------------|----------------------|
| nnenum (reference) | **40 / 0.46s**       | **168 / 1.52s**      |
| v4 base            | 583 / 36s            | 827 / 64s            |
| v4 + joint contract| 463 / 30s (5965 cheap-only) | 36960 cheap-only |
| v4 + spec-aware    | 531 / 35s            | 812 / 64s            |

Measured root cause: single min-area zono is **47x looser than truth** at the root
(sampled). nnenum uses the SAME relaxation but closes in 40 splits — so the gap is
NOT the root bound, NOT contraction (joint made it worse), NOT split selection
(spec-aware barely moved it). By elimination + the earlier ablation (`area_only`
= 6x stars), the remaining lever is nnenum's **multi-type zono prefilter**
(area ∩ ybloat ∩ interval with shared progressively-tightened layer bounds, per
run_overapprox_round). Replicating it faithfully = porting a good chunk of
nnenum's overapprox engine (3 coupled zonos, shared bound feedback).

**Bottom line:** my prototypes (v3 25-50s, v4 35-64s) are CORRECT but 25-40x
slower than nnenum because they use one weaker abstraction. Matching nnenum needs
its multi-type prefilter. AND: these acasxu cases are ALREADY solved in production
(vectorized input-split ~73s < 116s VNNCOMP timeout) — so this is a SPEED
nice-to-have, not a correctness gap. Recommend surfacing the decision rather than
porting more of nnenum.

## FIRST-SPLIT INSTRUMENTATION (2026-06-01) — isolates the gap exactly
Compared v4 vs nnenum split-by-split on prop_4 (both single-thread):
- nnenum first split: ReLU layer 1, neuron 6, pre-ReLU [-0.024210, 0.025270],
  **6 unstable neurons** at that layer.
- v4 RAW layer-1 (li=0): **same 6 unstable** {6,8,23,41,44,48}; neuron 6 =
  **[-0.024210, 0.025270] — IDENTICAL to 6 decimals.**
- Contraction matches: v4's later neuron-6 bound [-0.0242, 0.0046] shares nnenum's
  exact lower bound; the tighter upper is just because v4 split it 4th (after
  contracting 3 others). Box contraction is exact.

So abstraction + contraction + first-split bounds ALL MATCH. The ONLY divergence:
- Split ORDER (nnenum one-norm picks n6 first; v4 spec-aware picks n23) — tested,
  worth <10%.
- **Overapprox-to-output PRUNING strength.** nnenum: 40 splits, **0 LPs** (pure
  multi-type zono prunes subtrees after a few splits). v4: 531 splits, **312912
  LPs** (single min-area zono can't prune -> falls through to the LP rung at almost
  every node). This is THE lever, now isolated by elimination with matching bounds.

→ CONFIRMED: implement nnenum's multi-type zono overapprox-to-output
(area ∩ ybloat ∩ interval, shared progressively-tightened layer bounds). Not a bug
hunt anymore — the cheap rung just needs nnenum's stronger (intersected) zono.

## ROOT-BOUNDS COMPARISON (2026-06-01) — REVERSES the "weak zono" hypothesis
Multi-type rung barely helped (zono_close 82->89, splits 583 either way). So I
compared root overapprox OUTPUT bounds directly:
- nnenum single zono.area:  out0 [-1.92, 3.03], out3 [-1.97, 2.84]
- v4 multizono (intersect): out0 [-0.16, 0.73], out3 [-0.50, 0.94]
- true (sampled):           out0 [ 0.16, 0.26], out3 [ 0.09, 0.28]
(My single-area zono == nnenum's [-1.9,3.0] exactly; the multizono is the 5x
tighter one. Both SOUND — contain the true range.)

**My zono bounds are 5x TIGHTER than nnenum's, yet I split 583 vs nnenum's 40.**
So the gap is NOT bound tightness and NOT the prefilter. It is the SEARCH STRATEGY:
1. nnenum makes each split EXACT in the LP star (the neuron is fixed exactly);
   my v4 approximates the split with a per-coordinate box contraction, so my
   post-split overapprox tightens less per split.
2. nnenum splits the single best neuron (one-norm) and re-runs the full overapprox;
   my per-layer descent + loose-zono split selection grows a ~10x bigger tree.
3. v4 (cheap-first) is actually WORSE than v3 (LP-only: 310 splits) because the
   cheap rung's looser bounds drive worse split SELECTION.

## FINAL CONSOLIDATION
Bounds are NOT the problem (v3/v4 match or beat nnenum's). The 10-40x slowdown is
nnenum's SEARCH STRATEGY: exact-split star representation (splits as LP rows, not
box contraction) + one-norm split order + overapprox-guided pruning of the first
unstable layer. Matching it = reimplementing nnenum's core search, not a tweak.
AND these acasxu cases already pass in production (vectorized input-split ~73s <
116s timeout). RECOMMENDATION: stop the prototype, keep v3/v4 + this plan as the
documented research record, move to the regular-track queue. The one production-
relevant artifact is the gated dual-ascent contraction option (keep for later).

## FIRST-2-SPLITS TRACE + LP COUNTING (2026-06-01) — two big findings
**1. nnenum UNDERCOUNTS LPs.** Hooked glp_simplex: prop_4 reports total_lps=0 but
actually solves **2191 LPs**. nnenum is NOT avoiding LPs — `update_bounds_lp` runs
at every post-split node (use_lp = star_has_splits). Its speed is FAST GLPK
(warm-started simple LPs), not LP-avoidance. So the "0 LP" that drove the whole
"my zono is too weak" theory was a mirage.

**2. v3 (NOT v4) already matches nnenum's first 2 splits exactly:**
- nnenum: split1 L1 n6 (branching=6); split2 L1 n41 (branching COLLAPSES 6->1,
  LP proves 4 stable); split3 L3 (branching=4, picks 15).
- v3:     split1 L1 n6 (unst=6);     split2 L1 n41 (unst=1, SAME collapse);
  split3 L3 (unst=2, picks 33).
v3's LP-tight neuron_bounds selection reproduces the branching collapse. v3 is even
TIGHTER at L3 (2 vs 4 unstable). v4's cheap-zono selection was a DETOUR (loose
bounds -> worse). nnenum contracts the domain box (dim2 0.318->0.346); v3 uses the
equivalent box+C,d LP, same bounds.

## REVISED diagnosis (supersedes "search strategy is unbridgeable")
v3 already matches nnenum's split STRATEGY for the first 2 splits. The remaining
gap is TWO things, both measurable:
1. **Tree shape past split 2**: v3 310 splits vs nnenum 40 (~8x). v3 is tighter
   per-node yet grows a bigger tree -> diverges deeper; needs a full split-sequence
   diff to localize.
2. **Per-LP cost**: nnenum GLPK warm-started (~0.2ms/LP, 2191 LPs / 0.46s);
   v3 Gurobi cold (~80ms/split). ~7x per-split. This is most of the 25s vs 0.46s.

→ v3 is the right base. Next: deep-trace the v3 vs nnenum split SEQUENCE to find
where the tree blows up, and consider GLPK/warm-start for the per-LP cost.

## BLOWUP LOCALIZED (2026-06-01) — per-layer split histogram
            L1   L3   L5   L7    L9   total
- nnenum:   7    21   12   0     0    40
- v3:       10   40   63   149   48   310
**nnenum proves SAFE after layers 1,3,5 — NEVER splits L7/L9. v3 descends into
L7/L9 for 197/310 splits (64%).** So v3's overapprox FAILS TO PRUNE at the
L5-resolved nodes where nnenum's succeeds. That is the entire gap.

Counterintuitive: v3's ROOT triangle-LP bound (-0.07) is much TIGHTER than nnenum's
root zono ([-1.9,3.0]), yet v3 prunes WORSE deep in the tree. Hypotheses to test:
(a) v3's per-layer triangle relaxation compounds looser than nnenum's zono+star.lp
    intersection at depth (the zono keeps cross-layer affine correlations the
    per-layer triangle re-relaxes away);
(b) split-CHOICE divergence (at L3 v3 picks n33, nnenum n15 — different branching
    sets) leads v3 into subtrees that need L7/L9 work.
NEXT: reproduce one L5-resolved node, compare v3 triangle-LP overapprox margin vs
nnenum's, to decide (a) vs (b).

Also: WALL-CLOCK gap (25s vs 0.46s = 54x) = split-count 8x x per-LP cost ~7x
(Gurobi cold vs GLPK warm-started). Even matching splits leaves the per-LP gap.

## OVERAPPROX MARGIN AT THE L7-SPLIT NODE (2026-06-01) — resolves (a) vs (b)
Captured the first node v3 splits at li=3 (L7), 6 accumulated splits, and compared:
- triangle-LP (v3's) query margins: [-0.030,-0.029,-0.010,-0.011] -> NOT safe
- multi-type zono margins:          [-0.142,-0.153,-0.214,-0.317] -> NOT safe
v3's triangle-LP is the TIGHTER overapprox (-0.010 vs -0.21), yet NEITHER closes
this node -> it genuinely needs more splitting. So it is NOT (a) "v3's overapprox is
weaker". It is (b): **BaB EXPLORATION DYNAMICS.** nnenum never reaches a node like
this — its split path prunes the L5 parents first. v3's tighter bounds -> smaller
branching sets -> different one-norm picks -> it descends into near-miss
(-0.010-margin) nodes that nnenum's path avoids. The margins are tiny, so the tree
is sensitive to the exact split sequence.

## FINAL ANSWER to "why doesn't gurobi match nnenum"
1. nnenum isn't avoiding LPs (2191 real LPs, mis-reported 0). Its speed = fast
   warm-started GLPK + an efficient split tree.
2. v3 already matches nnenum's first 2 splits and has TIGHTER per-node bounds.
3. The 8x split gap is BaB exploration dynamics (sensitive near-miss nodes), NOT a
   weaker abstraction. Matching it = replicating nnenum's exact bound computation +
   split order (effectively re-running nnenum).
4. The wall-clock also carries a ~7x per-LP penalty (Gurobi cold vs GLPK warm) —
   a GENERAL vibecheck inefficiency worth noting beyond acasxu.

## DOMAIN-CONTRACTION TEST at the captured L7-node (2026-06-01)
User: "contract box domain of zono with all constraints, then recompute ub lb."
Tested THIS node's L7 unstable count three ways:
- zono over RAW [-1,1] domain (v4's cheap rung used this): **10 unstable**
- zono over CONTRACTED domain (all 6 constraints): **4 unstable**  <- user's step works
- v3 LP neuron_bounds (uses C,d directly): **3 unstable**  <- TIGHTEST
e.g. neuron 1: raw[-0.12,0.39] -> contracted[0.31,0.39] STABLE -> LP[0.34,0.39].

CONCLUSION: the user's domain-contraction is the right CHEAP mechanism (it's why
v4's raw-zono rung was bad: 10 vs 4) and is how nnenum keeps the zono prefilter
useful + fast. BUT v3's LP already BEATS it (3 < 4) — v3 is NOT missing the
re-tightening. So bounds are confirmed NOT the gap: v3 has the tightest bounds and
still splits 310 vs 40. The remainder is exploration dynamics (different branching
sets -> different one-norm picks -> different tree), which is path-dependent and
sensitive (near-miss -0.01 margins).

THE actionable takeaway: nnenum's SPEED comes from the cheap-zono-with-domain-
contraction + LP-only-where-needed recipe (mostly cheap zono, rare LP), NOT from
fewer splits via tighter bounds. A fast vibecheck relu-split prototype should use
that recipe; its split count will track v3's (~tightest-bound BaB), but each node
is cheap (GLPK/zono) instead of a full Gurobi neuron_bounds sweep.

RECOMMENDATION: stop the prototype OR (if pursuing speed) build the cheap-zono +
domain-contraction + LP-where-needed recipe (v5). acasxu passes in production.
Transferable insight: per-LP warm-start + cheap-zono prefilter. Keep gated dual-ascent.

## *** ROOT CAUSE FOUND (2026-06-01) — the SPEC CHECK, not bounds/dynamics ***
Side-by-side trace pinpointed the divergence: nnenum PRUNES (6T,41T) (0 paths with
(1,41,True)); v3 SPLITS it. (6T,41T) is genuinely SAFE (sampled +0.035). nnenum
proves it via star.lp in round 3. nnenum's split_overapprox is the IDENTICAL
triangle to v3's. So the difference is the SPEC CHECK:

prop_4 spec = 1 disjunct, 4 queries = a CONJUNCTION (unsafe: y0<=y1 ^ y0<=y2 ^
y0<=y3 ^ y0<=y4). At (6T,41T):
- v3 per-query: max query margin -0.026 -> 'not safe' (no single query holds)
- nnenum joint: the unsafe conjunct is INFEASIBLE -> SAFE

v3/v4 checked each query independently (needs ONE query >0 everywhere = sufficient
but NOT necessary). nnenum solves ONE LP with ALL unsafe constraints jointly
(exact). FIX: TStar.conjunct_safe — add all conjunct constraints, check LP
infeasible. Sound (exact on the over-approx, strictly tighter) AND cheaper (1 LP
vs N).

RESULT: prop_4 **310 -> 3 splits, 25s -> 0.74s** (even fewer than nnenum's 40,
because v3's bounds are tighter). THE per-query disjunctive spec check was the
entire 100x gap.

⚠ LIKELY A PRODUCTION BUG: if vibecheck's production spec check uses the same
per-query 'any disjunct query > 0' logic for CONJUNCTIVE unsafe specs, it is
massively over-splitting on acasxu (and any conjunctive-spec benchmark). CHECK
spec.py / VNNSpec.check + the input-split closure logic. This could be a large
production speedup, not just a prototype fix.

## PRODUCTION FIX (2026-06-01) — joint conjunctive-spec check
Affected benchmarks (multi-constraint conjuncts): acasxu(4), cersyve(2),
sat_relu(2), soundnessbench(12) [regular]; yolo(5)[extended]. All others
single-constraint (per-query == joint, unaffected). sat_relu+soundnessbench use
MILP (joint-correct already); acasxu+cersyve use input-split (the bugged closure).

FIXES:
1. `_zono_spec_lbs_and_open_qis` (graph-pipeline zono closure): added
   `_conjunct_unsafe_feasible_zono` — exact scipy LP joint feasibility over the
   zonotope generators. tests/test_conjunct_joint_spec.py (4 tests, fail-before).
2. Input-split batched closure (verify_graph ~line 9965): batched sum-certificate
   over CROWN input-linear bounds — box-min(Σ_q A_q·x+acc_q)>0 => conjunct
   infeasible. Sound, GPU, no-op for single-query. Setting
   input_split_joint_conjunct=True.

VALIDATION: acasxu integration 5/5 pass (2 SAT stay SAT — soundness preserved).
BENEFIT so far: LIMITED on acasxu input-split (3_3 prop_2: 79.1s->79.9s, no change).
The prototype's 310->3 was RELU-split-specific: input-split tightens via
box-splitting, so the joint check matters far less on the full box. The sum
certificate is also weaker than the prototype's exact triangle-LP joint check.

OPEN: the proper affected-benchmark SWEEP (AWS/server1) is needed to measure the
real cross-case effect (esp. cersyve UNSAT). Possible upgrade: exact per-leaf LP
(tighter than sum) if the sweep shows the sum is too weak.

## FIRST DIFFERENCE v3 vs nnenum (relu-split, prop_4, 2026-06-01) — TRACED
v3 trace: root SPLIT n6; [n6+] PRUNE; [n6-] SPLIT n8; [n6-,n8+] PRUNE; [n6-,n8-]
SPLIT n23; both children PRUNE. = 3 splits.
nnenum: root SPLIT n6; [n6+] SPLIT n41 (then 11 splits beneath n6+); ...= 40.

FIRST DIVERGENCE = the 2nd node, [n6 active]:
- v3 PRUNES it. Verified: per-query margins [-0.0555,-0.0626,-0.0575,-0.0963] (ALL
  negative -> per-query can't prove); the JOINT conjunct check says SAFE -> prune.
- nnenum SPLITS it. Verified: its overapprox round was CANCELED —
  "zono.area gens exceeds limit (> 50)" (OVERAPPROX_MIN_GEN_LIMIT=50). At [n6
  active] 5 L1 + deeper unstable neurons need >50 error generators, so nnenum
  ABANDONS the overapprox and splits instead. It re-tries the overapprox on the
  smaller children until the gen count fits, then prunes — ~11 splits beneath n6+.

WHERE NNENUM IS "NOT TIGHT": by CHOICE, not math. nnenum caps the overapprox at 50
generators (a speed/memory budget). On large nodes it gives up on the (sound,
tight) overapprox and splits. v3 has NO gen cap -> runs the full triangle-LP +
joint conjunct check -> prunes the big node directly.

BOTH SOUND. Splitting is always sound; the gen cap only trades pruning power for
per-node cost. v3's uncapped overapprox is a valid over-approximation. The
tradeoff: v3 = fewer-but-pricier nodes (big LPs), nnenum = more-but-cheaper nodes
(capped LPs) — which is why v3's wall-clock (~0.75s prop_4) is comparable to
nnenum's (0.46s) despite 3 vs 40 splits.

(Note: nnenum ALSO has the joint check — its get_violation_star (Specification =
conjunct, unsafe iff mat·y<=rhs for ALL rows). So the joint check is COMMON.)

## EXACT MATCH ACHIEVED (2026-06-01) — TWO nnenum speed-knobs explain 40 vs 3
Disabling nnenum's per-node speed heuristics makes it match v3 EXACTLY:
  default                                    -> 40 splits
  + OVERAPPROX_GEN_LIMIT_MULTIPLIER=None     -> 14 splits (gen cap 50 removed)
  + OVERAPPROX_BOTH_BOUNDS=True              ->  3 splits, 0.30s  == v3 (3, 0.75s)
With both off, nnenum PRUNES [n6 active] (same as v3); tree identical
(root n6; n6+ PRUNE; n6- split n8; ...).

THE TWO KNOBS (both nnenum per-node SPEED heuristics, both SOUND):
1. OVERAPPROX_GEN_LIMIT_MULTIPLIER / OVERAPPROX_MIN_GEN_LIMIT (cap ~50 error gens):
   on a large node the overapprox is CANCELED -> split instead. (40->14)
2. OVERAPPROX_BOTH_BOUNDS=False: nnenum LP-tightens only the rejection-relevant
   bound per neuron, using the looser zono bound for the OTHER side of the
   triangle -> looser triangle relaxation -> joint check fails on borderline
   nodes like [n6+]. Set True (LP both bounds) -> triangles as tight as v3's
   neuron_bounds -> prunes. (14->3)

v3 = nnenum with NO gen cap + BOTH bounds via full LP (v3.neuron_bounds always
computes both bounds for every neuron, no cap). nnenum's defaults trade pruning
power (more splits) for cheaper per-node LPs.

WHICH IS SOUND: BOTH. Splitting is always sound; a looser bound that fails to
prune just causes more splitting, never unsoundness. v3's tighter uncapped
overapprox is a valid over-approximation. Pure time-vs-splits tradeoff.

AND: at the SAME 3 splits, nnenum (0.30s) is FASTER than v3 (0.75s) — warm GLPK
vs cold Gurobi per LP. So the per-LP backend is the remaining v3 speed gap, NOT
the algorithm (which now provably matches).

## SOLVER / TIMING MEASUREMENTS (2026-06-01)
Profiled v3: neuron_bounds (per-neuron LP) = 97% of time; conjunct_safe 1%.
Drilled into "why slower at same splits, same solve time":
- **GLPK ≈ Gurobi raw simplex**: 142 vs 157 us/solve (26 var, 42 constr). The
  backend is NOT the lever; nnenum's edge is FEWER + lower-overhead solves.
- **Gurobi expression-build overhead is ~45%**: setObjective(bias+a_mat@e) =
  24.9ms vs e.Obj=coef_array = 14.2ms vs raw optimize() = 13.7ms. nnenum's GLPK
  sets coefs at C level (no Python expr). FIXED v3: e.Obj=coef -> direct.
- **Vectorized dual ascent (all 50 neurons, numpy batched + per-row sort)**:
  FAST (k=1 0.44ms, k=100 20ms vs LP loop 67ms) but STALLS — proves 28/50 (k=1,
  =box) .. 31/50 (k=100) vs LP's 35/50; bound gap 0.16-0.71. Sound but too loose
  to replace the LP for borderline neurons (the free box already gets 28). NOT a
  win. GPU port wouldn't change accuracy.
- **Selective LP (fast_bounds)**: box bound for box-stable neurons, LP only
  box-unstable. Sound (box-stable => LP-stable). 2x.
- **Single-bound (BOTH_BOUNDS=False)**: backfires for v3 (61->1505 splits) — box
  fallback too loose; nnenum's works only via its tighter zono fallback.

APPLIED to v3 (prop_3): 9.88s -> 3.88s (2.5x), SAME 61 splits, SAFE. Levers used:
selective LP + direct-obj. Remaining: MULTIPROCESSING (BaB is embarrassingly
parallel; nnenum runs N workers; ~Nx) — the biggest untapped lever. A forward
ZONO prefilter (tighter than box, like nnenum) would cut LP count further.

NOTE: v3 is the relu-split RESEARCH prototype; it is NOT the production path.
Production acasxu = input-split (with the joint-conjunct fix already shipped). The
acasxu SWEEP (task #27) validates the PRODUCTION fix vs VC-old vs nnenum, on
server1/AWS.

## LP-COUNT vs nnenum (prop_3, 2026-06-01)
v3 (fast+direct): 61 splits, 24562 LP solves, 3.76s.
nnenum:          169 stars,  9804 glp_simplex,  1.43s.
=> v3 does 2.5x MORE LPs despite FEWER splits. This is the SAME tradeoff as
everywhere: v3's tight triangle-LP bounds -> fewer splits but an LP PER LAYER
per node; nnenum's zono-first -> more splits but few LPs/node.
- closed-form box+halfspace CONTRACTION prefilter: 24562 -> 22686 (8% only). The
  per-coordinate single-halfspace contraction is too weak; borderline neurons
  need the joint LP. NOT enough.
- DUAL ASCENT as the bound (sound but loose): would INCREASE splits (cf.
  single_bound 61->1505). Keep as a gated option but expect more splits.
=> To match nnenum's LP count, v3 needs a MIN-AREA ZONO prefilter that AVOIDS the
per-layer LP (propagate a zono with no LP, LP only selectively / at the spec
check) — the v4 escalation, a real refactor of overapprox_from. + MULTIPROCESSING.

STATUS: v3 fast on easy/medium (prop_3 3.8s, prop_4 0.75s) but TIMES OUT on the
hard cases (3_3 prop_2). NOT yet ready for an all-fast acasxu sweep; needs the
zono prefilter + parallelism first.

## FIRST DIFFERENCE in BOUND COMPUTATION (2026-06-01) — read update_bounds_lp
With cap=widest the split trees ~match (v3 78 vs nnenum 169 on prop_3), so the
remaining gap is purely LP-per-node: v3 356 LP/split vs nnenum 58 (~6x). Reading
nnenum's update_bounds_lp_serial, the 6x = two mechanisms:
1. WITNESS (~2x): the concrete SIMULATION is the witness for one side
   (lputil.py:200,214-215). sim[i]<0 => lb<0 proven by a real achievable point,
   so nnenum LPs only the OTHER bound. both_bounds=False. v3 LPs both.
2. SELECTIVE-via-ZONO (~3x): nnenum LPs only BRANCHING neurons; its zono prefilter
   proves the rest stable. v3 LPs every BOX-unstable neuron (box looser than zono).

KEY: this is WHY v3's single_bound blew up (61->1505) — I used a BOX fallback for
the unprobed side (too loose -> triangle explodes). nnenum uses (a) concrete-point
WITNESS for stability + (b) the ZONO bound for the triangle's other side. The
witness NEEDS the zono; alone it's the same loose triangle that explodes.

TO MATCH nnenum's LP count, v3 needs its full bound engine: per-layer LAYER_BOUNDS
updated INCREMENTALLY (only constraint-affected neurons) via witnessed single-bound
LP, feeding a min-area ZONO prefilter. This is a real refactor of overapprox_from
(maintain carried layer_bounds + a concrete sim + a parallel zono), not a patch.

## ZONO PREFILTER + WITNESS — IMPLEMENTED + TESTED (2026-06-01) — NEGATIVE
Built overapprox_zono (min-area zono prefilter: LP only zono-unstable) + compute_sim
(concrete witness -> 1 LP/neuron). prop_3 (cap=widest):
- baseline (box prefilter):  78 splits, 27793 LP, 4.1s
- + min-area zono prefilter: 78 splits, 29573 LP (MORE), 4.2s
- + zono + witness:          1510 splits, 330586 LP, TIMEOUT
- nnenum:                    169 splits, 9804 LP, 1.4s
Why it FAILS:
1. v3's existing box prefilter already CONTRACTS the box vs C,d (triangle
   constraints) -> TIGHTER than a bare min-area zono -> the zono proves FEWER
   stable -> MORE LPs. Bare min-area zono is the wrong (looser) prefilter.
2. The witness blows up (78->1510) — witnessed side uses the loose zono bound for
   the triangle -> explodes (same failure as single_bound).
ROOT: nnenum's zono prefilter is NOT bare min-area — it is min-area +
CONTRACT_ZONOTOPE_LP (LP-contraction of the gen box vs constraints) + multi-type
intersection + carried incremental layer_bounds. That LP-contraction (which itself
uses LPs) is the missing tightness. A single min-area zono cannot match it.
=> Gated OFF by default (zono_prefilter=False); v3's default path unchanged. The
nnenum-grade bound engine remains a deep reimplementation, not a prefilter patch.

## TRACED (not speculated) the LP-profile difference (2026-06-01)
Hooked nnenum's update_bounds_lp with (depth, n_LPd) on prop_3:
- **root overapprox (depth 0): 0 neurons LP'd** — nnenum's shallow overapprox is
  PURE ZONO, no LP. It proves-or-splits on zono bounds.
- LP work (6183 neurons total via update_bounds_lp) is ALL at depth>0 — only once
  splits shrink the gen count below the cap so the star.lp round actually runs.
- v3 LPs at EVERY overapprox layer including the root.

CORRECTED conclusion: the LP gap is NOT (only) zono looseness — it is that nnenum
runs NO LP until a node is small enough for star.lp; v3 LPs everywhere. nnenum's
shallow phase = pure zono + cheap zono-bound splits; the LP triangle is reserved
for deep small nodes (the gen cap is what gates this).
=> the correct fix: at shallow (gen>cap) nodes v3 must NOT LP — use zono bounds +
a zono-based spec check + split; run the LP triangle only when gens fit the cap.
My zono_prefilter wrongly LPs the zono-unstable at every node. This is the v4
"zono-first / LP-on-small" escalation done right (with the joint spec check).

## nnenum BOUND ENGINE PORTED -> v3 NOW MATCHES (2026-06-01) — multizono=True
Decomposed the LP gap (v3 vs nnenum-tuned, prop_3, bound-neurons LP'd):
- box prefilter:                          11269  (1.65x)
- + multi-type zono (area/ybloat/interval): 11718 (worse — closed-form contract loose)
- + EXACT LP contraction of input gen box: 10799
- + FEED LP-tightened bounds into the zono relu (shared layer_bounds): 8839
- + INCREMENTAL CARRY (parent bounds -> child, re-LP only still-unstable): **7741**
- nnenum-tuned:                            **6843**
Result: bound-neurons 1.65x -> **1.13x**; time 3.82s -> **2.93s vs nnenum 2.79s (1.05x)**;
LP 22686 -> 17100. SOUND: prop_4 SAFE 3 splits 0.24s (= nnenum); SAT cases not
falsely verified (1_5 not-safe, 1_9 timeout — never SAFE).
Each step was TRACED, not guessed: multi-type+LPc+feed closed the prefilter-tightness
gap; incremental carry closed the per-node-recompute gap. Residual ~1.13x = nnenum's
early-reject (1 LP/neuron when the first side proves stable) — minor.
v3 engine (gated multizono=True): multi-type zono prefilter + exact LP contraction +
LP-bounds-fed-to-zono + incremental parent->child carry. Matches nnenum's algorithm
(splits) AND bound profile (~1.1x LP/time).

## RESIDUAL ~1.1x TRACED (2026-06-01) — early-reject + first-layer-LP
- Early-reject single-bound (LP one side, stop if it proves stable): implemented
  (single_bound=True under multizono). Minor: 17100->16409 LP (most zono-unstable
  are genuinely unstable -> both bounds needed). NOT the residual cause (residual
  is in NEURON count 7741 vs 6843, not solves/neuron).
- TRACED root per-layer neurons LP'd:
    v3:     L1=9  L3=5 L5=9 L7=19 L9=22 L11=46
    nnenum:       L3=5 L5=9 L7=19 L9=22 L11=46
  EXACT match L3-L11; the ONLY diff is v3 LPs 9 at L1, nnenum doesn't (~9/node x
  ~100 nodes = the ~900-neuron residual).
- Tried skipping the L1 LP -> EXPLODED (643 splits, timeout). TRACED reason: v3
  keeps splits as C,d CONSTRAINTS, so its first-layer zono (over the LP-contracted
  INPUT box) is LOOSE at deeper nodes; the L1 LP compensates. nnenum BAKES splits
  into the affine map (exact ReLU) -> tight first-layer zono -> no L1 LP needed.
=> the residual ~1.1x is a REPRESENTATION difference (C,d constraints vs baked-in
splits), CONFIRMED by the explosion when the LP is naively removed. Closing it
needs v3 to bake relu decisions into the affine map like nnenum (a deeper rep
change), not a missing trick. FINAL: v3 1.04x time / 1.13x neurons / 1.18x LP of
nnenum, sound (prop_4 3 splits 0.24s; SAT not falsely verified).
  + better split-order star BaB, or accept the existing input-split +
  adaptive-hybrid fallback for acasxu and move to the queue (sat_relu,
  soundnessbench, tllverifybench).
- Separately: schedule the `_batched_dual_ascent` overshoot soundness audit.

## PER-SPLIT LP DECOMPOSITION: contraction vs bounding (2026-06-01) — MEASURED
Question: "when you split, is the contraction LP the same? the subsequent
bounding? where do we diverge?" Instrumented BOTH (trace_lp_categories.py for
nnenum: wraps update_input_box_bounds=CONTRACT, minimize_output=BOUND,
minimize_vec=SPEC, + per-call lists; star_bab_v3 CONTRACT_SOLVES + MZ_CON/BND_CALLS).
Case 1_1 prop_3. nnenum: NUM_PROCESSES=1, TRY_QUICK_OVERAPPROX=False (isolate BaB),
both_bounds+nocap to match v3's tree.

|                | v3 (multizono+sb) | nnenum-tuned | ratio |
| splits/stars   | 78                | 64           | 1.22x |
| **contraction LP** | **1890**      | **744**      | **2.54x** v3 |
|  - per call    | 9.95 (190 calls)  | 5.90 (126 calls) |   |
| **bounding LP**| **18131**         | **12822**    | **1.41x** v3 |
|  - per call    | 74.3 (190 calls)  | 20.9 (612 calls) | diff granularity |
| spec LP        | 79                | 252          | 0.31x (v3 fewer) |
| total LP       | 20100             | 13818        | 1.45x |
| time           | 3.0s              | 2.15s        | 1.40x |

CONTRACTION is NOT the same (v3 2.5x more). MECHANISM (measured, not guessed):
- nnenum contracts INCREMENTALLY at the split: contract_lp(child, ±row, ±bias)
  with the ONE new halfspace (prefilter.py:316-317), and update_input_box_bounds
  WITNESS-PRUNES to only input dims the new constraint moves -> LP/call dist
  hist[2:21,3:3,4:11,5:14,6:26,7:20,8:13,9:6,10:7], mean 5.9, first6=[2,7,2,6,5,4].
  Only at split nodes (126 calls = 63 splits x 2 children).
- v3 RE-CONTRACTS the full input box from scratch every node: for j in range(k)
  min+max = exactly 2k=10 LP/node (first6=[0,10,10,10,10,10], the 0 is root),
  AND at descend nodes too (190 calls). => 9.95 vs 5.90 per call + more calls.

BOUNDING is NOT the same (v3 1.41x more, the larger ABSOLUTE gap: +5309 vs
contraction's +1146). Root (from the earlier per-layer trace, unchanged): v3 LPs
~1.48x more NEURONS (9491 vs ~6411) — dominated by the 9 neurons at L1 it bounds
per node that nnenum proves stable for free. (Call granularity differs: v3 = 1
overapprox call/node bounding all remaining layers; nnenum = 1 update_bounds_lp
invocation per layer per node, 612 of them — so per-call means aren't comparable,
only totals are.)

ROOT CAUSE (both channels, one cause): nnenum BAKES each ReLU decision into the
affine map (exact ReLU substitution -> shrinks the generator set), so per node its
zonotope is tight: (a) a new split perturbs only a few input-box bounds (witness-
pruned contraction) and (b) the first layer needs no LP. v3 keeps splits as C,d
HALFSPACE CONSTRAINTS on a fixed 5-gen input zono -> loose -> full re-contraction
+ extra L1 bounding LPs + 1.22x more splits. Same representation difference plan.md
already pinned; now DECOMPOSED into contraction(2.5x, witness) + bounding(1.4x, L1)
+ split-count(1.22x). Closing it requires v3 to bake relu decisions into the affine
map (deep rep change), not a per-channel patch.

## CORRECTION: the prior decomposition was CAP-ASYMMETRIC (2026-06-01)
User caught it: the 64-star "tree-matched" nnenum run used nocap (multiplier=inf,
min_gen_limit=1e9) while v3 kept gen_cap='widest'=50. So that run matched the tree
but NOT the cap -> v3 over-counted contraction relative to a cap-fair nnenum.

CAP-MATCHED (both cap=50, both single-bound) 1_1 prop_3:
|              | v3 (cap50,sb) | nnenum (cap50,sb=default) |
| splits/stars | 78            | 169          |
| contraction  | 1890          | 2067         |  ~EQUAL (nnenum slightly more: 169>78 stars)
| bounding     | 18131         | 7205         |  v3 2.52x
| spec         | 79            | 388          |
| total        | 20100         | 9660         |  v3 2.08x

Two corrections:
1. CONTRACTION TOTAL is ~EQUAL at matched cap (1890 vs 2067), NOT 2.5x. The 2.5x
   was the cap-asymmetry artifact (uncapped nnenum had only 64 stars -> fewer
   contraction calls). Per-CALL still differs (v3 9.95 full-box vs nnenum 6.15
   witness-pruned) but nnenum makes more split-calls (336=168x2) so totals wash out.
2. WITNESS-PRUNING is CAP-INDEPENDENT: contract LP/call mean 6.15 (cap50) vs 5.90
   (nocap), first6=[2,7,2,6,5,4] IDENTICAL. Robust mechanism, not a cap artifact.

THE REAL DIVERGENCE (cap-matched) is BOUNDING: v3 18131 vs nnenum 7205 = 2.52x.
And the trees invert: at the SAME cap=50 v3 splits FEWER (78 vs 169) because its
triangle-LP per-node bounds are tighter -> proves more per node -> 2x fewer splits,
but PAYS 2.5x the bounding LP/node (232/split vs 42/star) to get them. That is the
fundamental tradeoff, now cleanly isolated: v3 buys 2x fewer splits with 2.5x more
bounding LP. Net total LP: v3 2.08x nnenum. Time 3.0s vs 0.86s (nnenum default cap50
single-bound is the fast one; the nocap/both_bounds tuning was what slowed it to 2.15s).

## WITNESS-BASED CONTRACTION ported to v3 (2026-06-01) — UNCAPPED both
User: don't cap either; add witness contraction to v3; trace first splits to
confirm LP match. Ported nnenum's update_input_box_bounds(_new) faithfully into
_witness_contract(): per-dim witnesses (box corners at root), skip a box-bound
re-solve when its witness still satisfies the new split halfspace, batched 'pre'
loop to confirm unchanged dims with one shared LP. Threaded cstate=(n_done,zlo,
zhi,wlo,whi) parent->child through process()/overapprox_multizono. SOUND by
construction: skipping keeps the parent bound -> box only over-approximates.

1_1 prop_3, UNCAPPED (v3 gen_cap=None, multizono+single_bound+witness_contract):
  61 splits, SAFE, 2.86s. contraction 572 (was 1890 full-recontract -> 3.3x less),
  bounding 14791, spec 148, total 15511.
  CONTRACT LP/call: mean 3.86 over 148 calls (incl 38 DESCEND nodes now =0, were 10),
  nonzero (split-child) calls: 110, mean 5.2, first nonzero=[4,5,4,3,5,3,8,1].
nnenum nocap (126 child-contractions): mean 5.90, first6=[2,7,2,6,5,4].
=> per-child contraction now MATCHES in mechanism + magnitude (v3 5.2 vs nnenum
5.9), and descends are free. Total v3 572 < nnenum 744 (v3 has 61 vs 64 splits +
slightly more skips). NEXT: per-split side-by-side trace of first few splits.

## "DIFFERENT BRANCHERS" CLAIM REFUTED (2026-06-01) — they are IDENTICAL
User challenged "different branchers". Traced the actual split (layer, neuron,
score) in both. nnenum: SPLIT_ORDER=SPLIT_ONE_NORM -> score=min(hi,-lo), sort
DESCENDING, split branching_neurons[0] (prefilter.sort_splits + do_first_relu_split).
v3: scores=min(hi,-lo); ni=argmax. SAME formula, SAME direction.

1_1 prop_3 uncapped, first 10 splits (v3 li vs nnenum cur_layer, map li->4+3*li):
  v3:     44 27 46 28 8 37 35 | 33 6 | 22
  nnenum: 44 27 46 28 8 37 35 | 33 6 | 22   <- IDENTICAL neuron order
Scores match EXACTLY where the region is identical (44:.053 27:.0395 46:.0365
28:.013 37:.0061 35:.0022 6:.0443) and differ slightly once contraction has
accumulated (8: v3 .0069 vs nnenum .0177; 33: .1157 vs .1391; 22: .6931 vs .7447)
-- because v3's witness box-contraction argmin (Gurobi) != nnenum's (GLPK), so the
contracted-region bounds drift by ~1e-2 after a few splits.

=> The brancher is NOT the divergence source; it already matches. Trees still end
up different SIZES (v3 61 vs nnenum 93 splits) for the OTHER reason established
all along: v3's triangle-LP per-node bounds are TIGHTER than nnenum's zono+LP, so
v3 has fewer unstable neurons -> fewer splits. The accumulated bound drift flips
the argmax around split 11 (v3 backtracks to (1,6); nnenum stays at (10,22)).
Nothing to "make match" in the brancher. To get byte-identical trees you'd have to
match the BOUNDS engine (give up v3's tighter triangle LP), which is the wrong
trade. Per-split contraction RATE already matches (~5 LP/child both); that was the
question that mattered.

## "v3 TIGHTER -> FEWER SPLITS" — TRACED + CORRECTED + SOUNDNESS-CHECKED (2026-06-01)
User: "61 vs 93 is weird, elaborate with real trace." It WAS misleading. Two
separate effects were conflated:

(1) BOTH_BOUNDS POLICY (the big one, was a config artifact): I compared v3
single_bound (61) to nnenum both_bounds=FALSE (93). Not apples-to-apples. nnenum
both_bounds=FALSE LP-tightens ONE side and leaves the other at the looser ZONO
bound -> more apparent-unstable -> 93. nnenum both_bounds=TRUE LP-tightens both ->
64. v3's single_bound STILL LP-tightens both sides of genuinely-unstable neurons
(early-rejects only neurons proven stable on the 1st side) -> behaves like
both_bounds=TRUE. Matched policy: v3 61 vs nnenum 64. They track neuron-for-neuron
AND score-for-score through split 11 (both pick n6 score .0112), diverging at split12.
=> the real gap is 61 vs 64, NOT 61 vs 93. Child order identical (both
self_gets_positive=True -> active-first), so regions match -> fair comparison.

(2) GENUINE (tiny) TIGHTNESS RESIDUAL (61 vs 64): traced the first 7 split bounds.
Binding (score-determining) side matches EXACTLY for 6 of 7 first-layer neurons;
only neuron 8 differs: v3 lo=-0.0072 vs nnenum lo=-0.0177. SOUNDNESS CHECK
(independent scipy LP, raw weights, region {44+,27+,46+,28+}): TRUE min = -0.0072.
=> v3 is EXACT (LP over box∩halfspaces); nnenum's -0.0177 is its zono/box bound,
LOOSER than truth (sound over-approx, just not tight). v3's pre-LP zono was already
-0.0072 (3-zono intersection + incremental pbounds carry + exact-LP box contraction
beat nnenum's single zono there). For most neurons the zono is already tight on the
binding side so it doesn't matter; occasionally (n8) v3's exact bound proves a
neuron closer to stable -> trims a split.

CONCLUSION: the "1.5x bound-tightness gap" does NOT exist. Apples-to-apples it's
61 vs 64 (~5%), almost all of which is the both_bounds policy; the genuine
tightness residual is a handful of neurons where v3's exact LP beats nnenum's zono.
The "weird" instinct was right -- a real 1.5x gap between two LP methods would be
implausible; it was a config-comparison error on my part.

## "DID NNENUM RUN SIMPLEX FOR NEURON 8?" — TRACED (2026-06-01)
Instrumented minimize_output(output_index==8) + captured the exact split node.
At neuron 8's split node ({44+,27+,46+,28+}, nocap both_bounds run):
  layer_bounds[8] used for split: (-0.0177, 0.0403)
  live zono box_bounds[8]:        (-0.0177, 0.0409)
  sim[8]: -0.0011
ANSWER: simplex ran on ONE side only.
- UPPER side: LP'd (0.0403, tighter than zono 0.0409). 
- LOWER side (the score-determining/binding side): NOT LP'd -> layer_bounds_lo
  == live zono_lo == -0.0177 (the zono box-bound). The exact polytope LP is -0.0072
  (v3 + independent scipy).
WHY: nnenum's BRANCHING bound (prefilter.py:128 update_bounds_lp) is called WITHOUT
both_bounds -> defaults both_bounds=False, and OVERAPPROX_BOTH_BOUNDS does NOT reach
it (that flag only affects the overapprox/prune phase, not branching). In
update_bounds_lp_serial (both_bounds=False): ub_first = sim[i]<0 picks the side to
LP (sim[8]=-0.0011<0 -> LP the UPPER), LPs that ONE side, and uses the concrete
SIM point as the witness for the other side (lputil.py:214-215 verbatim: "for
enumeration, we only need a single bound since the simulation is the witness for the
other side"). The sim (-0.0011<0) already witnesses neuron 8 reaches negative, so
with LP'd upper (0.0403>0) the neuron is confirmed unstable -> split; the lower bound
is left at the looser zono -0.0177, never refined.
=> This is why both_bounds True/False give the SAME neuron-8 branching score 0.0177
(branching is single-sided regardless; both_bounds only changes the overapprox phase,
hence total splits 64 vs 93 but not the per-neuron branching bound). v3 LPs BOTH
sides of every unstable neuron -> exact -0.0072 -> marginally tighter -> the 61-vs-64.

## BRANCH single-bound IS the whole gap — proved by forcing both-bounds (2026-06-01)
User: "is that a setting in nnenum? could you make it one?" -> No existing setting:
prefilter.py:128 calls update_bounds_lp WITHOUT both_bounds (hardcoded default
False); OVERAPPROX_BOTH_BOUNDS doesn't reach branching. Made it a knob (1-line
override in the harness emulating Settings.BRANCH_BOTH_BOUNDS=True; real change =
add the flag + pass both_bounds=Settings.BRANCH_BOTH_BOUNDS at prefilter.py:128).

1_1 prop_3 nocap:
  branch single-bound (default): 93 stars; neuron 8 split lo=-0.0177 (ZONO), score .0177
  branch both-bounds (forced):   62 stars; neuron 8 split lo=-0.0069 (LP),   score .0069
  v3:                            61 splits; neuron 8 lo=-0.0069
With both-bounds branching, nnenum's first 12 splits become IDENTICAL to v3 -
same neurons AND scores incl. split 11 (1/7,6,.0112) and 12 (2/10,22,.6425), which
diverged under single-bound. Total 62 vs v3 61 (residual = 1 split, solver tie-break).

=> CONCLUSIVE: the entire 93-vs-61 "v3 splits fewer" was the branching single-bound
(sim-witness) shortcut, NOT a bound-engine tightness difference. v3 == nnenum with
both-bounds branching. nnenum leaves the non-sim side at the zono bound to save 1
LP/branch-neuron (its enumeration default); v3 LPs both -> exact bound -> the few
neurons (e.g. n8) where the zono is loose on the binding side don't over-split.
Making it a real nnenum Settings flag is a 2-line change if we ever want the A/B.

## SINGLE-BOUND IN V3 -> EXPLODES (2026-06-01) — the LP-both-sides is load-bearing
User: do nnenum's single-bound in v3, compare LP+timing. Added NNENUM_SINGLE: LP
one side (the zono-lean binding side), keep the ZONO bound for the other (don't
2nd-LP), neuron stays unstable -> split. 1_1 prop_3 uncapped+witness:
  v3 both-bounds:    61 splits, 15511 LP, 14791 bound, 2.88s, SAFE
  v3 nnenum-single:  >=601 splits, >=34182 LP, TIMEOUT (>90s, hit 600 cap, unfinished)
  nnenum single:     93 splits, ~9660 LP, 1.60s, SAFE
  nnenum both:       64 splits, ~13818 LP, 2.14s, SAFE
=> Adopting nnenum's single-bound makes v3 ~10x WORSE (explodes), NOT faster. SAME
algorithm gives nnenum 93 but v3 601+. Root cause (already established): v3 keeps
ReLU splits as C,d CONSTRAINTS over a fixed input zono -> its zono box bound on the
NON-LP'd side is much LOOSER than nnenum's, which BAKES splits into the affine map
(exact ReLU) -> tight zono. So the triangle built from [LP-side, zono-side] is loose
for v3 -> over-splits. nnenum's tight baked zono makes the zono-side good enough.
=> v3's LP-both-sides is LOAD-BEARING: it computes the exact polytope bound,
compensating for v3's looser representation. v3's sweet spot is both-bounds (61,
2.88s). nnenum can afford single-bound (1.60s) ONLY because its zono is already
tight. Confirms the rep difference is the real axis, not the bound policy: at
both-bounds both ~match (61 vs 64); at single-bound v3 explodes while nnenum holds.

## WHY v3 SINGLE-BOUND EXPLODED: it's a PREFILTER thing, not the relaxation (2026-06-01)
User: "is it just in the prefilter, not actual bounds?" YES. nnenum has TWO bound
computations:
- BRANCHING prefilter (prefilter.py:128): update_bounds_lp WITHOUT both_bounds ->
  single-bound. Only decides WHICH neurons to branch.
- OVERAPPROX relaxation (overapprox.py:388 tighten_bounds -> :478 both_bounds=
  Settings.OVERAPPROX_BOTH_BOUNDS): both-sided for the layers it propagates the
  triangle through (first overapprox layer inherits the prefilter bounds at :357,
  deeper layers :362-399 recompute both-sided).
ALSO: in the exact BaB split path, splitting a neuron BAKES the ReLU sign exact (no
relaxation) -> the loose single-bound never propagates; it only gates the decision
on a neuron about to become exact.
=> v3 conflates the two: overapprox_multizono's lo,hi BOTH decide branching AND build
the propagated triangle. So NNENUM_SINGLE loosened the ACTUAL relaxation -> explosion.
nnenum never does that: its propagated relaxation is always both-bounds. v3's default
(both-bounds in overapprox) is the correct analog of nnenum's overapprox. The
single-bound saving is only available to nnenum because it has a SEPARATE exact-split
branching phase whose loose bounds never propagate. To get it in v3 you'd have to
split the branching-decision bounds from the relaxation bounds (architectural), or
single-bound ONLY the split-layer neurons that become exact -- not the propagated ones.

## SINGLE-BOUND @ BRANCH-LAYER-ONLY also explodes (2026-06-01) — bounds are dual-use
Tried the targeted version: single-bound (keep zono non-binding side) ONLY at l==li
(the branch layer), both-bounds at l>li. 1_1 prop_3 uncapped+witness:
  v3 both-bounds:          61 splits, 15511 LP, 2.88s, SAFE
  v3 single@branch-only:   >=401 splits, 95275 LP, TIMEOUT (19s, hit 400 cap)  <- WORSE
  v3 full-single:          >=601 splits, TIMEOUT
WHY it still explodes: the branch SCORE is unchanged (v3 single_bound LPs the BINDING
side, which sets min(hi,-lo)), but the layer-li triangle relaxation built from
[LP-side, ZONO-side] is loose and PROPAGATES through the prune attempt's deeper
layers -> the overapprox prune fails far more often -> more splits. The split
SELECTION is fine; the prune RATE collapses. v3's layer-li bounds are DUAL-USE
(branch decision AND propagated prune relaxation), so you can't loosen them for the
branch saving without crippling the prune.
=> CLOSED: v3 cannot capture nnenum's single-bound LP saving in any form. nnenum gets
it because its split is EXACT (baked affine) and its prune is a SEPARATE both-bounds
overapprox -- a missed prune just costs one exact split, bounded. v3's split keeps
constraints + relies on the relaxation staying tight, so any loosening cascades.
v3's both-bounds (61 splits, 2.88s) is and stays its best config. The single-bound
axis requires the exact-affine-baking representation, not a bound-policy tweak.

## PARALLEL v3 implemented + threaded comparison (2026-06-01)
User: long-running cases, single vs multi-threaded, v3 vs nnenum (default+matched),
wall time + num LPs. Then full sweep; may need PGD/LP-witness for SAT cases.
Hard case: 2_2 prop_3 is TRIVIAL (0.2s, 3 splits) -- the user misremembered. Real
hard ones (v3 single-thread TIMEOUT @45s): 3_3 prop_2 (1451 splits in 45s),
4_2 prop_2 (874). Using 3_3 prop_2.

v3 parallelism (multiprocessing): serial shallow frontier-gen to depth ceil(log2(4W))
collects un-pruned nodes (deferred at the split point, carrying pbounds+cstate), then
mp.Pool(W) solves each subtree serially (each worker rebuilds context + own Gurobi env
via verify(start_node=...)). Aggregate: SAFE iff all safe; sum LPs/splits. Smoke test
1_1 prop_3: serial 2.96s/15511LP vs 4-worker 1.72s/16533LP (61 splits both, +6.6% LP
from re-deriving the 8 frontier nodes). Correct.

nnenum timing runner (nnenum_timed.py): default = set_control_settings (TRY_QUICK on,
single-bound branch, gen70x1.5, LP-timeout 0.02); matched = nocap+both_bounds+TRY_QUICK
off. LP count: single-proc res.total_lps is BROKEN (22, 0) -> lean monkeypatch counter
(validated: matched-1 = 13818, == earlier categorized). Multi-proc uses res.total_lps
(shared Value). NOTE: nnenum multi-proc does MORE work (SPLIT_IF_IDLE force-splits to
feed idle workers): 1_1 prop_3 default nproc1=94 stars vs nproc4=505 stars. So nnenum
parallel is NOT a clean parallelization of the serial tree -- it's a more-aggressive
search that parallelizes. Will report as-is.

3_3 prop_2 results so far:
  v3 parallel-16: SAFE 67.8s, 6487 splits, 579829 LP, frontier=40
  (v3 serial, nnenum default-1/matched-1 running; 16-core nnenum runs after)

## THREADED COMPARISON 3_3 prop_2 — full matrix (2026-06-01)
| engine config        | wall   | LP    | stars/splits |
| v3 serial (1)        | 152.1s | 570K  | 6487 |
| v3 parallel-16       |  67.8s | 580K  | 6487  (2.24x)  frontier=40 |
| nnenum default  1    |  26.8s | 420K  | 10295 |
| nnenum default 16    |   3.0s | 177K  | 18083 (8.9x) |
| nnenum matched  1    |  96.3s | 563K  | 7179 |
| nnenum matched 16    |   6.0s | 168K  | 15214 (16x) |
Notes: nnenum LP single(lean counter) vs multi(res.total_lps) are different counters
-> within-nnenum single-vs-multi LP not comparable (multi shows FEWER LP w/ MORE stars).
v3 single-vs-multi LP IS comparable (same counter): 570K->580K (+1.7% frontier redo).
KEY GAPS:
1. nnenum default 1-proc (26.8s) >> v3 serial (152s): nnenum's tuned config (TRY_QUICK,
   single-bound branch, LP-timeout 0.02, gen70) is ~5.7x faster single-threaded. v3's
   both-bounds + Gurobi-per-node is heavier.
2. v3 parallel SCALES POORLY (2.24x/16) vs nnenum (8.9-16x): v3 uses a STATIC frontier
   partition (Pool.map over 40 fixed subtrees) -> bound by the largest subtree, no
   rebalance. nnenum has a dynamic work queue + SPLIT_IF_IDLE (force-split to feed idle
   workers) -> near-linear. To fix v3 needs work-stealing / recursive frontier, not a
   static partition. Testing deeper frontier next (more, smaller nodes).

## DEEPER FRONTIER doesn't help (2026-06-01) — static partition is the ceiling
v3 parallel-16 on 3_3 prop_2 by frontier_depth: d6=67.8s(40 nodes), d9=86.0s(123),
d12=100.5s(266). DEEPER = SLOWER: serial frontier-gen grows (more prune attempts
before the parallel phase, Amdahl) while load-balance barely improves (work is
concentrated in a few subtrees a static partition can't break up). 2.24x is the
static-frontier ceiling here. To match nnenum's 9-16x, v3 needs WORK-STEALING
(dynamic queue; idle workers re-split big subtrees) -- nnenum's SPLIT_IF_IDLE + work
queue. Noted as the next parallel lever; static frontier shipped + measured.

## SAT via LP-witness + PGD-refine (2026-06-01)
PGD random-restart is too weak for acasxu's tiny unsafe slivers (1_2 prop_2: 20/300k
random pts unsafe, margin -4e-4). Implemented: at each spec check (ATTACK mode), solve
the DEEPEST-violation LP (min t s.t. w.y+b<=t) over the relaxed star, take the input
coords -> concrete check on the REAL net; if not a real cex, PGD-REFINE from that seed
(gradient descent on real-net margin). A hit raises CexFound -> sound SAT (verified by
concrete forward eval). Results: 1_2 prop_2 SAT@split1 5.5s, 2_2 prop_2 SAT@4 10s,
1_3 prop_2 SAT@71 27s. All CORRECT + cex-verified; UNSAT (1_1 prop_2/3) stay SAFE.
Slower than nnenum (0.2s) on tiny-region SAT (refine fires late when relaxation loose),
but all under timeout. Good enough for the sweep.

## FULL ACASXU SWEEP launched (2026-06-01)
sweep_acasxu.py: 186 official instances, v3 (parallel-16 + attack, gen_cap=None,
multizono+witness_contract+single_bound) vs nnenum (default control settings, 16-proc).
Verdicts read from per-case results-files (NOT exit code), per CLAUDE.md. Flags
DISAGREE (critical: one says unsat other sat -> soundness bug) and v3_timeout.
TO_CAP=120s. v3 CLI (star_bab_v3.py net prop --timeout --workers --results-file) +
nnenum_timed.py both write vnncomp verdicts (unsat/sat/timeout). Smoke test: 1_2 prop_2
both 'sat' AGREE; 1_1 prop_4 v3 'unsat'. Results -> sweep_acasxu.csv (incremental).

## SWEEP early status (2026-06-01): correct but v3 slow
First rows: 1_1..1_5 prop_1 all AGREE (unsat) v3 vs nnenum. NO soundness issues.
But v3 16-40s/case (parallel-16) vs nnenum 1.2s -> v3 ~15-30x slower across the board
(per-LP Gurobi overhead + less efficient search than nnenum's tuned control config).
At ~20-40s/case the full 186-case sweep ETA ~2-3h (running detached, sweep_acasxu.csv).
Verdict correctness is the headline (agreeing); speed gap is the known per-LP/search
overhead, consistent with the 3_3 prop_2 single-thread result (v3 152s vs nnenum 27s).

## SWEEP setup bug fixed + restarted (2026-06-01)
First sweep hit NORESULT/NORESULT on 1_7 prop_1: 31 of the acasxu onnx are gzip-ONLY
(load_acasxu + nnenum's load_onnx both need the .onnx). gunzipped all 31; restarted
the full sweep clean. Verdicts before the bug all AGREEd (1_1..1_6 prop_1 unsat). ETA
~2-3h at v3's ~20-40s/case. Watch sweep_acasxu.csv; grep DISAGREE for soundness.

## FULL SWEEP COMPLETE (2026-06-01) — v3 SOUND+COMPLETE, 0 disagreements
186 official acasxu instances, v3 (parallel-16+attack) vs nnenum (default 16-proc):
  179 AGREE, 0 DISAGREE (no false verdicts -- the headline correctness result).
  v3: 138 unsat, 46 sat, 2 timeout (1_9 prop_7 @118s, 2_9 prop_8 @176s).
  5 inconclusive = NNENUM NORESULT/error on prop_5-10 specials (harness issue on
  nnenum side; v3 gave verdicts there). Saved sweep_acasxu_DONE.csv.
Note: 2_9 prop_8 ran 176s > 120 cap -> parallel per-worker timeout doesn't respect
the global deadline (fix in the work-stealing rewrite).

## (a) BestBdStop=0 — MEASURED, no help (2026-06-01)
3_3 prop_2 single-threaded: off=150.6s/570258LP, on=153.1s/570258LP. IDENTICAL LP
count, 1.6% SLOWER (per-LP setParam overhead). Confirms: Gurobi simplex on 5-50 var
LPs hits optimum in a few iterations -> stopping at the sign cuts nothing. v3's cost
is per-node model rebuild + LP count, not simplex iters. BestBdStop is a dead end.

## (b) WORK-STEALING implemented (2026-06-01) — replaces static frontier
Scrapped the static-frontier Pool. New: shared mp.Queue seeded with root; N long-lived
workers each build context once, loop {pull node; process with LOCAL budget
FRONTIER_DEPTH=depth+chunk; push overflow nodes back to queue; update atomic
`outstanding` counter}. Termination: outstanding hits 0 -> push N sentinels. SAT: shared
`found` flag -> all exit; cex via CexFound in-worker. Timeout: shared wall-clock DEADLINE
(time.time(), cross-process). Dynamic rebalancing -> big subtrees redistribute. Smoke:
1_1 prop_3 nw1=3.5s nw8=1.37s, SAFE 61 splits (correct). 3_3 prop_2 scaling test running.

## "HARD" SET defined (2026-06-02, user) — saved hard_set.txt
8 cases to test on later: 3_3 prop_2 (canonical hard), v3 timeouts (1_9 prop_7,
2_9 prop_8), sweep-inconclusive where nnenum gave no result (5_7 prop_1, 1_1 prop_5,
1_1 prop_6, 3_3 prop_9, 4_5 prop_10). Will run this set when user says ready.

## WORK-STEALING + the 16x CEILING is HARDWARE (2026-06-02)
Rewrote parallelism: static frontier -> true work-stealing (local DFS stack per
worker; donate half to a shared mp.Queue only when it's starving (qsize<nw); steal
when local empty; termination by idle-count==nw + queue empty; wall-clock DEADLINE).
Children-deferred (carry parent bounds) so LP count == serial exactly (570258, no
inflation). Correct (1_1 prop_3 SAFE/61, 1_2 prop_2 SAT).
3_3 prop_2 nw=16: ~6x (chunk=1 25s/5.9x), CPU 854% (=8.5 of 16 cores busy).
ROOT CAUSE of <16x is HARDWARE, not the algorithm: local box = Core Ultra 7 265H,
HYBRID: 6 P-cores@5.3GHz + 8 E-cores@4.6GHz + 2 LP-E@2.5GHz, +throttling (scaling 79%).
Serial baseline runs on ONE 5.3GHz P-core; 16 heterogeneous throttled cores give
~10x theoretical / ~6-8x real -> 16x is PHYSICALLY IMPOSSIBLE here for CPU-bound work.
16 independent v3 jobs: 3.53s alone -> 7.95s x16 (2.25x slowdown) confirms it.
=> must validate true scaling on server1 (uniform i9, 16 equal cores). Frontier-seed
variant was worse (474% CPU); reverted to pure root-seed local-stack work-stealing.

## WORK-STEALING SCALING CURVE (local, 3_3 prop_2) — algorithm is GOOD, HW is the cap
  nw  wall   speedup  efficiency
   1  150.6s  1.0x     100%
   2   80.0s  1.9x      94%   (2 P-cores)
   4   45.8s  3.3x      82%   (4 P-cores)
   6   35.2s  4.3x      71%   (6 P-cores)
   8   29.7s  5.1x      63%   (+2 E-cores)
  16   24.3s  6.2x      39%   (+8 E/LP-E cores)
=> near-linear (82-94%) while adding the 5.3GHz P-cores; efficiency collapses as the
slower E-cores (4.6) and LP-E (2.5) join + throttling under full load. The algorithm
scales well per fast core; 16x needs 16 UNIFORM full-speed cores, which this hybrid
laptop (6P+8E+2LP) does NOT have. server1 abandoned (i9 single-thread ~12x slower /
loaded -> serial baseline timed out, no clean numbers). Conclusion: ~80-90% per-core
efficiency on uniform fast cores would give ~13-14x on a real 16-core box; the 6.2x
here is the hybrid+throttling hardware, not the work-stealing.

## ENV VARS + IDLE% + nnenum table (2026-06-02)
BUG: v3 imported numpy BEFORE vibecheck's __init__ -> OPENBLAS_NUM_THREADS unset ->
OpenBLAS ran 16 threads. FIXED: set OPENBLAS/OMP/MKL/NUMEXPR/VECLIB=1 at top of
star_bab_v3.py before `import numpy`. Impact MARGINAL (6.2->6.5x) -- acasxu's 50x5
matmuls stay single-threaded below OpenBLAS's threshold, so no real oversubscription.
Gurobi: all 4 models already Threads=1 (verified). Good hygiene regardless.

IDLE% (3_3 prop_2, instrumented work vs steal-wait time per worker):
  nw=8:  28% idle, work=172s (+10% vs 157 serial -> throttling)
  nw=16: 42% idle, work=230s (+46% throttling). nw=16 loses to BOTH idle + throttle.
  If idle->0: nw=16 would be 230/16=14.4s -> ~11x (throttling-capped on this HW).

SCALING TABLES (3_3 prop_2, same hybrid laptop):
  v3 WS:  nw 2/4/8/16 eff 94/85/66/41%  (speedup 1.9/3.4/5.3/6.5x, base 157s)
  nnenum: np 2/4/8/16 eff 98/92/78/53%  (speedup 1.9/3.7/6.2/8.5x, base 27.3s)
nnenum ~12pts more efficient at every level. Its edge: SPLIT_IF_IDLE (force-split to
keep cores busy -> less idle) + lighter GLPK (less per-LP work -> less throttle
pressure). Both decline on the hybrid cores. v3 does LESS total work (no redundant
splits) so if idle were closed v3 could MATCH/BEAT nnenum (~11x ceiling). Next: cut idle.

## IDLE CUT: chunk=0 + nw=14 -> 7.7x (2026-06-02)
Idle-responsive donation (donate when idle.value>0, give oldest/biggest half): no
change alone. chunk=0 (defer CHILDREN immediately = finest grain) cut idle 43%->31%
-> 7.4x. Sweet-spot search (chunk=0): nw8=5.7x(71%), nw12=7.3x, nw14=7.7x(best),
nw16=7.5x. The 2 LP-E cores @2.5GHz HURT nw=16 (more idle, little compute) -> nw=14
(6P+8E) is best. Defaults now chunk=0.
PROGRESSION: 6.2x -> 6.5x(BLAS) -> 7.4x(chunk0) -> 7.7x(nw14). vs nnenum 8.5x.
Residual gap: ~32% idle (BaB tail) + ~44% throttle-inflation (hybrid cores). Throttle
caps even zero-idle at ~11x here. To reach nnenum's 8.5x: SPLIT_IF_IDLE (redundant
work to fill tail) -- a philosophy change (v3 minimizes work; nnenum fills cores).

## SPLIT_IF_IDLE: measured, NO difference -> reverted (2026-06-02)
Added force-split-when-idle in the prune branch. Result: splits IDENTICAL (6487),
timing identical (nw14 7.7->7.6x, nw16 7.6->7.4x). It NEVER triggered: v3 prunes at
deep nodes whose layer-li neurons are already stable (no unstable neuron to split on),
unlike nnenum where the EXPENSIVE overapprox is what SPLIT_IF_IDLE distributes. v3's
overapprox is one cheap pass -> nothing to distribute. Reverted per "keep only if it
helps". FINAL v3 parallel: 7.7x @ nw14 chunk=0 (vs nnenum 8.5x); gap = tail-idle (31%)
+ hybrid-core throttle (+44% work, ~11x HW ceiling). Algorithm sound; HW is the cap.

## PHASE MICROBENCHMARK v3 vs nnenum (2026-06-02) — 1_1 prop_3 serial
Added PHASE_T timers in v3 (gated PHASE_ON). nnenum via its TIMING_STATS Timers tree.
  phase              v3 (Gurobi)            nnenum (GLPK)
  LP solve (bound)   2.20s 65% 14791@0.149ms  0.32s 6183@0.052ms   <- 6.9x gap
  spec check         0.24s  7% 148@1.65ms      0.08s              <- 3x
  gurobi model build 0.21s  6%                 ~0 (persistent LP)
  relax build (zono) ~0.13s                    ~0.18s             ~even
  other (matmul/rec) 0.58s 17%                 small
  TOTAL              3.40s                      0.86s             <- 4x
LP-solve 6.9x gap = 2.4x MORE LPs (v3 both-bounds branch + L1 rep) x 2.9x SLOWER
per-LP (Gurobi 0.149ms vs GLPK 0.052ms on tiny 50-var LPs).
GUROBI TUNING TESTED (3000x tiny LP, warm reused model): default 0.116ms, Presolve=0
0.116 (no change), Method=0 primal 0.195 (WORSE), warm 0.196. => Gurobi per-LP NOT
tunable below ~0.12ms; GLPK 0.052ms is inherently ~2x faster. Server1 invalid for this
(its Gurobi WLS license = 15x slower per model: 12.56s vs 0.82s/2000 LPs).
CONCLUSION: v3's 4x single-thread deficit is LP-solve-dominated, split ~evenly between
(a) 2.4x LP COUNT [needs affine-baking rep change, hard] and (b) 2.9x PER-LP ENGINE
[needs GLPK/lightweight LP swap, not a Gurobi tweak]. Neither is a quick knob. Model
build (6%) is NOT the bottleneck (earlier hypothesis wrong).
