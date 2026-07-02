# vibecheck clean-slate design study

Status: analysis only, no code changes. Written 2026-07-01 from a full survey of the
codebase (58.8k lines, 324 settings keys). Part 1 inventories every verification and
falsification method in the tree. Part 2 proposes how the same functionality could be
organized in a small, low-duplication codebase.

## Part 1: method inventory

### 1.1 Sound bounding methods

| # | Method | Implementations today | Gurobi? | Notes |
|---|--------|----------------------|---------|-------|
| B1 | Interval (IBP) forward | `verify_milp._ibp_forward_graph_batched` | no | used as cheap BaB refresh |
| B2 | Forward zonotope | `zonotope.py` (dense numpy + torch), `patches_zonotope.py` (conv patches), `batched_zono.py` (leading-B batch) | no | 3 implementations |
| B3 | Forward LiRPA linear bounds | `forward_lirpa.py` (interpreted, batched, streaming, traceable), `bounded_module.py` (codegen-compiled clone) | no | 2 implementations, same math |
| B4 | Backward CROWN | `alpha_crown._crown_backward_matrix` (dense general), `alpha_tighten` (per-target layer tightening), `patches_crown` (conv patches, min-area), `attn_crown` (attention/bilinear) | no | 4 implementations |
| B5 | alpha-CROWN (optimized slopes) | `alpha_crown.run_alpha_crown[_batched]`, per-disjunct in phase 2.5, warm-started per query in phase 8, `attn_crown_alpha` | no | batched, GPU, per-open-disjunct |
| B6 | beta-CROWN (split-constraint dual) | `alpha_crown` `split_beta`, `attn_crown` beta/rbeta (richest: ReLU + bilinear value splits), `verify_milp._crown_bab_noreforward` beta_params | no | 3+ sites |
| B7 | Closed-form LP (box + halfspaces) | `box_halfspace.lagrangian_min` (1 cut, exact O(n log n)), `lagrangian_n.lag_subgrad_min` (N cuts, subgradient) | no | Gurobi-avoidance |
| B8 | Dual ascent on the alpha-zono LP | `dual_ascent_bab.py`, `fast_dual_ascent/` (topk successor, torch.compile, ~53x faster) | no | validated to match Gurobi LP verdicts; state built 3 ways (`verify_gen_lp.precompute_gen_state`, `gen_state_backward`, `reverse_g`/`reverse_batched`) |
| B9 | LP triangle relaxation (solver) | `verify_gen_lp.solve_spec`, per-neuron tighteners in `verify_graph` phase 1 / `verify_milp` | yes | phase 7 + phase 1 tightening |
| B10 | Exact MILP (big-M ReLU binaries) | `verify_milp.py`, phase 8 racing in `verify_graph` | yes | the only exact/complete solver |
| B11 | Per-layer bound tightening | phase 1 `bab_refine` cascade (LP/MILP/probe), `alpha_tighten` (GPU), INVPROP-style output constraints | mixed | three coordinate forms (alpha / phase1 / alpha_zono) |

### 1.2 Branch-and-bound / splitting (10 drivers, 4 split types)

| # | Driver | Splits | Score | Bound engine | Queue |
|---|--------|--------|-------|--------------|-------|
| S1 | `verify_zono_bnb._run_bnb` | input dims | eval-internal + PGD steer | fwd zono + CROWN | list BFS/DFS |
| S2 | `verify_graph._input_split_batched` (+multi-sub) | input dims, top-K -> 2^K | Smart-Branching `|lA|*width` + domain clipping | batched fwd LiRPA / zono + batched CROWN/alpha | GPU tensor frontier |
| S3 | `_verify_nonlinear_graph._split_dim` | input dims | spec-sensitivity x width | fwd zono + spec.check | list DFS |
| S4 | `_verify_trig_nonlinear_split` | nonlinear-op range OR input dim (hybrid) | bbps column slack | fwd zono under op_clamps | list DFS |
| S5 | `_zono_relu_split_close` | ReLU neuron + bilinear value | BaBSR `min(-lo,hi)*|ew|`, chord slack | fwd zono + CROWN + attn beta/alpha | heapq worst-first |
| S6 | `_zono_input_split_close` | input dims | `|w@G|` column | fwd zono | heapq worst-first |
| S7 | `verify_milp._crown_bab_noreforward` | ReLU neurons | BaBSR intercept + kFSB + multilevel | batched alpha+beta CROWN, no reforward | heapq batched |
| S8 | `verify_milp._ibp_crown_bab` | ReLU neurons | ew-weighted width, kFSB | batched IBP reforward + alpha-CROWN | heapq batched |
| S9 | `attn_crown.attn_beta_bab` | ReLU + exp-input value | ew-weighted BaBSR / slack | no-reforward plane+beta CROWN, batch Adam | heapq batched |
| S10 | `dual_ascent_bab.verify_query_dual_ascent_bab` | ReLU + nonlinear (halfspace pairs) | pre-scored upstream (BaBSR) | batched dual ascent | batched tensor frontier |

Shared math lives in `nonlinear_split_planes.py` (planes, split_point, bilinear axis
score) and `nonlinear_split_dual.py`, but queue management, deadline checks, heartbeat,
OOM-halving, BaBSR scoring, and PGD-on-subdomain are copy-adapted per driver. S5/S6 and
S7/S8 are near-duplicate skeletons. All four score families are the same quantity in
disguise: adjoint-weighted relaxation slack that the split would remove.

### 1.3 Falsification (10 descent loops + 2 non-gradient)

| # | Loop | Backend | Oracle | Distinctives |
|---|------|---------|--------|--------------|
| F1 | `pgd.pgd_attack_general` | traced own graph (GPU) | spec.check | canonical: restarts batched, OSI init, optimizer modes, lr decay, per-restart disjunct, plateau, budget |
| F2 | `pgd.pgd_attack_from_init` | same | spec.check | caller-seeded (MILP/DA witnesses) |
| F3/F4 | `verify_zono_bnb._pgd_attack[_graph]` | dense folded / graph | none (raw) | pairwise-only margins, near-clones |
| F5/F6 | `verify_hybrid_acasxu._simple_pgd[_batched]` | graph | none | 10k restarts, OOM-halving, per-leaf boxes |
| F7 | `onnx_torch_runner.pgd_via_onnx` | onnx2torch | none | last resort when own forward unavailable |
| F8 | `surrogate_pgd.surrogate_attack` | quant surrogates (STE/fakequant/saturating) | ORT-CPU | multi-input boxes, strict buffer |
| F9 | `sign_attack.sign_attack` | onnx2torch + Sign-STE | ORT-CPU | Adam+ExpLR, vertex init, preact penalty |
| F10 | `torch_attack.torch_attack` | onnx2torch genuine | ORT-CPU | Adam+ExpLR |
| F11 | `cctsdb_yolo` | ORT | ORT | exhaustive integer grid (complete) |
| F12 | `monotone_invert` | analytic | pipeline | output inversion, no loop |

Each loop re-derives: spec -> (W,b) constraint matrices, min-over-disjuncts
max-over-constraints margin, init, projection/clamp, step rule, witness check. What
actually varies is backend x oracle x init x accept-threshold x schedule.

### 1.4 Orchestration and infrastructure

- `verify_graph.py` (13.7k lines): routing + a ~2570-line `_run_pipeline` with phases
  0 / 0.5 / 1 / 2 / 2.5 / 2.6 / 3 / 3.5 / 7 / 8 / 9 interleaving attack, bound, LP,
  MILP-or-dual-ascent, and final attack. Per-disjunct open-set tracking threads through
  phases (good idea, buried in the monolith).
- SAT chokepoint (`_validate_sat_witness` + `_sat_disposition` + `_finalize` + the
  `main._verify` re-check): a genuinely good, single funnel. Keep.
- Memory chunking: reimplemented at ~76 sites in verify_graph, 58 in forward_lirpa,
  52 in dual_ascent_bab, 12+ files total, with ~24 settings knobs.
- Settings: 324 keys, most of which select among duplicate paths.
- Infra that is already clean and worth keeping as-is: ONNX loader + OP_REGISTRY graph,
  VNNLIB v1/v2 loaders, network-pair merge, nonlinear-augment transpile, config
  auto-detection, results-file discipline, CE emission, both CLIs.

### 1.5 Gurobi dependency, precisely

Only `verify_milp.py`, `verify_gen_lp.py`, `gurobi_util.py` import Gurobi. Removing it
loses exactly two things: solver-exact MILP verification and Gurobi-solved LP
relaxations. The dual-ascent path already reproduces Gurobi LP-relaxation verdicts
(validated match) and beta-CROWN plus splitting recovers completeness in the limit for
piecewise-linear nets. Half a dozen modules exist mainly to avoid Gurobi
(box_halfspace, lagrangian_n, dual_ascent_bab, fast_dual_ascent, gen_state_backward,
reverse_g/reverse_batched).

## Part 2: clean-slate architecture

Design goals, from discussion: all split types coexist under one score function; forward
zono and backward CROWN both sparse and patch-structured, always on; no hard Gurobi
dependency; alpha tightening per open disjunct; one PGD engine; one settings tree; runs
on small GPUs by chunking, just slower.

### 2.1 Layer 0: two data types everything shares

**`LinMap`**: a linear map with multiple physical layouts (dense | conv-patches |
index-sparse | diagonal) behind one interface: compose-with-layer, abs-row-sum
(concretization), matvec, slice-rows. Patches and sparsity stop being alternative
propagators and become layouts of the one map type. Materialize-to-dense is the explicit
fallback, chosen by the memory budget, never by a config flag.

**`RelaxLib`**: one relaxation object per nonlinear op (relu, sigmoid, tanh, sin, cos,
pow, exp, reciprocal, bilinear/McCormick, floor, sign). Each provides: sound lower/upper
planes parameterized by alpha; the clamp semantics of a range split; the split-point
rule; the slack of its current relaxation (for branching scores); and beta hooks for
split-constraint duals. Both propagation directions consume the same objects, so a new
op is added exactly once. (Per CLAUDE.md, planes are closed-form or bracketed-Newton,
never sampled.)

**Memory service**: `chunked(fn, items, bytes_per_item)` in one module. Predictive
first (declared cost per item x free VRAM with a safety factor), optimistic fallback
second (catch CUDA OOM at this one wrapper, empty_cache, halve, retry, re-raise loudly
below a floor). All 12 files' ad-hoc chunk logic and the ~24 chunk knobs collapse into
this. Small-GPU support is then automatic: same code, more chunks.

### 2.2 Layer 1: two propagators, one attack engine

- **Forward propagator** (one implementation): symbolic affine state over input noise
  plus accumulated relaxation generators, generic over LinMap layout. Zonotope is the
  symmetric case; forward-LiRPA's asymmetric planes are the general case. Batched over
  domains by construction (the leading-B dimension is not a variant, it is the shape).
  torch.compile replaces the hand-codegen `bounded_module`.
- **Backward propagator** (one implementation): CROWN adjoint over the same graph,
  same RelaxLib, generic over LinMap (patches mode = today's patches_crown). alpha
  (slopes), beta (split constraints on any op, ReLU or bilinear or range), and gamma
  (output constraints / INVPROP) are optimizer-visible tensors of the one propagator,
  not separate files. Warm-startable per domain and per disjunct.
- **Attack engine** (one implementation): the F1 recipe as the core loop with four
  plug points: forward backend (traced graph | onnx2torch with op-patch hooks for
  Sign-STE and quant surrogates | ORT for validation), init strategy (uniform | vertex
  | center | OSI | seeded), accept policy (strict | boundary | near-miss | strict
  buffer), and projection (box | per-leaf batched boxes | fixed-dim masks | multi-input
  boxes | search-only widening). Everything submits attack jobs to it: warmup, BaB
  leaves, dual-ascent witness refinement, per-disjunct targeting. The discrete-grid
  enumerator stays as one small non-gradient strategy beside it.

### 2.3 Layer 2: one BaB over one domain type

A **Domain** = input box + op clamps (ReLU phase fixes, nonlinear range splits,
bilinear boxes all in one dict) + beta/lambda state + current lb + open-disjunct mask.
Candidate **actions** = split input dim k | fix ReLU (L,j) | split op range (name,j).
One **score**: adjoint-weighted slack removed, which is what SB, BaBSR, bbps, and the
chord-slack scores each compute for their own action type. Scoring all action types in
one ranking is what lets input, ReLU, and nonlinear splits genuinely coexist; the
verifier stops needing an up-front "input-split net vs ReLU-split net" routing decision
(today's `input_split_max_dims` fork).

One **search loop**: batched GPU tensor frontier (the S2/S10 idiom), worst-first
pickout of a memory-budgeted batch, bound cascade per leaf (cheap forward -> backward
alpha/beta with warm start -> optional dual-ascent LP with the domain's halfspaces),
attack on the most promising leaves, deadline from a single Budget object. The five
heapq drivers and three list drivers disappear.

### 2.4 Layer 3: verifier cascade without Gurobi

Per open disjunct target: forward bound -> backward alpha-CROWN (per-disjunct alphas)
-> BaB with beta and dual-ascent leaf certification. Pure torch end to end. Gurobi
becomes an optional plugin implementing the same `QueryVerifier` protocol (useful for
exact MILP on small FC nets and as a cross-check oracle in tests), imported lazily,
never required. The closed-form halfspace LPs remain as the dual-ascent inner kernels.

### 2.5 Layer 4: a small orchestrator

Replace the phase cascade with a scheduler over three primitives (attack, bound,
branch) and one piece of state: the open-disjunct set with per-disjunct lb and history.
Loop: while budget remains and disjuncts are open, pick the cheapest promising move
(attack new seeds, tighten alphas, or advance BaB) for the worst disjunct. Phases 0,
0.5, 1, 2, 2.5, 2.6, 3, 3.5, 7, 8, 9 are then schedules, not code paths. Config
auto-detection survives but selects budgets and a few strategy weights instead of
choosing among duplicate pipelines.

Special routes shrink to strategy plugins at this level: quant-surrogate attack,
sign-STE attack, discrete enumeration, monotone inversion, per-disjunct subbox
decomposition, network-pair and nonlinear-augment (which are already front-end rewrites
and stay untouched).

### 2.6 Settings and infra

One typed settings tree mirroring the layers: `attack.*`, `bound.*`, `branch.*`,
`memory.*`, `output.*`, plus per-benchmark yaml overrides. Estimate 50-70 keys instead
of 324, since most current keys select between duplicated paths that no longer exist.
Keep unchanged: ONNX front end, spec loaders, pair merge, augment transpile, SAT/ORT
validation chokepoint, results-file discipline, CLI (legacy + VNN-LIB standard),
config detection skeleton.

### 2.7 Size estimate and what is genuinely at risk

Rough core: LinMap+RelaxLib+memory ~2k; forward+backward propagators ~2.5k; attack
engine ~1k; BaB ~1.5k; dual-ascent verifier ~1.5k; orchestrator ~1k; front/back ends
kept ~4k. Total ~13-15k lines against 58.8k today, with one implementation per concept.

Honest risks: (1) exact MILP wins on some hard small-FC UNSAT instances; mitigated by
the optional plugin. (2) The four CROWN implementations encode benchmark-specific
performance tricks (no-reforward vs IBP-refresh, fp32-search/fp64-certify, codegen);
these must become options of the one propagator, and parity needs per-benchmark
verdict+time gates before deleting anything. (3) The unified score needs calibration
across action types (slack units differ); start by normalizing each family's score to
estimated-lb-improvement on probe splits.

### 2.8 Realistic path

Strangler, not rewrite-in-place: build the core under `vibecheck2/` (or a new repo)
with the old pipeline kept as the verdict oracle; port one benchmark family at a time
gated on verdict parity and time-ratio budgets using the existing integration-test
harness; delete old paths only when their last benchmark ports. The front end, spec
machinery, validation chokepoint, and CLIs move over unchanged on day one, so the
rewrite is only the middle ~40k lines.

## Part 3: architecture detail

### 3.1 Module tree

```
core/
  graph.py          # DAG of ops (ONNX-loaded), forks and merges first-class
  linmap.py         # LinMap: dense | conv-patches | index-sparse | diagonal
  relax.py          # RelaxLib: per-op planes, alpha params, split semantics, slack
  memory.py         # budget service: chunked(fn, items, bytes_per_item), OOM backstop
  forward.py        # symbolic forward state (zono / asymmetric planes), DAG-native
  backward.py       # CROWN adjoint walk, alpha/beta/gamma tensors, DAG-native
  attack.py         # one PGD engine: backend/init/accept/projection plug points
  domain.py         # Domain (box + clamps + duals + lb), action enumeration + score
  search.py         # batched BaB frontier, pickout, bound cascade, deadline
  dual_lp.py        # dual-ascent leaf certifier (fast_dual_ascent topk successor)
  scheduler.py      # open-disjunct loop over attack/bound/branch primitives
  budget.py         # one Budget object: wall deadline, phase shares, heartbeat
frontend/           # onnx_loader, optimizer, vnnlib v1/v2, pair merge, augment  (ported as-is)
verdict/            # spec margins, SAT/ORT chokepoint, CE emit, results file    (ported as-is)
cli/                # legacy + VNN-LIB standard CLIs                             (ported as-is)
handlers/           # one-off benchmark strategies USING the core (see 3.6)
solvers/gurobi.py   # optional exact-MILP plugin, lazy import, test oracle
configs/            # per-benchmark yaml, config detection
```

Core target is ~10k lines. `handlers/` is allowed to be ugly; core is not.

### 3.2 General graphs: forks, merges, residuals

Both propagators are DAG-native from day one (today `batched_zono` raises on merges
and several paths assume sequential nets; that assumption is banned in core).

- Forward: state per tensor edge; a fork copies the reference; a merge (Add of two
  branches) aligns generator IDs so shared noise terms recombine exactly (the existing
  `DenseZonotope.add` semantics). Generator IDs are global to the run, not per-branch.
- Backward: adjoints of a forked tensor sum over consumers; the walk is a reverse
  topological sweep over the DAG, not a layer list. Residual blocks then need no
  special casing anywhere else.
- Test nets for this: cifar100/tinyimagenet resnets, ml4acopf `-residual` variants,
  soundnessbench `model_residual`, dist_shift concat nets, nn4sys forked duals.

### 3.3 Generator lifecycle: grow, reduce, drop, continue

Forward symbolic state grows one generator per unstable nonlinearity. Policy, driven
by the memory service rather than config flags:

1. **Grow** while the projected cost (generators x width of remaining layers, a cheap
   shape-only estimate) fits the budget.
2. **Reduce** when near budget: consolidate the K least important generator columns
   into per-neuron interval slack (importance = column norm |G_j|, optionally weighted
   by a one-shot backward adjoint estimate). Sound over-approximation, standard
   zonotope order reduction. Patches columns consolidate per-patch.
3. **Drop and continue** when even reduced state is too big (user-observed pattern:
   zono forward + CROWN backward works until generators explode): concretize the
   symbolic state to bounds at a cut layer, then restart a fresh symbolic state from
   that box. Backward CROWN treats the cut as a virtual input. Multiple cuts allowed;
   this is what makes vggnet-class nets fit small GPUs, just with looser bounds.
4. The same estimate decides dense materialization vs staying in patches.

The cut decision is a soundness-preserving accuracy/memory tradeoff, so it can be
auto-tuned: try to verify with the bounds obtained; if open disjuncts remain and
budget allows, redo the worst cut with a higher generator allowance (BaB-like restart
at the representation level).

### 3.4 Nonlinear ops: bounding and splitting as one contract

Each RelaxLib entry implements one interface; the table is the complete nonlinear
surface of the current tree:

| Op | Planes (bound) | Split semantics | Score contribution |
|----|----------------|-----------------|--------------------|
| relu | triangle / adaptive slope (alpha) | phase fix (2 children) | BaBSR intercept |
| sigmoid / tanh | tangent-chord, bracketed Newton | range split at 0 or midpoint | column slack |
| sin / cos | monotone-piece chords | range split at extremum | column slack |
| pow / exp / reciprocal | convex tangent + chord | range split at midpoint | chord slack |
| bilinear (Mul, MatMul-var, Div) | McCormick | box split on wider factor | McCormick gap |
| floor / sign | step envelope | integer-piece split | piece count |
| softmax / attention | composed exp+recip+sum (no monolith op) | via constituents | via constituents |

Rules carried over from CLAUDE.md: planes are closed-form or provably bracketing,
never sampled; adversarial sampling is used to TEST planes, not to compute them. The
split-point rule and slack live with the op (today: `nonlinear_split_planes.py`), so
the unified BaB scorer ranks a sigmoid range split against a ReLU fix against an input
split with no per-family driver.

### 3.5 The bound cascade per BaB batch, concretely

For a picked batch of domains (one GPU tensor, leading dim B):

1. Interval forward under clamps (cheap, prunes trivially-verified leaves).
2. Symbolic forward (zono) under clamps with the generator lifecycle of 3.3.
3. Backward alpha-CROWN from the open disjuncts' output rows, warm-started alphas
   and betas from the parent domain, K Adam steps (K small, e.g. 10 to 20).
4. Leaves still open and worth it (lb close to 0): dual-ascent LP certification with
   the domain's split halfspaces (`dual_lp.py`).
5. Attack the most promising leaves (lowest lb margin) with seeds from the LP primal
   and the parent's best adversarial point.

Every step is optional per budget; on tiny nets 1+3 alone match today's fast paths.
fp32 search with fp64 re-certification before pruning (the attn_crown trick) is a
cascade option, not a separate implementation.

### 3.6 handlers/: one-off benchmarks on top of the core

Explicit non-goals for core; each is a small strategy file consuming core APIs:

- `handlers/quant_surrogate.py`: smart_turn INT8 path. Builds float/fakequant/
  saturating STE surrogates, runs core attack.py with the surrogate backend and
  ORT oracle. (today: surrogate_pgd + saturating_quant, ~1.4k lines)
- `handlers/mega_disjunct.py`: nn4sys-style specs with thousands of disjuncts each
  carrying its own input subbox. Groups disjuncts by subbox, runs the core verifier
  per group with batched fast-CROWN pre-screening (today: _verify_per_disjunct_
  subboxes, 812 lines in verify_graph).
- `handlers/discrete_enum.py`: cctsdb_yolo integer patch-position enumeration.
- `handlers/monotone_invert.py`: sigmoid-head output inversion (ml4acopf route).
- `handlers/sign_bnn.py`: Sign-STE attack config (mostly just attack.py options).
- `handlers/hybrid_acasxu.py`: only if the unified input-split BaB does not already
  subsume it (expected to; acasxu is the easiest family in the matrix).

Handlers register with config detection the same way configs do now. A handler may
pre/post-process (like pair merge and nonlinear augment already do) but must emit
verdicts through the one chokepoint.

## Part 4: testing plan

### 4.1 Ground truth sources

- `~/repositories/vnncomp2026_results_official/{vibecheck,alpha_beta_crown}/results.csv`
  (3446 joined instances, 28 benchmarks). Agreement of both tools = assumed ground
  truth; disagreements are hand-classified (all current ones are tolerance-boundary
  cases or soundnessbench planted-CE instances, see 4.4).
- Every `sat` is additionally self-validating via the ORT replay chokepoint, so sat
  ground truth never depends on another tool.

### 4.2 Tier 0: core unit tests (no benchmarks, seconds)

- RelaxLib soundness per op: planes bracket the function on dense adversarial samples
  including endpoints and extrema (samples VALIDATE, never define, the bound).
- LinMap layout equivalence: dense vs patches vs sparse give bit-compatible
  concretizations on random conv stacks.
- Forward/backward duality: backward CROWN lb equals the direction-adaptive forward
  reconstruction (the existing alpha_crown invariant) on random DAGs with forks,
  merges, residuals.
- Generator lifecycle: reduce and drop-and-continue are sound (bounds only widen)
  and deterministic; a forced 1-generator budget still verifies a trivial property.
- beta/gamma monotonicity: adding a split constraint or output constraint never
  loosens the bound; zero multipliers reproduce the plain bound exactly.
- Attack engine: each backend/init/accept combination finds the planted CE on a
  2-layer toy net; witness always passes the ORT chokepoint.
- Memory service: predictive path never OOMs on a synthetic 100x overestimate; the
  fallback path halves, logs, and re-raises at the floor.
- Determinism: fixed seed -> identical verdict, bounds, and CE bytes across runs.

### 4.3 Tier 1: curated parity matrix (minutes, per merge)

Picked from the joined results (regenerate with scratch/clean_slate/pick_parity_cases.py (current output: scratch/clean_slate/named_cases_2026.txt); source of truth is
the two results.csv files). Three cases per benchmark where available: easiest
agreed sat, easiest agreed unsat, hardest vibecheck-solved (borderline). Selected
anchors, chosen to cover net types:

| Class | Instance (net / prop) | Expected | Note |
|---|---|---|---|
| FC input-split | acasxu 2_1/prop_2 sat 2.9s; 4_8/prop_3 unsat 3.5s | sat/unsat | easy smoke |
| FC pairs | monotonic 4_1/instance_5 sat; isomorphic 1_9/instance_16 unsat; iso 2_6/instance_3 unsat 98.6s | | pair merge + borderline |
| conv resnet | cifar100 resnet_medium idx_5001 sat 3.8s; idx_4429 unsat 10s; idx_8945 unsat 87.9s (ab timeout) | | residual + vc-only win |
| conv large | vggnet16 spec0_suit sat 41s; spec3_canoe unsat 40s; spec14_mink unsat 147.5s; spec15_tiger_beetle (vc=error today) | | generator-drop stress; fix the error case |
| transformer | vit ibp_3_3_8_3949 unsat 94.4s; ibp_3_3_8_6769 (vc timeout, ab 52s) | | attention beta BaB; a target to win back |
| maxpool quant | traffic_signs model_30_idx_7573 sat 4.6s; model_48_idx_10645 (vc unknown, ab sat 10s) | | target to win back |
| nonlinear ops | lsnc quadrotor2d_state_51 sat / _33 unsat; ml4acopf 14_ieee prop1 sat 1.8s / prop4 unsat 1.9s; 300_ieee prop3 unsat 23.4s (ab timeout) | | trig/sigmoid planes |
| nonlinear spec | adaptive_cruise instance_6 sat / instance_14 unsat / instance_42 (ab-only unsat, target) | | augment route |
| mega-disjunct | nn4sys lindex_1 unsat 2.6s; mscn_2048d_dual cardinality_1_960 unsat 80.2s (ab errors); cardinality_1_1 (ab-only 12.9s, target) | | handler |
| big conv BaB | tinyimagenet idx_9262 sat; idx_7473 unsat; idx_62 unsat 67.2s (ab timeout) | | |
| misc easy | cersyve, cgan2026, collins_rul, cora, dist_shift, linearizenn, malbeware, metaroom, safenlp, sat_relu, tll, yolo easy pairs | | one sat + one unsat each |

Gate per instance: verdict matches expectation; sat witness passes the vendored
competition checker; wall time within 1.5x of old vibecheck (ratchet down over time).

### 4.4 Tier 2: boundary and soundness gates (nightly)

- Tolerance-boundary set (current vc/ab disagreements, all expected-behavior):
  ml4acopf 118/300_ieee prop3+prop4 (within-tol sat vs unsat both accepted),
  adaptive_cruise instances 11/30/41/44, nn4sys mscn_2048d cardinality_0_500.
  Gate: verdict is sat-with-valid-witness OR unsat OR unknown; a sat here must be
  CORRECT or CORRECT_UP_TO_TOLERANCE under vnncomp_cex_v2; never a rejected witness.
- soundnessbench (planted CEs): never output unsat on an instance with a planted CE
  (model_residual property_025/048 are current vc sat wins; keep them). This is the
  primary anti-unsoundness benchmark.
- `--disable-sat-finding` probe on all Tier-1 sat cases: bounds path must never say
  unsat (existing soundness gate, kept).
- Full old-vs-new differential: run both pipelines on a rotating random 100-instance
  sample; any verdict flip unsat->sat or sat->unsat is a release blocker until
  hand-classified.

### 4.5 Tier 3: performance and small-GPU (weekly / pre-release)

- Time-ratio dashboard on the full 3446-instance matrix vs the 2026 official runs.
- Small-GPU emulation: cap the memory service at 4 GB and 2 GB; gate = identical
  verdicts on Tier 1 (slower is fine, OOM or verdict change is not).
- torch.compile on/off parity run.

### 4.6 Milestone order

1. core/{graph, linmap, relax, memory, forward} + Tier 0. Exit: point-prop parity
   with ORT on every benchmark net (the existing vnncomp point-prop test style).
2. backward.py (alpha) + verdict/ port. Exit: Tier-1 easy-unsat rows that today
   verify without BaB.
3. attack.py. Exit: all Tier-1 easy-sat rows; CE bytes validated.
4. domain/search/dual_lp (BaB with unified score). Exit: borderline rows; acasxu,
   cifar100, tinyimagenet full parity.
5. Generator lifecycle + patches always-on. Exit: vggnet16 incl. the current error
   case; 4 GB small-GPU gate.
6. attention plane/beta options in relax+backward. Exit: vit parity, then the two
   ab-only vit/traffic wins as stretch targets.
7. handlers/ ports (quant_surrogate, mega_disjunct, discrete_enum, monotone_invert).
   Exit: full Tier-1 + Tier-2 green; old pipeline deleted benchmark by benchmark.

## Part 5: implementation log (updated as built; measurements, branch clean-slate)

Status after the first build night (2026-07-02, src/vibecheck2/):

- M1-M4a landed: IR/LinMap/RelaxLib/memory/forward, backward alpha/beta CROWN,
  attack engine, input-split BaB, relu-split BaB (no-reforward), subbox
  decomposition, pair merge + nonlinear augment front ends, mega-disjunct
  screening. 42 Tier-0 tests.
- Measured decisions worth keeping:
  - Per-edge CROWN-refined intermediates (identity queries, chunked, only the
    interval-ambiguous neurons) are THE decisive tightener: acasxu BaB domains
    at depth 13 were 85% open under zono intermediates, 0% under refined ones;
    tinyimagenet resnet unsat went unknown -> 6s (v1 15.5s, abcrown timeout).
  - ABC-style input clipping + per-restart-disjunct PGD targeting + OSI init
    recovered the remaining acasxu rows (186/186, prop_7 sat 3.1s).
  - VNNLIB constraints are NON-strict: candidate acceptance must be <= 0
    (sat_relu rows sit exactly on the boundary); the ORT chokepoint stays the
    only authority.
  - onnx.shape_inference must be the ND-shape oracle; v1's recorded shapes go
    stale around broadcasts (ml4acopf concat declared 160 vs ORT 186).
  - alpha/beta tensors per (domain, query) explode on conv nets; share across
    queries ((B,1,n)) when the full size passes ~1GB.
- Categories at parity (60s dev budget): acasxu 186/186, easy/sat suites 33/33,
  fast-category sweep running clean (0 misses in the first 160), ml4acopf +
  adaptive_cruise + nn4sys(mscn_128d/lindex) + pairs + cgan/lsnc anchors pass.
- Open: conv borderline rows (cifar100/tinyimagenet hard unsat) need relu-BaB
  bound quality (per-domain intermediates refresh / patches); vit needs the
  bmm McCormick adjoint (softmax decomposition + interval bmm are in); vgg
  needs the generator lifecycle + patches (M5); handlers for quant surrogate /
  cctsdb enumeration / monotone invert not started; augment sat CEs validate
  on the augmented net pending the strict original-spec disposition.
