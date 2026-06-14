# challenging_certified_training_2026 — vibecheck benchmark record

VNNCOMP 2026 regular track. **60 instances** across 6 IBP-trained CNN7s
(5 Conv + 2 Gemm, pure ReLU, linear chain — no forks): cifar10 eps2,
cifar10 eps8, tinyimagenet eps1, each in a **normal** and a **wide**
variant. Per-instance timeouts 30 / 120 / 550 s. `instances.csv` ships
CRLF line endings (bash sweeps need `tr -d '\r'`).

The benchmark name is literal: the nets are *certified-trained* (IBP),
so they are saturated — very few unstable neurons, but the ones that
remain sit on a knife-edge. The hard queries need branch-and-bound, and
the right BaB engine depends on the perturbation size.

## The core idea: a structural 3-way routing gate

There is no single engine that wins across the benchmark. The selector
is a **structural gate** in `verify_graph.py` keyed on input perturbation
width (`milp_route_pert_threshold`) and budget — not a hybrid pipeline:

1. **HIGH input uncertainty** (cifar10 **eps8**, mean box width ~0.31) →
   `milp_verify`'s GRAPH path: **IBP Phase 1 + α-CROWN + no-reforward
   β-CROWN BaB**. The large perturbation loosens the forward zonotope
   enough that CROWN's triangle relaxation + per-domain β-CROWN BaB wins.
   This closes the hard eps8 UNSAT queries (9566_s3 unsat ~91 s, 8231_s3
   unsat ~41 s) that nothing else here closes.
2. **tinyimagenet** (12288-dim input) → same milp_verify engine via the
   **input-dim gate** (`phase1_ibp_input_dim_threshold: 8000`); its zono
   generators OOM a 24 GB GPU, so IBP Phase 1 is mandatory.
3. **LOW input uncertainty + long budget** (cifar10 **eps2**, ~0.078, the
   120/550 s harder samples) → the graph pipeline's **GPU dual-ascent
   BaB** (`phase8_fast_dual_ascent`, sound far-probe cert, ~3 µs/node,
   factored shared-`a_g` GEMM → multi-million-node frontiers without OOM).
   The no-reforward β-CROWN BaB *explodes* on these loose roots (stale
   intermediate bounds → frontier blowup); dual-ascent brute-forces them
   (e2c s2/s3 unsat 36/39 s, e2w s3 87 s).
4. **LOW input uncertainty + short budget** (eps2, the easy 30 s samples)
   → the fast layers Phase 1 (zono + CROWN), which verifies them outright
   at +1.3 in ~3 s. Routing these through the IBP+BaB engine regressed
   them (IBP root far looser → timeout); the budget sub-gate
   (`milp_route_dualascent_min_budget: 60`) keeps them on the fast path.

eps8 (~0.31) vs eps2 (~0.078) separate with a 4× gap; the 0.15 threshold
sits safely between. The gate overrides `milp_force_graph_path` /
`milp_force_ibp_phase1` per-instance for the conv nets; the config values
are the tinyimagenet (input-dim-gated) defaults.

## Algorithmic wins vs the α,β-CROWN reference

### 1. No-reforward β-CROWN BaB (`_crown_bab_noreforward`, verify_milp.py)

The eps8 nets have **tight α-CROWN roots** (9566 q3: IBP→α reaches −0.065).
The validated recipe — four load-bearing pieces:

- **No-reforward**: keep the root α-tight intermediate bounds; per domain,
  intersect *only the split neuron's own bound* with the clamp. Do **not**
  IBP-re-forward (that loosens deeper layers on tight-root nets: split
  −0.065 → −0.8).
- **BaBSR triangle-intercept split score**: `|clamp(ew, max=0)| ·
  (−lo·hi / (hi−lo))` — the ReLU triangle's max relaxation gap (ABC's
  `babsr_score` intercept term). NOT width (width plateaus).
- **Per-domain `ew`** (the final bug): recompute the backward coefficients
  with *this domain's* split-tightened bounds, not the stale root `ew`.
  Scoring deep domains with the root `ew` picks wrong neurons → plateau.
- **kfsb candidate evaluation + multilevel**: per open domain, take the
  top-`prefilter` candidates per layer by score, evaluate each candidate's
  2 children with a cheap `cand_iters`-step α+β bound, branch the top
  `multilevel` simultaneously (2^multilevel children).

**Feed the BaB a CLEAN, self-consistent per-query base.** Tightening the
intermediate bounds *more* (joint-α merge, cross-query cumulative bounds,
a separate `tighten_big_layers` pass) — though all sound — makes the
per-domain bounds noisier and the BaBSR scoring worse, so the same query
swings 41→449 domains and times out. Fix: `sb_bab_base` = Phase-1 bounds +
only query-*independent* tightening; each query's BaB runs on
`sb_bab_base ∩ that-query's-own root_bounds`. This made it deterministic.

cifar10 must use **IBP Phase 1, not zono**: zono is a tighter enclosure
but yields a *worse* α-CROWN root here (9566 q3: IBP→α −0.066 closes;
zono→α −0.295). Config routes both families through IBP.

### 2. GPU dual-ascent for the loose-root eps2 cluster

Measured: ABC closes eps2_cnn7 s2 in 29 s *without* `clip_interm_domain`
(ablated on=45 s / off=29 s — clip is not the key; it's standard β-CROWN).
Our roots are comparable (−0.379 ≈ ABC −0.360) and we *do* have per-spec
α. The real gap was the no-reforward BaB keeping stale intermediate bounds
on loose roots → frontier explodes (18k domains / 1421 s). The dual-ascent
brute-forces it instead — the right engine for loose-root small-eps.

### 3. Targeted SAT-finding PGD (focus on open disjuncts)

Two SAT cases were missed (idx5613_s3 tinyimagenet, ABC sat 16 s; and
**idx1247_s9 eps8_cnn7, ABC sat 8 s — the only real ABC win on the whole
benchmark**). Root cause: the milp graph path ran ONE joint-loss PGD over
all disjuncts. On idx5613, Phase 1 left only **4 of 199 open** (at high
indices); their gradient was diluted among the 195 verified and the attack
never descended the open basins → sat=False, then the slow tightening ate
the budget → unknown@117 s. Fix: when `0 < n_open <=
milp_graph_targeted_pgd_max_open`, restrict the attack to the open
disjuncts with `per_restart_disj=True` (each gets dedicated restarts) and
a larger pool. The small-open gate is a SAT signal so a hard UNSAT case
with many open disjuncts skips the sweep; the time cap + plateau give-up
bound the cost on few-open UNSAT cases (measured: 9566_s3 +1.5 s, 8231_s3
+1.9 s, idx1038_s3 +1.4 s — all still verify well within budget).

**Restart count matters — the eps8 nets have NARROW CE basins.** At
`milp_graph_targeted_pgd_restarts=128`, idx5613 was 8/8 but idx1247 was
only ~4/5 reliable (the focused PGD missed the basin on some seeds and the
case fell through to a timeout). OSI / diversified init (ABC's trick) did
NOT help (6/8). **256 restarts gives 8/8 on idx1247** (CE in ~1 s) and
keeps idx5613 at 5/5 — so the config sets 256. Still batched + sub-second.

### 4. Memory-aware caps (avoid 24 GB OOM, no soundness loss)

- **α start-node cap is a MAX-LAYER memory budget**
  (`milp_graph_alpha_start_mem_elems`): the non-chunked per-target α
  backward peaks at `n_targets × max_layer × 4 B` (cifar10-normal L1 =
  65536² × 4 = 17.2 GB, measured). A `n_targets × n_input` model is wrong
  and OOMs the wide cnn7 (131072-neuron layers = 68 GB). At 5 G elems the
  normal-L1 stays a start node; wide/tiny giant layers route to the
  chunked tightener.
- **α-refresh layer cap** (`phase1_alpha_refresh_mem_elems`): the wide
  cnn7's 131072-neuron layers OOM the per-target α-refresh backward. At
  5 G the wide big layers keep their Phase-1 bounds (the normal cnn7's
  65536-neuron L1 at 4.3 G still fits). This *reduces* memory rather than
  disabling α-refresh wholesale — measured to fix the two eps2_wide OOM
  errors (idx4031_s9, idx8961_s8) with no verdict regression.

## Final score

**Full 60-case sweep** (A10G, `cct2026_fullsweep_final`, verdicts from
`--results-file`):

| Solver | Solved / 60 | breakdown |
| --- | --- | --- |
| **vibecheck** | **32** | 13 sat + 19 unsat (incl. idx7018_s4); the rest timeout/unknown |
| **AB-CROWN** | **33** | 12 sat + 21 unsat; 9 timeout, 18 NONE (OOM) |

**Head-to-head (both swept on the A10G):** 32 ties · 27 both-fail · **0
we-win · 1 ABC-win**. We are **behind ABC by 1** (`cct2026_fullsweep_final`
+ the idx7018 fix vs `cct2026_abc_fullsweep`):
- **idx9074_s6** (eps2_wide): ABC unsat 78 s; us timeout. Investigated by
  ABC ablation like idx7018, but it's the OPPOSITE problem — a **root-bound
  gap**, not throughput: our dual-ascent root is -0.69 vs ABC's -0.37 (1.9×
  looser), so its frontier explodes. Re-routing idx9074 to the milp-graph
  IBP+α engine DOES recover ABC's tight root (-0.37, and q2 of 2 closes), but
  the remaining query q6's BaB frontier still **explodes** (4800+ domains,
  growing) where ABC keeps its bounded (≤159) via β-CROWN + last-2-layer
  kfsb — a branching/β-strength gap our no-reforward BaB can't match even
  from the tight root, with fast throughput, or with split_deepest. Genuine
  limitation; left unsolved (this is the 1 remaining ABC win).

**idx7018_s4 — CLOSED (the ABC ablation paid off).** I twice wrongly
concluded it needs ABC's per-domain `optimize_interm_bounds` (interm-refresh).
A rigorous ABC ablation **refuted that**: disabling ABC's `clip_interm_domain`
left its node count unchanged (1109→1114) and 20 s faster. Our ROOT BOUNDS
already MATCH ABC's (per-layer unstable counts identical, root lb -0.175 vs
-0.172). ABC closes in ~1110 domains via β-CROWN + last-2-layer kfsb; our
`_crown_bab_noreforward` is the SAME kind of engine but was ~17× slower per
domain (batch 16 + 20 α-iters) so it only reached 521 domains before timing
out. **Speed-tuning it (batch 192, α-iters 8, cand 2) closes idx7018 in
~383 s / 3545 domains** — the bound climbs monotonically and the frontier
collapses (1048→8). Wired as a **route-based config**
(`milp_graph_ibp_bab_large_net_dim: 8000` + `_large_*` params): tinyimagenet
(n_in 12288) gets the high-throughput params; the small tight-root eps8 cnn7s
keep the default tight params (fewer iters TIMED OUT 9566_s3 — they need few
domains + tight per-domain bounds, the opposite of idx7018). Verified: 9566_s3
unsat 92 s, 8231_s3 unsat 44 s (no regression), idx7018_s4 unsat 383 s.

Both ABC-wins were cases ABC closes via β-CROWN BaB. ABC also OOMs 18
wide/large cases that we can't solve either (both-fail), so neither tool is
complete on this benchmark. **At 32 < 33 we are 1 short of ABC** (idx9074_s6);
closing idx7018 + idx9074 would tie. See the closeability investigation.

By family (vibecheck solved / 10): eps2_cnn7 5, eps2_wide 4, eps8_cnn7 5,
eps8_wide 4, tinyimagenet_cnn7 4, tinyimagenet_wide 9.

Fixes vs the earlier partial sweep v5 (42/60):
- **idx5613_s3** (tinyimagenet wide, SAT): unknown → **sat** (targeted PGD).
- **idx1247_s9** (eps8_cnn7, SAT): timeout → **sat** (targeted PGD, ~4 s vs
  ABC's 8 s) — a case ABC also solves, so it lifts us from a loss to a tie.
- **idx4031_s9, idx8961_s8** (eps2_wide): error (OOM) → clean **timeout**
  (memory cap; ABC also OOMs these). The eps2_wide dual-ascent overran the
  deadline → NOFILE on idx2558/idx8961/idx9074; the Phase-8 per-chunk
  deadline check (`fast_verify_dual.py`) converts those to clean `timeout`.

**Known loss vs AB-CROWN: idx7018_s4** (tinyimagenet_eps1_cnn7) — ABC unsat
77 s (β-CROWN BaB with per-domain `optimize_interm_bounds`); we return
unknown at 550 s (the no-reforward BaB keeps stale intermediate bounds on a
~-1.8 root). See the closeability investigation below. The full ABC sweep
will enumerate the complete win/loss/tie set.

**ABC cross-check of the original 15 sweep-v5 miss cases**
(`cct2026_abc_misscheck`, 550 s): **14 hard-for-both, 1 (idx1247_s9) was an
ABC win — now fixed.** (This subset did NOT cover the 18 cases new to the
full sweep — idx7018_s4 is among those, hence the loss above.)

| family | cases | ABC verdicts |
| --- | --- | --- |
| eps2_cnn7 | idx1911/4470/6005/9776_s* | 4× timeout (556–559 s) |
| eps2_cnn7 | idx5700_s4 | OOM (11 s) |
| eps2_wide | idx4031_s9, idx8961_s8, idx2558_s7 | 3× OOM |
| eps2_wide | idx3310_s5 | NOFILE (36.8 s) |
| eps8_cnn7 | idx1742_s4/s5/s6, idx4051_s8, idx8570_s7 | 5× timeout (555–558 s) |
| eps8_cnn7 | **idx1247_s9** | **sat 8 s → FIXED (we sat ~1 s)** |

(That table was the original 15-case misscheck; the full 60-case ABC sweep
later surfaced idx7018_s4 and idx9074_s6 as the two real ABC wins — idx7018
is now closed, idx9074 remains the one loss. See the Final score above.)

## Lessons learned (worth carrying to the next benchmark)

1. **Ablate the reference before claiming "X is the difference."** Twice I
   concluded idx7018 needed ABC's per-domain `optimize_interm_bounds`
   (a multi-day feature). A direct ABC ablation — disable that knob, re-run —
   showed the node count was *unchanged* and it was 20 s *faster*. The claim
   was speculation; the ablation cost ~6 min and turned a dead-end into a
   one-config win. Same method then diagnosed idx9074 as the *opposite*
   problem. Instrument the reference; don't infer from behavior.
2. **"Bounds gap" vs "search gap" is the first fork — measure both roots.**
   idx7018: our root bounds *matched* ABC's (same per-layer unstable counts,
   root -0.175 vs -0.172) → the gap was purely BaB throughput (we were 17×
   slower per domain). idx9074: our root was 1.9× looser (-0.69 vs -0.37) →
   a root gap. Dump per-layer unstable counts + root lb for both tools first;
   it tells you whether to fix bounding or search.
3. **One config can't fit opposite case shapes — route by structure.** The
   tight-root eps8 cnn7s need *few domains + tight per-domain bounds* (high
   α-iters); large-input tinyimagenet needs *many fast domains* (big batch,
   few iters) — fewer iters TIMED OUT 9566. The fix was a route gate
   (`milp_graph_ibp_bab_large_net_dim`) by input dim, not a compromise config.
4. **Report `timeout` vs `unknown` honestly at the source.** A complete BaB
   that runs out of budget is a timeout, not an "unknown"; emit it where the
   deadline is detected, not via a brittle elapsed-vs-budget threshold in the
   CLI.
5. **Dead ends, with evidence (so they aren't re-litigated):** per-neuron MILP
   tightening ≤ α-CROWN here (the ReLU relaxation isn't the dominant looseness
   — exact L1 MILP beat α by 3 of 7464 neurons); the GPU dual-ascent swept
   65.7 M nodes without closing idx7018 (the *per-node bound*, not node count,
   was the wall — until we matched ABC's bound by routing to the β-CROWN BaB).

## Knobs (`configs/challenging_certified_training_2026.yaml`)

Routing gate:
- `milp_route_pert_threshold: 0.15` — eps8 (>0.15) → IBP+BaB; eps2 → graph.
- `milp_route_dualascent_min_budget: 60` — eps2 long-budget → dual-ascent;
  short-budget (30 s) → fast layers Phase 1.
- `phase1_ibp_input_dim_threshold: 8000` — tinyimagenet (12288) → IBP.
- `milp_force_graph_path: true`, `milp_force_ibp_phase1: true` —
  tinyimagenet (input-dim-gated) defaults; the pert gate overrides these
  per-instance for the conv nets.

No-reforward β-CROWN BaB (eps8 + tinyimagenet engine):
- `milp_graph_ibp_bab_enabled: true`, `milp_graph_ibp_bab_no_reforward:
  true`, `milp_graph_ibp_bab_split_deepest: false`.
- `milp_graph_ibp_bab_batch: 16`, `_alpha_iters: 20`, `_root_alpha_iters:
  60`, `_cand_iters: 8`, `_prefilter: 12`, `_multilevel: 3`.
- `milp_graph_alpha_enabled: true`, `milp_graph_tighten_big_layers: true`,
  `milp_graph_alpha_start_mem_elems: 5e9`,
  `phase1_alpha_refresh_mem_elems: 5e9`.

Dual-ascent (eps2 long-budget engine):
- `phase8_use_dual_ascent_gpu: true`, `phase8_fast_dual_ascent: true`,
  `max_tighten_layer: 0`, `zono_lift_enabled: false`. KEEP CROWN — do NOT
  set `phase2_crown_enabled: false` (that returned worst=−inf on the wide
  net, a confirmed bug).

Targeted SAT-finding PGD:
- `milp_graph_targeted_pgd_max_open: 16` — focus the attack only when ≤16
  disjuncts remain open (SAT signal).
- `milp_graph_targeted_pgd_restarts: 128`, `_budget: 8.0`.

Shared:
- `phase8_high_bin_fallback: false` — the high-bin MILP fallback builds a
  giant Gurobi model with no mid-build deadline check and overruns the
  timeout.
- `pgd_restarts: 30`, `phase26_pre_cascade_total_frac: 0.03`,
  `phase26_pre_cascade_per_spec_cap: 1.0`,
  `phase26_pgd_per_spec_min_per_spec: 0.1`.

## Reproducing a single case

```bash
# eps8 hard UNSAT (no-reforward β-CROWN BaB)
.venv/bin/python -m vibecheck.main \
  --net path/to/challenging_certified_training_2026/1.0/onnx/cifar10_eps8_cnn7.onnx \
  --spec path/to/.../vnnlib/cifar10_eps8_cnn7/cifar10_eps8_cnn7_idx9566_sample3.vnnlib \
  --config configs/challenging_certified_training_2026.yaml \
  --device gpu --timeout 120 --results-file /tmp/r.txt --verbose

# tinyimagenet SAT (targeted PGD)
.venv/bin/python -m vibecheck.main \
  --net path/to/.../onnx/tinyimagenet_eps1_wide_cnn7.onnx \
  --spec path/to/.../vnnlib/tinyimagenet_eps1_wide_cnn7/tinyimagenet_eps1_wide_cnn7_idx5613_sample3.vnnlib \
  --config configs/challenging_certified_training_2026.yaml \
  --device gpu --timeout 120 --results-file /tmp/r.txt --verbose
```

The case needs a 24 GB GPU (A10G). server1's 10 GB OOMs the wide /
tinyimagenet forward.

## Full sweep

Sweep scripts on the A10G: `~/persistent_runs/` (`abc_misscheck.sh`,
`test_idx5613.sh`, `test_eps8_regr.sh`, `test_tiny_regr.sh`). Verdicts
MUST come from `--results-file`, never exit code. The benchmark's
`instances.csv` is CRLF — `tr -d '\r'` before parsing.

## Integration test coverage

`tests/integration/test_challenging_certified_training_2026.py`:
- `idx3736_s1` (eps8, SAT, ~7 s) — PGD root path.
- `idx9566_s3` (eps8, UNSAT, ~91 s) — the flagship no-reforward β-CROWN
  BaB case; needs IBP Phase 1 + self-consistent per-query base + batched α.
- `idx8231_s3` (eps8 wide, UNSAT, ~41 s) — wide-net BaB.
- `idx5613_s3` (tinyimagenet wide, SAT, ~7 s) — targeted PGD (4 of 199
  disjuncts open); pins the `milp_graph_targeted_pgd_*` wiring.

## Known unsolved cases

The hardest 550 s samples (eps2_cnn7 s4/s6/s7/s8/s9, several eps2_wide and
eps8_cnn7 tail samples). Cross-checked against α,β-CROWN where data exists,
these are hard for **both** tools — ABC returns OOM / NOFILE / timeout on
every checked case (none are ABC wins). Full ABC cross-check pending
(`cct2026_abc_misscheck` sweep). These need either a faster per-node BaB
kernel or a tighter root than IBP+α gives on the loose eps2 roots.
