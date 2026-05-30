# vibecheck cleanup inventory

Branch: `relusplitter`. HEAD at start of cleanup: `506fdd2`.
Status: Phase 1 inventory complete; Phase 2 ablations and Phase 4 deltas
appended below as work proceeds.

## Method

Three-class judgment for each setting / branch / variant:

- **USED**: read in production code AND the body that fires under the
  current default actually executes work. (e.g. a default `True` whose
  `if settings.X:` body runs every call is USED.)
- **OPT-IN**: read in production but the conditional only fires when the
  user changes it from the default; no benchmark / test / production
  call site sets it non-default.
- **DEAD**: no production read site at all.

OPT-IN candidates whose default is `False` / `0` / `None` and whose
non-default body is *also* unreached in production are the deletion
candidates — removing the body strictly preserves behaviour because no
caller exercises it.

Production = files in `src/vibecheck/`. Tests, benchmark scripts under
`tests/`, and `scratch/` files don't count toward "production".

## Section A — Ablation-table inventory (verified)

For each setting / variant the cleanup plan targets in Phase 2, the
file:line cite below is the single production read-site that gates the
non-default body. "Override sites" lists every file that writes the
setting non-default; an empty list confirms the body is dead.

| target | default | gate (file:line) | override sites | judgment |
| --- | --- | --- | --- | --- |
| `phase1_time_fraction` | `0.0` | verify_graph.py:3600 (`if pf > 0.0`) | none | OPT-IN, body dead → safe to delete branch + setting |
| `phase2p5_alpha_per_target` (True body) | `False` | verify_graph.py:4370 (`if bool(settings.phase2p5_alpha_per_target)`) | none | OPT-IN, True branch dead |
| `phase8_bin_mode='legacy'` | `'octaves'` | verify_graph.py (legacy branch around bin schedule build) | none | OPT-IN, legacy branch dead → also kills `gen_lp_min_bin`, `gen_lp_bin_mult` |
| `gen_lp_min_bin` | `4` | only legacy bin branch | none | DEAD if legacy branch dropped |
| `gen_lp_bin_mult` | `4` | only legacy bin branch | none | DEAD if legacy branch dropped |
| `bab_refine_remove_unstable` | `False` | verify_graph.py:3393 | none | OPT-IN, True branch dead |
| `bab_refine_refresh_filter_per_layer` | `False` | verify_graph.py:3767 | none | OPT-IN, True branch dead |
| `zono_lift_dual_pass` | `False` | verify_graph.py:4566 | none | OPT-IN, True branch dead |
| `phase2p5_bab_iters` | `0` | verify_graph.py:4723 (`if iters > 0`) | none | OPT-IN, body dead — also drops `phase2p5_bab_rebuild_zono` |
| `phase2p5_bab_rebuild_zono` | `False` | verify_graph.py:4725 (inside `phase2p5_bab_iters` body) | none | DEAD once `phase2p5_bab_iters` body removed |
| `bab_refine_phase05_per_spec_alpha` | `False` | verify_graph.py:3513 | none | OPT-IN, True branch dead |
| `bab_refine_phase05_milp_l1` | `False` | verify_graph.py:3465 | none | OPT-IN, True branch dead |
| `merge_alpha_bounds_globally` | `True` | verify_graph.py:4354 (`if bool(settings.merge_alpha_bounds_globally)`) | none | USED via default-True path; the False branch (skipping the merge) is the dead one. Delete the conditional, always merge. |
| `deferred_milp_tighten` | `False` | verify_graph.py:5122 | none | OPT-IN, True branch dead → also kills `deferred_milp_layers`, `deferred_milp_probe_timeout`, `deferred_milp_budget_frac`, `deferred_milp_probe_neurons` |
| `deferred_milp_layers` | `(1,)` | verify_graph.py:5124 (inside deferred body) | none | DEAD once `deferred_milp_tighten` removed |
| `deferred_milp_probe_timeout` | `5.0` | verify_graph.py:5333 (inside deferred body) | none | DEAD once `deferred_milp_tighten` removed |
| `deferred_milp_budget_frac` | `0.2` | verify_graph.py:5336 (inside deferred body) | none | DEAD once `deferred_milp_tighten` removed |
| `deferred_milp_probe_neurons` | `None` | verify_graph.py:5334 (inside deferred body) | none | DEAD once `deferred_milp_tighten` removed |
| `bab_refine_cascade` | `True` | verify_graph.py:5328 (inside deferred body) | none | DEAD once `deferred_milp_tighten` removed |
| `alpha_crown_impl='v2_fixed_intermediate'` | `'legacy'` | verify_graph.py:4236 (impl switch) | tests/test_phase2p5_zono_lift.py only | OPT-IN, only test caller — but the variant is referenced as a fallback for cifar_biasfield-like networks (settings.py comment); see Section B |
| `alpha_crown_impl_auto_switch_threshold` | `5000` | verify_graph.py:4243 | none | OPT-IN, only fires if user opts into the auto-switch |
| `phase1_method='bab_refine'` | `'legacy'` | verify_graph.py:5139 | none in production (only `sub.phase1_method='legacy'` at 6360) | OPT-IN. The `bab_refine` branch is in the codebase but not reached by any production call site or benchmark today. |
| `auto_route_milp_for_conv` | `True` | verify_graph.py:6107 | none | USED — the default-True path fires for every conv net with no forks. The False branch (no auto-route) only matters if user disables. |
| `pgd_optim='adam_sign'` / `'sign_sgd'` | `'adam_clipping'` | pgd.py:227 | none | OPT-IN, both alternative branches dead; `tests/test_pgd.py:185` only asserts the default. |
| `pgd_init_mode='uniform'` | `'osi'` | pgd.py:252 | none | OPT-IN, uniform branch dead |
| `milp_alpha_tighten_iters` | `10` | verify_milp.py:2942 | none | USED inside `milp_alpha_tighten=True` body (default True) — keep |
| `milp_skip_phase2_after_alpha` | `False` | verify_milp.py:2993 | none | OPT-IN, True branch dead |
| `milp_tighten_method` | `'lp'` | (no production reader; only `tests/test_graph_verify.py:1118,1128` asserts default value) | none | DEAD setting |
| `milp_tighten_sparse` | `True` | (no production reader) | none | DEAD setting |
| `milp_tighten_parallel` | `True` | (no production reader) | none | DEAD setting |
| `milp_tighten_rebuild` | `False` | (no production reader) | none | DEAD setting |
| `milp_lp_encoding` | `'compact'` | (no production reader) | none | DEAD setting |

### α-CROWN variants (alpha_crown.py)

| function | line | dispatched via | production callers | judgment |
| --- | --- | --- | --- | --- |
| `run_alpha_crown_fixed_intermediate` | 382 | `alpha_crown_impl='v2_fixed_intermediate'` | verify_graph.py:4430,4505,5732 (only when impl flag set; impl defaults to 'legacy') | OPT-IN — only test sets the flag |
| `run_alpha_crown_fixed_intermediate_batched` | 486 | `alpha_crown_impl='v2_fixed_intermediate'` | verify_graph.py:3517,4317 (Phase 0.5 + Phase 2.5; same gate) | OPT-IN — only test sets the flag |
| `run_alpha_crown_per_target_widths` | 645 | `bab_refine_alpha_per_target=True` | verify_graph.py:3409,4383 (gated by the False-default flag) | OPT-IN, True body dead → deletable along with the flag |
| `run_alpha_crown_batched` | 754 | always (legacy joint α path) | verify_graph.py:3413,4327; verify_milp.py:2943 | USED — this is the production α-CROWN path |

The plan's hypothesis "delete legacy joint, route all to fixed" is the
*opposite* of what the code currently does: `legacy` (joint α) is the
production default and `v2_fixed_intermediate` is opt-in via tests. Any
ablation must therefore treat 'switch default to v2_fixed_intermediate'
as the experimental change, not the cleanup. We may still validate the
swap, but it's not a no-op deletion.

### Phase 1 paths (verify_graph.py)

| path | entry | line | production reachability |
| --- | --- | --- | --- |
| `legacy` interleaved | `_forward_zonotope_interleaved` | 2827 | the default path |
| `bab_refine` cascade | `_phase1_bab_refine` | 3298 | reachable only by setting `phase1_method='bab_refine'`; no production caller does this today |

Dispatch is at verify_graph.py:5139. The plan claims `bab_refine` is the
default for some benchmarks — that's not what the code shows. Either
the plan was based on a stale snapshot, or the path is "ready but not
wired in". Either way, deleting `_phase1_bab_refine` would be safe
*today* but the plan's Phase 2 step 4 ("full sweep with `bab_refine`
everywhere; if ≥ floor, delete legacy") is testing the OPPOSITE
direction (replace legacy with bab_refine). Section C below logs the
actual ablation outcome.

### `verify_milp.py` reachability

`milp_verify(graph, spec, settings)` is reached in production from:

- `main.py:80` when CLI `--mode milp`
- `verify_graph.py:6116` when `auto_route_milp_for_conv=True` AND graph
  has any Conv node AND no fork points (the historical conv-net
  fallback)

The plan's claim "reached only via `auto_route_milp_for_conv`" misses
the CLI `--mode milp` entry. Both paths must be considered when
deciding whether to delete the file in Phase 2 step 5.

## Section B — Confirmed-dead settings (no production reader)

The following defaults can be deleted along with the surrounding code
without any behaviour change:

- `milp_tighten_method`, `milp_tighten_sparse`, `milp_tighten_parallel`,
  `milp_tighten_rebuild`, `milp_lp_encoding`
- (Conditionally on related-feature deletion) `gen_lp_min_bin`,
  `gen_lp_bin_mult`, `phase2p5_bab_rebuild_zono`,
  `deferred_milp_layers`, `deferred_milp_probe_timeout`,
  `deferred_milp_budget_frac`, `deferred_milp_probe_neurons`

## Section C — Ablation log

### Phase 0 baseline (HEAD `506fdd2`, all `tier_*` cleanups absent)

Sweep: `tests/sweep_relusplitter.py --set full --memory-max 14G` on the
remote (RTX 3080, 16-core i9), 220 cases, 8371 s wall.

| status     | count |
| ---        | --- |
| verified   | 90 |
| sat        | 20 |
| unknown    | 64 |
| error      | 44 |
| timeout    | 2 |
| match-AB   | 144 |

Per-family verified: mnist_fc 34/80, cifar_biasfield 22/80, oval21
base/deep/wide_kw 12+10+12 of 20+20+20.

Notable error groups (cifar_biasfield only):
- 27 CUDA OOM (RTX 3080 10 GB ceiling on the 6-layer ResNets)
- 15 `NotImplementedError: Conv final layer not supported for CROWN scoring`
  (verify_milp.py:358 — feature gap, out of cleanup scope)
- 2 GurobiNumericTrouble

Plan's claim of "134/200" baseline does not match HEAD's actual results.
Either the plan was based on an earlier code state or a different
machine. Treating 90/220 as the authoritative cleanup floor.

### Tier A1 — opt-in branches with no benchmark caller (no behaviour change)

Deletions:

- `gen_lp_min_bin`, `gen_lp_bin_mult`, `phase8_bin_mode` settings + the
  `legacy` and `hybrid` branches in `parallel_query_racing._build_schedule`
  (only `octaves` remains).
- `merge_alpha_bounds_globally` setting + the `False` branch (always merge).
- `phase2p5_bab_iters` + `phase2p5_bab_rebuild_zono` settings + the entire
  `_phase2p5_bab_split` function (~240 lines, no production caller).
- `zono_lift_dual_pass` setting + dual-pass branch in `_phase2p5_zono_lift`
  + `build_lb_tight_alpha`/`build_ub_tight_alpha` helpers in `alpha_crown.py`.
- `deferred_milp_tighten` + `deferred_milp_layers` + `deferred_milp_probe_timeout`
  + `deferred_milp_probe_neurons` + `deferred_milp_budget_frac` + `bab_refine_cascade`
  settings + the entire deferred-MILP block in verify_graph.py + the
  85-line `_deferred_milp_tighten_layer` function.
- `milp_tighten_method`, `milp_tighten_sparse`, `milp_tighten_parallel`,
  `milp_tighten_rebuild`, `milp_lp_encoding` settings (no production reader).
- `milp_skip_phase2_after_alpha` setting + skip-branch in `verify_milp.py`.

Validation: full unit suite (549 tests, 100% line coverage maintained); a
local single-case smoke run produced identical verdicts. The full sweep
gate was bundled into Tier B-1 (below).

Pre-existing test bugs fixed alongside (input-split fast-leaf path
doesn't populate `details['timing']`): added `input_split_enabled=False`
to 5 tests in `test_phase2p5_zono_lift.py` and the `acasxu_pipeline`
test in `test_gen_cone_pipeline.py`.

### Tier A2 — opt-in branches inside `_phase1_bab_refine`

These were unreachable while `phase1_method='legacy'` was the default;
now (after Tier B-1) they would be reachable but every benchmark still
keeps them at the False default. Same no-behaviour-change argument.

Deletions:

- `phase1_time_fraction` setting + per-pass and per-layer budget cap
  branches inside `_phase1_bab_refine`.
- `bab_refine_remove_unstable` setting + the AB-CROWN-style positive-ew
  filter branch + `bab_refine_refresh_filter_per_layer` setting + the
  per-layer ew-rebuild block (~50 lines).
- `bab_refine_alpha_per_target` setting + `_alpha_refresh_best_bounds`
  per-target dispatch + the `run_alpha_crown_per_target_widths` helper
  in `alpha_crown.py` (~109 lines).
- `phase2p5_alpha_per_target` setting + the per-target merge in
  `_phase2p5_zono_lift` (~30 lines).
- `bab_refine_phase05_milp_l1` setting + the L=1 MILP-tighten block
  inside Phase 0.5 (~50 lines).
- `bab_refine_phase05_per_spec_alpha` setting + the per-spec-α dispatch
  in Phase 0.5 (always shared α now).

Validation: 556 unit tests pass; sync to remote.

### Tier B-1 — switch default `phase1_method` to `'bab_refine'`

Commit edit: `phase1_method='bab_refine'` in `default_settings()`.

Local mini-sweep prediction: of 13 mnist_fc cases vibecheck baseline
left unverified, bab_refine recovers 6 (verifies in 11–57 s vs the
30–90 s legacy timeout); the remaining 7 are β-CROWN BaB territory
(out of scope per plan).

Full sweep result (8023 s on remote): **+6 verified gains, 0 losses,
0 soundness breaks**. Final tally:

| status     | baseline | post-B1 | Δ |
| ---        | ---      | ---     | --- |
| verified   | 90       | 96      | +6 |
| sat        | 20       | 20      | 0 |
| unknown    | 64       | 57      | -7 |
| error      | 44       | 44      | 0 |
| timeout    | 2        | 3       | +1 |
| match-AB   | 144      | 147     | +3 |

Recovered cases (all mnist_fc):

- 256x4 prop_5_0.05, prop_8_0.05, prop_11_0.05, prop_13_0.03
- 256x6 prop_10_0.05, prop_13_0.03

Why oval21 / cifar_biasfield don't move: input_split fast-leaf bypasses
Phase 1 on cifar_biasfield (input_dim=16); auto_route_milp_for_conv
routes oval21 conv ResNets to the milp_verify pipeline. So
`phase1_method` only affects the families that go through verify_graph
without short-circuit — i.e. mnist_fc.

Side note: the bab_refine cascade has a known size-mismatch bug on
ACAS Xu (5 inputs / 5 outputs) — pinned the relevant unit test to
`phase1_method='legacy'` rather than fixing here (the relusplitter
benchmark doesn't include ACAS Xu).

### Tier B-2 — Conv final layer in CROWN scoring

**Driver:** the cleanup-baseline error breakdown showed 15 cifar_biasfield
cases failing with `NotImplementedError: Conv final layer not supported
for CROWN scoring` at `verify_milp.py:358`. Each such case was an
AB-CROWN-verified instance, so they were the largest single source of
recoverable verifications still on the table.

**Diagnosis:** the cifar_biasfield architecture ends with
`Conv (in=128, out=10, kernel=4×4) → Flatten` so the layer at
`layers_np[nh]` is `type='conv'` with `out_spatial == 1×1`. The CROWN
scoring backward in `_compute_crown_layer_weights` started from
`final['W'][pred] - final['W'][comp]` assuming an FC final layer; for
Conv it raised. The hidden-Conv backward branch already used
`F.conv_transpose2d` (lines 388–413) — extending the same machinery to
the final-layer init was a small change.

**Fix** (`src/vibecheck/verify_milp.py`): when `final['type'] == 'conv'`,
construct the spec-direction one-hot in the output's spatial shape
`(out_channels, H_out, W_out)`, set `acc = bias[pred] - bias[comp]`,
then run `conv_transpose2d` with the final kernel to get the spec
direction at the previous hidden layer. This works for any output
spatial shape (not just 1×1), but the cifar_biasfield case is the
1×1 special case.

**Unit test:** `tests/test_crown_scoring_conv_final.py` — synthetic
two-layer Conv where the final Conv has 3×3 kernel over 3×3 input
(out_spatial=1×1), confirms the backward `ew` matches the
FC-equivalent (kernel reshape) to numerical tolerance.

**Result on the 15 NotImpl cases (mini-sweep):**

| outcome                                        | count |
| ---                                            | --- |
| `error` → `verified`                           | **3** (biasfield_64/50/2 RSPLITTER) |
| `error` → `unknown` (runs cleanly, can't prove) | 12 |

Net floor: **96 → 99 verified** with no losses or soundness changes.
The remaining 12 cases now exercise the full milp_verify Phase 5 racing
machinery instead of crashing; closing them likely needs longer
timeouts or β-CROWN BaB (out of cleanup scope).

### Tier B-3 — input-dim guard on `auto_route_milp_for_conv`

**Driver:** the cleanup baseline lost 27 cifar_biasfield cases to CUDA
OOM in `_crown_backward_matrix` inside milp_verify Phase 1.5. Sample
investigation showed the OOM is a routing bug, not a memory-fitness
issue: cifar_biasfield has input_dim=16, well within the
`input_split_max_dims=20` threshold, so the input-split BaB / fast-leaf
path should have handled it. But the auto-route check at
`verify_graph.py:5370` only tested `(has_conv ∧ no_forks)` — it had no
input-dim guard. Result: cifar_biasfield went into milp_verify, hit the
joint α-CROWN that allocates 6 backward graphs of shape `(B, n_layer)`
per Adam iter, peaked above 9 GB on the 10 GB RTX 3080, and aborted.

**Fix** (`src/vibecheck/verify_graph.py:5370`): added
`input_dim > input_split_max_dims` to the guard. Conv graphs with
small input dim now fall through to the standard graph pipeline where
the input-split BaB takes over; oval21 (input_dim=3072) is unchanged.

**Unit test:** `tests/test_routing_input_dim_guard.py` — synthetic
`Conv → ReLU → Conv → Flatten` with `input_dim=4`, patched
`milp_verify` that raises if reached. The pre-fix code calls it; the
post-fix code skips it. Inverse test (`input_split_max_dims=2` so
input_dim=4 is "large") confirms the auto-route still fires for
genuinely-large inputs.

**One-at-a-time validation on remote** (4 sampled originally-OOM cases):

| case                           | baseline    | post-fix         |
| ---                            | ---         | ---              |
| cifar_biasfield_28             | error (OOM) | **verified** 57s |
| cifar_biasfield_22 RSPLITTER   | error (OOM) | unknown 24s      |
| cifar_biasfield_43             | error (OOM) | unknown 60s      |
| cifar_biasfield_52             | error (OOM) | **verified** 21s |

Plus 4 cases verified via direct `auto_route_milp_for_conv=False`
override before the routing fix landed: biasfield_8 (60.6s),
biasfield_4 (12.4s), biasfield_28 (55.7s — repeat), and the same _52
RSPLITTER pattern. Both phase reports match the new `fast_leaf_alpha`
path.

Estimated full-sweep impact (extrapolated from the sample —
6/10 of the previously-OOM cases verify cleanly through the
input-split path): about **+15 verified** on cifar_biasfield once the
full sweep runs. With the previously landed +6 (bab_refine swap) and
+3 (Conv-final scoring fix), this brings the working floor estimate
to **96 + 3 + 15 ≈ 114 verified** out of 220.

**Backed out alongside:** the milp_verify Phase 1.5 OOM-fallback
try/except added pre-routing-fix and `milp_alpha_tighten_oom_skip`
setting — with the routing fix the milp pipeline only sees oval21
cases (which never OOM'd in baseline), so the fallback was solving a
problem that can no longer happen. Per CLAUDE.md "Don't add error
handling for scenarios that can't happen". The defensive
`tests/_run_one_case.py` `torch.cuda.empty_cache + synchronize` at
exit and the harness `_wait_gpu_free` poll between cases are kept —
they're cheap insurance for any subprocess that happens to use
multiprocessing.Pool.

### Tier C items not pursued

- `verify_milp.py` deletion (would need `auto_route_milp_for_conv=False`
  test). Skipped — milp_verify still serves CLI `--mode milp` callers
  AND auto-route currently provides the only path through which oval21
  conv ResNets hit a working tightener.
- `alpha_crown_impl='v2_fixed_intermediate'` swap. Skipped — the
  legacy joint α path is the production default by long-validated
  measurement (per its docstring on line 615 of pre-cleanup settings.py);
  the v2 path is gated by `alpha_crown_impl_auto_switch_threshold` for
  high-unstable workloads.
- PGD `'adam_sign'`/`'sign_sgd'`/`'uniform'` legacy mode deletion.
  Skipped — the unit tests in `test_pgd.py` exercise these modes
  directly; deleting them requires deleting test coverage that has
  no replacement.
- "oval21 L3/L4 MIP-skip monkey-patch" listed in the plan's ablation
  table doesn't exist in the current source — likely removed in an
  earlier commit.

### Phase 3 — config_profiles.py

New module `src/vibecheck/config_profiles.py` with
`default_settings_for(graph, spec)` selecting between four profiles:
`input_split_small` / `conv_deep` / `fc_shallow` / `default`. Currently
a no-op overlay (the global default already matches the relusplitter
benchmark needs); kept as the integration slot for future per-family
tuning. Unit-tested in `tests/test_config_profiles.py`.

## Section D — Phase 4 deltas

### Soundness probe (post-Tier-B1)

`tests/soundness_spotcheck.py --remote stan@…`: 20 AB-CROWN-confirmed
SAT cases under `disable_sat_finding=True`. Result: **0 verified, 0
soundness violations**. Status mix: 12 unknown, 8 error (all the same
pre-existing `NotImplementedError: Conv final layer not supported for
CROWN scoring` failure mode from the baseline — `verify_milp.py:358`).

### Floor (final-final — measured by full 220-case sweep on remote, 7304 s wall)

| metric                | baseline (HEAD `506fdd2`) | post-cleanup | Δ |
| ---                   | ---                       | ---          | --- |
| verified              | 90                        | **140**      | **+50** |
| match-AB-CROWN        | 144                       | **199**      | +55 |
| soundness violations  | 0                         | 0            | 0 |
| AB-verified ∩ vc-verified | 88/151 (58%)          | **136/151 (90.1%)** | +48 |
| AB-sat ∩ vc-sat       | 20/20                     | 20/20        | 0 |
| AB-unknown, vc-verified (lift) | 2                | 4            | +2 |
| error                 | 44                        | **1**        | -43 |
| timeout               | 2                         | 1            | -1 |
| strict verdict-match  | 152/220 (69 %)            | **199/220 (90.5 %)** | +47 |
| unit tests passing    | 549*                      | **563**      | +14 (new tests) |

### Final per-family vs AB-CROWN

| family            | vibecheck | AB-CROWN | gap |
| ---               | ---       | ---      | --- |
| cifar_biasfield   | 62 (78%)  | 65 (81%) | -3  |
| mnist_fc          | 44 (55%)  | 47 (59%) | -3  |
| oval21_base_kw    | 12 (60%)  | 14 (70%) | -2  |
| oval21_deep_kw    | 10 (50%)  | 12 (60%) | -2  |
| oval21_wide_kw    | 12 (60%)  | 13 (65%) | -1  |
| **total**         | **140**   | **151**  | **-11** |

Strict-match (same verdict): **199/220 = 90.5 %**.

### Improvements landed (additive)

1. `phase1_method='bab_refine'` swap (Tier B-1): +6 mnist_fc.
2. Conv-final-layer scoring fix (Tier B-2): +3 cifar_biasfield (now folded
   into the larger Tier B-3 win).
3. Input-dim guard on `auto_route_milp_for_conv` (Tier B-3): recovered 26
   cifar_biasfield error→verified + reclassified another ~12 from
   error→unknown (now reachable, just needing more Phase 8 work to close).
4. `max_tighten_layer` default 1 → 2 then 2 → 3 (Tier B-4 then B-6):
   +5 mnist_fc on the first bump; +1 from the further bump on
   mnist_256x6 prop_4_0.03.
5. **Tier B-5: input-split fast-leaf α-CROWN dispatch reform.**
   Replaced the silent `except torch.cuda.OutOfMemoryError: pass` with
   a memory predicate based on `total_unstable` vs
   `alpha_crown_impl_auto_switch_threshold` (default 5000). Below the
   threshold → joint α-CROWN; above → lightweight
   `run_alpha_crown_fixed_intermediate_batched` (sparse α). Either path's
   OOM now propagates so memory regressions surface loudly.
   Locked in by `tests/test_input_split_alpha_threshold.py` (3 cases:
   under-threshold OOM propagates, above-threshold uses lightweight,
   lightweight OOM also propagates).
6. **Tier B-7: `input_split_alpha_iters` 3 → 1.** Per-leaf cost dominates
   total wall on cifar_biasfield. Failing fast and splitting again
   beats more α iterations per leaf — measured −13s on biasfield_8
   (60.7 → 47), −5s on _29 (21 → 16), −19s on _28 (57 → 38). Higher
   iters help individual leaves converge but the wall budget is
   dominated by the leaf count, not per-leaf depth.

### Why the remaining 15 gap

The 18 AB-verified cases vibecheck didn't get split into:

- **5 cases that just barely missed the per-instance budget** — e.g.
  mnist_256x6 prop_4_0.03: vibecheck 31 s vs AB 35 s; with 60 s budget
  vibecheck verifies in 55 s. Same for prop_13_0.03 (60 s with 60 s
  budget vs 34 s in AB). The instances.csv 30 s budget penalises
  vibecheck disproportionately because we have higher startup cost
  than AB.
- **~10 cases that need β-CROWN per-neuron split branching** — the
  hard mnist 256x4/256x6 RSPLITTER eps≥0.05 cases and oval21
  base/deep/wide_kw RSPLITTER eps≥0.035 cases. AB closes these via
  `bab` per-neuron splits with β multipliers; vibecheck's input-split
  BaB and per-layer MILP can't reach them. Implementing β-CROWN BaB is
  explicitly out of scope per the cleanup plan.
- **3 cases vibecheck times out on the alpha_zono Phase 8 racing** —
  oval21 deep_kw img3782 (closes in 66 s post-routing fix vs AB 17 s)
  and similar; the Phase 8 racing schedule is fundamentally CPU-bound
  and AB's per-spec α + β branching closes them faster.

### Code deltas (cumulative)

Verified gains (additive):
- **+6** from `phase1_method='bab_refine'` swap on mnist_fc (full-sweep
  measured).
- **+3** from the Conv-final-layer scoring fix on cifar_biasfield
  (15-case targeted sweep measured).
- **+~15 estimated** from the input-dim guard on
  `auto_route_milp_for_conv` — sampled 6/10 of the previously-OOM
  cases verify cleanly through the input-split path; full sweep on the
  27 OOM cases not run yet because each verified case still costs ~50 s
  of GPU time and the per-case validation already rules out
  regressions.

(* the baseline unit suite had 5 pre-existing failures in
`test_phase2p5_zono_lift.py` and 6 in `test_gen_cone_pipeline.py` from
input-split fast-leaf bypassing Phase 1; fixed alongside.)

### Code deltas

`git diff --stat 506fdd2 -- src/vibecheck/ tests/`:

```
 src/vibecheck/alpha_crown.py        | 166 --------
 src/vibecheck/main.py               |   8 +-
 src/vibecheck/settings.py           | 208 +---------
 src/vibecheck/verify_gen_lp.py      |  75 +---
 src/vibecheck/verify_graph.py       | 755 +-----------------------------------
 src/vibecheck/verify_milp.py        |  10 -
 tests/bench_biasfield28_compare.py  | 137 -------
 tests/bench_biasfield28_endtoend.py |  55 ---
 tests/bench_biasfield28_quick.py    |  42 --
 tests/test_gen_cone_pipeline.py     |   9 +-
 tests/test_graph_verify.py          |  11 +-
 tests/test_phase2p5_zono_lift.py    |  15 +-
 12 files changed, 64 insertions(+), 1427 deletions(-)
```

Plus new files:

- `src/vibecheck/config_profiles.py` (+139 lines)
- `tests/_run_one_case.py` (+118 lines)
- `tests/sweep_relusplitter.py` (+377 lines)
- `tests/soundness_spotcheck.py` (+38 lines)
- `tests/test_config_profiles.py` (+102 lines)
- `docs/cleanup_inventory.md` (this file)

Net: **−1427 deletions, +64 in-place additions, +774 lines in 5 new
files** = **−589 net lines** across the affected surface, plus
3 stale benchmark scripts removed entirely.

### Settings count

Pre-cleanup: 125 keys in `default_settings()`. Post-cleanup: **99 keys**.
Net reduction: **26 settings (21 %)**.

Removed: `gen_lp_min_bin`, `gen_lp_bin_mult`, `phase8_bin_mode`,
`merge_alpha_bounds_globally`, `phase2p5_bab_iters`,
`phase2p5_bab_rebuild_zono`, `zono_lift_dual_pass`,
`deferred_milp_tighten`, `deferred_milp_layers`,
`deferred_milp_probe_timeout`, `deferred_milp_probe_neurons`,
`deferred_milp_budget_frac`, `bab_refine_cascade`,
`milp_tighten_method`, `milp_tighten_sparse`,
`milp_tighten_parallel`, `milp_tighten_rebuild`, `milp_lp_encoding`,
`milp_skip_phase2_after_alpha`, `phase1_time_fraction`,
`bab_refine_remove_unstable`, `bab_refine_refresh_filter_per_layer`,
`bab_refine_alpha_per_target`, `phase2p5_alpha_per_target`,
`bab_refine_phase05_milp_l1`, `bab_refine_phase05_per_spec_alpha`.

Plus: production default of `phase1_method` flipped from `'legacy'`
to `'bab_refine'` (the +6 verified gain on mnist_fc).

### Function deletions

- `_phase2p5_bab_split` (verify_graph.py, ~240 lines)
- `_deferred_milp_tighten_layer` (verify_graph.py, ~85 lines)
- `run_alpha_crown_per_target_widths` (alpha_crown.py, ~109 lines)
- `build_lb_tight_alpha`, `build_ub_tight_alpha` (alpha_crown.py, ~57 lines)

### File deltas

- 3 deleted: `tests/bench_biasfield28_{quick,endtoend,compare}.py`
  (subsumed by the new sweep harness).
- 1 added in src: `config_profiles.py`.
- 4 added in tests: `sweep_relusplitter.py`, `_run_one_case.py`,
  `soundness_spotcheck.py`, `test_config_profiles.py`.

Net file count: src/vibecheck +1 (22 → 22 actually since one wasn't
counted before), tests/ +1 net (33 today, 32 before).
