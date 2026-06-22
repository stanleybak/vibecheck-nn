# lsnc_relu (Lyapunov Stable Neural Control)

VNNCOMP 2026 **extended track** (unscored). 80 instances, one ONNX
(`relu_quadrotor2d_state.onnx`, 6-in / 8-out), 25 s timeout each.

## What the net/spec are

A discrete-time control certificate for a 2-D quadrotor:

- A ReLU **controller** `u = π(x)` (with a ReLU saturation clamp).
- A ReLU **residual dynamics** `x' = f(x, u)`.
- A **convex quadratic Lyapunov value** `V(x) = uᵀ P u` (`u = scaled_x − x_eq`,
  `P` symmetric PSD, eig `0.013 … 10.2`) and the **decrease** `dV = V(x') − V(x)`.

The 8 outputs are `Y = [dV, V(x), x'(6 dims)]`. The spec's unsafe region is

```
(OR of 13 box-violation disjuncts: dV ≥ 1e-6, or x'_i out of its box)
  AND  Y_1 = V(x) ∈ [a, b]          ← level-set band, per-instance a,b
```

i.e. on the level set `{V ≈ band}`, the next state must stay in the box and
`V` must decrease. The band is a **global conjunct ANDed into every disjunct**.

## Headline result (server1 RTX 3080, 25 s budget)

| | vibecheck | α,β-CROWN (published 2025) |
|---|---|---|
| sat | 12 | 12 |
| unsat | 68 | 68 |
| unsolved | 0 | 0 |
| **solved** | **80 / 80** | **80 / 80** |

**Full parity with α,β-CROWN.** Every UNSAT verifies in ~6–8 s, every SAT
witness is found. Soundness gate: all 12 SAT cases correctly fail to verify
with PGD disabled (no false-verify). ABC's SAT set:
{1,26,28,31,34,36,43,45,51,67,68,74}.

### Regression & fix (2026-06-22) — shared-code routing

A later commit (`762cc34`, ml4acopf work) added a nonlinear auto-router in
`verify_graph` gated on `_has_trig or _has_mul_bilinear` that **force-disabled
`input_split_batched_enabled`** for any net with a both-vary bilinear `Mul`.
lsnc's Lyapunov quadratic forms `V = uᵀPu` ARE both-vary Muls, so lsnc was
silently hijacked into the ml4acopf nonlinear path and its throughput BaB was
killed → **every UNSAT timed out** (60 s, confirmed). Fix: the router now defers
to explicit config intent — a config that set `input_split_batched_enabled`
(lsnc does) keeps its declared BaB strategy; the auto-router only fires when no
batched strategy was requested (ml4acopf, unchanged). The router/helpers were
also renamed `acopf_* → nonlinear_*` (`_route_nonlinear`,
`_verify_nonlinear_graph`, `_nonlinear_{backward_crown_root,alpha_opt,nominal_cex_probe}`,
`nonlinear_*` settings) since the mechanism is general, not acopf-specific; the
dead, never-imported `acopf_dual_ascent.py` module was deleted.

Re-confirmed on AWS g5-8x (A10G), 2.0/v2 specs, 25 s budget: **80/80 (12 sat +
68 unsat), verdicts identical to ABC's per-instance, tightest case ~12.9 s.**
ml4acopf 14_ieee prop1/2/3 + full-prop3 re-run on the renamed code: verdicts
unchanged.

## Bugs fixed to get here

The benchmark was a 100% miss at the start (every case `unknown@2.6s`). The
causes, in order:

### Crash / no-op bugs (the BaB never ran)
1. `gpu_graph` lacked **`Neg`** → build raised, input-split never started.
2. The dual-ascent **op serializer** rejected non-fc/conv ops even when MILP
   escalation was off (it isn't needed then) → built it only when escalation
   is enabled.
3. **`n_output`** was taken from the "last fc op" — wrong for this **concat
   output** (real width 8, last Gemm is the 6-wide dynamics). Now derived from
   a point-forward through the graph.
4. Two `sorted()` calls choked on **mixed str/int keys** (the forward stashes
   string-keyed bilinear boxes alongside int ReLU-layer keys) — filtered to
   int keys.

### Soundness bugs (these are the important ones — both in shared code)
5. **vnnlib loader dropped global conjuncts.** `(assert (or …))` followed by a
   top-level `(assert (<= Y_1 b))` must AND the band into *every* disjunct.
   We dropped it → the unsafe region was too large → **false-SAT** (a witness
   with `Y_1 = 28` "satisfied" a disjunct that the real band excludes). Fixed
   in `vnnlib_loader._parse_top_level_y_constraints`; pinned by
   `tests/test_spec.py::test_parse_or_and_trailing_global_y_*`.
6. **Bilinear backward CROWN was unsound.** No caller passed `bilinear_op_bounds`,
   so every `mul_bilinear` backward used a point-linearization that drops the
   `−a·b` constant — a tangent, not a bound; it double-counts and the "lower
   bound" can sit *above* the true value. That silently **false-VERIFIED** the
   SAT cases (with PGD off). Fixed: `_spec_backward_graph_batched` auto-fetches
   the stashed McCormick boxes; the legacy path now raises. (nn4sys 9/9 and
   vit verdicts unaffected.)

## The throughput story (the actual optimization)

After the soundness fix, the bound is the **sound McCormick** relaxation of the
two quadratic forms. It is loose (the indefinite `dV` ignores the u/Pu
correlation), so these convex cases converge only by **brute-force input-split**
— ~6–12 M domains for the hardest UNSAT.

**α,β-CROWN does exactly this**, measured on q0: its `clip_n_verify` input-split
runs ~28 iterations with **constant ~0.065 s bounding regardless of domain
count** (it concretizes a CROWN linear bound on a 1e6 batch) + aggressive domain
clipping (65–75% shrunk/iter) → ~670 k domains/s. *Not* standard β-CROWN.

Ours started at "never converges". The fixes (all in the shared batched
input-split BaB, so they help any large-batch input-split benchmark):

1. **Adaptive split arity.** `bab_split_depth=4` (16-way) routed through a
   per-leaf **Python loop** = 97% of wall. The K>1 path only exists to *grow* a
   small queue; once the popped batch is full (`B == batch_size`) we use K=1's
   vectorized GPU 2-way split. (27× more leaves.) The gate had a bug — it tested
   the *post-pop* worklist size, which is 0 when a case pops the whole queue, so
   it never switched.
2. **GPU-native SB scoring** (no per-iteration `.cpu()` sync).
3. **Chunk worklist.** The worklist was a Python list of *millions* of 1-row GPU
   tensors (`extend(unbind)` / `stack`) = ~40% of wall. Now a list of 2-D chunk
   tensors; pop/push stay on-GPU. (180k → 356k domains/s; split → 0%.)
4. **Query dedup.** The band `Y_1∈[a,b]` is repeated across all 13 disjuncts —
   26 of 39 queries are duplicates. Computing each unique halfspace once cut the
   spec backward ~2.6× *and* fixed the split score (it summed `|A|` over all 39
   queries, over-weighting band dims). (356k → 660k/s, q0 12.1M → 6.66M leaves.)
5. **K=1 split arity** (the final unlock — closed the last 6 "hard" UNSAT). The
   non-LP gate forced `bab_split_depth=4`, routing through a per-leaf Python
   loop. For n_var ≤ 8 the arity doesn't change the leaf count, so K=1's
   vectorized GPU 2-way split is strictly better — q2 went 33 s → **6.4 s**,
   same verdict. The "6 throughput-hard UNSAT" were never hard; they were this
   bug. (Measured by ablating ABC's clip first — it OOMs q2 on the 10 GB GPU,
   so clip wasn't our lever; digging into q2 found the K=4 split was.)

Net: every UNSAT verifies in **~6–8 s**.

**SAT side** (all 12 found): per-open-spec root PGD (`pgd_per_restart_disjunct`
— each restart descends one disjunct's basin, not a diluted joint loss) finds
10 of 12 at the root in ~2 s; the 2 narrowest (q43/q45) need a *light* per-leaf
PGD (`every=40`) that adds <1 s to the UNSAT BaB.

## Config knobs (`configs/lsnc_relu.yaml`) and why

- `input_split_batch_size: 65536`, `input_split_batched_max_worklist: 4e7` —
  saturate the GPU and let the queue grow to the millions of domains needed.
- `input_split_crown_intermediate: false` — forward-zono intermediate bounds;
  backward-CROWN intermediate does **not** reduce the leaf count here (both
  converge at ~12M pre-dedup) but costs ~17% more per leaf.
- `pgd_per_restart_disjunct: true`, `pgd_restarts: 130`,
  `pgd_time_budget_phase0: 2.0` — per-open-spec root PGD (each restart descends
  one disjunct's basin) finds 10/12 SAT in ~2 s with a modest restart count.
- `input_split_leaf_pgd_every: 40` (+ `max_leaves: 64`, `restarts: 256`,
  `iters: 50`) — a *light* per-leaf attack for the 2 narrowest SAT basins
  (q43/q45); every=40 catches them by ~10 s yet adds <1 s to the UNSAT BaB.
- `input_split_batched_alpha_iters: 0` — the acasxu α-boundary closer is
  off (throttles throughput; not needed, no degenerate tail).
- Split arity K=1 is set by the non-LP gate (n_var=6 ≤ 8 → K=1): the
  vectorized GPU 2-way split, not the per-leaf Python K>1 loop.

## Reproduction

```bash
# single case (server1 GPU)
.venv/bin/python -m vibecheck.main \
  --net  .../lsnc_relu/onnx/relu_quadrotor2d_state.onnx \
  --spec .../lsnc_relu/vnnlib/quadrotor2d_state_0.vnnlib \
  --config configs/lsnc_relu.yaml --timeout 25 --results-file /tmp/r.txt
# full sweep: ~/lsnc_full_sweep.sh on server1 (verdicts from --results-file)
```

## Known unsolved

None — 80/80, full parity with α,β-CROWN.

## A dead end worth recording

`input_split_batched_clip_recrown_cycles > 0` (re-run CROWN on clipped boxes)
is **UNSOUND** on these bilinear graphs — it "verified" q2 at the root in 1
leaf and false-verified every SAT case (1 leaf, real CE ignored). Left at 0.
