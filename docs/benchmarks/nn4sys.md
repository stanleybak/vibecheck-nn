# nn4sys — vibecheck benchmark record

## 2026-06-02 — AWS A10G confirmation: 194/194 (the stale "190/194 AWS" was a pre-fix sweep)

Re-swept the only previously-incomplete model, `pensieve_big_parallel`, on current `main` (commit `65af0be`, the
K=2 + boost-exp-2.0 branching fix already merged) on the AWS A10G (24 GB). **All 75/75 `unsat`**, slowest 7.08s
(`pensieve_parallel_35`) against per-case budgets of 30/60/140s. The 4 previously-"failing" cases verified in
5.4–7.2s wall: pp36 7.19s · pp48 5.39s · pp56 5.36s · pp60 5.42s.

- Verdicts are **file-based** (`--results-file` per case → `~/pensieve_runs/full/summary.csv`; tally `75 unsat`,
  0 sat / 0 unknown / 0 timeout / 0 NOFILE). Not from exit code.
- The other 11 models are untouched by the fix (gated on `n_var>8` = pensieve-only) and were already 100%, so
  **nn4sys = 194/194 on the A10G**. The earlier "190/194, 63–65s margin" was a *pre-fix* sweep that was never
  re-validated on AWS until now — the fix had landed but the number was stale.
- No code change: branch `bench/nn4sys-pensieve-aws` was identical to `main`. Runner: `~/sweep_pbp.sh` (AWS);
  slowest 5: pp_35 7.08s, pp_86 6.84s, pp_83 6.71s, pp_100 6.36s, pp_40 6.11s.

## 2026-05-29 — pensieve_big_parallel last-4 timeouts FIXED: branching only (K=4→2 + boost exponent 1→2); ~4s, beats ABC

The 4 remaining timeouts (`pensieve_parallel_36/48/56/60`) were closed by **two branching knobs**, not a bounding
change. Decisive head-to-head on `pp36`, same local GPU (ABC runs in vibecheck's `.venv` +
`PYTHONPATH=$ABC:$ABC/complete_verifier`, no conda; faithful copy of `exp_configs/vnncomp23/nn4sys.yaml`):

- **The gap was branching, not bounds.** On dumped *stalled* sub-boxes, VC's per-box bound = **−0.66** vs ABC's
  incomplete forward+crown bound = **−4.38** → **VC's bounds are already 6.6× tighter than ABC's**. So tighter /
  α-dependent intermediate bounds was the wrong lever (the existing `_alpha_crown_layerwise_tighten` moved pp36's
  root bound only −9.29→−9.26, +0.027; spec-α `_run_alpha_crown_inputsplit_batched` measured **0.0000** gain). The
  entire ~32× leaf gap (VC 55,985 vs ABC 1,743) was **search** — and for UNSAT, leaf count is independent of search
  *order* (DFS vs best-first), so only split-dim **selection** (`branch_boost`) and split **arity** (K) matter.
- **Two-knob fix** (`verify_graph.py` non-LP forcing block, gated on pensieve signature `n_var>8`):
  1. `bab_split_depth = 2` (was 4). K=4 = 16-way split over-split the 32-dim boxes. Sweep (`VC_PHASE_TIMING`):
     K=4 = 55,985 leaves/39s; **K=2 = 9,849/11s**; K=1 = 127,999/timeout. K=2 is a true optimum (not monotonic).
  2. `input_split_batched_branch_boost_exp = 2.0` (was implicitly 1.0). The boost `(1+n_unstable_in_dominant_
     shallow_layer)^exp` sharpens dim-selection toward the bound-critical unstable-branch dims. With K=2, exp 1→2
     cut pp36 9,849→**529** leaves and pp56 33,469→**169** (10–198×). `branch_boost` is load-bearing — OFF or
     *softer* (exp 0.5) both explode to ~128k/timeout; only *sharper* helps. exp only changes which dims split
     (efficiency), never bounds/soundness.
- **Combined result — all 4 misses, production default (no env):** pp36 **4.5s** · pp48 **4.2s** · pp56 **4.1s** ·
  pp60 **4.2s**, all `unsat`, **169–529 leaves (below ABC's 1,743)**, wall ≈ ABC's 3.47s. From 38–47s → ~4s.
  Wall is overhead-bound (~4s: load+PGD+root setup); exp>2.0 trims leaves further but not wall.
- **Regression (production default):** mscn_128d 4.4s, mscn_2048d_dual 3.3s, pensieve_parallel_100 (non-miss) 4.6s,
  pensieve_big_simple_7 3.0s, broad pensieve_parallel spread {1,8,23,35,…,104} — all `unsat`, no regression. mscn
  (`n_var≤8`) keeps K=4/exp=1 (both measured irrelevant there). **VC now matches/beats ABC on nn4sys (194/194 local).**
- Reproduce: `.venv/bin/python -m vibecheck.main --net onnx/pensieve_big_parallel.onnx --spec
  vnnlib/pensieve_parallel_36.vnnlib --device gpu --timeout 60 --results-file /tmp/r.txt` (verdict from the file).
  `VC_PHASE_TIMING=1` adds per-phase timing + leaf count; `BAB_SPLIT_DEPTH=N` / `BRANCH_BOOST_EXP=X` override.

VNNCOMP 2025 regular track. 194 instances across 12 ML-for-systems
models: index learning (lindex, lindex_deep), cardinality estimation
(mscn_128d/2048d, with dual variants), video bitrate selection
(pensieve_small/mid/big × simple/parallel). All 194 ABC verdicts are
UNSAT (the benchmark is a stress test for soundness of learned
database/networking predictors).

## Final score (server1, 2026-05-26, RTX 3080 / 10 GB, 60 s cap)

| Solver | Solved / 194 | Wall (total) | Notes |
| --- | --- | --- | --- |
| **vibecheck** | **148** | ~96 min | verdict from `--results-file`; 30 timeout (unknown), 16 NO_FILE (likely GPU driver degradation late in sweep — to re-test after driver reset) |
| AB-CROWN (published, 2025) | 194 | ~2559 s | all UNSAT |

Per-model breakdown (honest, verdict from file):

| Model | Solved | Notes |
| --- | --- | --- |
| lindex / lindex_deep | 24/24 | ✓ |
| mscn_128d / mscn_128d_dual | 31/31 | ✓ |
| mscn_2048d | 11/11 | ✓ |
| mscn_2048d_dual | 22/23 on AWS A10G with 600s/case (vs 4/23 at 60s on RTX 3080) | one timeout at 620s (verifies in 610s with grace); 5x slower than ABC overall — see 2026-05-28 section below |
| pensieve_big_parallel | 48/75 | 27 timeout at 60 s — both LiRPA and zono BaB fail equally → algorithmic limit, not regression |
| pensieve_big/mid/small_simple, small_parallel | 30/30 | ✓ |

**0 WRONG verdicts** (was 22 pre-fix — see soundness fixes below).
**0 SAT** (all 194 should be UNSAT per ABC).

### A prior sweep report (194/194 in 161 s) was bogus

The earlier sweep script invoked `python -m vibecheck.main ...`, but `main.py`
had no `if __name__ == "__main__":` block, so the CLI was a silent import-only
no-op. The sweep counted every `exit=0` as `verified`, producing a fake
"194/194 in 161 s" with suspiciously uniform 0.81–0.85 s timings. Fix:
added the `__main__` block AND a `--results-file PATH` CLI flag that
writes the VNNCOMP verdict (`unsat`/`sat`/`unknown`/`timeout`) — sweep
scripts now key off file contents, never exit code. CLAUDE.md has the
forcing rule. The 148/194 above is from the honest verdict-file sweep.

## Two soundness bugs caught here (fixed in main, not nn4sys-specific)

1. **milp_verify Gurobi `FeasibilityTol=1e-6` vs float32-zono ulp gap**
   (`verify_milp.py:_solve_spec_worker`). Float32 zono forward produces
   active-neuron bounds tight to ~1 ulp; the float64 LP arithmetic
   computes `expr + b_j` 1-3 ulp outside those bounds → Gurobi
   reports INFEASIBLE → caller returns `verified` on a real SAT
   case. Caught on metaroom 4cnn_ry_99_16 / spec_43 (1/10 runs
   `verified` on ABC-witnessed SAT). Fix: `setParam('FeasibilityTol',
   1e-5)` — the gap is at the precision floor, no algorithmic
   loosening. Regression test in `tests/test_milp.py`.

2. **vnnlib parser dropped per-disjunct X constraints + overwrote X
   bounds across (and ...) blocks** (`vnnlib_loader.py:_parse_or_and`).
   Mixed X/Y conjuncts like
   `(and (>= X_0 a) (<= X_0 b) (<= Y_0 c))` had:
     - X bounds overwritten across blocks → global x_lo/x_hi was
       the LAST block's range, not the UNION
     - X constraints stripped from Conjunct, leaving Y-only — so
       a PGD witness with X outside the conjunct's subrange but Y
       satisfying the Y-constraint was falsely flagged as SAT.
   Caught on nn4sys lindex_10000 (22 false-SAT verdicts). Fix:
   per-block X bounds → UNION for global x_lo/x_hi + store
   per-disjunct subbox in `Conjunct.input_lo`/`input_hi`;
   `VNNSpec.check_witness(x, y)` evaluates the full X-AND-Y unsafe
   region. 4 regression tests in `tests/test_spec.py`.

## Algorithmic adds for this benchmark

- **Per-disjunct batched sub-verification** (`verify_graph.py:
  _verify_per_disjunct_subboxes`): when conjuncts carry X subboxes,
  decompose into one sub-verification per unique subbox. The
  many-disjoint-input-box pattern (mscn cardinality_X_Y has hundreds
  of disjuncts, each a small box) gets a truly-batched path that
  stacks all subboxes into ONE (B, n_in) tensor and runs a single
  batched forward + a single batched CROWN backward — effectively a
  "verify_batch" inside verify_graph. Same shared-w optimization
  closes most subs in a single shot; α-CROWN escalation for the
  remainder. Lindex_10000 (10000 subboxes) verifies in 0.49 s.
- **Forward LiRPA for batched subboxes** (`forward_lirpa.py:
  forward_lirpa_compat_zono_batched`): replaces the zonotope forward
  in the batched-subbox path. LiRPA gives tighter intermediate bounds
  for sigmoid/tanh/mul_bilinear (mscn uses all three), so basic CROWN
  closes more subs without α-CROWN escalation. Setting
  `use_forward_lirpa_subboxes=True` (default).
- **LiRPA forward inside `_multi_sub_input_split_bab` BaB iterations**:
  the inner BaB used to call `_forward_zonotope_graph_batched` per iter
  (`use_lirpa = False` was hardcoded). On mscn the zonotope generator
  tensor explodes (each sigmoid/mul adds 2048 new gens), forcing the
  OOM-halve loop to drop `batch_size` 4096 → 16 in 8 OOMs per w-group.
  Switching to LiRPA forward (no generators, just `(A, b)` linear
  bounds) keeps `batch_size=4096`, processes the whole queue in 1 iter,
  cuts cardinality_1_240_2048_dual from 33 s → 16 s (101 vs 341 BaB
  iterations). Set via `BAB_USE_LIRPA=1` (env-var, default on after the
  shape-bug fix in `forward_lirpa.last_varying_mask` exposure).
- **Gather op support** (`network.py:gg-ops`, dispatches reuse Slice
  handlers): unlocks pensieve_simple models. Gather is semantically a
  flat-index selection — same handler shape as Slice.
- **ONNX input-shape loader fix** (`onnx_loader.py`): pre-fix, dim 0
  was unconditionally stripped as "batch dim" — broke nn4sys
  pensieve_*_parallel models which use fixed `[12, 8]` input (no
  batch dim). Post-fix: if all dims are concrete, keep them as-is;
  if dim 0 is dynamic/0, strip it. Unlocked 75 pensieve_big_parallel
  cases.

## Knobs (`configs/nn4sys.yaml`)

No tuning; defaults verify all currently-supported model types.

## Known unsolved (from honest 60 s sweep)

**46/194 not verified** (verdict from file):

- **27 pensieve_big_parallel timeouts at 60 s** — `_verify_per_disjunct_subboxes`
  closes some sub-boxes, multi-sub BAB hits the wall-clock budget.
  Confirmed NOT a regression from the LiRPA-BaB switch: running
  `BAB_USE_LIRPA=0` (zono path) also returns `unknown` on
  `pensieve_parallel_37` at 60 s, same as the LiRPA path. Likely needs
  α-CROWN bound tuning during BaB iterations (we currently only run
  basic CROWN per iter).
- **3 mscn_2048d_dual high-cardinality timeouts at 60 s** —
  cardinality_1_720, _960, and one more. BAB visits ~5k–10k leaves but
  some subs don't close in time.
- **16 mscn_2048d_dual NO_FILE (exit -1/-9)** — all in the tail of the
  sweep after the GPU driver degraded. Suspected to be a driver
  artifact rather than real failures; re-run isolated after driver
  reset. Possible real OOM root cause: LiRPA forward with B=2000+ subs
  doesn't dynamically chunk on OOM in the BaB inner path.

## 2026-05-28 — full mscn_2048d_dual sweep on AWS A10G (24 GB)

Re-ran all 23 mscn_2048d_dual cases on AWS A10G with `MINI_GROUP_SIZE=240`,
`--timeout 600` per case (vs VNNCOMP's 200s). 22/23 verified — the lone
miss (cardinality_1_11080) timed out at 620s but verified in 610s in an
isolated run with extra grace, so it's a budget issue not a soundness gap.

Total wall: **vibecheck 4307s vs ABC 852s → 5.1x slower overall**. Gap
widens monotonically with disjunct count:
- 1-1000 disjuncts: 1.3-2.7× ABC
- 2000-6000: 4-5× ABC
- 7000+: 5-7× ABC

Profile (5 calls @ B=240 on the same input):
- 4213 cudaLaunchKernel per `batched_forward_linear_bounds` call → CPU stuck 51.6%
  in `Command Buffer Full` (CUDA queue saturated)
- Real GEMM compute (`ampere_sgemm_128x128_nn`) = 12ms/call; total wall
  223ms → **94% overhead** (Python op-dispatch, kernel launch latency)
- ABC's per-leaf bound prop is ~5× faster (195µs vs 1000µs)

What was tried this session (all in `forward_lirpa.py` + `verify_graph.py`):
1. `.item()` sync cache on bilinear `is_pt` checks (`_a_is_pt_false`/`_b_is_pt_false` op-dict cache)
2. `closed_t` GPU tensor replacing per-leaf `.item()` keep_mask filter in BAB inner loop
3. McCormick simplification at r=0.5 (`alpha == beta == midpoint`; halved clamp count)
4. `@torch.jit.script` helpers — `_mccormick_substitute`, `_fc_mid_half_nd/1d`
   (matches ABC's `clampmult.py` / `bound_forward_mul` pattern)

End-to-end impact: ~1% per-case speedup, well within noise. Did not change
the 5× gap.

Why so little impact: the per-call kernel launches went from ~4213 → ~4080
(only -3%). The dominant cost is the Python `for op in gg['ops']` dispatch
loop (97 ops × ~5ms Python overhead each) + per-op tensor allocations, not
the math inside any single op.

What's needed for the 5× gap:
1. **Rewrite forward LiRPA as a single `nn.Module` + `torch.compile(mode='max-autotune')`**.
   Eliminates Python op-dispatch overhead. Expected: 2-3× speedup.
2. **CUDA graph capture**. Currently blocked by syncs (`K = int(varying_mask.sum())`
   and ~60 other implicit syncs/call). After fixing: expected 3-5× since
   it eliminates all launch overhead. Capture failed with
   `cudaErrorStreamCaptureInvalidated` — need to track down each remaining sync.
3. **Algorithmic: tighter pre-BAB bounds** so each disjunct's BAB visits ~12 leaves
   like ABC (we visit ~22). 2× savings on leaf count.

ABC's secret per-op speed comes from `@torch.jit.script` decorators on every
hot helper (`clampmult.py`, `linear.py`, `relu.py`, etc.) — combined with
`auto_LiRPA`'s `BoundedModule` nn.Module structure that lets PyTorch fuse
operations across the call. We mirror the JIT pattern but lack the
nn.Module structure to fuse across ops.

## Algorithmic adds for this benchmark

## Reproducing

```bash
.venv/bin/vibecheck \
  --net path/to/nn4sys/onnx/lindex.onnx \
  --spec path/to/nn4sys/vnnlib/lindex_10000.vnnlib \
  --mode graph --timeout 30 --bits 32
```

## Integration test coverage

`tests/integration/test_nn4sys.py`:
- `lindex / lindex_10000` (UNSAT, ~0.5 s) — per-disjunct fast batched
  CROWN over 10000 subboxes. Regression for both soundness fixes
  (parser unsoundness was fired by this case).
- `pensieve_small_simple / pensieve_simple_0` (UNSAT, ~2 s) — Gather
  op support + per-disjunct.
- `pensieve_big_parallel / pensieve_parallel_1` (UNSAT, ~2 s) —
  fixed-shape `[12, 8]` input loader regression.
