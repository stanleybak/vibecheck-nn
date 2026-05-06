# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working style

**These instructions take priority when they conflict with default conciseness guidance.**

Claude chooses the approach that correctly and completely solves the problem, even when that takes longer than a quicker alternative. The correct implementation is the goal.

Claude communicates clearly, with detail matched to the complexity of the work. Conciseness guidelines apply to chat messages — they apply to how Claude talks, not to the thoroughness of code changes or implementation work. Claude can lead with the answer in explanations while keeping the underlying implementation thorough.

### Scope and adjacent code

Claude matches the scope of changes to what was requested, and also addresses closely related issues when fixing them is clearly the right move. When adjacent code is broken or contributes to the problem at hand, Claude fixes it as part of the work.

### Code quality

Claude adds error handling at real boundaries where failures can actually occur — I/O, parsing, network calls, user input, external systems.

Claude uses judgment about abstraction: extract when duplication causes real maintenance risk, and leave things inline when extracting would be premature.

Claude does the work a careful senior developer would do, including edge cases. Claude completes tasks fully rather than leaving them half-done.

### Exploration and investigation

When exploring a codebase, researching a problem, or running a subagent task, Claude is thorough. Completeness takes priority over raw speed. Claude includes code snippets whenever they provide useful context.

### Tone

Claude's tone is clear and appropriately detailed for the complexity of the work.

### Evidence over speculation

When stating a causal claim — e.g. "we're slower because of X", "the bound is loose because of Y", "ABC verifies this via Z" — Claude provides evidence: a measurement, a code reference with line numbers, or a directly-observed log line. Phrases like "probably", "I think", "presumably", "almost certainly" are warning signs that a claim is unsupported. If the evidence isn't in hand, Claude either (a) gathers it (instrument, run, read source) before making the claim, or (b) explicitly labels the statement as a hypothesis to test, not a conclusion. This applies especially to comparisons against α,β-CROWN or other reference verifiers — read their source/run them with debug prints, don't infer from behaviour.

## What is vibecheck?

A zonotope-based neural network verification tool. Given an ONNX network and a VNNLIB specification (input bounds + output property), it determines whether the property is provably satisfied ("verified") or "unknown" using abstract interpretation with zonotope domains.

## Development Commands

**Always use the venv (`.venv/bin/python`) for all commands.**
**Never run `git commit` or `git push` — the user handles all git operations.**
**Never kill `tmux` or `claude` processes.** When cleaning up after interrupted or runaway runs, only kill the specific PIDs you started (tracked via `run_in_background`) or narrow the `pkill` pattern to the offending script. Broad `pkill -f python` or `kill $(pgrep ...)` sweeps will take down the tmux session hosting Claude itself.
**Run memory-uncertain experiments under a cgroup cap.** When benchmarking code that builds large tensors (conv weight propagation, full G matrices, big LPs) — or running a new/refactored path where peak memory is unknown — wrap the command in `systemd-run --user --scope -p MemoryMax=8G` (adjust the cap for the laptop's RAM; leave ≥25% headroom for the desktop + Claude Code + tmux). If the process exceeds the cap the OOM killer terminates only the transient cgroup, not the host shell. Past incident: a `precompute_gen_state` run with synthetic all-unstable bounds exploded n_gens and OOM-killed tmux and Claude Code along with it.
**Single-threaded BLAS**: `__init__.py` sets `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` via `setdefault` before numpy is imported. Multi-threaded OpenBLAS causes massive scheduling jitter on the small matrices typical in verification (100x variance between runs) and is slower overall. No manual env vars needed; override with explicit env vars if desired.

**Never silently swallow OOM.** When a GPU/CPU OOM fires inside the pipeline, the user wants to know — silent fallbacks hide real problems (e.g. a forward path that should fit on 10 GB suddenly doesn't, or a cache line grew in a recent commit). Default behavior is to re-raise `torch.cuda.OutOfMemoryError` (and `MemoryError`) so the exception surfaces to the CLI. The only code paths that may catch-and-retry are ones the user has explicitly authorised via `settings.raise_on_oom=False` (e.g. a benchmarking loop that wants to record "OOM" as an outcome and move to the next instance). When writing new code that touches large tensors, do NOT add a try/except around the allocation just to degrade to a slower path — let it raise.

```bash
# Setup
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"

# Run all tests
.venv/bin/python -m pytest tests/ -v

# Run a single benchmark test
.venv/bin/python -m pytest "tests/test_graph.py::test_vnncomp_benchmark[acasxu_2023/ACASXU_run2a_1_1_batch_2000]" -v -s

# Run the verifier
.venv/bin/vibecheck --net model.onnx --spec property.vnnlib
```

## Architecture

The verification pipeline flows: **ONNX loading → graph construction → zonotope propagation → spec check**.

The codebase uses **object-oriented dispatch**: each ONNX op type is a `GraphNode` subclass (in `network.py`) that implements `infer_shape()` and `zonotope_propagate()`. The verifier loop simply iterates in topological order and calls these methods.

- **`network.py`** — Core graph representation. `ComputeGraph` holds a DAG of `GraphNode` subclass instances keyed by tensor name. Each subclass (`ConvNode`, `ReluNode`, `AddNode`, `TransposeNode`, `SliceNode`, etc.) implements its own shape inference and zonotope propagation. Also contains: `OP_REGISTRY` (maps ONNX op strings to subclasses), `_find_shared_gens()` for fork/merge tracking. Use `print(graph)` for a structural summary showing topo indices, shapes, predecessor/successor connections, and fork points.

- **`onnx_loader.py`** — Loads ONNX models into `ComputeGraph`. Parses ONNX nodes into the right `GraphNode` subclass via `OP_REGISTRY`, performs constant folding, topological sort, shape inference (`node.infer_shape()`), and BatchNorm folding into preceding Conv/Gemm. Sets `SplitOutput` shapes from parent `Split` params.

- **`spec.py`** — OOP specification types. `VNNSpec` holds input bounds + disjunction of `Conjunct`s (DNF). Each `Conjunct` contains `Constraint` (threshold: `Y_i >= val`) or `PairwiseConstraint` (`Y_comp >= Y_pred`). `VNNSpec.check(output_lo, output_hi)` evaluates margins — positive means verified safe.

- **`vnnlib_loader.py`** — VNNLIB file parsing. `load_vnnlib(path)` reads `.vnnlib` / `.vnnlib.gz` files and returns a `VNNSpec`. Supports pairwise output constraints, threshold constraints, and `(or (and ...))` disjunctive normal form with mixed input/output constraints.

- **`zonotope.py`** — `DenseZonotope` represents sets as `{center + G @ e | ||e||_inf <= 1}`. Methods: `propagate_linear()` (FC/Conv), `apply_relu()` with three relaxation types (`min_area`, `y_bloat`, `box`), `copy()` for fork points, `add(other, shared_gens)` for skip connection merges. The add method splits generators into shared prefix (added element-wise) and branch-specific suffix (concatenated).

- **`verify.py`** — Thin dispatch loop: `zonotope_verify(graph, spec)` iterates topo order calling `node.zonotope_propagate()`, then calls `spec.check()` on the output bounds.

- **`verify_milp.py`** — MILP verification pipeline using Gurobi. Strategy: (1) GPU zonotope + CROWN for initial bounds, (2) per-layer tightening (MILP for conv, per-worker LP for FC), (3) spec MILP with racing escalation. Key functions: `_build_base_model()` / `_build_spec_model_compact()` (Gurobi model building), `_tighten_layer_parallel()` (parallel bound tightening), `_racing_escalation()` (doubling bin schedule, races feasibility 1T vs optimization (n-1)T per level), `_solve_spec_worker()` (builds cascading MILP from scratch per worker, compact LP encoding for non-binary neurons), `score_neurons_ew_frac()` (|ew|×frac scoring). Settings: `milp_scoring` ('ew_frac'|'crown'|'crown_lp_fractional'), `milp_lp_per_worker` (True = each worker builds own LP model for FC layers).

- **`verify_zono_bnb.py`** — Branch-and-bound verification with zonotope/CROWN abstract domains. Also exposes `_forward_zonotope_graph`, `_spec_backward_graph`, and `_make_slopes` used by the graph pipeline.

- **`verify_graph.py`** — Graph-mode pipeline (zonotope + CROWN + interleaved LP/MILP) for DAG-structured networks. Shares op-walking and dead-neuron propagation between a readable `_build_reference` LP builder and a batched `_build_optimized` MVar+sparse builder; both preserve the Bug #1 fix (conv/fc with all-dead inputs still emits a fixed bias variable, never `None`). All Gurobi solves flow through `optimize_checked`. Phase 2.5 (`_phase2p5_zono_lift` + `_forward_keep_pre_gpu`) runs between Phase 2 CROWN and Phase 7 LP: for each still-open disjunct, iteratively tightens pre-ReLU bounds via the closed-form box+halfspace LP (`vibecheck.box_halfspace`), then re-computes CROWN LB on the tightened bounds. Bounds stay query-local (sound only for the counterexample region) and are never merged into the shared `bounds_by_relu`. Gated by `settings.zono_lift_enabled` (default True). Phase 1 has two modes — `phase1_method='legacy'` (default) is the historical interleaved-forward + per-layer tightener; `phase1_method='bab_refine'` is the α,β-CROWN bab-refine cascade (forward zono → per-layer MILP-tighten with sliding window K + batched α-CROWN refresh → merge into global bbr). On mnist_fc 256x6 prop_5 the cascade verifies in **87 s** vs the legacy pipeline's 392 s and α,β-CROWN's reported 124 s. Sliding window K is set via `settings.bab_refine_window` (default 1, matches AB-CROWN) or the `VC_TIGHTEN_WINDOW` env var.

- **`box_halfspace.py`** — Closed-form LP for the polytope `{e ∈ [-1,1]^n : a·e ≤ β}`. Given a linear objective `d·e + c0`, `lagrangian_min` / `lagrangian_max` compute the tight primal value in `O(n log n)` via the 1D concave Lagrangian dual (breakpoints at `λ*_i = −d_i/a_i`). Matches Gurobi LP to ~1e-7 and is ~30× faster per solve on a 4K-generator CIFAR100 ResNet pass. `tighten_all_layers` vectorizes over layers and keeps pre-ReLU G on GPU, transferring only unstable rows to CPU.

- **`verify_gen_lp.py`** — Dense generator-based LP/MILP encoding for spec verification. Separates the linear transformation (G matrix per op, propagated via batched GPU conv) from the zonotope domain (generators e ∈ [-1, 1]). Produces a much smaller model than the per-neuron builders (e.g. ~4K vars vs ~104K on CIFAR100 ResNet medium) while giving identical LP triangle bounds. Selected via `settings.phase8_milp_mode` ('find_sat' | 'infeasibility' | 'alpha_zono_bnb' (default) | 'alpha_zono_infeasibility'); the α-zono variants build a tighter per-query parallelogram via `alpha_crown.forward_zono_dir_adaptive`.

- **`onnx_optimizer.py`** — Semantics-preserving graph rewrites applied after ONNX loading, called from `ComputeGraph.optimize(settings)`. `fold_relusplitter` collapses the expanded `Conv(C→2C)→ReLU→Conv(2C→C,1×1)→ReLU` pattern back into a single `Conv(C→C)→ReLU` (exact because ReLU(z) − ReLU(−z) = z). `fuse_gemm_reshape_conv` fuses `Gemm→Reshape→Conv` into a single equivalent FC layer so the backward pass sees fewer, larger layers.

- **`settings.py`** — Configuration for BnB and MILP verification. `default_settings()` returns a DotMap with defaults.

- **`main.py`** — CLI entry point. Exit code 0 = verified, 1 = unknown.

- **`gurobi_util.py`** — `optimize_checked(model, user_callback=None)` wraps `model.optimize()` with a message callback that scans for numeric-trouble warnings (`Markowitz tolerance tightened`, `variables dropped from basis`, `switch to quad precision`, `max constraint violation`). Raises `GurobiNumericTrouble` if any are captured.

## Gurobi solve convention

**Never call `model.optimize()` directly.** Always use `optimize_checked(model)` from `gurobi_util.py` (or pass a callback via its `user_callback=` argument if the caller also needs MESSAGE/MIPNODE events). Rationale: on numerically fragile models (e.g., the dense gen-LP formulation on deep conv nets), Gurobi can silently certify wrong bounds — `NumericFocus=2` has been observed to return `ObjBound=+0.034` on a problem whose true bound is `-0.355`, with no queryable attribute flagging the issue. The log-stream warnings captured by `optimize_checked` are the only reliable signal, and raising on them turns silent soundness failures into loud errors. Callers that genuinely want to tolerate trouble (e.g., a tightening pass that will re-solve with sparser formulation on failure) should catch `GurobiNumericTrouble` explicitly.

## Gen-LP/MILP entry forms — must dispatch by `entry['form']`

Two coordinate systems are in use for unstable-neuron entries that flow through `_dependency_cone` and the per-target gen-cone LP/MILP builders:

- **`form='alpha'`** — produced by `verify_gen_lp.precompute_gen_state` / `_gen_cone_state`. The new generator column for unstable neuron k carries coefficient `1.0` and is paired with the alpha-form variable `a_k ∈ [0, hi_k]` (the actual ReLU output). LP/MILP builder: `_build_gen_cone_lp` in `verify_graph.py`.
- **`form='phase1'`** — produced by `_record_zono_pre_relu_rows` (the live-zonotope **piggyback** path) and propagated via `state_from_phase1`. The new column carries coefficient `μ_k` and is paired with `e_new_k ∈ [-1, 1]` in the parallelogram `y_k = λ_k·z_k + μ_k·(1+e_new_k)`. LP/MILP builder: `_build_gen_cone_lp_phase1` in `verify_graph.py` or `_build_phase1_lp` in `verify_gen_lp.py`.
- **`form='alpha_zono'`** — same coordinate system as `phase1` but with α-CROWN tightened `(λ, μ)`. Builder: `_build_alpha_zono_lp`.

The two coordinate systems are NOT interchangeable. Feeding `phase1`-tagged rows into the alpha-form builder produces a sound but materially looser bound (mnist_fc_256x6 prop_5 L=1 j=1: `−14.886` from the mismatch vs `−11.531` from a consistent dispatch), and once was a real production bug — see `tests/test_gen_cone_form_dispatch.py`.

When adding new code that consumes a `gen_rows_by_layer` / `unstable_list` entry: read `entry['form']` and dispatch to the matching builder. Both `_build_gen_cone_lp` and `_build_gen_cone_lp_phase1` assert on the entry's form tag and refuse to silently encode a mismatch. Producers MUST stamp `entry['form']` so this assertion is meaningful — the regression test pins the contract.

## Testing

Tests use pytest. Unit tests cover zonotope math and individual op propagation. Integration tests load real ONNX networks from vnncomp benchmarks (discovered via `instances.csv`), run point propagation, and validate against onnxruntime. On soundness failure, per-node comparison identifies the divergent op. External benchmark paths configured in `tests/paths.yaml` (gitignored, template at `tests/paths.yaml.template`).

### During development

Run unit tests with coverage after any code change:

```bash
# Unit tests only (fast, ~2s) — must be 100% line coverage
.venv/bin/python -m pytest tests/ -k "not vnncomp" --cov=src/vibecheck --cov-report=term --cov-report=html
```

### When user asks to run all tests

Run the full suite including vnncomp integration:

```bash
# Full suite: unit tests + vnncomp regular track (~1 min)
.venv/bin/python -m pytest tests/ -k "not extended" --cov=src/vibecheck --cov-report=term --cov-report=html
```

Coverage report at `htmlcov/index.html`.

### Coverage rules

The goal is **100% line coverage from unit tests alone** (without vnncomp). Current status: 300 unit tests, 100% coverage, 1586 statements.

- **Never use `# pragma: no cover`** — write tests for every line instead.
- **Never use defensive `try/except`** — use assertions so the passing path gets coverage. If a condition can't actually occur, remove the dead code rather than testing it.
- **Remove dead code** rather than writing tests for unreachable branches. If a branch is provably unreachable (e.g., a loop that always returns), delete it.
- **Synthetic ONNX models** for testing onnx_loader parsing branches: create models with `onnx.helper` in `test_onnx_ops.py`. Each ONNX op type should have a test that creates a minimal model exercising that parsing path.
- **Inline VNNLIB text** for testing the parser: use `parse_vnnlib_text()` with strings in `test_spec.py`. No temp files needed.
- **Small real ONNX files** for loading tests: use ACAS Xu (tiny FC), cersyve (fork points), cifar100 (ResNet+BN fold) from the vnncomp benchmarks in `test_loading.py`.

### How to run

```bash
# Unit tests only — fast (~2s), must be 100% coverage
.venv/bin/python -m pytest tests/ -k "not vnncomp"

# Full correctness check — unit tests + vnncomp regular track (~1 min)
.venv/bin/python -m pytest tests/ -k "not extended"

# Extended track (currently has some known failures)
.venv/bin/python -m pytest tests/ -k "extended"
```

## Experiment runs — cache `details` to `/tmp`

When running a verify_graph / milp_verify / zonotope_bnb_verify experiment for the user, **pickle the returned `details` dict** (and anything else needed to answer follow-up questions — stdout capture, per-layer stats, the settings used) to a named file under `/tmp/vibecheck_runs/`. The user often asks for different views of the **same run** ("what was phase 7 timing?", "how many open queries?", "what's the unstable count at L3?"); re-running a 60 s benchmark to re-derive numbers that are already in `details` is wasteful.

Conventions:
- Cache path: `/tmp/vibecheck_runs/{short_slug}.pkl` (e.g. `cifar100_5127_sidx_993_milp_l1.pkl`). Include the instance id and config in the slug so cache entries don't collide.
- When answering a follow-up from cache, **say so explicitly** — e.g. "cached from the earlier run" — so the user knows I didn't re-run.
- For a **new experiment** (different instance, different settings, code change that invalidates old results), always re-run and overwrite the cache.
- `details` already contains: `timing`, `per_layer_timing`, `neuron_stats.per_layer`, `phase7[qi]`, `racing[qi]`, `phase`, `n_splits`, `remaining`. Prefer reading these over re-parsing stdout.

## Remote GPU Machine

A remote machine with an NVIDIA RTX 3080 (10 GB VRAM) is available via SSH at `ssh stan@100.83.144.97`. If the user asks, use it for GPU profiling, benchmarking, and testing CUDA workloads that benefit from a desktop-class GPU (320W TDP, ~760 GB/s memory bandwidth). Files can be copied with `scp` or `rsync`.

- **Specs**: 64 GB RAM, i9-11900KF (8C/16T), CUDA 13.0, driver 580.126.09.
- **Remote working dir (vibecheck checkout)**: `~/Desktop/temp/vibecheck-temp` — the package is installed in editable mode; its interpreter is `~/Desktop/temp/vibecheck-temp/.venv/bin/python` and the CLI entry-point is `~/Desktop/temp/vibecheck-temp/.venv/bin/vibecheck`. If the venv is missing, recreate it with `python3 -m venv ~/Desktop/temp/vibecheck-temp/.venv && ~/Desktop/temp/vibecheck-temp/.venv/bin/pip install -e ~/Desktop/temp/vibecheck-temp[dev]`.
- **VNNCOMP benchmarks** (ONNX + vnnlib files): `~/repositories/vnncomp2025_benchmarks/benchmarks/` — same tree as locally (e.g. `cifar100_2024/{onnx,vnnlib}`). Also a duplicate at `~/Desktop/temp/vnncomp2025_benchmarks/` (symlink-friendly).
- **VNNCOMP reference results** (α,β-CROWN verdicts, counter-examples, results.csv): `~/repositories/vnncomp2025_results/alpha_beta_crown/2025_*/results.csv` — use this to pick α,β-CROWN-provable ("unsat") cases or match verdicts.
- **α,β-CROWN source** (for behaviour comparisons / diff checks): `~/Desktop/temp/abcrown/alpha-beta-CROWN_vnncomp2025`.
- **Keeping the remote in sync**: when you need the current local code on the remote, `rsync -az --exclude '.venv' --exclude 'htmlcov' --exclude '__pycache__' /home/stan/repositories/vibecheck/ stan@100.83.144.97:~/Desktop/temp/vibecheck-temp/` is the pattern. Re-run `pip install -e .` after if `pyproject.toml` changed.

## Active investigations — keep iterating, don't stop to ask

When the user directs "keep going" or "implement it," the default is:

- Keep implementing and running explorations until the goal is reached
  OR a hard structural impossibility is DOCUMENTED with evidence.
- Don't stop after a single negative result to ask whether to proceed.
  Run the next reasonable experiment.
- Multi-iteration, multi-pass, or iterative refinement is OK even when
  slower than the target reference — correctness first, then optimize.
- Before each big implementation, write a SMALL TEST on a toy problem
  to validate correctness; then scale.
