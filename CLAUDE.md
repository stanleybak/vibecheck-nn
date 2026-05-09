# CLAUDE.md

Guidance for Claude Code working in this repository.

## Working style

Correctness over speed. Match scope to what was asked, but fix adjacent broken code when that's clearly the right move. Add error handling at real boundaries (I/O, parsing, external systems), not as defensive padding. Don't extract abstractions until duplication causes real maintenance risk.

When making causal claims (e.g. "X is slower because Y", "AB-CROWN does Z"), back them with a measurement, code reference with line numbers, or directly observed log line. "Probably", "I think", "presumably" are warning signs — either gather evidence first or label the statement as a hypothesis. Especially for comparisons against α,β-CROWN: read the source or run it with debug prints, don't infer from behaviour.

## What is vibecheck?

A zonotope-based neural network verification tool. Given an ONNX network and a VNNLIB spec, decides whether the property is provably satisfied ("verified") or "unknown" using abstract interpretation with zonotope domains.

## Operational rules

- **Use `.venv/bin/python`** for all commands. **Never run `git commit` or `git push`** — the user handles git.
- **Don't broad-kill processes.** `pkill -f python` or `kill $(pgrep ...)` will take down the tmux session hosting Claude. Only kill specific PIDs you started (tracked via `run_in_background`) or narrow the pattern to the offending script.
- **Run memory-uncertain experiments under a cgroup cap:** `systemd-run --user --scope -p MemoryMax=8G ...`. Without a cap a runaway tensor allocation kills tmux and Claude along with the process.
- **Single-threaded BLAS** is set in `__init__.py` (`OMP_NUM_THREADS=1` etc.) before numpy import — multi-threaded OpenBLAS causes huge run-to-run jitter on small matrices.
- **Never silently swallow OOM.** Default behavior is to re-raise `torch.cuda.OutOfMemoryError` and `MemoryError`. Only catch when `settings.raise_on_oom=False` (e.g. a benchmarking loop that records OOM as an outcome).

## Architecture

Pipeline: **ONNX loading → graph construction → zonotope propagation → spec check**. Object-oriented dispatch — each ONNX op is a `GraphNode` subclass with `infer_shape()` and `zonotope_propagate()`.

- **`network.py`** — `ComputeGraph` DAG. Each `GraphNode` subclass (Conv, ReLU, Add, Slice, etc.) implements its own propagation. `OP_REGISTRY` maps ONNX strings to subclasses. `print(graph)` for a structural summary.
- **`onnx_loader.py`** — Loads ONNX into `ComputeGraph`. Constant folding, topo sort, shape inference, BatchNorm folding into preceding Conv/Gemm.
- **`spec.py` / `vnnlib_loader.py`** — VNNLIB parsing into `VNNSpec` (input bounds + DNF disjunction of `Conjunct`s). `VNNSpec.check(lo, hi)` returns margins.
- **`zonotope.py`** — `DenseZonotope` with `propagate_linear()` (FC/Conv), `apply_relu()` (min_area / y_bloat / box relaxations), and `add()` for skip-connection merges.
- **`verify.py`** — Thin dispatch loop for the basic zonotope path.
- **`verify_milp.py`** — MILP pipeline: GPU zono+CROWN → per-layer tightening → spec MILP with racing escalation. `_tighten_layer_parallel`, `_racing_escalation`, `_solve_spec_worker`.
- **`verify_zono_bnb.py`** — BnB with zono/CROWN. Also exposes `_forward_zonotope_graph`, `_spec_backward_graph`, `_make_slopes` used by the graph pipeline.
- **`verify_graph.py`** — Graph-mode pipeline (zono + CROWN + interleaved LP/MILP). Phase 2.5 (`_phase2p5_zono_lift`) tightens pre-ReLU bounds via the closed-form box+halfspace LP, query-local. Phase 1 has `legacy` (interleaved-forward + per-layer tightener) and `bab_refine` (α,β-CROWN-style cascade with sliding-window MILP-tighten + α-CROWN refresh).
- **`box_halfspace.py`** — Closed-form LP for `{e ∈ [-1,1]^n : a·e ≤ β}`. `lagrangian_min`/`max` solve in O(n log n) via the 1D concave Lagrangian dual; matches Gurobi LP to numerical tolerance, much faster.
- **`verify_gen_lp.py`** — Generator-based LP/MILP encoding. Smaller model than per-neuron builders, identical LP triangle bounds. `phase8_milp_mode` ∈ {find_sat, infeasibility, alpha_zono_bnb, alpha_zono_infeasibility}.
- **`onnx_optimizer.py`** — Semantics-preserving rewrites: `fold_relusplitter` collapses `Conv(C→2C)→ReLU→Conv(2C→C,1×1)→ReLU` back to `Conv(C→C)→ReLU` (exact: ReLU(z) − ReLU(−z) = z). `fuse_gemm_reshape_conv` merges `Gemm→Reshape→Conv`.
- **`gurobi_util.py`** — `optimize_checked(model)` wraps `model.optimize()` with a callback that captures numeric-trouble warnings and raises `GurobiNumericTrouble`.
- **`settings.py`** — `default_settings()` returns a DotMap. **`main.py`** — CLI; exit 0 = verified, 1 = unknown.

## Gurobi convention

**Never call `model.optimize()` directly. Use `optimize_checked(model)`.** Rationale: numerically fragile models can silently certify wrong bounds (observed cases of wrong-sign `ObjBound` with no queryable flag). The log-stream warnings captured by `optimize_checked` are the only reliable signal — raising on them turns silent soundness failures into loud errors. Callers that want to tolerate trouble (e.g., re-solve with sparser formulation) should catch `GurobiNumericTrouble` explicitly.

## Gen-LP/MILP entry forms — must dispatch by `entry['form']`

Three coordinate systems flow through `_dependency_cone` and the per-target builders:

- **`form='alpha'`** — from `precompute_gen_state` / `_gen_cone_state`. New gen column for unstable k carries coefficient 1.0, paired with `a_k ∈ [0, hi_k]`. Builder: `_build_gen_cone_lp`.
- **`form='phase1'`** — from `_record_zono_pre_relu_rows` (live-zono piggyback) via `state_from_phase1`. New column carries μ_k, paired with `e_new_k ∈ [-1, 1]` in the parallelogram `y_k = λ_k·z_k + μ_k·(1+e_new_k)`. Builder: `_build_gen_cone_lp_phase1` / `_build_phase1_lp`.
- **`form='alpha_zono'`** — same coordinate system as `phase1` but with α-CROWN tightened `(λ, μ)`. Builder: `_build_alpha_zono_lp`.

The systems are NOT interchangeable. Mixing them gives a sound but materially looser bound — once a real production bug; pinned by `tests/test_gen_cone_form_dispatch.py`. Producers must stamp `entry['form']`; the builders assert on it.

## Testing

Unit tests cover zonotope math and individual op propagation. Integration tests load ONNX from vnncomp benchmarks (paths in `tests/paths.yaml`, gitignored), run point propagation, validate against onnxruntime.

```bash
# Unit tests only — fast (~2s); must remain 100% line coverage
.venv/bin/python -m pytest tests/ -k "not vnncomp" --cov=src/vibecheck --cov-report=term

# Full correctness check (unit + vnncomp regular)
.venv/bin/python -m pytest tests/ -k "not extended"
```

100% line coverage from unit tests alone is the goal. **No `# pragma: no cover`** — write the test instead. **No defensive `try/except`** — assert so the passing path gets coverage. **Remove dead code** rather than testing unreachable branches. Use synthetic ONNX (`onnx.helper`) for op-parsing tests and inline VNNLIB strings for parser tests.

## Experiment runs — cache `details` to /tmp

When running a verify experiment for the user, pickle the returned `details` dict to `/tmp/vibecheck_runs/{slug}.pkl`. The user often asks for different views of the same run ("what was Phase 7 timing?", "unstable count at L3?"); re-running a 60s benchmark to re-derive numbers already in `details` is wasteful. Include the instance id and config in the slug. When answering from cache, say so explicitly. For new experiments (different instance/settings/code change), re-run and overwrite.

## Remote GPU machine

`ssh stan@100.83.144.97` — RTX 3080 (10 GB), 64 GB RAM, 16-thread i9. Use for GPU profiling and benchmarks.

- **Vibecheck checkout**: `~/Desktop/temp/vibecheck-temp` (editable install, `.venv/bin/python`).
- **VNNCOMP benchmarks**: `~/repositories/vnncomp2025_benchmarks/benchmarks/`.
- **VNNCOMP reference results** (α,β-CROWN verdicts): `~/repositories/vnncomp2025_results/alpha_beta_crown/2025_*/results.csv`.
- **α,β-CROWN source** (for behaviour comparisons): `~/Desktop/temp/abcrown/alpha-beta-CROWN_vnncomp2025`.
- **Sync from local**: `rsync -az --exclude '.venv' --exclude '__pycache__' /home/stan/repositories/vibecheck/ stan@100.83.144.97:~/Desktop/temp/vibecheck-temp/`. Re-run `pip install -e .` on remote if `pyproject.toml` changed.

## Active investigations — keep iterating

When the user says "keep going" or "implement it": keep going until the goal is reached or a structural impossibility is documented with evidence. Don't stop after one negative result to ask whether to proceed — run the next reasonable experiment. Multi-iteration / multi-pass refinement is fine even when slower than the reference; correctness first, then optimize. Before each big implementation, write a small toy-problem test to validate correctness before scaling.
