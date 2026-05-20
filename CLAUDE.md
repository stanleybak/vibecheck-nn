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

Three test categories:

1. **Unit tests** — zonotope math + individual op propagation. Synthetic ONNX (`onnx.helper`) + inline VNNLIB strings. Must remain 100% line coverage. **No `# pragma: no cover`**, **no defensive `try/except`** (assert so the passing path gets coverage), **remove dead code** rather than testing unreachable branches.
2. **VNNCOMP point-propagation tests** — load ONNX from vnncomp benchmarks (paths in `tests/paths.yaml`, gitignored), run point propagation, validate against onnxruntime.
3. **Per-benchmark verdict regressions** — live in `tests/integration/<benchmark>.py`, pytest-marked `@pytest.mark.integration`. Each merged benchmark contributes ~3 cases (1 SAT if cracked + 2 hard UNSAT). Every benchmark merge re-runs *all* prior integration cases — catches cross-benchmark regressions.

```bash
# Unit tests only — fast (~2s); must remain 100% line coverage
.venv/bin/python -m pytest tests/ -k "not vnncomp" -m "not integration" --cov=src/vibecheck --cov-report=term

# Per-benchmark verdict regressions
.venv/bin/python -m pytest tests/integration -m integration

# Full correctness check (unit + vnncomp regular)
.venv/bin/python -m pytest tests/ -k "not extended"
```

## Experiment runs — cache `details` to /tmp

When running a verify experiment for the user, pickle the returned `details` dict to `/tmp/vibecheck_runs/{slug}.pkl`. The user often asks for different views of the same run ("what was Phase 7 timing?", "unstable count at L3?"); re-running a 60s benchmark to re-derive numbers already in `details` is wasteful. Include the instance id and config in the slug. When answering from cache, say so explicitly. For new experiments (different instance/settings/code change), re-run and overwrite.

## Remote GPU machines

Two servers, identical hardware (RTX 3080 / 10 GB, 64 GB RAM, 16-thread i9). Mirrored layout — same paths on both. Software setup may diverge during ad-hoc experiments; treat server1 as canonical and re-sync server2 when in doubt.

- **server1** (canonical): `ssh stan@100.83.144.97`
- **server2**: `ssh stan@100.107.254.48`

server1 → server2 SSH is keyed (server1's `~/.ssh/id_ed25519`), so direct rsync works for fan-out: `ssh stan@100.83.144.97 'rsync -az ~/path/ stan@100.107.254.48:~/path/'`.

Shared layout on each server:

- **Vibecheck checkout**: `~/Desktop/temp/vibecheck-temp` (editable install, `.venv/bin/python`).
- **VNNCOMP benchmarks**: `~/repositories/vnncomp2025_benchmarks/benchmarks/`.
- **VNNCOMP reference results** (α,β-CROWN verdicts): `~/repositories/vnncomp2025_results/alpha_beta_crown/2025_*/results.csv`.
- **α,β-CROWN source** (for behaviour comparisons): `~/Desktop/temp/abcrown/alpha-beta-CROWN_vnncomp2025`. Conda env: `~/miniconda/envs/abcrown/bin/python`.
- **Sweep scripts + results**: `~/persistent_runs/{scripts,results}` (sweep_sxs.py, runner_p25off.py, sxs_v* result dirs).
- **Sync from local**: `rsync -az --exclude '.venv' --exclude '__pycache__' /home/stan/repositories/vibecheck/ stan@<server>:~/Desktop/temp/vibecheck-temp/`. Re-run `pip install -e .` on the server if `pyproject.toml` changed.

**Long-running GPU sweeps may degrade the driver** — after a 6-hour cifar100+tinyimagenet sweep, server1's NVIDIA driver hit "Unable to determine the device handle" (load avg pinned at 5.0 from D-state threads). Recovery requires either `sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia` (asks user to run) or reboot. Verdicts going from `verified` → `error_no_result` near sweep end are a tell.

**Use both servers in parallel for sweeps** — they have identical hardware (RTX 3080) and benchmark + ABCROWN + vibecheck checkouts are mirrored. Split work by benchmark or by case index to halve wall time. `scratch/sweep_sxs.py` supports `SWEEP_CATEGORIES=cifar100_2024` to run only one benchmark per server, and `SWEEP_OUT_DIR` to direct output to a server-specific results dir. Sweep launch pattern: launch in `tmux` on each server with nohup-style logging to a file (NOT tmux pipe-pane — it killed a sweep mid-run during the v6 cifar100 + tinyimagenet measurement). Periodically reload reference results to merge per-server outputs.

## Active investigations — keep iterating

When the user says "keep going" or "implement it": keep going until the goal is reached or a structural impossibility is documented with evidence. Don't stop after one negative result to ask whether to proceed — run the next reasonable experiment. Multi-iteration / multi-pass refinement is fine even when slower than the reference; correctness first, then optimize. Before each big implementation, write a small toy-problem test to validate correctness before scaling.

When implementing a multi-phase plan, **push through every phase in sequence**. Treat each phase's stated gate as the only stopping condition (regression → revert that single ablation, then continue). Do not pause between phases to ask "should I keep going?" — only pause if a gate genuinely fails after a sensible revert, or if a destructive irreversible action requires explicit authorization. Tasks (TaskCreate) are good for tracking phases but resolving them is not a checkpoint to stop at.

## Benchmark optimization workflow

Each VNNCOMP **regular-track** benchmark is optimized on its own branch, then squash-merged to `main`. Goal: beat AB-CROWN on every regular-track benchmark.

1. **Branch**: `git checkout -b bench/<benchmark>` from `main`.
2. **Config**: create `configs/<benchmark>.yaml` containing ONLY the overrides on top of `configs/default.yaml`. Keys map 1:1 to `Settings` attrs (no hidden mapping). Loaded explicitly with `--config configs/<benchmark>.yaml`; if no `--config`, fall back to `default_settings_for(graph, spec)`.
3. **Optimize on the remote GPU**: run vibecheck + AB-CROWN live side-by-side; cross-check against the published `~/repositories/vnncomp2025_results/alpha_beta_crown/2025_<benchmark>/results.csv`. Fix any obvious misses.
4. **Stuck-case rule**: if a case won't crack after 1-2 attack angles, surface a diag (timing breakdown, open-spec count, phase outcome) back to the user rather than spinning indefinitely.
5. **Integration tests**: add `tests/integration/test_<benchmark>.py` with **3 hard cases — 1 SAT we cracked + 2 hard UNSAT we verified** (each pinned with `max_wall_s` ~1.5× observed for regression detection). `@pytest.mark.integration`. Every merge re-runs *all* prior benchmarks' integration cases.
6. **Per-benchmark README** at `docs/benchmarks/<benchmark>.md` capturing: (a) final score (vc + abc-server + abc-published, with timestamp + sweep id), (b) algorithmic wins vs published reference, (c) any benchmark-specific knobs in `configs/<benchmark>.yaml` and *why* they're there, (d) reproduction commands (single case + full sweep), (e) integration test cases with rationale, (f) known unsolved cases. This is the canonical record for the benchmark; the YAML + tests are the runnable artifacts.
7. **Pre-merge gap report**: before squash-merging, present (a) cases still unsolved, (b) any visible AB-CROWN wins, (c) score delta vs published AB-CROWN results — to the user for feedback. Do not merge until they approve.
8. **Squash-merge** to `main`; delete branch.

Allowed references: read auto_LiRPA / AB-CROWN source (`~/Desktop/temp/abcrown/alpha-beta-CROWN_vnncomp2025` on remote) or run them with debug prints — especially for non-ReLU activations (tanh, sigmoid, GELU, MHA) — then re-implement.

### Track split (authoritative: `~/repositories/vnncomp2025_results/SCORING-SMALL-TOL/settings.py:30-59`)

**Regular track (16) — what we score on**: acasxu_2023, cersyve, cgan_2023, cifar100_2024, collins_rul_cnn_2022, cora_2024, dist_shift_2023, linearizenn_2024, malbeware, metaroom_2023, nn4sys, safenlp_2024, sat_relu, soundnessbench, tinyimagenet_2024, tllverifybench_2023.

**Extended track (10) — out of scope for scoring**: cctsdb_yolo_2023, collins_aerospace_benchmark, lsnc_relu, ml4acopf_2023, ml4acopf_2024, **relusplitter**, traffic_signs_recognition_2023, vggnet16_2022, vit_2023, yolo_2023.

### Already optimized

cersyve · cifar100_2024 · mnist_fc (historical, regression-only) · tinyimagenet_2024 · relusplitter (extended; kept as deliverable).

### Queue (alphabetical, regular track only)

acasxu_2023 · cgan_2023 · collins_rul_cnn_2022 · cora_2024 · dist_shift_2023 · linearizenn_2024 · malbeware · metaroom_2023 · nn4sys · safenlp_2024 · sat_relu · soundnessbench · tllverifybench_2023.
