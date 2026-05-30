# CLAUDE.md

Guidance for Claude Code working in this repository.

## Working style

Correctness over speed. Match scope to what was asked, but fix adjacent broken code when that's clearly the right move. Add error handling at real boundaries (I/O, parsing, external systems), not as defensive padding. Don't extract abstractions until duplication causes real maintenance risk.

When making causal claims (e.g. "X is slower because Y", "AB-CROWN does Z"), back them with a measurement, code reference with line numbers, or directly observed log line. "Probably", "I think", "presumably" are warning signs — either gather evidence first or label the statement as a hypothesis. Especially for comparisons against α,β-CROWN: read the source or run it with debug prints, don't infer from behaviour.

## What is vibecheck?

A zonotope-based neural network verification tool. Given an ONNX network and a VNNLIB spec, decides whether the property is provably satisfied ("verified") or "unknown" using abstract interpretation with zonotope domains.

## Operational rules

- **Use `.venv/bin/python`** for all commands. **Never run `git commit` or `git push`** — the user handles git.
- **Sweep verdicts MUST come from `--results-file` contents, NEVER from exit code or stdout.** The CLI accepts `--results-file PATH` and writes a single VNNCOMP-style line: `unsat` (verified), `sat` (counterexample), `unknown`, or `timeout`. Sweep scripts MUST: (a) pass `--results-file /tmp/r.txt`, (b) read the file after the subprocess, (c) refuse to count a case if the file is missing or contents are anything other than the expected verdict. This actually shipped: a sweep that "verified 194/194 in 161 s" was an empty no-op loop — `python -m vibecheck.main` ran the module body without calling `main()` (no `if __name__ == "__main__":` block existed), exited 0 silently, and the sweep counted every case as verified. Suspicious uniform timing (e.g. all 0.81–0.85 s for cases of widely varying size) is a tell — investigate before reporting any speedup claim. Trust the verdict file, nothing else.
- **Don't broad-kill processes.** `pkill -f python` or `kill $(pgrep ...)` will take down the tmux session hosting Claude. Only kill specific PIDs you started (tracked via `run_in_background`) or narrow the pattern to the offending script.
- **Run memory-uncertain experiments under a cgroup cap:** `systemd-run --user --scope -p MemoryMax=8G ...`. Without a cap a runaway tensor allocation kills tmux and Claude along with the process.
- **Long-running or memory-uncertain experiments go to server1, NOT local.** Local hosts the tmux that Claude Code runs in — any process that OOMs or runs the GPU dry kills the whole session (lost context, mid-flight work). Local is for quick (~30s) smoke tests only. Anything that loops over benchmark cases, allocates large gen-tensors, or uses GPU memory at scale → `ssh "$SERVER1_HOST"` and run there. Same goes for `pkill`/`kill` from local — only safe to kill processes on the remote server, never on local.
- **Single-threaded BLAS** is set in `__init__.py` (`OMP_NUM_THREADS=1` etc.) before numpy import — multi-threaded OpenBLAS causes huge run-to-run jitter on small matrices.
- **Never silently swallow OOM.** Default behavior is to re-raise `torch.cuda.OutOfMemoryError` and `MemoryError`. Only catch when `settings.raise_on_oom=False` (e.g. a benchmarking loop that records OOM as an outcome).
- **Bounds for nonlinear ops MUST be symbolic / provably-sound, never sampling-based.** This applies to ReLU, sigmoid, tanh, softmax, GELU, attention, layernorm — any nonlinear activation or transformer op. Sampling N points and taking max/min of the gap is *not* a sound upper/lower bound (the worst case may lie between samples). Use either: (a) closed-form bounds derived from monotonicity / convexity / concavity of the function, or (b) Newton / binary-search on a monotone condition that provably brackets the worst case (auto_LiRPA's `precompute_relaxation` pattern in `tanh.py`). When in doubt, verify soundness by sampling many adversarial points and checking that NONE violate the bound (this is the test, not the bound itself).
- **No silent exception swallowing.** `except Exception: pass`, `except: pass`, and bare `except Exception` blocks (even with a debug-only print gate) are FORBIDDEN. They hide real bugs and route us back to the same class of failures as silent op-skipping. The pattern manifests in several disguises — all banned:
  - `try: ...; except Exception: pass` (no logging, no re-raise)
  - `try: ...; except Exception: <debug print if env var>; pass`
  - `try: ...; except Exception as e: continue` in a loop
  - Broad `except (Exception, RuntimeError, ValueError, ...)` that catches more than the documented expected error
  
  The only acceptable forms:
  - Catch ONE specific exception type that you have a documented reason for (e.g. `except torch.cuda.OutOfMemoryError:` with a comment that fallback is per-sub) — narrow exception class is the proof that you understood what can fail.
  - Catch + log + re-raise, when you need context added before propagation.
  - Catch only when `settings.raise_on_oom=False` (existing benchmarking-loop convention — also narrow type).
  
  A real bug shipped from this (forward LiRPA wire-in silently disabled because an outer `except Exception: pass` swallowed the upstream `DotMap` falsy-default issue — the path the user thought was active was actually never running). When auditing or writing new code, search for `except Exception` and treat every hit as a bug-in-waiting. If you find one in code you didn't write, fix it as part of your change.
- **Op dispatches must `else: raise NotImplementedError`. Never silently skip an op.** Every `elif t == ...` chain (forward zono, CROWN backward, PGD forward, MILP builder, alpha_crown adaptive, etc.) must terminate in an explicit `else: raise NotImplementedError(f'...: unsupported op {t!r} at {name!r}')`. A real soundness bug (linearizenn `prop_10_10`) shipped because `_spec_backward_graph` had no `else`: Slice/Concat ops silently fell through, backward `ew` died mid-chain, `ew_at[input]` stayed zero, and `spec_lb = acc + 0` was vacuously positive — we declared `verified` on a SAT case ABC's witness easily satisfied. The same rule applies to bias-handling branches: never have `if bias is not None: acc += ...` without considering whether silently dropping the bias is sound. Adding the raise turns latent unsoundness into a loud failure that surfaces the missing op immediately.

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

Two GPU machines available. Pick based on availability — neither is preferred over the other.

### server1 (RTX 3080, 10 GB)

Connect: `ssh stan@100.107.254.48` (RTX 3080 / 10 GB, 64 GB RAM, 16-thread i9;
hostname `reliablesystems-ubuntu`). This is a Tailscale IP — only reachable
inside the owner's tailnet, so it's fine to keep in git. `$SERVER1_HOST` also
works if set. Steady-state α-CROWN throughput ~1 s/freeze when healthy.

**Sometimes overheats / GPU "falls off the bus" (Xid 79)** under sustained load — `nvidia-smi` returns "No devices were found", verdicts go from `verified` → `error_no_result` near sweep end. Recovery requires either `sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia` (asks user to run) or full reboot. Xid 154 ("Node Reboot Required") means modprobe won't help — reboot only.

Layout on server1:

- **Vibecheck checkout**: `~/Desktop/temp/vibecheck-temp` (editable install, `.venv/bin/python`).
- **VNNCOMP benchmarks**: `~/repositories/vnncomp2025_benchmarks/benchmarks/`.
- **VNNCOMP reference results** (α,β-CROWN verdicts): `~/repositories/vnncomp2025_results/alpha_beta_crown/2025_*/results.csv`.
- **α,β-CROWN source** (for behaviour comparisons): `~/Desktop/temp/abcrown/alpha-beta-CROWN_vnncomp2025`. Conda env: `~/miniconda/envs/abcrown/bin/python`.
- **Sweep scripts + results**: `~/persistent_runs/{scripts,results}` (sweep_sxs.py, runner_p25off.py, sxs_v* result dirs). Use this directory for sweep logs — survives reboot (unlike `/tmp`).
- **Sync from local**: `rsync -az --exclude '.venv' --exclude '__pycache__' /home/stan/repositories/vibecheck/ "$SERVER1_HOST":~/Desktop/temp/vibecheck-temp/`. Re-run `pip install -e .` on the server if `pyproject.toml` changed.

### AWS g5 (A10G, 24 GB)

Connection details live in env vars (NOT here — keeps the hostname/keyfile out of git):
- `$AWS_GPU_HOST` — e.g. `ubuntu@ec2-...compute-1.amazonaws.com`
- `$AWS_GPU_PEM` — path to the .pem key in `~/.ssh/`

Connect: `ssh -i "$AWS_GPU_PEM" "$AWS_GPU_HOST"`. User starts/stops via AWS console — Claude only SSHes in, never starts/stops via API. 24 GB GPU mem means ABC + vibecheck fit on cases that OOM on server1.

Env vars are set in `~/.profile`. If unset, see `AWS_SETUP.txt` at repo root for the runbook (driver install, venv, repo sync, cron). The setup doc itself is generic — no hostname in it either.

**Idle-shutdown protocol (AWS only)**: if the user has been idle for **31+ minutes** while an AWS sweep or other GPU work might still be holding the instance up, Claude MUST proactively SSH in, check `nvidia-smi` + running processes, and if nothing useful is running, run `sudo shutdown -h now` to stop the billing clock (~$1/hr otherwise burns silently overnight). Don't shut down if a justifying long sweep is producing results.

**Reset the AWS idle counter on every interaction.** The instance runs `/usr/local/bin/idle-shutdown.sh` every 5 min and counts toward shutdown when GPU<5% AND `who` shows no interactive ssh. Claude's batch `ssh host 'cmd'` calls do NOT register in `who`, so they accumulate idle seconds even while Claude is actively working. To avoid Claude's own AWS box shutting down mid-task, **every** AWS ssh call must remove `/tmp/idle_since` — bake it into the command itself. Pattern:
```
ssh -i "$AWS_GPU_PEM" "$AWS_GPU_HOST" 'sudo rm -f /tmp/idle_since; <real command>'
```
Cheap to do, prevents losing 20 min of work to a shutdown that fires while a long-running script is just slow rather than idle.

(There used to be a server2 for parallel sweeps. Removed from this guide because its RTX 3080 showed 30× slower and highly variable α-CROWN timing — apparent thermal/power degradation. Don't fall back to it.)

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
8. **Squash-merge** to `main`. Keep the `bench/<benchmark>` branch around (do NOT delete) — useful as a rollback point and for going back to inspect non-squashed history later.

Allowed references: read auto_LiRPA / AB-CROWN source (`~/Desktop/temp/abcrown/alpha-beta-CROWN_vnncomp2025` on remote) or run them with debug prints — especially for non-ReLU activations (tanh, sigmoid, GELU, MHA) — then re-implement.

### Track split (authoritative: `~/repositories/vnncomp2025_results/SCORING-SMALL-TOL/settings.py:30-59`)

**Regular track (16) — what we score on**: acasxu_2023, cersyve, cgan_2023, cifar100_2024, collins_rul_cnn_2022, cora_2024, dist_shift_2023, linearizenn_2024, malbeware, metaroom_2023, nn4sys, safenlp_2024, sat_relu, soundnessbench, tinyimagenet_2024, tllverifybench_2023.

**Extended track (10) — out of scope for scoring**: cctsdb_yolo_2023, collins_aerospace_benchmark, lsnc_relu, ml4acopf_2023, ml4acopf_2024, **relusplitter**, traffic_signs_recognition_2023, vggnet16_2022, vit_2023, yolo_2023.

### Already optimized

acasxu_2023 · cersyve · cgan_2023 · cifar100_2024 · collins_rul_cnn_2022 · cora_2024 · dist_shift_2023 · linearizenn_2024 · malbeware · metaroom_2023 · mnist_fc (historical, regression-only) · nn4sys (partial 46/194 — see docs) · tinyimagenet_2024 · relusplitter (extended; kept as deliverable).

### Queue (alphabetical, regular track only)

safenlp_2024 · sat_relu · soundnessbench · tllverifybench_2023.
