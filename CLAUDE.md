# CLAUDE.md

Guidance for Claude Code working in this repository.

## Working style

Correctness over speed. Match scope to the request, but fix adjacent broken code when that's clearly right. Add error handling at real boundaries (I/O, parsing, external systems), not as padding.

Back causal claims ("X is slower because Y", "AB-CROWN does Z") with a measurement, a code reference (`file:line`), or an observed log line — not "probably"/"I think". Especially for comparisons against α,β-CROWN: read its source or run it with debug prints, don't infer from behavior.

## What is vibecheck?

A zonotope-based neural network verifier. Given an ONNX network and a VNNLIB spec it decides `unsat` (verified), `sat` (counterexample), or `unknown`/`timeout`, via abstract interpretation over zonotope domains plus CROWN/α-CROWN, LP/MILP (Gurobi), and BaB.

## Hard rules (these prevent real disasters / soundness bugs)

- **Use `.venv/bin/python`** for everything (keeps the single-threaded BLAS set in `__init__.py` consistent). **Never `git commit`/`push`** — the user handles git.
- **The verdict is the `--results-file` contents — NEVER the exit code or stdout.** Pass `--results-file`, read it after the subprocess, refuse to count a case whose file is missing or holds an unexpected value. A sweep once "verified 194/194 in 161 s" while running an empty no-op loop (the module body ran without `main()`); suspiciously uniform timings are the tell. Trust the file, nothing else. (Exit codes are only 0=verified, 1=unknown, 2=error.)
- **Acceptable verdicts for a SAT-region property: both `unsat` (proof) and a within-tolerance `sat` are acceptable; prefer proving `unsat`.** A *within-tolerance counterexample* is a point whose spec margin is within `COUNTEREXAMPLE_ATOL` (1e-4) of violating — VNNCOMP's scorer accepts it. So on a tolerance-boundary instance (true margin ~0 ± atol, e.g. ml4acopf 14_ieee prop3), emitting the within-tol `sat` is fine when we can't prove `unsat`; don't treat it as a false-sat or "stop and fix" it. Policy: a *clear* CE (margin < −atol) → return `sat` immediately; a within-tol CE → write it early (so a timeout can't lose it) but **keep searching** for a clear CE or an `unsat` proof (which overrides it); never let a later `timeout`/`unknown` overwrite a `sat` already found. (`verify_graph._sat_disposition` + `main._emit_result`.)
- **Never broad-kill processes.** `pkill -f python` / `killall python` takes down the tmux session hosting Claude and any batch orchestrator. Match the verifier narrowly (`vibecheck\.main`) or kill only specific PIDs you started.
- **Cap memory-uncertain experiments:** `systemd-run --user --scope -p MemoryMax=8G ...`. A runaway tensor alloc otherwise OOM-kills the session. Long/heavy/GPU-scaled work goes to a remote box, not local (local hosts the tmux Claude runs in).
- **Never silently swallow OOM.** Re-raise `torch.cuda.OutOfMemoryError` / `MemoryError`; catch only when `settings.raise_on_oom=False` (benchmarking loops that record OOM as an outcome).
- **No silent exception swallowing.** `except Exception: pass` / `continue` (even behind a debug print) is forbidden — it once hid a real bug (forward LiRPA silently disabled). Allowed only: catch ONE specific type you have a documented reason for, or catch + log + re-raise. Treat any `except Exception` you find as a bug to fix.
- **Op dispatches end in `else: raise NotImplementedError(...)` — never silently skip an op.** A missing `else` once let Slice/Concat fall through, zeroed a backward chain, and declared `verified` on a SAT case. Same for bias branches: never drop a bias without proving it's sound.
- **Nonlinear-op bounds must be symbolic / provably sound — never sampling-based.** ReLU, sigmoid, tanh, softmax, GELU, attention, layernorm: use closed-form bounds (monotonicity/convexity) or a Newton/binary-search that provably brackets the worst case (see auto_LiRPA `tanh.py`). Sampling N points and taking max/min is NOT a sound bound — the worst case can lie between samples. Validate by checking many adversarial points violate nothing (that's the test, not the bound).

## Architecture

Pipeline: **ONNX load → graph construction → zonotope propagation → spec check.** OO dispatch — each ONNX op is a `GraphNode` subclass with `infer_shape()` + `zonotope_propagate()`; `OP_REGISTRY` maps op strings to subclasses (`network.py`). `print(graph)` gives a structural summary.

- **`onnx_loader.py`** — ONNX → `ComputeGraph`: constant folding, topo sort, shape inference, BatchNorm folding. `onnx_optimizer.py` does semantics-preserving rewrites (relusplitter fold, gemm/reshape/conv fusion).
- **`spec.py` / `vnnlib_loader.py`** — VNNLIB → `VNNSpec` (input bounds + DNF of `Conjunct`s); `VNNSpec.check(lo, hi)` returns margins.
- **`zonotope.py`** — `DenseZonotope`: `propagate_linear()` (FC/Conv), `apply_relu()` (relaxations), `add()` (skip merges).
- **Verify entry points** — `verify_graph.py` (production graph mode: zono + CROWN + interleaved LP/MILP), `verify_milp.py` (MILP pipeline), `verify_zono_bnb.py` (BnB; also the forward/backward graph builders), `verify.py` (basic zono path).
- **Bounding / search** — `alpha_crown.py` / `alpha_tighten.py` (α-CROWN), `forward_lirpa.py` / `bounded_module.py` (forward linear bounds), `box_halfspace.py` / `lagrangian_n.py` (closed-form halfspace LP), `dual_ascent_bab.py`, `batched_zono.py` (vectorized input-split), `pgd.py` (counterexample search).
- **Config / CLI** — `settings.py` (`default_settings()` → DotMap), `config_loader.py` (`--config <yaml>`, keys map 1:1 to Settings attrs), `config_profiles.py` (`default_settings_for(graph, spec)` when no `--config`), `main.py` (CLI).

### Two correctness traps

- **Gurobi: never call `model.optimize()` directly — use `optimize_checked(model)`** (`gurobi_util.py`). Numerically fragile models can silently certify wrong bounds; the log-stream warnings it captures are the only reliable signal, and it raises `GurobiNumericTrouble` on them.
- **Gen-LP/MILP: dispatch by `entry['form']`** (`'alpha'` / `'phase1'` / `'alpha_zono'`). The three coordinate systems are NOT interchangeable — mixing gives a sound-but-looser bound (once a real bug; pinned by `tests/test_gen_cone_form_dispatch.py`). Producers stamp `entry['form']`; builders assert on it.

## Testing

```bash
# Unit — synthetic ONNX/VNNLIB, ~2 s, must stay 100% line coverage (no pragma, no defensive try/except)
.venv/bin/python -m pytest tests/ -k "not vnncomp" -m "not integration" --cov=src/vibecheck --cov-report=term
# Per-benchmark verdict regressions (need a local benchmark clone via tests/paths.yaml)
.venv/bin/python -m pytest tests/integration -m integration
```

Three categories: (1) **unit** — zono math + per-op propagation; (2) **vnncomp point-prop** — load real ONNX, validate against onnxruntime; (3) **integration** — per-benchmark verdict regressions in `tests/integration/test_<benchmark>.py` (`@pytest.mark.integration`), ~3 pinned cases each (1 SAT if cracked + 2 hard UNSAT); every merge re-runs all prior cases.

## Benchmark campaign & remote GPUs

The per-benchmark optimization process (branch-per-benchmark, configs, integration pins,
soundness gate, pre-merge gap report) and the remote-GPU infrastructure (server1 / AWS g5
hosts, idle-shutdown discipline, sweep economy, `details` caching) live in
**`docs/benchmarks/overall.md`**. Read it before starting benchmark work or touching a remote
box. Per-benchmark records are `docs/benchmarks/<name>.md`. Current 2026 benchmark lists,
status, and known upstream-benchmark bugs are in the **`vnncomp2026`** skill
(`.claude/skills/vnncomp2026/SKILL.md`).

Three essentials worth keeping in front of you: heavy/long/GPU-scaled work runs on a **remote
box, never local** (local hosts the tmux Claude runs in — an OOM there kills the session);
every AWS `ssh` must `sudo rm -f /tmp/idle_since` or the idle-shutdown fires mid-task; and
**never keep anything you care about in the remote `/tmp`** — a stop/start (e.g. idle-shutdown,
which changes the public IP too) **wipes `/tmp`**, so diagnostic scripts, oracles, and results
must live under `~/persistent_runs/` (survives reboot) or, better, be kept in the local repo
(e.g. `scratch/`) and `rsync`'d up, so they can be re-pushed after any restart.
