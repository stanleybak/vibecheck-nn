## What is vibecheck?

A zonotope-based neural network verification tool. Given an ONNX network and a VNNLIB specification (input bounds + output property), it determines whether the property is provably satisfied ("verified") or "unknown" using abstract interpretation with zonotope domains.

## Development Commands

**Always use the venv (`.venv/bin/python`) for all commands.**
**Never run `git commit` or `git push` — the user handles all git operations.**
**Never kill `tmux` or `claude` processes.** When cleaning up after interrupted or runaway runs, only kill the specific PIDs you started (tracked via `run_in_background`) or narrow the `pkill` pattern to the offending script. Broad `pkill -f python` or `kill $(pgrep ...)` sweeps will take down the tmux session hosting Claude itself.
**Single-threaded BLAS**: `__init__.py` sets `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` via `setdefault` before numpy is imported. Multi-threaded OpenBLAS causes massive scheduling jitter on the small matrices typical in verification (100x variance between runs) and is slower overall. No manual env vars needed; override with explicit env vars if desired.

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

- **`verify_zono_bnb.py`** — Branch-and-bound verification with zonotope/CROWN abstract domains.

- **`settings.py`** — Configuration for BnB and MILP verification. `default_settings()` returns a DotMap with defaults.

- **`main.py`** — CLI entry point. Exit code 0 = verified, 1 = unknown.

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

## Remote GPU Machine

A remote machine with an NVIDIA RTX 3080 (10 GB VRAM) is available via SSH at `ssh stan@100.83.144.97`. Use for GPU profiling, benchmarking, and testing CUDA workloads that benefit from a desktop-class GPU (320W TDP, ~760 GB/s memory bandwidth). Files can be copied with `scp` or `rsync`.

- **Working directory**: `~/Desktop/temp/vibecheck-temp` (create if needed)
- **Specs**: 64 GB RAM, i9-11900KF (8C/16T), CUDA 13.0, driver 580.126.09
- When running Python code remotely, create a venv there and install dependencies as needed.
