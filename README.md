# vibecheck

Vibe-Coded Neural Network Verification Tool — graph branch.

A zonotope-based neural network verifier. Given an ONNX network and a VNNLIB
spec, it decides whether the property is provably satisfied (`unsat` /
"verified"), refuted by a counterexample (`sat`), or `unknown`, using abstract
interpretation with zonotope domains plus CROWN / α-CROWN, LP/MILP, and BaB.

## Setup

Use [uv](https://docs.astral.sh/uv/) — a single binary that pulls a prebuilt
Python and produces a normal venv (the system Python on recent distros is often
too new for the torch wheels):

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a venv with a known-good Python and install vibecheck editably.
# uv venv has no pip by default, so point VIRTUAL_ENV at the venv and use
# `uv pip install`.
uv python install 3.12
uv venv --python 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[dev]"
```

Invoke the tool and tests with `.venv/bin/python` (keeps the single-threaded
BLAS setup in `__init__.py` consistent and avoids env drift).

## Usage

```bash
.venv/bin/python -m vibecheck.main --net model.onnx --spec property.vnnlib
```

Common flags (see `--help` for the full list):

- `--config configs/<benchmark>.yaml` — per-benchmark overrides on top of
  `default_settings()`. When omitted, `default_settings_for(graph, spec)`
  auto-detects a profile.
- `--results-file PATH` — write a single VNNCOMP-style verdict line (`unsat` =
  verified, `sat` = counterexample, `unknown`, or `timeout`). **This is the
  authoritative verdict** — sweep scripts must read it rather than infer from
  the exit code.
- `--timeout SECONDS` — BnB timeout (default 30).
- `--device {gpu,cpu}`, `--bits {16,32,64}`, `--mode {graph,bnb}`.

Exit codes: `0` = verified, `1` = unknown, `2` = error (a verdict line is still
written to `--results-file` when set).

## Tests

```bash
# Unit tests only — fast (~2s); must remain 100% line coverage
.venv/bin/python -m pytest tests/ -k "not vnncomp" -m "not integration" \
    --cov=src/vibecheck --cov-report=term

# Per-benchmark verdict regressions (needs the vnncomp benchmark mirror)
.venv/bin/python -m pytest tests/integration -m integration

# Full correctness check (unit + vnncomp regular)
.venv/bin/python -m pytest tests/ -k "not extended"
```

The integration and vnncomp tests read external benchmark paths from
`tests/paths.yaml` (gitignored). Create it once:

```bash
cp tests/paths.yaml.template tests/paths.yaml
# then edit it to point at your vnncomp2025_benchmarks / _results clones
```

### Running a single test

Integration cases are parametrized by their `desc` string, so `pytest -k`
selects one. To run just **acasxu net 1_1, prop_3** (a fast UNSAT case; the `-k`
terms are AND-ed, so each is a separate word):

```bash
.venv/bin/python -m pytest tests/integration/test_acasxu_2023.py \
    -k "1_1 and prop_3" -m integration -v
```

To run that same instance through the CLI directly (bypassing pytest) — the
ONNX/VNNLIB paths are relative to your `vnncomp_benchmarks` clone, and
`--config` selects the production acasxu path:

```bash
BENCH=/path/to/vnncomp2025_benchmarks/benchmarks/acasxu_2023
.venv/bin/python -m vibecheck.main \
    --net  "$BENCH/onnx/ACASXU_run2a_1_1_batch_2000.onnx" \
    --spec "$BENCH/vnnlib/prop_3.vnnlib" \
    --config configs/acasxu_2023.yaml \
    --timeout 120 \
    --results-file /tmp/r.txt
cat /tmp/r.txt   # -> unsat  (i.e. verified)
```

Any unit test runs the same way, e.g. a single test function:

```bash
.venv/bin/python -m pytest tests/test_zonotope.py::test_propagate_fc -v
```
