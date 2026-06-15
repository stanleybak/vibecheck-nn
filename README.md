# vibecheck

`vibecheck` is a vibe-coded toolkit for solving neural network verification problems. Given an ONNX neural network and a VNNLIB property, it tries to decide whether the property is provably true or refuted by a counterexample.

The underlying verification algorithms are complementary: an initial pass uses Gurobi LP/MILP to compute tight neuron bounds; zonotope abstract interpretation is combined with CROWN / α-CROWN slopes to produces tight overapproximations; a high-performance GPU-enabled branch-and-bound search with an optimal-step dual-ascent solver then explores millions of splits per second (on some problems).

## Setup

Use [uv](https://docs.astral.sh/uv/) for setup and configuration:

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.12
uv venv --python 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[dev]"
```

The tool and tests can then be invoked with `.venv/bin/python`.

## Usage

```bash
.venv/bin/python -m vibecheck.main --net model.onnx --spec property.vnnlib
```

Common flags (see `--help` for the full list):

- `--config configs/<benchmark>.yaml` — per-benchmark overrides on top of
  `default_settings()`. When omitted, a profile is auto-detected from the network
  and spec.
- `--results-file PATH` — write a single VNNCOMP-style verdict line (`unsat` =
  verified, `sat` = counterexample, `unknown`, or `timeout`). **This is the
  authoritative verdict** — read it rather than inferring from the exit code.
- `--timeout SECONDS` — tool timeout (default 30).
- `--device {gpu,cpu}`, `--bits {16,32,64}`, `--mode {graph,bnb}`.

Exit codes: `0` = verified, `1` = unknown, `2` = error (a verdict line is still
written to `--results-file` when set).

## Tests

```bash
# Unit tests — no external data, ~1-2 min (drop --cov for a faster run)
.venv/bin/python -m pytest tests/ -k "not vnncomp" -m "not integration" \
    --cov=src/vibecheck --cov-report=term

# Per-benchmark verdict regressions (need a local benchmark clone; see below)
.venv/bin/python -m pytest tests/integration -m integration
```

The unit tests build synthetic ONNX/VNNLIB inline and need no external data — they
run on a fresh clone. Only the **integration** and **vnncomp point-propagation**
tests read benchmark paths from `tests/paths.yaml` (gitignored).

Run a single unit test by node id, or a single integration case by its
parametrized `desc` (the `-k` terms are AND-ed):

```bash
.venv/bin/python -m pytest tests/test_zonotope.py::test_propagate_fc -v
.venv/bin/python -m pytest tests/integration/test_acasxu_2023.py -k "1_1 and prop_3" -m integration -v
```

## Running specific benchmarks

The integration tests — and any direct CLI run on competition models — load
ONNX/VNNLIB from a local clone of the VNNCOMP benchmarks kept elsewhere on your
system. The sets are published per year as `stanleybak/vnncomp<year>_benchmarks`
— e.g. [vnncomp2025_benchmarks](https://github.com/stanleybak/vnncomp2025_benchmarks)
and [vnncomp2026_benchmarks](https://github.com/stanleybak/vnncomp2026_benchmarks).
Clone one and unpack its models:

```bash
git clone https://github.com/stanleybak/vnncomp2025_benchmarks.git
cd vnncomp2025_benchmarks
./setup.sh        # downloads + unpacks the per-benchmark onnx/vnnlib
```

> **Gotcha:** `setup.sh` seeds the network generator from the clone's directory
> name, and on some machines that seed fails to build a few of the largest
> networks. If it errors on a big benchmark, rename the clone directory (which
> changes the seed) and re-run — an upstream benchmark-repo quirk, not a vibecheck
> issue.

To enable the integration/point-prop tests, point `tests/paths.yaml` at the clone:

```bash
cp tests/paths.yaml.template tests/paths.yaml
# edit `vnncomp_benchmarks:` to the clone root, e.g. ~/repositories/vnncomp2025_benchmarks
```

To run a single instance through the CLI, point `--net` / `--spec` at files in the
clone and select the matching config:

```bash
BENCH=~/repositories/vnncomp2025_benchmarks/benchmarks/acasxu_2023
.venv/bin/python -m vibecheck.main \
    --net  "$BENCH/onnx/ACASXU_run2a_1_1_batch_2000.onnx" \
    --spec "$BENCH/vnnlib/prop_3.vnnlib" \
    --config configs/acasxu_2023.yaml \
    --timeout 120 --results-file /tmp/r.txt
cat /tmp/r.txt   # -> unsat (verified)
```

Each benchmark's `instances.csv` lists its `(onnx, vnnlib, timeout)` triples — pick
any row to reproduce a specific case.
