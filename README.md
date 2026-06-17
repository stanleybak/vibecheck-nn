<p align="center">
  <img src="https://raw.githubusercontent.com/stanleybak/vibecheck-nn/master/vibecheck.png" alt="vibecheck logo" width="640">
</p>

**Vibecheck** is a high-performance vibe-coded neural network verification tool. Given an ONNX neural network and a VNNLIB specification, Vibecheck tries to prove the property or find a counterexample. It solves the same problem as established verifiers like  [α,β-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN) and [Marabou](https://github.com/NeuralNetworkVerification/Marabou), hopefully faster and on larger networks.

The underlying verification algorithms in Vibecheck are complementary, specified in configuration files based on a problem's complexity and timeout. They include Gurobi LP/MILP for tight neuron bounds, zonotope abstract interpretation combined with CROWN / α-CROWN slopes for tight overapproximations, and a high-performance GPU-enabled branch-and-bound search with an optimal-step dual-ascent solver that can (sometimes) explore millions of splits per second.

## Setup

Use the [uv](https://docs.astral.sh/uv/) package manager for setup and configuration:

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

- `--config configs/<benchmark>.yaml`: per-benchmark overrides on top of
  `default_settings()`. When omitted, a profile is auto-detected from the network
  and spec.
- `--results-file PATH`: write a single VNNCOMP-style verdict line (`unsat` =
  verified, `sat` = counterexample, `unknown`, or `timeout`). **This is the
  authoritative verdict**; read it rather than inferring from the exit code.
- `--timeout SECONDS`: tool timeout (default 30).
- `--device {gpu,cpu}`, `--bits {16,32,64}`, `--mode {graph,bnb}`.

Exit codes: `0` = verified, `1` = unknown, `2` = error (a verdict line is still
written to `--results-file` when set).

## Tests

```bash
# Unit tests: no external data, ~1-2 min (drop --cov for a faster run)
.venv/bin/python -m pytest tests/ -k "not vnncomp" -m "not integration" \
    --cov=src/vibecheck --cov-report=term

# Per-benchmark verdict regressions (need a local benchmark clone; see below)
.venv/bin/python -m pytest tests/integration -m integration
```

The unit tests build synthetic ONNX/VNNLIB inline and need no external data, so
they run on a fresh clone. Only the **integration** and **vnncomp
point-propagation** tests read benchmark paths from `tests/paths.yaml`
(gitignored).

Run a single unit test by node id, or a single integration case by its
parametrized `desc` (the `-k` terms are AND-ed):

```bash
.venv/bin/python -m pytest tests/test_zonotope.py::test_propagate_fc -v
.venv/bin/python -m pytest tests/integration/test_acasxu_2023.py -k "1_1 and prop_3" -m integration -v
```

## Running specific VNNCOMP benchmarks

The integration tests (and any direct CLI run on competition models) load
ONNX/VNNLIB from a local clone of the VNNCOMP benchmarks kept elsewhere on your
system. The sets are published per year as `VNN-COMP/vnncomp<year>_benchmarks`,
e.g. [vnncomp2025_benchmarks](https://github.com/VNN-COMP/vnncomp2025_benchmarks)
and [vnncomp2026_benchmarks](https://github.com/VNN-COMP/vnncomp2026_benchmarks).
Clone one and unpack its models:

```bash
git clone https://github.com/VNN-COMP/vnncomp2025_benchmarks.git
cd vnncomp2025_benchmarks
./setup.sh        # downloads + unpacks the per-benchmark onnx/vnnlib
```

> **Gotcha:** `setup.sh` seeds the network generator from the clone's directory
> name, and on some machines that seed fails to build a few of the largest
> networks. If it errors on a big benchmark, rename the clone directory (which
> changes the seed) and re-run. This is an upstream benchmark-repo quirk, not a
> vibecheck issue.

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

Each benchmark's `instances.csv` lists its `(onnx, vnnlib, timeout)` triples; pick
any row to reproduce a specific case.
