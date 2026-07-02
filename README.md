<p align="center">
  <img src="https://raw.githubusercontent.com/stanleybak/vibecheck-nn/main/vibecheck.svg" alt="vibecheck logo" width="640">
</p>

The **vibecheck** formal verification tool is a high-performance, vibe-coded decision procedure for neural networks. Given an ONNX neural network and a VNNLIB specification, vibecheck tries to prove the property or find a counterexample. It solves the same open-loop neural network verification problem as established verifiers like [α,β-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN), [Marabou](https://github.com/NeuralNetworkVerification/Marabou), and [NNV](https://github.com/verivital/nnv), hopefully faster and on larger networks. Does your neural network pass the vibecheck?

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

vibecheck implements the VNN-LIB standard's solver CLI (Chapter 5):

```bash
vibecheck verify <query.vnnlib> --network NAME=<model.onnx> [--timeout SECONDS] \
                 [--serialise-assignments DIR]
vibecheck supports <capability>        # e.g. --onnx-operators
vibecheck --name | --version
```

`verify` prints the verdict (`sat`/`unsat`/`unknown`/`timed-out`) as the first
stdout line, followed only by the satisfying assignment for `sat` (progress goes
to stderr); `--serialise-assignments DIR` writes the assignment as ONNX
TensorProtos instead. In `supports` output, a `*` after an identifier marks
partial support, with a short note on the same line.

Example, on the bundled ACAS-Xu network (`examples/`) with a property that holds.
Everything except the final `unsat` is progress on stderr (trimmed here):

```console
$ vibecheck verify examples/prop_1.vnnlib --network N=examples/ACASXU_run2a_2_2_batch_2000.onnx --timeout 60
Auto-config: acasxu_2023.yaml | rule 5: low input-dim (<=20) FC (input-split) | Low-dimensional ReLU FC (ACAS-Xu family): batched input-split BaB, hybrid ACAS-Xu path off.
Loading network: examples/ACASXU_run2a_2_2_batch_2000.onnx
  22 ops, 6 ReLU layers, 0 fork points, input shape: (1, 1, 1, 5)
Loading spec: examples/prop_1.vnnlib
  1 constraint(s), 1 disjunct(s)
Running graph verification (device=gpu, impl=optimized, profile=auto:acasxu_2023.yaml(rule 5), timeout=60.0s)...
[pgd] no CE: restarts=100 iters=100/100 gap(best_margin)=+4.012e+00 elapsed=0.29s
[branch] iter=0 split X_1 (width=1.0000) leaves=1
...
[branch] iter=7 split X_2 (width=0.1250) leaves=17

Result: verified
  Time: 2.24s
unsat
```

And a violated property — stdout is the verdict plus the satisfying assignment:

```console
$ vibecheck verify examples/prop_2.vnnlib --network N=examples/ACASXU_run2a_2_2_batch_2000.onnx --timeout 60  2>/dev/null
sat
X float32 [1,1,1,5]
0.6208617091178894
-0.01862180233001709
-0.024315983057022095
0.47021740674972534
-0.46439468860626221
Y float32 [1,5]
0.023801768198609352
-0.021306384354829788
0.023232858628034592
-0.016359567642211914
0.023021360859274864
```

The `Auto-config:` line shows config selection: with no `--config`, vibecheck
picks a bundled per-benchmark config (`configs/*.yaml`) from the structure of
the network and spec (input dim, conv/transformer/nonlinear ops, network-pair
kind) and logs which rule fired. Override it with `--config configs/<name>.yaml`,
or override any single setting with `--set KEY=VALUE` (repeatable; wins over the
config).

The legacy flat CLI (`vibecheck --net model.onnx --spec property.vnnlib
--results-file out.txt`, the form the VNNCOMP harness drives) is unchanged; see
`vibecheck --help`.

## Tests

```bash
# Unit tests: no external data, ~1-2 min (drop --cov for a faster run)
.venv/bin/python -m pytest tests/ -k "not vnncomp" -m "not integration" \
    --cov=src/vibecheck --cov-report=term

# Per-benchmark verdict regressions (need a local benchmark clone; see below)
.venv/bin/python -m pytest tests/integration -m integration
```

The unit tests build synthetic ONNX/VNNLIB inline and need no external data, so
they run on a fresh clone. Only the **integration** tests read benchmark paths
from `tests/paths.yaml` (gitignored).

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

To enable the integration tests, point `tests/paths.yaml` at the clone:

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

## Contributors

* [Stanley Bak](https://stanleybak.com) (lead)
* Doug Wehbe (testing)
