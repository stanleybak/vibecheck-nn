# test

VNNCOMP **test** category — the toolchain smoke check, not a scored benchmark. It
exists to confirm the install + `prepare_instance.sh`/`run_instance.sh` pipeline
produces correct verdicts on a trivial set before the real benchmarks run.

## Instances (v1 / 1.0)

Five instances (`benchmarks/test/1.0/instances.csv`, 60 s timeout each):

| onnx | spec | net | expected | result |
| --- | --- | --- | --- | --- |
| `test_nano` | `test_nano.vnnlib` | `Relu(0.5·x)`, 1-D | unsat | **unsat** (1.6 s) |
| `test_tiny` | `test_tiny.vnnlib` | tiny 1-D ReLU | unsat | **unsat** (1.4 s) |
| `test_small` | `test_small.vnnlib` | tiny 1-D ReLU | unsat | **unsat** (1.3 s) |
| `test_sat` | `test_prop.vnnlib` | ACAS Xu 1-7, prop 3 | sat | **sat** (1.6 s) |
| `test_unsat` | `test_prop.vnnlib` | ACAS Xu 1-6, prop 3 | unsat | **unsat** (2.6 s) |

Measured 2026-06-15 on branch `vnncomp2026_work`, `configs/test.yaml`, default
device (RTX 3080). `test_nano` is `unsat` because `relu(0.5·x) ≥ 0` can never meet
the spec's `Y ≤ −1`; `test_tiny`/`test_small` ask for `Y ≥ 100`, unreachable for
their tiny weights. `test_sat`/`test_unsat` are ACAS Xu property 3 ("COC is
minimal") on nets 1-7 and 1-6 respectively.

## Config

`configs/test.yaml` carries **no tuning** — the default profile already returns
all five correct verdicts. The config exists so `run_instance.sh` selects an
explicit, stable profile for the `test` category, and pins the single behavior the
set relies on:

- `disable_sat_finding: false` — Phase-0 PGD finds the `test_sat` (ACAS Xu 1-7)
  counterexample. Everything else is `default_settings()`.

## Reproduction

```bash
B=~/repositories/vnncomp2026_benchmarks/benchmarks/test/1.0

# single case
.venv/bin/python -m vibecheck.main \
    --net "$B/onnx/test_sat.onnx" --spec "$B/vnnlib/test_prop.vnnlib" \
    --config configs/test.yaml --timeout 30 --results-file /tmp/r.txt
cat /tmp/r.txt   # -> sat

# all five
while IFS=, read -r onnx vnnlib to; do
  .venv/bin/python -m vibecheck.main --net "$B/$onnx" --spec "$B/$vnnlib" \
    --config configs/test.yaml --timeout 30 --results-file /tmp/r.txt
  printf "%-16s -> %s\n" "$(basename "$onnx")" "$(head -1 /tmp/r.txt)"
done < "$B/instances.csv"
```

## Known unresolved — v2 (2.0) specs not yet parsed

The `benchmarks/test/2.0` specs use the new VNNLIB **v2** syntax
(`(vnnlib-version <2.0>)`, `declare-network`, `declare-input`, `X[0]`/`Y[0]`
indexing). `vnnlib_loader.py` is regex-based on the v1 `X_<n>`/`Y_<n>` form and
does not recognize v2 — a v2 spec raises `ValueError: No input bounds found in
VNNLIB` (`vnnlib_loader.py:65`) and `run_instance.sh` reports `error`. Use the 1.0
instances until the loader learns v2 (the first real VNNCOMP-2026 compatibility
task — wire vibecheck to the common parser at
`vnncomp2026_benchmarks/scripts/vnnlib_parser.py`, or extend the loader; the two
versions were cross-checked to agree, and v2 is the version to prefer when both
exist).
