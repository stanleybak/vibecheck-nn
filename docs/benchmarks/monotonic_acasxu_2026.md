# monotonic_acasxu_2026

Extended-track, **2.0/** (v2 vnnlib). 50 instances, 100 s budget. **Monotonicity** of an ACAS Xu
controller: along a designated input coordinate, an output must move monotonically; a
counterexample is a pair of inputs that violates it. vibecheck **matches α,β-CROWN: 50/50 sat**
(all real violations), ~3 s each.

## The benchmark

- **Pair format:** the net field is `[('f', net), ('g', net)]` with **f = g (the same ACAS Xu
  net)**. `main._maybe_network_pair` merges it into one ONNX (`network_pair.py`) that evaluates
  the net at two correlated input points: `x_g = clamp(x_f − δ·e_k, LO, HI)` with `δ ≥ 0` on the
  monotone coordinate `k` (the other coordinates equal). The merge is onnxruntime-oracle-gated.
- **Spec:** input box on `x_f` + `x_g[k] ≤ x_f[k]`, output e.g. `Y_f[3] < Y_g[3]` — i.e.
  decreasing input `k` should not increase output 3. A `sat` witness is an `(x_f, x_g)` pair
  where it does (monotonicity broken). The authoritative spec is the **`.gz`** (the loose
  `instance_*.vnnlib` may be a stale sibling; `network_pair` reads `.gz` first).

## ABC reference

α,β-CROWN: **50/50 sat** (`vnncomp2026_results/alpha_beta_crown/2026_monotonic_acasxu_2026`).
These are shallow, real violations — found quickly by CE search; no per-benchmark config needed
(default `input_split_small` profile + Phase-0/leaf PGD).

## Results (vibecheck vs ABC)

| | vibecheck | α,β-CROWN |
|---|---|---|
| sat | **50 / 50** | 50 / 50 |
| wall | ~3 s each | — |

Per-instance: identical (all sat). 0 misses, 0 conflicts. Nothing to improve — the benchmark is
maxed for both tools. Each witness is the merged net's `(x_f, x_g)` input pair; the strict `<`
boundary is handled by the sound non-strict-closure encoding + PGD/ORT confirmation
(`network_pair._parse_output_atoms` + `verify_graph._sat_disposition`).

## Reproduce

```bash
.venv/bin/python -m vibecheck.main \
  --net "[('f','onnx/ACASXU_run2a_2_2_batch_2000.onnx'),('g','onnx/ACASXU_run2a_2_2_batch_2000.onnx')]" \
  --spec vnnlib/instance_0.vnnlib --timeout 100 --results-file out.txt
# (run with cwd = the benchmark 2.0/ dir so the relative onnx path resolves)
```

Integration pin: `tests/integration/test_monotonic_acasxu_2026.py`.

## Key unresolved issues

- None — VC matches ABC at 50/50.
