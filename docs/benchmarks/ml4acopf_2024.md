# ml4acopf_2024

AC optimal-power-flow physics nets: Gemm/ReLU MLP blocks fused with the AC
power-flow equations — bilinear products (V·V, V·V·cos, V·V·sin), self-product
squares (V²), `Sin`/`Cos`, `Sigmoid`, `Floor`. Three model variants per grid:
the FULL physics net (live `Sin`/`Cos`/`Sigmoid`) and two `linear-residual` /
`linear-nonresidual` variants that bake the trig as a ReLU-PWL lookup (so the
graph is bilinear + ReLU, no live trig op).

## Head-to-head vs α,β-CROWN (14_ieee, `--timeout 120`)

**0 soundness conflicts, 0 ABC-only misses on 14_ieee.** Per-prop:

| prop | VC verdict | how | vs ABC |
|---|---|---|---|
| prop1 | `sat` (linear) / `unsat` (full) | nominal-point CE probe (`_nonlinear_nominal_cex_probe`) — the spec is violated at the operating point; full-net center is safe | ABC's clean-input attack does the same |
| prop2 | `unsat` (+3.977) | backward-CROWN root + topo-order intermediate refinement (`_nonlinear_backward_crown_root`) | **beats** ABC's init-CROWN (+3.956) |
| prop3 | `unsat` | linear: ReLU-slope α-CROWN closes lb(Y_5); **FULL net: α-CROWN closes Y_6 at +9.88e-7** (see below) | matches ABC (+8.64e-7) |
| prop4-14 | `unsat` | α-CROWN / backward-CROWN root | parity |

The hard **118_ieee / 300_ieee** props time out for BOTH VC and α,β-CROWN (deep
nested power-flow bilinears); no ABC-only misses there.

## Method

Production routes ACOPF graphs (detected by `Sin`/`Cos`/`Mul`-bilinear) through a
self-contained sound pipeline (`verify_graph._verify_nonlinear_graph` and helpers),
because the 9-phase LP/MILP pipeline and the batched input-split don't support
trig/bilinear end-to-end:

- **Forward zonotope** (`_forward_zonotope_graph`, float64) with the DeepZ
  **affine-band** transformer for `Sigmoid`/`Tanh`/`Sin`/`Cos`/`Floor`
  (`nonlinear_relax.zono_affine_transform`): `y = λx+μ` (gens scaled by λ,
  preserving input correlation) + one fresh δ error generator per element.
  Bounds are symbolic/closed-form (critical-point enumeration), never sampled.
- **α-CROWN** (`_nonlinear_alpha_opt`): Adam-optimizes a per-neuron relaxation slope
  α∈[0,1] for every Sqr/Sigmoid/Tanh/Sin/Cos and the ReLU lower-slopes, through
  the differentiable forward zonotope, against the worst-disjunct spec margin.
- **Backward-CROWN root** (`_nonlinear_backward_crown_root`) with topo-order
  per-node intermediate-bound refinement — gated on `not _has_trig` and a
  net-size cap (the small linear nets only; closes 14_ieee prop2 beating ABC).
- **SAT**: nominal-point CE probe + `_pgd_attack_general` (ORT-validated, never
  a false sat) + the nonlinear-split BaB (`verify_graph._verify_nonlinear_graph`).
- **BaB**: input-split (few varying dims) / nonlinear-pre-activation split
  (118/300) when the root doesn't close.

Two soundness fixes were load-bearing: the **gather/slice backward adjoint must
be scatter-ADD** (`index_add_`, not `index_copy_` — fan-out gathers in the
power-flow equations otherwise silently drop contributions); and the global
**column-ID merge invariant** (concat must not alias independent noise symbols).

## Full-net prop3: the `rel_pad` fix (within-tol-sat → unsat)

Full-net prop3's binding atom is `Y_6 >= 0.300001` with `Y_6 = 0.6·sigmoid(z6) −
0.3` and z6 a **saturated** sigmoid (pre-act ≈ [8.18, 19.31]). VC's forward
α-CROWN plateaued at margin **−1.56e-7** (within-tol `sat`) no matter how α
tuned the slope. Root cause: `zono_affine_transform`'s `rel_pad` — a sound
inflation covering the FLOAT ROUNDING of `λ·center+μ`, hardcoded `1e-6` (sized
for float32, eps≈1.2e-7). The ACOPF path runs **float64**, where 1e-6 is ~10
orders too large: a CONSTANT symmetric band bloat (~2e-6 at value≈1) that **no
slope/α removes** (α tunes λ; the pad is added on top), capping the margin.

**Fix** (`nonlinear_relax.py`, dtype-aware default): `rel_pad = 1e-6 if float32
else 1e-12` (1e-12 is ~4 orders above float64 eps, soundly covers float64
rounding without swamping sub-1e-6 margins). The existing α-CROWN then closes
prop3 at **+9.88e-7**, matching α,β-CROWN's +8.64e-7 (sound: ORT true margin
≈+1.2e-6). Dead ends (all reverted): backward-CROWN root (−0.0265), an α-tangent
backward sigmoid (−4.70e-7, worse than forward-α), and the "δ-decorrelation"
thesis (falsified — a no-δ backward gives the same −4.70e-7).

## Reproduction

```bash
B=.../vnncomp2025_benchmarks/benchmarks/ml4acopf_2024
.venv/bin/python -m vibecheck.main \
  --net $B/onnx/14_ieee_ml4acopf.onnx \
  --spec $B/vnnlib/14_ieee_prop3.vnnlib \
  --results-file /tmp/p3.txt --timeout 120 --verbose
# -> unsat ; log: [acopf_alpha] verified at iter 6, margin=+9.88e-7
```

## Soundness gate

Sweep with `pgd_restarts:0` + `--disable-sat-finding` → every decided case is
`unsat`/`timeout` (no `sat`); any `sat` re-validated by point-prop through
onnxruntime. The dtype-aware pad TIGHTENS bounds (sound by construction:
1e-12 ≫ float64 rounding ~1e-16); canary — linear prop1 (a true `sat`) stays
`sat` after the fix (no false-verify). Full unit suite: 1048 passed / 0 failed.

## Integration tests

`tests/integration/test_ml4acopf_2024.py` (4 cases, default settings — the
acopf path auto-triggers on the graph, no config): 14_ieee prop1 `sat`, prop2
`unsat` (backward-CROWN +3.977), prop3 `unsat` (ReLU-α, linear-residual), and
**FULL-net prop3 `unsat`** (the dtype-aware band-pad fix → α-CROWN closes,
matches ABC). Unit: `tests/test_nonlinear_relax.py::test_zono_affine_transform_pad_is_dtype_aware`
pins the pad fix + soundness; `tests/test_gather_backward_adjoint.py` pins the
scatter-ADD gather adjoint.

## Known unsolved

118_ieee / 300_ieee hard props (prop2/3/4 etc.) — both VC and α,β-CROWN time
out (deeply nested power-flow bilinears; the per-node refinement is grossly
insufficient at 1696/3804 output dims). No ABC-only misses.
