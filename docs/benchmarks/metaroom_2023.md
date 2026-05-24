# metaroom_2023 — vibecheck benchmark record

VNNCOMP 2023 regular track. 100 instances across two CNN model
families (`4cnn_*` / `6cnn_*`, 3×32×56 RGB input → 20 classes).
Layer count: 4cnn = 2 Conv + 1 Gemm; 6cnn = 4 Conv + 1 Gemm.

95 UNSAT + 5 SAT + 1 ABC timeout per published results.

## Final score (server1, 2026-05-24, RTX 3080 / 10 GB, 60 s budget)

| Solver | Solved / 100 | Wall (total) | Notes |
| --- | --- | --- | --- |
| **vibecheck** | **99** | **~93 s** | 5 SAT + 94 UNSAT |
| AB-CROWN (published, 2025) | 99 | ~892 s | + 1 ABC timeout — tie |

**1 shared unsolved**: `6cnn_ry_39_6 / spec_idx_119` is `unknown` for
both ABC (217 s wall, exceeded 210 s budget) and vibecheck (4 open
disjuncts at LB ∈ {-21, -33, -44, -21} with 1934 unstable neurons after
α-CROWN; full-MIP fallback at 200 bins plateaus). Neither solver
verifies this case in budget.

Wall: 93 s vs AB-CROWN's 892 s — **~10× faster** at parity.

## Algorithmic adds for this benchmark

- **Same auto-route fix as malbeware**: 44 of 100 instances initially
  crashed with `RuntimeError: tensor a (57344) must match tensor b
  (5376)` in `_evaluate_region`'s `apply_relu` — the historical
  `milp_verify` pipeline (auto-routed for conv nets > 20 input dims)
  has a zonotope shape bug on these `_tz_` models. The graph
  pipeline handles them fine.

## Knobs (`configs/metaroom_2023.yaml`)

- `auto_route_milp_for_conv: false` — bypass the broken historical
  conv routing.
- `input_split_enabled: false` — 5376-input image too high-dim for
  input-split BaB; default to the graph pipeline.

## Reproducing a single case

```bash
.venv/bin/vibecheck \
  --net path/to/metaroom_2023/onnx/4cnn_ry_0_0_no_custom_OP.onnx \
  --spec path/to/metaroom_2023/vnnlib/spec_idx_100_eps_0.00000436.vnnlib \
  --mode graph --timeout 60 --bits 32 \
  --config configs/metaroom_2023.yaml
```

## Full sweep

```bash
.venv/bin/python scratch/metaroom_smoke.py
```

## Integration test coverage

`tests/integration/test_metaroom_2023.py`:
- `4cnn_ry_0_0 / spec_100` (SAT, ~0.6 s) — root-PGD on 2-Conv net.
- `6cnn_tz_35_5 / spec_176` (UNSAT, ~2 s) — `_tz_` model regression
  for the auto-route fix (pre-fix crashed with shape mismatch).
- `6cnn_ry_0_0 / spec_140` (UNSAT, ~5 s) — `_ry_` 4-Conv model.

## Known unsolved cases

1 case: `6cnn_ry_39_6 / spec_idx_119_eps_0.00000436` — ABC also
times out (217 s wall vs 210 s budget). Algorithmic plateau, not
budget on our side: 4 open disjuncts at LB ∈ {-21, -33, -44, -21}
even with full 210 s budget.
