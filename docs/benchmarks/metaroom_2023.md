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
even with full 210 s budget. The net is a 6-Conv CNN, **~172 K ReLUs**,
on a **wide input box** (161 of 5376 dims vary, width up to 0.52 — the
camera-pose perturbation; `eps_…436` is unrelated). Confirmed **UNSAT**:
2000-restart × 500-iter PGD (both α, per-restart-disjunct) finds no CEX.

### Dual-ascent experiment (2026-06-08, `bench/metaroom_dual_ascent`)

Routed this case through the GPU dual-ascent Phase-8 BnB instead of the
default conv→`milp_verify` auto-route (`auto_route_milp_for_conv: false`
+ `phase8_use_dual_ascent_gpu: true` + the cifar100 lean-Phase-1 knobs).
Result: **strict improvement but still not solved** — `timeout` (honest)
vs the default `milp_verify` path which on the local 7.5 GB GPU *crashes*
with `GurobiNumericTrouble`. Dual-ascent **closes 13/19 disjuncts**; the
5 hardest (q2/q4/q8/q10/q16) hit `stop=oom` on the frontier (~5–7.6 M
nodes) with LBs stuck at −35 … −76, and the high-bin MILP fallback also
times out there. Raising `phase8_dual_ascent_max_iter` 1→20 and the
line-search `K` 256→512 did not change the frontier blow-up.

**Re-run on a 24 GB A10G (AWS) settles it: relaxation-bound, NOT
memory-bound.** With 3–4× the GPU memory the frontier grew ~5× before
OOM (q8: 67 M nodes / 33.5 M frontier vs 6.9 M on the 7.5 GB card), yet
the verdict is unchanged — **still 13/19, the same 5 disjuncts open**,
and 10× more explored nodes moved the bounds by ≈3 (q2 LB −42→−39, q4
−63→−61). So BnB is at diminishing returns: the α-zonotope relaxation
gap on these 5 disjuncts is too large for branching to close at any GPU
size. Closing this case needs a *tighter relaxation / intermediate
bounds* (the per-layer MILP-tighten path, which here crashes with
`GurobiNumericTrouble` and is slow), not more nodes.

**No production config change.** Dual-ascent does not *solve* the case;
switching metaroom's routing risks the 99 already-solved instances; and
ABC times out too, so there is no scoring upside. Recorded as a
documented dead end — the open lever is bound *tightness*, not compute.
