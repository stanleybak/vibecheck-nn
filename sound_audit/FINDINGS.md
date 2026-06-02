# Sigmoid/tanh soundness audit (2026-05-30)

Triggered by: dist_shift's sigmoid bounds falsely verifying ABC-confirmed-SAT
cases (self-contradiction: vibecheck PGD finds a CEX its own bounds certify away).

## Blast radius

| benchmark | track | sigmoid/tanh nets | SAT cases on those nets | impact |
| --- | --- | --- | --- | --- |
| **dist_shift_2023** | regular | mnist_concat, mnist_generator | 7 (mnist_concat) | **UNSOUND** — ≥3/7 SAT cases falsely `verified` (PGD off). Since merge `6789823`. |
| cgan_2023 | regular | 2 nets (Sigmoid+Tanh) | **0** | no wrong verdict possible (all-UNSAT); sound primitives. Latent only. |
| nn4sys | regular | mscn_128d/2048d(+dual) | **0** | same — all-UNSAT instances; no scoring impact. Latent only. |
| collins_aerospace | extended | 1 | not probed | out of scope (extended track) |
| ml4acopf_2024 | extended | 3 | not probed | out of scope (extended track) |

## Key result: the shared relaxation PRIMITIVES are SOUND

Sample-tested `_sigmoid_tanh_chord_parallelogram` and `_sigmoid_tanh_linear_bounds`
(verify_zono_bnb.py) over 480 intervals (centers −8..8, widths 0.01..16), 4000
samples each:

```
sigmoid: worst parallelogram violation = 2.1e-07   linear = 1.6e-06
tanh:    worst parallelogram violation = 1.2e-07   linear = 1.8e-06
```

All within numerical tolerance ⇒ the relaxation math is sound. **The dist_shift
unsoundness is NOT the relaxation** — it's in how dist_shift's pipeline *wires*
the (sound) sigmoid primitive into the zono/CROWN/spec propagation.

## Pinpoint so far (ablation on the 2 confirmed-unsound SAT cases, sat-finding off)

- `zono_lift_enabled=False` → still `verified` (NOT Phase 2.5 zono-lift)
- `phase8_use_dual_ascent_gpu=False` → still `verified` (NOT dual-ascent)

⇒ unsoundness is in the **core base-bounds path** (forward-zono/CROWN propagation
of the sigmoid layer, or the spec backward, or the phase-8 sigmoid encoding), not
the tightening overlays. Next step to localize: forward an index1285 CEX
layer-by-layer and find the first layer whose certified bound it violates.

## Bottom line

- **Scoring impact: dist_shift only** (and only with sat-finding disabled; in
  production PGD finds the SAT cases first → correct `sat`). The c521691 crash
  currently masks it entirely (fails safe).
- **cgan/nn4sys: no wrong verdicts** (all-UNSAT sigmoid instances) — but their
  sigmoid bounds are latently unprobeable (no SAT case to self-contradict).
- The fix is localized to dist_shift's sigmoid-layer wiring, NOT a broad
  relaxation rewrite.
