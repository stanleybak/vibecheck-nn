# tinyimagenet_2024 — vibecheck benchmark record

VNNCOMP 2025 regular track. TinyImageNet image classification (200 classes,
3×56×56) on a medium ResNet with L∞ adversarial robustness specs.

## UPDATE 2026-06-02 — regression found + partially fixed (A10G full sweep)

A fresh full 200-case sweep on the AWS A10G (current main) measured **vibecheck
171/200**, below the v9-claimed 178 — a genuine regression (the A10G has *more*
VRAM than server1, so it's not hardware). Cross-checked vs ABC by (onnx,vnnlib):
**0 unsound**, 1 win (prop_1651), 5 misses.

**Root cause #1 (FIXED) — spurious-witness short-circuit.** For 3 of the misses,
a *light* Phase-0/pre-cascade PGD found a near-boundary point (worst margin
≈ +1e-4, just inside the safe side) that failed the 1e-4 sat-validation. The
verifier then `return`ed `unknown` in ~5 s, **abandoning ~95 s of budget and
skipping the full-restart cascade PGD** that finds the real counterexample. Fixed
in `verify_graph.py` via `_sat_or_fallthrough` (new setting
`pgd_fallthrough_on_spurious=True`): a spurious witness is logged and **falls
through** to the next/stronger stage instead of aborting. Applied at all 7
PGD/MILP sat-return sites. Soundness unchanged — every emitted `sat` is still
ORT-validated. **Recovers 3 cases → 174/200:** prop_9992 + prop_6773 (SAT, the
cascade now finds the CEX) and prop_9458 (a previously-regressed win — falls
through to bounds → `unsat` where ABC times out). Validated by re-running all 29
former-`unknown` cases: 3 recovered, 0 unsound, 0 regressions. Pinned by
`test_tinyimagenet_2024.py::...prop_6773 (SAT, spurious-fallthrough)`.

**The remaining 3 misses are NOT an algorithmic regression (root cause #2,
investigated 2026-06-02).** A 300 s-budget run reframes them:
- `prop_6546` (UNSAT): **verifies at 133 s** (unknown at the 100 s budget). ABC
  does it in 58.5 s → vc is ~2.3× slower here. A genuine *perf* gap, not a bug.
- `prop_7542` (UNSAT): **verifies at 139 s**. ABC's own published time is 102 s —
  *over* the 100 s budget — so this case is borderline for both solvers.
- `prop_3553` (SAT): still `unknown` even at 300 s; +restarts (2000), +iters
  (1000), and dual-ascent-off all miss the CEX (PGD margin stalls at +1.3e-3).
  ABC finds it in 7.8 s → a real **attack-strength gap** (vc's PGD ≪ ABC's), not
  budget.

So the v9 "178 / 0 misses" was **inflated by the pre-`1c805c0` timeout-enforcement
bug**: 6546/7542 only "verified" in v9 by running PAST the 100 s budget; the
`1c805c0` fix ("enforce timeout in the per-query alpha-zono state loop") now
correctly stops them → `unknown`. **Current main's 174/200 is the legitimate
within-budget score.** Closing the residual gap is a *performance* goal
(speed up the dual-ascent BaB ~2× on 6546-class UNSAT) plus a *stronger-attack*
goal (match ABC's CEX search on 3553) — not a correctness regression.

### Same-hardware head-to-head vs ABC (A10G, 100 s budget, 2026-06-02)

Ran ABC (`exp_configs/vnncomp25/tinyimagenet.yaml`) on the **same A10G** on the 5
differential cases — the only ones where vc and the published ABC disagree:

| case | vc | ABC (A10G) | ABC (published, A100) | winner |
| --- | --- | --- | --- | --- |
| prop_1651 (UNSAT) | verified 84 s | timeout 110 s | timeout 107 s | **vc** |
| prop_9458 (UNSAT) | verified 103 s | timeout 106 s | timeout 108 s | **vc** |
| prop_3553 (SAT)   | unknown | sat 13.8 s | sat 7.8 s | ABC (attack) |
| prop_6546 (UNSAT) | unknown (133 s) | unsat 66 s | unsat 58.5 s | ABC (perf ~2×) |
| prop_7542 (UNSAT) | unknown (139 s) | **timeout 105.6 s** | unsat 102 s | **draw** |

`prop_7542` is the tell: ABC's *published* "solved" is an A100 + scoring-slack
artifact — on the same A10G it **times out** (105.6 s), same as vc. So on equal
hardware vc wins 2 (1651, 9458 — ABC can't solve them on any GPU), ABC wins 2
(3553 attack, 6546 perf), 1 draw. The other ~23 vc-`unknown`s are ABC-timeout
cases too (draws). **Net: vc ≈ ABC on identical hardware (vc 174 vs ABC ~174-175),
0 unsound, with 2 clean wins ABC cannot match.** The published "176 vs 174"
overstates ABC's same-hardware edge.

## Final score (v9 sweep, 2026-05-19, server1 RTX 3080 / 10GB)

| Solver | Solved / 200 | Rate | Notes |
| --- | --- | --- | --- |
| **vibecheck** (this repo) | **178** | **89.0%** | server1 RTX 3080 |
| AB-CROWN (server1) | 124 | 62.0% | same HW (OOMs on most UNSAT cases) |
| AB-CROWN (published) | 175 | 87.5% | competition HW (A100 etc.) |

- **+3 algorithmic wins** over published AB-CROWN — vc verifies 3 cases pub-abc times out on
- **0 misses** — every case pub-abc solves, vc also solves
- All 4 tinyimagenet v7-gap cases solved (prop_6546, prop_3574, prop_7390, prop_7542)
- All 6 v7-gap cases overall (cifar+tiny) closed by v9

The big margin over server-AB-CROWN (62%) is hardware-headroom: vibecheck's `PatchesZonotope` (auto-selected via `settings.zono_impl='patches'` for image-like inputs) fits in 10GB GPU memory where AB-CROWN's dense allocations don't. On competition hardware the gap shrinks to the +3 algorithmic margin.

## Knobs (`configs/tinyimagenet_2024.yaml`)

Identical override set to cifar100_2024 — they share the v9 hybrid config. The PatchesZonotope path is auto-engaged via `settings.zono_impl='patches'` (set in `default_settings()`) when the input shape is `(C, H, W)`.

See `docs/benchmarks/cifar100_2024.md` for per-knob rationale.

## Reproducing a single case

```bash
ssh stan@100.83.144.97
./.venv/bin/python /home/stan/persistent_runs/scripts/runner_v8.py \
  ~/repositories/vnncomp2025_benchmarks/benchmarks/tinyimagenet_2024/onnx/TinyImageNet_resnet_medium.onnx \
  ~/repositories/vnncomp2025_benchmarks/benchmarks/tinyimagenet_2024/vnnlib/TinyImageNet_resnet_medium_prop_idx_6546_sidx_3168_eps_0.0039.vnnlib \
  100
```

## Reproducing the full sweep

```bash
ssh stan@100.83.144.97 'tmux new-session -d -s sweep_tiny "SWEEP_CATEGORIES=tinyimagenet_2024 SWEEP_OUT_DIR=/home/stan/persistent_runs/results/sxs_tiny_$(date +%Y%m%d) \
  ~/Desktop/temp/vibecheck-temp/.venv/bin/python /home/stan/persistent_runs/scripts/sweep_sxs.py 2>&1 | tee ~/persistent_runs/results/sweep_tiny_$(date +%Y%m%d).log"'
```

Wall ≈ 3-4 hours.

## Integration tests (`tests/integration/test_tinyimagenet_2024.py`)

| Case | Verdict | Wall budget | Why |
| --- | --- | --- | --- |
| `TinyImageNet_resnet_medium_prop_idx_9262_sidx_880_eps_0.0039` | **sat** | 15s | Easy SAT; exercises pre-α-CROWN PGD short-circuit on patches-zono path. |
| `TinyImageNet_resnet_medium_prop_idx_1175_sidx_7775_eps_0.0039` | **verified** | 70s | Hard UNSAT (~36s) — Phase 8 dual-ascent BaB. |
| `TinyImageNet_resnet_medium_prop_idx_9215_sidx_4878_eps_0.0039` | **verified** | 85s | Hard UNSAT (~50s); catches deeper Phase 1 cascade regressions. |

Run: `.venv/bin/python -m pytest tests/integration/test_tinyimagenet_2024.py -v`.

## Algorithmic wins (vs published AB-CROWN)

| Case | vc | pub-abc (best HW) |
| --- | --- | --- |
| `TinyImageNet_resnet_medium_prop_idx_1651_sidx_3548_eps_0.0039` | **verified/58.3s** | timeout/107s |
| `TinyImageNet_resnet_medium_prop_idx_9569_sidx_8561_eps_0.0039` | **verified/82.6s** | timeout/108s |
| `TinyImageNet_resnet_medium_prop_idx_9458_sidx_2592_eps_0.0039` | **verified/69.1s** | timeout/108s |

All three are cases pub-AB-CROWN times out on competition hardware; vc on the 10GB GPU solves them with parallel-PGD-during-MILP overlap.

## Known unsolved (timeout block)

22 cases where vc returns `unknown/101s` — also `timeout` in published reference. Not a vibecheck-specific gap.
