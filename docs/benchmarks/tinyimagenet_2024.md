# tinyimagenet_2024 — vibecheck benchmark record

VNNCOMP 2025 regular track. TinyImageNet image classification (200 classes,
3×56×56) on a medium ResNet with L∞ adversarial robustness specs.

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
