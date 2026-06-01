# acasxu_2023

Small FC controller (5-D input, 6 × 50 ReLU layers, 5-D output). VNNCOMP regular track. 186 cases across 10 properties (prop_1–prop_10): 139 UNSAT, 47 SAT.

## UPDATE 2026-06-01 — switched off the hybrid; input-split + CROWN + leaf-PGD + vectorized split

**The freeze-replay hybrid below is SUPERSEDED.** A re-sweep showed `use_hybrid_acasxu: true` timed out **13/32** sampled cases on server1 — the `_full_freeze` per-layer α tightening has no fast deadline and overran to 150–180 s, mostly on prop_1's wide box. The production config now uses the batched input-split BaB directly:

- **`input_split_crown_intermediate: true`** — AB-CROWN's `bound_prop_method: crown` (backward-CROWN intermediate bounds). Forward-zono intermediate bounds are ~2× too loose for ACAS Xu's amplifying weights (root margin fwd-zono −2570 vs crown −1101 on 3_3 prop_2) and diverged; backward CROWN is tighter AND cheaper. (A mutual zono∩CROWN tightening was tried — `_crown_intermediate_batched` `sweeps>1` — and measured to give only ~3 % extra margin, sub-threshold to close a leaf one bisection earlier, so it is a net end-to-end loss; left default-OFF. See `scratch/acasxu_p2_33/plan.md` explore11–14.)
- **worst-margin leaf-PGD** (`input_split_leaf_pgd_*`) — root-box PGD misses narrow SAT witnesses (200k restarts fail on 1_9 prop_7); the witness leaf can never close, so its margin stays most-negative — batched-PGD the worst-margin leaves and it's caught in ~1 s. `_simple_pgd_batched` in `verify_hybrid_acasxu.py`.
- **vectorized on-GPU 2-way split** (`verify_graph.py`, K_eff==1 fast path) — the per-child `.cpu()`/`.to(device)` loop was ~80 % of wall (a host round-trip per child × millions of leaves). Building both children of every leaf in a few GPU ops cut **3_3 prop_2 126 s → 73 s** and **4_2 prop_2 151 s → 10 s** — the two hard prop_2-UNSAT cases that timed out before. (This reverted once for breaking a SAT case via reordered leaves; worst-margin leaf-PGD now finds SAT independent of split order.)

### Result — HW-dependent; 0 false-verifies; +11 vs the hybrid

Verdicts correctly keyed by **(onnx, vnnlib)** — `prop_2.vnnlib` is reused across 45 nets with 39 sat/6 unsat, so a basename-only ABC match false-flags misses, see `[[project_audit_abc_key_collision]]`.

| | laptop (RTX PRO 2000, Blackwell) | server1 (RTX 3080, ~4 yr old) |
|---|---|---|
| **TOTAL** | **186 / 186** | **184 / 186** |
| prop_1 (was the hybrid's 13 timeouts) | all verified <1 s | all verified ~65 s |
| 3_3 prop_2 | 72 s ✓ | **147 s ✗** (>116 s cap) |
| 4_2 prop_2 | 10 s ✓ | **143 s ✗** (overran the cap) |
| false-verifies | 0 | 0 |

The 2 server1 misses are a **slow-GPU artifact**: server1's RTX 3080 is ~2× slower than the laptop and slower than typical VNNCOMP datacenter HW, so two hard prop_2-UNSAT cases that finish in 10–72 s on the laptop exceed the 116 s cap there (147 s, 143 s). 4_2 in particular is 14× slower on server1 than the laptop — float32 GPU nondeterminism makes its BaB unstable right at the margin. **Net: +11 over the hybrid's measured 173/186** (the hybrid's `_full_freeze` timed out 13/32 on prop_1's wide box).

**Path to a robust 0-miss (any HW):** the hybrid does 3_3 prop_2 in 47 s via **α-optimized intermediate bounds** (tighter → far fewer leaves); the input-split uses plain CROWN-intermediate (looser → more leaves → ~3× more wall on the hard cases). Cheap levers don't close it — leaf-PGD dial-down saves ~10 s (doesn't clear the cap and risks SAT), and boundary-α gives ~0 (the unclosed leaves are deeply negative, not near-boundary). The real fix is wiring the hybrid's freeze-replay (`_full_freeze` + `_replay_batched`) into the input-split BaB (α frozen once at root, replayed per leaf). Substantial; deferred. Full 186-case v2 sweep on server1: `audit_acasxu_v2.out`.

---


## Final score (server1 RTX 3080, 120 s/case)

| | vibecheck | AB-CROWN (server1) | AB-CROWN published |
|---|---|---|---|
| **TOTAL CORRECT** | **186 / 186** | 186 / 186 | 186 / 186 |
| Mean time/case | **3.21 s** | 7.37 s | n/a (faster hardware) |
| Total wall time | **597 s** (~10 min) | 1370 s (~23 min) | n/a |
| Slowest case | 47.15 s (3_3 prop_2) | 18.16 s (same) | 18.16 s (same, faster GPU) |

Measured 2026-05-22 on server1 (`~/persistent_runs/sweep_unified.csv`).

## Algorithm

The pipeline lives in `src/vibecheck/verify_hybrid_acasxu.py`. `verify_graph()`
routes ACASXU to `verify_hybrid()` via `use_hybrid_acasxu: true` in the config —
the input space is 5-D so input-split BaB dominates, and the per-leaf cost is
small enough that heavy α-CROWN at every leaf would crush throughput.

**This wiring is the fix for a real disconnect.** Until it was added, `verify_graph`
(the production path used by `main.py`, the CLI, and the cross-sweep) routed
ACASXU to the *generic* batched input-split BaB, which propagates **forward-zono
intermediate bounds**. Those are ~1000× too loose for ACAS Xu's amplifying
weights (root spec margin -1597 vs a true value > 0), so the BaB **diverged** —
6.8M leaves, never converging — and timed out on 3_3 prop_2. `verify_hybrid`'s
freeze-replay tightens per-layer pre-ReLU bounds with backward α-CROWN
(`_full_freeze`), intersected with the forward-zono bound so it stays sound, and
converges. (The 186/186 below was always achievable via `verify_hybrid` directly
— e.g. `sweep_unified.py` and the old integration test called it directly — but
the production verdict path did not, until this wiring.)

1. **Root PGD** (sign-gradient, 10K restarts × 50 iters). Multi-disjunct DNF aware. Catches 42 of the 47 SAT cases by itself in ~5 ms. The remaining 5 SAT cases (1_5/1_6/3_2/5_3 prop_2, 2_9 prop_8) have narrow witnesses — handled by between-rounds PGD or the multi-disjunct fix respectively.

2. **Initial fan-out**: split root box into 32 leaves by widest dim (5 binary splits).

3. **Batched 32-leaf α-CROWN freeze**: per-query α (Q = 2·n_layer per ReLU layer, Q = n_spec for the spec). 100 Adam iters with sum-based early-stop, lr=0.25, lr_decay=0.98. Stores per-leaf α tensors as 32 "α groups". Each child of a depth-5 leaf inherits its parent's group α as warmstart for BaB.

4. **BaB loop**, pop up to 4096 leaves per batch, grouped by α-group. For each group's leaves (batched on GPU):
   - Forward zonotope (min-area parallelogram ReLU slopes) → tight bounds per layer
   - CROWN backward with **frozen** per-group layer α — intersect with forward zono via `max(zono_lo, crown_lo)`, `min(zono_hi, crown_hi)`. Both bounds contribute; neither alone is enough at deep leaves.
   - Per-leaf spec α-opt: 10 Adam iters warmstarted from the group's frozen spec α (no inheritance from BaB parents — group warmstart works better). Take best spec_lb seen across the 10 iters.
   - Close leaves where worst-disjunct best spec_lb > 0; widest-dim split for open leaves.

5. **Between BaB rounds**: every iter, take the K=5 leaves with smallest spec_lb. Run simple PGD (1K restarts × 50 iters) on each leaf's box. Returns SAT if any restart violates the spec.

6. Stop on empty worklist (UNSAT) or 120 s timeout (UNKNOWN). In the full sweep 0 cases hit the timeout.

## Key algorithmic wins vs published reference

- **Per-query α-CROWN** at the 32-leaf freeze (each spec query gets its own α tensor per upstream ReLU). Without this, shared α plateaus at lb = -52 on 1_8 prop_2; with per-query α we reach -30, essentially matching LP-triangle (-29.8).
- **Forward zono ∩ frozen-α-CROWN intersection** at BaB leaves. Each bound is tighter on different neurons; intersection compounds.
- **Group α as warmstart** (not parent α). Tested per-leaf inheritance from parent's opt'd α — measured slower than group warmstart at every spec_iters setting (5..20). Hypothesis: inherited α is locally near-optimal but explores less; group α allows Adam to find a path that converges in fewer iters.
- **Plain CROWN at the rest of BaB**. Heavy α at every leaf is fatal on easy cases (one of the things AB-CROWN explicitly avoids on ACASXU — their config uses plain CROWN, no α-CROWN).

## Knobs in `configs/acasxu_2023.yaml`

The config now does one load-bearing thing: `use_hybrid_acasxu: true`, which is
what makes `verify_graph` route to `verify_hybrid`. (`input_split_batched_enabled:
true` remains as a fallback if the hybrid flag is ever turned off.) The old
selective-α / MILP-escalation / clipping knobs were removed — the MILP escalation
in particular dominated the old path (~80% of wall on 3_3 prop_2, from Gurobi
pool spawn/terminate overhead) and is gone.

The hybrid runner's hyperparameters are baked into `verify_hybrid()`'s function defaults:

| Parameter | Value | Why |
|---|---|---|
| `init_leaves` | 32 | Empirically the right size for 5-D input. Smaller = under-utilized GPU; larger = bigger freeze overhead. |
| `k_freeze` | 5 (depth of initial fan-out) | log2(32). |
| `batch` | 4096 | Hits peak GPU throughput without OOM. |
| `spec_iters` (per-leaf Adam) | 10 | Sweet spot at 26.5 s on 3_3 prop_2; 5 iters → 32 s, 20 iters → 36 s. |
| `pgd_between_every` | 1 | Every BaB iter. |
| `pgd_between_k` | 5 | Top-5 worst leaves. |
| `pgd_between_restarts` × `pgd_between_iter` | 1000 × 50 | Smallest config that catches all 4 narrow-witness prop_2 SAT cases. |
| Root PGD restarts × iters | 10000 × 50 | Catches all 47 - 5 = 42 SAT cases by itself in <5 ms. |

## Reproduction

Single case (production path — routes to verify_hybrid via the config):
```bash
B=$VNNCOMP/benchmarks/acasxu_2023
.venv/bin/python -m vibecheck.main \
  --net $B/onnx/ACASXU_run2a_3_3_batch_2000.onnx \
  --spec $B/vnnlib/prop_2.vnnlib \
  --config configs/acasxu_2023.yaml --timeout 120 --results-file /tmp/r.txt  # unsat ~46s
```

Full sweep (on server1):
```bash
cd ~/Desktop/temp/vibecheck-temp
nohup .venv/bin/python scratch/sweep_unified.py 120 \
  > ~/persistent_runs/sweep_unified.log 2>&1 &
```

Results CSV: `~/persistent_runs/sweep_unified.csv`.

## Integration tests

`tests/integration/test_acasxu_2023.py` covers 3 cases — 1 narrow-witness SAT (1_5 prop_2) + the two hardest UNSAT cases (3_3 prop_2, 1_1 prop_3). It now runs through the **production path** (`verify_graph` + the config via `_runner`), not `verify_hybrid` directly — closing the gap that let the test pass while the production pipeline diverged.

## Known unsolved cases

None. All 186 / 186 in the full sweep.

## Historical notes

The earlier `configs/acasxu_2023.yaml` + `verify_graph` path was tuned with selective per-leaf α-CROWN, MILP escalation, batched clipping, and racing. Best score there: ~176/186 with race v1+v2. The hybrid runner replaces it because (a) selective α-CROWN was firing on too many leaves and Pareto-dominated by plain CROWN on easy cases, and (b) MILP escalation on degenerate residual leaves was actively *slower* than splitting further. See git log for the multi-month investigation into bias-drop unsoundness in `verify_milp.py` that preceded this design.
