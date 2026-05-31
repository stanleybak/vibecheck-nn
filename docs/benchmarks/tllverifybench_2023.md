# tllverifybench_2023

VNNCOMP regular track. 32 instances of **Two-Level-Lattice (TLL)** networks:
2-D input → deep `MatMul+Add` chains → 8 ReLU layers → 1-D output; spec a single
halfspace `Y_0 <= c`. The weights are **ternary** ({−1,0,1}) and **sparse**
(2-8 nonzeros/row) — each neuron is a min/max of a few earlier neurons (the
lattice). 17 SAT, 15 UNSAT.

## Score

- **vibecheck: 32/32** (sat 17/17, unsat 15/15) — full parity with AB-CROWN.
- Slowest case ~29 s (N=M=56), most < 5 s; competition timeout is 600 s.
- Progression: 24/32 default → 29/32 (enable batched input-split) → **32/32**
  (naive branching).
- (vc sweep: `scratch/sweep_tll.sh configs/tllverifybench_2023.yaml`, 2026-05-31;
  ABC ref `~/repositories/vnncomp2025_results/alpha_beta_crown/2025_tllverifybench_2023/results.csv`.)

## How vibecheck solves it — and the diagnosis that found the fix

AB-CROWN uses **batched input-split BaB, NAIVE branching, no Gurobi/β/dual-
ascent** (`exp_configs/vnncomp23/tllVerifyBench.yaml`: `branching.method: naive`,
`bound_prop_method: forward+backward`, `batch_size: 1500`, `merge_linear`).

vibecheck started at 24/32 (the pure-ReLU net routed to the single-leaf
`fast_leaf` path → timeout). Two findings:

1. **Enable the batched input-split** (`input_split_batched_enabled`). It existed
   ("AB-CROWN-style, ~100× throughput") but was gated to bilinear nets. 24→29/32.

2. **Use naive branching** (`input_split_batched_branch_sb: false`). The 3 hard
   UNSAT cases stayed `unknown`. A methodical comparison against ABC ruled out
   every bound hypothesis:
   - **Per-layer unstable counts (root box) MATCH ABC** — 2_1: vibecheck 2022 vs
     ABC 2012 total unstable (L0 exact 788, L1 562, L2+ ~100%, magnitudes
     ±1025–1559). We are **not looser** at the root.
   - **LP-exact L1** (cheap — the weights are sparse): 562→**546** unstable.
     Negligible — the [−2,2]² box is too wide for *any* bound (CROWN/LP/MILP/α)
     to matter; input-split is the only lever.
   - **α**: cracked one case with smart branching but not the others; ABC uses
     no α here (~10 ms/domain). Not the cause.
   - **The real cause was BRANCHING.** Our smart split-dim heuristic
     (`branch_sb`) over-splits these lattice nets **~290×**
     (2_1: 89,735 leaves → timeout). ABC's naive branching: **305 leaves, 3 s**.
     Switching to naive closes both plateau cases (2_1 3 s, 4_0 7 s) → 32/32,
     **no α / no merge_linear / no MILP needed**.

## Config (`configs/tllverifybench_2023.yaml`)

- `input_split_batched_enabled: true` — batched input-split BaB.
- `input_split_batched_branch_sb: false` — naive branching (smart over-splits TLL).

## Reproduce

```bash
B=$VNNCOMP/benchmarks/tllverifybench_2023
.venv/bin/python -m vibecheck.main \
  --net $B/onnx/tllBench_n=2_N=M=24_m=1_instance_2_1.onnx \
  --spec $B/vnnlib/property_N=24_1.vnnlib \
  --config configs/tllverifybench_2023.yaml --timeout 60 --results-file /tmp/r.txt  # unsat ~3s
bash scratch/sweep_tll.sh configs/tllverifybench_2023.yaml 60   # full sweep -> 32/32
```

## Integration tests (`tests/integration/test_tllverifybench_2023.py`)

- N=M=8 instance_0_3 (SAT), N=M=8 instance_0_0 (UNSAT),
  N=M=16 instance_1_2 (UNSAT), N=M=24 instance_2_1 (UNSAT — the hard case the
  naive-branching fix closes; pins the regression).

## Known unsolved

None.

## Notes

`input_split_batched_alpha_all_leaves` (a boolean added while chasing the bound
hypothesis — run batched α-CROWN on every unclosed leaf, not just the
eps-boundary band) is a general tightener kept default-OFF; tllverifybench does
NOT use it (the fix was branching). `merge_linear` was tested on the actual
merged net and does NOT help (it is exact, only saves compute).
