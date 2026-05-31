# tllverifybench_2023

VNNCOMP regular track. 32 instances of **Two-Level-Lattice (TLL)** networks:
2-D input → deep `MatMul+Add` chains (3 consecutive linear layers per ReLU) →
8 ReLU layers → 1-D output. Spec: a single halfspace `Y_0 <= c`. 17 SAT, 15
UNSAT (per the AB-CROWN reference).

## Score

- **vibecheck: 30/32** (sat 17/17, unsat 13/15) with `configs/tllverifybench_2023.yaml`.
- AB-CROWN published: 32/32 (~2–11 s/case).
- Progression: 24/32 default → 29/32 (enable batched input-split) → 30/32
  (apply batched α-CROWN to all unclosed leaves).
- (vc sweep: `scratch/sweep_tll.sh configs/tllverifybench_2023.yaml`, 2026-05-31;
  ABC ref `~/repositories/vnncomp2025_results/alpha_beta_crown/2025_tllverifybench_2023/results.csv`.)

## How vibecheck solves it (the ABC comparison that found the levers)

AB-CROWN uses **batched input-split BaB — no Gurobi, no β-CROWN, no dual-ascent**
(`exp_configs/vnncomp23/tllVerifyBench.yaml`: `branching.method: naive`,
`input_split.enable`, `bound_prop_method: forward+backward`, `batch_size: 1500`,
`merge_linear`). Run locally: easy cases ~369 domains/2s; the hardest (N=M=40
instance_4_0) **still needs 22,015 domains / 11s** — they are genuinely hard,
ABC just bounds each batch in ~10 ms.

Two levers (both config-only — no code beyond a new boolean setting):

1. **Batched input-split** (`input_split_batched_enabled`). vibecheck routed the
   pure-ReLU TLL net to the SINGLE-LEAF `fast_leaf` path (one 3-iter α-CROWN per
   leaf, ~3 s/leaf → timeout). Its batched path (`_input_split_batched`,
   GPU-parallel, AB-CROWN-style) existed but was gated to bilinear nets.
   Enabling it: 24→29/32.

2. **α-CROWN on all unclosed leaves** (`input_split_batched_alpha_all_leaves`,
   new boolean, default OFF). The batched α-CROWN only fired on leaves within an
   eps-band of 0; the deep net's per-leaf backward-CROWN lb sits at −30..−180,
   so it never fired and the input-split exploded (45k+ leaves). Tightening
   every unclosed leaf cuts that ~200× on the cases it helps: 29→30/32 (closes
   N=M=24 instance_2_0 at 4 s, leaves 45,745→231).

## Config (`configs/tllverifybench_2023.yaml`)

- `input_split_batched_enabled: true`, `input_split_batched_branch_sb: true`.
- `input_split_batched_alpha_all_leaves: true`, `input_split_batched_alpha_iters: 20`.

## Reproduce

```bash
B=$VNNCOMP/benchmarks/tllverifybench_2023
.venv/bin/python -m vibecheck.main \
  --net $B/onnx/tllBench_n=2_N=M=16_m=1_instance_1_2.onnx \
  --spec $B/vnnlib/property_N=16_2.vnnlib \
  --config configs/tllverifybench_2023.yaml --timeout 60 --results-file /tmp/r.txt  # unsat ~3s
bash scratch/sweep_tll.sh configs/tllverifybench_2023.yaml 60   # full sweep
```

## Integration tests (`tests/integration/test_tllverifybench_2023.py`)

- N=M=8 instance_0_3 (SAT), N=M=8 instance_0_0 (UNSAT), N=M=16 instance_1_2
  (UNSAT, deep net cracked by the batched + α path).

## Known unsolved (2) — and the next lever

N=M=24 instance_2_1, N=M=40 instance_4_0 (both UNSAT). ABC verifies them in
3.9 s / 11.4 s with **10,735 / 22,015 domains**; vibecheck generates **~15×
more** domains (156k+) and times out — our per-domain bound is that much looser,
and even α-CROWN-on-all-leaves doesn't close the gap (and adds Adam-iter cost).

The lever is **`forward+backward` CROWN** (ABC's `bound_prop_method`): a *fast*
forward-CROWN pass combined with backward-CROWN, which is both **tighter** (15×
fewer domains) **and faster** (~10 ms/batch, no per-leaf α optimization) than
vibecheck's forward-zono + backward-CROWN (+ slow α-opt). vibecheck has no
standalone forward-CROWN today (only forward-zono and the slow α-refresh), so
adding it is the substantial next step for these two cases. (merge_linear was
tested on the actual merged net — does NOT help; it is exact, only saves
compute. β-CROWN / Gurobi / dual-ascent are NOT what ABC uses here.)
