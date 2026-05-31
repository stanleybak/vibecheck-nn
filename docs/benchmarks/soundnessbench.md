# soundnessbench

VNNCOMP regular track, SOUNDNESS-STRESS benchmark. 50 instances share ONE conv
net (`128 → Gemm(12288) → ReLU → Reshape(3,64,64) → Conv×6 → Gemm(384)`,
~241K ReLUs; first two conv ReLUs are 98304 wide). Each `model_i.vnnlib` is a
different 128-D input box (width 1.0/dim) hiding a different planted
counterexample. **EVERY instance is SAT** — the benchmark exists to catch
*unsound* verifiers: a `verified`/`unsat` here certifies a property that is
actually false. (AB-CROWN's results CSV lists one `unsat`, but that row is the
`test/test_nano` sanity instance, not a soundnessbench model.)

## Score

- **vibecheck: 50 sat / 50 — full parity with AB-CROWN.**
- AB-CROWN: 50/50 sat (~6–125 s/case, median ~80 s).
- vibecheck cracks via Phase-0 PGD; 49 cases at ~30–90 s on a single α=0.01, and
  the last (model_26) needs a second α=0.05 (two-way multi-α, below). Slowest
  case ~131 s, under the 150 s budget.
- (vc sweep: `~/persistent_runs/scripts/sweep_soundnessbench.sh` on server1,
  2026-05-31; ABC ref `2025_soundnessbench/results.csv`.)

## How vibecheck solves it — it's an ATTACK benchmark, not a bound benchmark

AB-CROWN's own config (`exp_configs/vnncomp25/soundnessbench.yaml`) is
**attack-only**: `pgd_restarts=250, pgd_steps=1000, pgd_alpha=0.005,
pgd_lr_decay=0.997`, `double_fp: true`, with `complete_verifier: auto` as an
unused fallback. The planted CEXs are found by **deep PGD**, not bound
propagation. Two things had to be true for vibecheck to match it:

1. **Route to verify_graph, attack first** (`auto_route_milp_for_conv: false`).
   The conv auto-route sends the net to `milp_verify`, whose dense forward
   zonotope OOMs at the first 98304-wide conv ReLU — it adds one noise symbol
   per unstable neuron → a ~98304×110720 generator matrix ≈ 43 GB. That path is
   never needed: every instance is SAT, so we want the witness, and Phase-0 PGD
   runs *before* any bound propagation. The zono is never reached.

2. **Make Phase-0 PGD as deep as AB-CROWN's.** vibecheck's default PGD
   (`pgd_iter=100`, `pgd_alpha_frac=0.25` → step 0.125 = 25× AB-CROWN's 0.005 on
   the width-1 box, 10 s budget) is far too coarse / short to descend into the
   narrow planted basin — it returned `unknown`. Mapping AB-CROWN's knobs onto
   vibecheck's (`pgd_iter=1000`, `pgd_alpha_frac=0.01` since `eps_input` is the
   half-width 0.5, `pgd_lr_decay=0.997`, `pgd_time_budget_phase0=180`) cracks
   them.

**Why not the zono→CROWN→α-CROWN→targeted-PGD pipeline** (used for
tinyimagenet)? It doesn't apply here: the spec is a SINGLE disjunct (a
12-constraint conjunction `Y_j ≥ c_j`), so there are no unsat disjuncts to
eliminate and focus PGD on; and there is no unsat case to verify. α-CROWN bounds
on the width-1 box over 241K ReLUs are far too loose to eliminate any constraint
without full input-split BaB. Mirroring AB-CROWN (deep PGD) is the right tool.

## Soundness

Every `sat` is **independently onnxruntime-validated**: `_finalize` runs the PGD
witness through `_validate_sat_witness` (ORT `InferenceSession`), checks it is
in-box AND that its ORT output satisfies the unsafe condition, and downgrades any
spurious witness to `unknown`. So we cannot false-`sat`, and we structurally
cannot false-`unsat` (Phase-0 PGD returns before any verified verdict is
possible). This is exactly the property the benchmark checks.

## Config (`configs/soundnessbench.yaml`)

`auto_route_milp_for_conv: false` + deep-PGD knobs (`pgd_iter=1000`,
`pgd_lr_decay=0.997`, `pgd_time_budget_phase0=145`) + two-way multi-α
(`pgd_alpha_multi: true`, `pgd_alpha_multi_fractions: [0.01, 0.05]`,
`pgd_restarts: 500`).

## Reproduce

```bash
B=$VNNCOMP/benchmarks/soundnessbench
.venv/bin/python -m vibecheck.main --net $B/onnx/model.onnx \
  --spec $B/vnnlib/model_0.vnnlib --config configs/soundnessbench.yaml \
  --timeout 190 --results-file /tmp/r.txt          # -> sat ~20-35 s
bash ~/persistent_runs/scripts/sweep_soundnessbench.sh   # full sweep (server1)
```

## Integration tests (`tests/integration/test_soundnessbench.py`)

model_0, model_1, model_2 — three SAT cases that deep-PGD cracks (~30 s).
Expected `sat`; a regression that emitted `verified`/`unsat` (false-verify) or
`unknown` (lost the attack) fails them.

## The model_26 straggler — and the two-way multi-α fix

49/50 crack on a single α=0.01 (the ABC-matched step). **model_26 does not** —
its planted basin is missed by α=0.01 (and by 0.002 and 0.25) but **hit by
α=0.05** (cracks in ~60 s). No single α solves all 50.

The naive fix (4-way `pgd_alpha_multi` at 600 restarts) *regressed* model_6: that
case is slow (~102 s on α=0.01) and the 4-way split starved it to 150 restarts
of α=0.01 → `unknown` at 158 s. The fix is a **two-way** split `[0.01, 0.05]` at
**500 restarts** → 250 each: model_6 keeps its full 250×0.01 density (cracks
129 s) *and* model_26 gets 250×0.05 running concurrently (cracks 60 s). Verified
50/50, all < 150 s (slowest model_8 131 s). Cost: ~2.4× slower on easy cases
(still far under budget). model_26 is a hard instance for AB-CROWN too — it gets
it `sat` in 106.6 s vs its ~80 s median.

## Known unsolved

None. 50/50.
