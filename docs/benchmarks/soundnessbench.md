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

- **vibecheck: 49 sat / 50** — the lone miss is `model_26`, a hard instance
  AB-CROWN also spends 106 s on (see Known unsolved).
- AB-CROWN: 50/50 sat (~6–125 s/case, median ~80 s).
- vibecheck cracks via Phase-0 PGD at ~30–90 s/case — **faster than AB-CROWN**
  where both solve.
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
`pgd_restarts=250`, `pgd_lr_decay=0.997`, `pgd_alpha_frac=0.01`,
`pgd_time_budget_phase0=180`).

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

## Known unsolved

`model_26` (1/50). Deep PGD ran the full 250 restarts × 1000 steps (~109 s)
without finding the planted CEX, then fell through to the zono (OOM) → `unknown`.
It is a hard instance for AB-CROWN too: AB-CROWN gets it `sat` in **106.6 s**
(vs its ~80 s median). The two tools differ in one knob I have not yet matched —
AB-CROWN runs soundnessbench in **float64** (`double_fp: true`); the planted
basin may need that precision. Candidate fixes (untested): `--dtype float64`,
multi-α restarts (`pgd_alpha_multi`), or more restarts. Tracked as the one
straggler; everything else is `sat`.
