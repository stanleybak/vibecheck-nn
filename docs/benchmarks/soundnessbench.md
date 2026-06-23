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

3. **Persist all the budget on attack (`pgd_phase0_persist_until_budget: true`,
   `pgd_seed: 0`).** A single 500-restart batch is *flaky*: it cracks model_6's
   planted basin only ~90 % of the time (measured 9/10 over seeds 0–9; seed=6
   misses), and the A10G full sweep drew an unlucky init → MISSED model_6
   (`unknown` @ 23.5 s) while server1 hit it. A batch takes only ~35 s of the
   145 s budget, so the fix is to **keep relaunching fresh-init batches until
   the budget is spent** (~4 rounds → miss probability ~1e-4). Verified: the
   unlucky seed=6 recovers on round 3 (sat @ 115 s); the lucky seed=0 returns
   on round 0 (~32 s). `pgd_seed: 0` just makes round 0 reproducible across
   machines (mirrors AB-CROWN's `reset_seed_after_precompile`). Full local
   re-sweep: 50/50, 0 miss.

   Each batch attacks all disjuncts with **per-restart disjunct targeting**
   (`per_restart_disj`): restart *r* descends only disjunct *r % n*'s loss, so
   every disjunct gets dedicated restarts instead of one joint loss all
   restarts share (the witness screen still accepts a CEX from any disjunct).
   This is a no-op for soundnessbench (single-disjunct conjunction) but is the
   right objective for multi-disjunct SAT specs that may use the same
   persist-until-budget path.

   *Why not just bound-prop as a backup?* The cascade can close **zero**
   soundnessbench cases (all SAT) and its dense zono OOMs (~43 GB), so when PGD
   exhausts the budget without a witness we skip the cascade and report
   `unknown` directly — all remaining time goes to attack, never to a doomed
   bound-prop pass.

   *Note on `unknown` vs `error`.* On the A10G, model_6's `unknown` was a true
   "PGD found no witness", not a crash — the dense zono is never reached once
   PGD runs first. If the dense zono *were* reached and OOM'd, the default
   `raise_on_oom=True` surfaces it as `error`, never a silent `unknown` (the
   one batched-path leaf that swallowed a batch=1 OOM into `unknown` now honors
   `raise_on_oom` too).

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
`pgd_restarts: 500`) + deterministic restarts (`pgd_seed: 0`) + persist-until-
budget attack (`pgd_phase0_persist_until_budget: true` — relaunch batches until
the budget is spent, then `unknown` + skip the OOM-prone cascade).

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

---

## 2026 update — `soundnessbench_2026` adds soundness-stress UNSAT-looking instances; **AB-CROWN is UNSOUND on them, vibecheck is correct**

The 2026 set (`soundnessbench_2026`, 60 instances on `model.onnx` and the residual
variant `model_residual.onnx`) adds instances whose property *looks* UNSAT and
that AB-CROWN reports `unsat` — but which are soundness-stress cases. **On at
least `property_009/012/034/042` (all on `model_residual.onnx`), AB-CROWN's
`unsat` is UNSOUND, and vibecheck's `unknown` is the correct, sound behavior.**

Established 2026-06-23 (AWS g5, float64):

- AB-CROWN verifies all four `unsat` with **init CROWN only** (no β, no BaB;
  "Verified with initial CROWN!", ~2 s).
- For `property_012` it verifies by refuting atom `Y_277 ≥ 5.125132`, claiming
  `Y_277 ≤ 5.05`. **But `onnxruntime` (the authoritative output oracle) gives
  `Y_277 = 7.388109` at an input inside the VNNLIB box** — the bound excludes
  reachable values, so the `unsat` is unsound. vibecheck's bound on `Y_277` is
  `7.84` (sound), so vibecheck cannot prove `unsat` → returns `unknown` (correct).
- Smoking gun: on the minimal single-atom spec `Y_277 ≥ 5.125132`, AB-CROWN's
  bound propagation says `unsat` while its own attack says `sat` (PGD finds the
  witness in 0.47 s) — an internal contradiction.
- Root cause: the residual net's **carrier branches** (input → `Reshape(1,8,16)`
  → 3 stacked convs, no intervening ReLU → ReLU → `MatMul` → added to output).
  Those carrier-ReLU pre-activations are an exact linear function of the input
  with range ≈ `[-88, 98]`, but AB-CROWN's intermediate bounds for them come out
  **degenerate** (e.g. `lo == hi == 2.39`, excluding the reachable `[-0.19, 4.38]`
  confirmed by sampling), poisoning the output bound on `Y_277`. vibecheck
  computes these carrier bounds exactly.

**Reproduced on the latest AB-CROWN** (fresh clone `e5c7e17` 2026-06-16 +
auto_LiRPA `5a098e8` 2026-06-10, run locally on CPU): same `initial CROWN bounds
[-5.049972]` → `Verified with initial CROWN!` → `unsat`. NOT an old-branch
artifact — the current release still false-verifies `demo_Y277.vnnlib`.

**Bug report package** (send to the soundnessbench / AB-CROWN maintainers):
`scratch/abc_soundness_bug/` — `model_residual.onnx`, `demo_Y277.vnnlib`,
`counterexample_X.npy`, `validate_counterexample.py`, `README.md`.
`md5(model_residual.onnx) = 51f54444f97ba39ddd9a89fd7b446307`.

**Scoring implication:** these are AB-CROWN false-verifications (heavy VNNCOMP
penalty), **not** vibecheck losses. The SCORING report must not credit AB-CROWN
for soundnessbench `unsat`s; vibecheck's `unknown`/`sat` is the sound outcome.
The earlier "4 cases need BaB to prove unsat" framing is wrong — proving them
`unsat` would itself be unsound. The right goal is to find the planted CEs and
return `sat`; the witnesses are tight needles (PGD reaches conjunct min-margin
≈ −2e-4) that neither tool's attack currently finds.

## Known unsolved (2026)

- `property_009/012/034/042` (residual model): vibecheck returns `unknown`
  (sound); the planted counterexamples are needles below current attack reach.
  AB-CROWN false-verifies them `unsat`. Closing them for vibecheck means a
  stronger *attack* (not a soundness-violating `unsat` proof).
