# isomorphic_acasxu_2026

Extended-track, **2.0/** (v2 vnnlib). 50 instances, 100 s budget. **Network-equivalence** of an
original ACAS Xu controller `f` vs a retrained/perturbed copy `g`: a counterexample is an input
where some output differs by more than 0.05 (`sat` = not equivalent), otherwise `unsat`
(equivalent). vibecheck **matches α,β-CROWN exactly: 37 unsat + 2 sat + 11 timeout**, 0 misses,
0 conflicts.

## The benchmark

- **Pair format:** each `instances.csv` row's net field is a list `[('f', onnx/original/…), ('g',
  onnx/perturbed/…)]`. `main._maybe_network_pair` merges the pair into **one** ONNX
  (`network_pair.py`): the 5-D input feeds both ACAS Xu stacks (`x_f = x_g`), and the merged
  outputs are the signed per-output diffs `Y_g[j] − Y_f[j]`. The merge is gated by an
  onnxruntime oracle (merged-vs-separate ≤ 1e-3) — a bad merge raises, never silently verifies.
- **Spec:** v2 `(or …)` over the 5 outputs — `Y_g[j] > Y_f[j]+0.05` **or** `Y_g[j] <
  Y_f[j]−0.05` for some j, i.e. `|diff_j| > 0.05`. The merged v1 spec is a 10-way output OR.

  > **Stale-file note:** the *unzipped* `instance_*.vnnlib` files in the 2.0/ dir are an older
  > revision with a contradictory `(and …)`; the authoritative **`.gz`** (newer) has the fixed
  > `(or …)`. `network_pair._read_vnnlib_text`/`_load_onnx` read `path+'.gz'` first, so VC
  > verifies the correct property regardless of the stale loose sibling.

## ABC reference

α,β-CROWN: 37 unsat, 2 sat, **11 timeout** (`vnncomp2026_results/alpha_beta_crown/
2026_isomorphic_acasxu_2026`). The merged net is two 6×50 ReLU stacks, so VC uses the
input-split + CROWN path (the same family as acasxu_2023); no per-benchmark config is needed —
the default `input_split_small` profile already matches ABC.

## Results (vibecheck vs ABC)

| | vibecheck | α,β-CROWN |
|---|---|---|
| unsat | 37 | 37 |
| sat | 2 | 2 |
| timeout | 11 | 11 |

Per-instance: identical verdicts; **0 misses, 0 conflicts**. The 2 sat are barely-over-threshold
needles (VC's witness diff `Y_0 ≈ 0.051`).

## The 11 timeouts (investigated — both tools miss the same 11)

These are **tight-margin** equivalence cases: the worst-case output diff sits right at the 0.05
threshold. Measured (`scratch/iso_maxdiff2.py`, 300k-sample + gradient refinement validated
against the 2 known sat needles): the 11 reach max|diff| **0.016–0.042, all below 0.05** → they
are **tight-UNSAT**, not findable sats. Two measured dead-ends for closing them:

- **acasxu-style boundary α-CROWN config** (backward-CROWN intermediate bounds + wide boundary
  α band, the recipe that collapses single-acasxu leaf counts): **0/11 closed** — these aren't
  degenerate-boundary-leaf cases.
- **Stronger CE search:** 0/11 reach 0.05 → no needle to find.

**Root cause:** the merged net concatenates two deep stacks and CROWN bounds them
*independently* before subtracting, so the bound on `g − f` is far looser than the true ~0.04 —
input-split BaB can't refine the 5-D box enough in 100 s to prove ≤ 0.05. ABC hits the same
wall. Closing them would need **coupled / difference-aware bounding** (relax `g − f` directly,
exploiting that f and g share the input) — a genuine algorithmic addition, not a hyperparameter.

## Reproduce

```bash
# via the pair harness (scratch/vibe_run_pair_benchmark.py) or directly:
.venv/bin/python -m vibecheck.main \
  --net "[('f','onnx/original/ACASXU_run2a_2_2_batch_2000.onnx'),('g','onnx/perturbed/ACASXU_run2a_2_2_batch_2000_perturbed_0.onnx')]" \
  --spec vnnlib/instance_0.vnnlib --timeout 100 --results-file out.txt
# (run with cwd = the benchmark 2.0/ dir so the relative onnx paths resolve)
```

Integration pin: `tests/integration/test_isomorphic_acasxu_2026.py`.

## Key unresolved issues

- The 11 tight-UNSAT timeouts need difference-aware bounds (above); deferred (ABC misses them
  too, so it's a parity-preserving research item, not a regression).
