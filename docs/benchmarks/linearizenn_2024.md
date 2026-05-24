# linearizenn_2024 — vibecheck benchmark record

VNNCOMP 2025 regular track. 60 instances across 11 `AllInOne_*.onnx`
networks (1 SAT + 59 UNSAT per ABC; vibecheck matches). Each model is
a 4-input deep ReLU MLP (`256→256→256→256→16→8→8→2`) with a parallel
linear **skip branch** via `Slice(input)→MatMul`, joined by `Concat`
before a final `MatMul` to a 2-D output.

The Slice + Concat skip-branch pattern hit two real bugs pre-fix:

1. **Soundness:** `_spec_backward_graph` (and several sibling backward
   dispatches) had no `else:` clause on the op-type chain, so unhandled
   Slice/Concat ops silently fell through. The backward `ew` died at
   the Concat output, never reached the input, and the final spec_lb
   was `acc + 0 · xl + 0 · xh = acc` — vacuously positive. We declared
   `verified` on `prop_10_10`, whose ABC counterexample (Y_0 ≈ 40.27)
   trivially satisfies the spec.
2. **Speed:** Once soundness was restored, fast-leaf input-split BaB
   timed out on the deeper models (`AllInOne_120_120` plateau in ~30s,
   stuck at depth ~7 because per-leaf forward+CROWN was too serial).

## Final score (server1, 2026-05-24, RTX 3080 / 10 GB, 30 s budget)

| Solver | Solved / 60 | Wall (total) | Notes |
| --- | --- | --- | --- |
| **vibecheck** | **60** | **~19 s** | 1 SAT + 59 UNSAT |
| AB-CROWN (published, 2025) | 60 | ~473 s | all verified |

**0 misses.** Wall: 19 s vs AB-CROWN's 473 s — **~25× faster** at full
coverage.

## Algorithmic adds for this benchmark

- **Slice / Concat ops everywhere** — added handlers to
  `_forward_zonotope_graph`, `_forward_zonotope_graph_batched`,
  `_spec_backward_graph`, `_spec_backward_graph_batched`,
  `_forward_batch_graph` (PGD), three `alpha_crown.py` dispatches
  (`_crown_backward_matrix`, per-query backward,
  `forward_zono_dir_adaptive`), and the gen-LP-builder loops in
  `verify_gen_lp.py`. Plus matching gg-ops registration in
  `network.py` (`flat_idx` for Slice, `axis` for Concat).
- **Universal `else: raise NotImplementedError(...)`** on every op
  dispatch site. Previous code silently fell through on unknown ops,
  hiding the linearizenn unsoundness for months until ABC's
  counterexample caught it. Now any future unhandled op fails loudly
  on the first execution.

## Knobs (`configs/linearizenn_2024.yaml`)

- `input_split_batched_enabled: true` /
  `input_split_batched_clip_enabled: true` /
  `input_split_batch_size: 4096` — cersyve-style batched leaf
  processing. The unbatched fast-leaf path serialized forward+CROWN
  per BaB leaf, dominating wall time. Batching the leaves through one
  GPU graph forward at depth=k+1 cuts per-leaf cost ~30× and finishes
  the hardest models in <0.5s.

No other overrides; default settings handle the rest.

## Reproducing a single case

```bash
.venv/bin/vibecheck \
  --net path/to/linearizenn_2024/onnx/AllInOne_120_120.onnx \
  --spec path/to/linearizenn_2024/vnnlib/prop_120_120.vnnlib \
  --mode graph --timeout 30 --bits 32 \
  --config configs/linearizenn_2024.yaml
```

## Full sweep

```bash
.venv/bin/python scratch/linearizenn_smoke.py
```

## Integration test coverage

`tests/integration/test_linearizenn_2024.py`:
- `prop_10_10` (SAT, ~0.5 s) — regression for the slice/concat
  soundness bug. Pre-fix this case declared `verified` (unsound);
  ABC's counterexample produces Y_0 ≈ 40.27 ≥ 40.204.
- `prop_120_120` (UNSAT, ~0.3 s) — batched input-split BaB
  on the deepest model.
- `prop_120_120_4` (UNSAT, ~0.3 s) — extra batched-BaB regression.

## Known unsolved cases

None.
