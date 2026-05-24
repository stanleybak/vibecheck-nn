# dist_shift_2023 — vibecheck benchmark record

VNNCOMP 2025 regular track. 72 instances on a single model
(`mnist_concat.onnx`) — an encoder MLP that compresses MNIST input
through a Sigmoid, then a classifier MLP that consumes the Sigmoid
output to predict the digit class. Specs are robustness assertions
on the cls/output.

The Sigmoid sitting in the middle of the active forward path was the
hard part: it adds slack noise vars that have to align with the
α-zono state's column accounting, and several backward/forward paths
needed sigmoid/tanh handlers that didn't exist before.

## Final score (server1, 2026-05-24, RTX 3080 / 10 GB, 30 s budget)

| Solver | Solved / 72 | Wall (total) | Notes |
| --- | --- | --- | --- |
| **vibecheck** | **72** | **~155 s** | 7 SAT + 65 UNSAT |
| AB-CROWN (published, 2025) | 72 | ~515 s | all verified |

**0 misses** — full crack. Wall 155 s vs AB-CROWN 515 s — **~3.3×
faster** at full coverage.

## Algorithmic adds for this benchmark

Several silent assumptions in the gen-LP MILP pipeline (built for pure
ReLU nets) broke on mnist_concat's mid-stream Sigmoid:

- **Sigmoid in `forward_zono_dir_adaptive`**
  (`alpha_crown.py`) — **chord-parallelogram relaxation**: solve for
  (α, β, γ) such that σ(x) ⊆ [α·x + β - γ, α·x + β + γ] over [lo, hi]
  via chord slope + symmetric critical-point analysis
  (`_sigmoid_tanh_chord_parallelogram` in `verify_zono_bnb.py`). Encoded
  as `y = α·x + β + γ·e_new` — **preserves input correlation** through
  the α slope (only γ is fresh slack). Pre-fix box-relax (just `mu·e_new`)
  scored 62/72; parallelogram pushed to 69/72.
  Soundness tests: `tests/test_sigmoid_parallelogram.py` (16 tests
  covering pure-convex/concave, mixed straddle, tiny/wide/random
  intervals; 500 intervals × 2000 samples each).
- **`state_from_alpha_zono` Sigmoid handling**
  (`verify_gen_lp.py`) — Sigmoid layers don't go in `unstable_list`
  (gen-LP MILP can't binarise sigmoid), but `cur_n_gens` still has to
  advance past them so downstream ReLU e_new_col indices align.
  Defensive `e_new_col >= n_gens` guard skips any misaligned entries.
- **`_per_neuron_adaptive_bounds` Sigmoid handler**
  (`verify_graph.py`) — closed-form linear backward via
  `_sigmoid_tanh_linear_bounds`.
- **`bab_refine` cascade cap at first Sigmoid layer**
  (`verify_graph.py`) — gen-LP MILP encodes ReLU triangles only;
  cap `max_layer` to `first_sigmoid_idx - 1` so MILP doesn't try to
  build a dependency cone through Sigmoid.
- **`forward_point` Sigmoid/Tanh handlers**
  (`verify_gen_lp.py`) — for witness validation.
- **`_phase8_score_box_halfspace` disabled** in the cora-style config —
  the score function still references state's n_gens accounting that
  breaks for Sigmoid; ew*frac fallback works fine.
- **`box_halfspace.tighten_all_layers` 1-elem layer fix** — 1-neuron
  layers (e.g. Sigmoid output of a generator network) collapse to 1D
  arrays; defensive `np.atleast_1d` keeps the code path 2D.

## Knobs (`configs/dist_shift_2023.yaml`)

- `zono_lift_enabled: true` — Phase 2.5 zono-lift uses the box+halfspace
  LP to tighten unstable pre-ReLU bounds per query. With box-relax it
  regressed (62 → 55); with the parallelogram (which routes input
  correlation through α-slope) it cracks the last 3 plateau cases and
  closes the benchmark at 72/72.
- `phase8_score_box_halfspace: false` — score_box_halfspace_delta_lb
  still has unfixed indexing bugs around Sigmoid; ew*frac scoring is
  used instead.
- `phase8_use_dual_ascent_gpu: true`, `phase8_dual_ascent_max_iter: 1`,
  `phase8_min_budget_frac: 0.5` — cifar100/cora-style Phase 8 config.
- `phase2_crown_enabled: false`, `pgd_middle_enabled: false` — skip
  no-op phases for cora/cifar100 family.

## Reproducing a single case

```bash
.venv/bin/vibecheck \
  --net path/to/dist_shift_2023/onnx/mnist_concat.onnx \
  --spec path/to/dist_shift_2023/vnnlib/index7901_delta0.13.vnnlib \
  --mode graph --timeout 30 --bits 32 \
  --config configs/dist_shift_2023.yaml
```

## Full sweep

```bash
.venv/bin/python scratch/dist_shift_smoke.py
```

## Integration test coverage

`tests/integration/test_dist_shift_2023.py`:
- `index4739` (SAT, ~0.05 s) — PGD root path.
- `index7901` (UNSAT, ~2 s) — Sigmoid forward + α-zono state alignment.
- `index6957` (UNSAT, ~1 s) — Phase 8 scoring with Sigmoid skip.

## Known unsolved cases

None.
