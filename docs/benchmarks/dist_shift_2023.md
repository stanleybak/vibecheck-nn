# dist_shift_2023 — vibecheck benchmark record

## UPDATE 2026-06-02 — soundness re-confirmed on AWS A10G (current main)

Re-verified that the gen-LP soundness bug (orphan `[0,0]`-column collapse in
`_build_alpha_zono_lp`, fixed by `4bc8097`) is fully closed, not just for the two
cases originally caught.

- **Soundness probe** (`--disable-sat-finding`, production config) on **all 7
  ABC-`sat` cases** — index 4739 / 1282 / 1285 / 9211 / 4312 / 4272 / 7534 — every
  one returns `unknown`. **0 unsound of 7.** (Prior record only validated 1285 &
  4312; the other 5 are confirmed here.) A `verified`/`unsat` here would mean the
  bounds path falsely closed a genuine counterexample.
- **Full 72-case production sweep** (config `dist_shift_2023.yaml`, verdict from
  `--results-file`, cross-checked vs the published ABC verdicts): **72/72 match —
  7 sat + 65 unsat, 0 unsound / 0 mismatch / 0 incomplete.**

Root cause recap: the spec-MILP / gen-LP reserves a generator column per zonotope
noise symbol; `state_from_alpha_zono` reserves columns for unstable neurons it
*skips* (the sigmoid γ-slack + `mnist_concat`'s ~740 generator-subnet ReLUs not in
`unstable_list`). Pinning those to `[0,0]` collapsed each neuron's parallelogram
`λ·z + μ·(1+e_new)` to its center line — an unsound ReLU under-approximation that
let the MILP binarization cut off a real CEX → false `verified`. Masked in
production because PGD finds the CEX first. Fix = every column free in `[-1,1]`
(`verify_gen_lp.py:1095-1097`), pinned by
`tests/test_genlp_relaxation_equals_zonotope_alpha_zono`. Reproduce the probe:
`python -m vibecheck.main --net onnx/mnist_concat.onnx --spec
vnnlib/index4312_delta0.13.vnnlib --config configs/dist_shift_2023.yaml
--disable-sat-finding --timeout 60 --results-file /tmp/r.txt` → must be `unknown`.

## UPDATE 2026-06-01 — 9 UNSAT misses closed → 72/72

A completeness audit found **9 UNSAT misses** (all `mnist_concat`, ABC solves in 8–13 s, we returned `unknown` in ~6 s). Two root causes, both fixed:

1. **Routing gate blocked the intended input-split path.** The config already set `input_split_batched_enabled: true`, but `verify_graph.py:7736` gates the input-split family on **total** input dim (`n_in ≤ input_split_max_dims`, default 20). `mnist_concat` is **792-dim but only 8 vary** (an 8-D latent; the 784 image pixels are fixed), so 792 > 20 silently routed every case to the dual-ascent `_run_pipeline` instead. That path can't reduce the **sigmoid relaxation slack** (it only splits ReLUs — "sigmoid γ slack is not in the split set"), which is exactly what kept the bound negative. Input-split narrows the 8 sigmoid inputs and closes them. **Fix: `input_split_max_dims: 800`** activates the intended path — index9733/2481 0→verified in <3 s (faster than ABC). 8 of 9 verify in <20 s; index112 in 152 s (within the 300 s timeout).

2. **Dual-ascent witness-check dimension bug** (the masked crash). `_da_witness_check` mapped the LP primal witness `e` (in the **input-generator** subspace = 8 varying dims) back to real input as if it were full-dim (792) → `RuntimeError`, swallowed by `main.py`'s top-level handler as `unknown`. Fixed to scatter the sparse witness into the varying dims (sound — only finds genuine in-box counterexamples). Still valuable for the fallback path / other benchmarks.

Re-audit: **0 MISSES of 72**, 0 false-verifies; the SAT cases stay `sat`, and the soundness probe (index4312, sat-finding off) correctly returns `unknown`.

---

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
