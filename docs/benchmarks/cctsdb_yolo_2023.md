# cctsdb_yolo_2023

Extended-track. 39 instances (1.0 ≡ 2.0). YOLO traffic-sign **patch** robustness on the
CCTSDB dataset. The property is a finite enumeration of integer patch placements, not a
continuous-input bounding problem (α,β-CROWN also solves it with a custom handler).
**Status: SOLVED — vibecheck 39/39 (28 sat + 11 unsat) = ABC, complete by patch-position
enumeration; 0 misses, 0 conflicts.** (The raw ONNX still can't be loaded/bounded directly —
that was the wrong frame; see below.)

## The benchmark

- **Models:** `patch-1.onnx`, `patch-3.onnx` — a 64×64×3 patch is scattered into a fixed
  background, run through a ~30-Conv YOLO detector, and YOLO post-processing (box decode +
  argmax + min/max selection) is **baked into the ONNX**. Input is flat `[12296]` (the 12 288
  patch pixels + 8 placement scalars). Ops include `ScatterND`, `Range`, `Where`, `Equal`,
  `Expand`, `ConstantOfShape`, `ArgMax`, dynamic `Min`/`Max`/`Clip` — i.e. data-dependent
  control flow, not a plain feed-forward net.
- **Spec:** 28 sat (a patch that flips/creates a detection) + 11 unsat (robust).

## Why vibecheck can't run it (measured)

Both of vibecheck's ONNX front-ends reject these graphs:

- **`vibecheck.onnx_loader.load_onnx` → `IndexError`** during shape inference — the
  control-flow ops (`ScatterND`/`Range`/`Where` with dynamic shapes) aren't modeled by the
  graph builder, and bounding them soundly would require new symbolic handlers for each
  (every op dispatch must be sound or `raise`, so silently skipping is not an option).
- **`onnx2torch.convert` → `NotImplementedError: Dynamic value of min/max is not implemented`**
  — so even the attack-only path (onnx2torch + PGD + ORT-validate, which cracks
  collins_aerospace and traffic_signs) has no autograd graph to differentiate.

But the raw ONNX was **never meant to be bounded** — the benchmark ships a custom model
loader. How α,β-CROWN actually solves all 39 (read from its source,
`complete_verifier/custom/custom_yolo_CCTSDB_verification.py` + `exp_configs/vnncomp23/
cctsdb_yolo.yaml`, which sets `onnx_loader: Customized(...)` + `complete_verifier:
Customized(...)`):

1. **Extract the conv backbone** with `onnx.utils.extract_model` — for `patch-1` the subgraph
   `364 → (461, 463)` (reg + cls heads), **discarding** the patch-placement preprocessing
   (ScatterND/Range/Where) and the box-decode/ArgMax tail. The extracted backbone is plain
   Conv/ReLU and loads fine.
2. **The perturbation is discrete and finite.** The vnnlib fixes the entire image and only
   varies two *integer* patch positions (`X_12288`, `X_12289` ∈ [0,62]) → a **63×63 ≈ 3844**
   grid. ABC `torch.arange`s the ranges, applies a zero-patch per position, and runs the
   backbone over the whole batch.
3. **Verification = concrete inference + a custom property** (`RecoveredYOLO`): re-implement
   the post-processing in clean torch (dynamic `Min`/`Max` → stacked `torch.min/max`,
   box-decode → `bbox_iou`, `ArgMax` detached) and compute `class_match · IoU`; the instance is
   `safe` iff `(score > rhs)` holds for **all** positions, else `unsafe`. So it is **complete by
   enumeration** — no CROWN/BaB needed.

## vibecheck approach — `src/vibecheck/cctsdb_yolo.py` (simpler than ABC, and sound)

We go one step **simpler** than ABC: since onnxruntime runs the full original ONNX (all the
control-flow ops), there's no need to extract the backbone or re-implement the post-processing.
The handler (gated on config `cctsdb_yolo: true`, dispatched by `main._maybe_cctsdb_yolo`):

1. Parse the vnnlib; the **free** input dims (where `hi>lo`) are the patch positions. Assert
   they are **integer-valued** and the grid size ≤ `cctsdb_max_positions` (else raise — not a
   discrete-patch instance).
2. Enumerate every integer position; for each, set the position dims and run the **ORIGINAL**
   net on **ORT-CPU** → detection score `Y`.
3. Decide via the spec margin (`_worst_margin_np`): a **clear CE** (margin<0) → `sat` (return
   that placement); all positions safe → `unsat` (complete); a measure-zero within-tol position
   → within-tol `sat`. ~1.6 s/instance (sat early-exits; unsat enumerates all 3844 ≈ 14 s).

The verdict is decided **only** by the original model on ORT-CPU (the scoring engine) — no
re-implemented post-processing to trust. **Soundness** rests on the benchmark's discrete
semantics: the patch position is an integer pixel offset (we enumerate every integer in
`[lo,hi)`, matching ABC). Net-agnostic — `patch-1` and `patch-3` use the same code.

## Results (vibecheck vs ABC)

Full sweep (39 instances, `configs/cctsdb_yolo_2023.yaml`, ORT-CPU; stop-on-miss-**or**-conflict
vs ABC; verdicts from `--results-file`):

| | vibecheck | α,β-CROWN |
|---|---|---|
| sat | 28 | 28 |
| unsat | 11 | 11 |
| **total** | **39 / 39** | 39 / 39 |

**0 misses, 0 conflicts** (no VC-unsat-where-ABC-sat = no false-unsat; no VC-sat-where-ABC-unsat
= no false-sat). Validated on `patch-1` and `patch-3`, sat and unsat. The earlier "out of reach"
was the wrong frame — it assumed the raw ONNX had to be bounded.

Integration pin: `tests/integration/test_cctsdb_yolo_2023.py` (1 unsat + 1 sat). Unit:
`tests/test_cctsdb_yolo.py` (100 % cov). Reference prototype that confirmed ABC's own
extract-backbone recipe: `scratch/cctsdb_proto.py`.
