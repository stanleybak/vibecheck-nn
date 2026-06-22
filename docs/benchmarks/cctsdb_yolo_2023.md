# cctsdb_yolo_2023

Extended-track. 39 instances (1.0 ≡ 2.0). YOLO traffic-sign **patch** robustness on the
CCTSDB dataset. α,β-CROWN solves all 39 (28 sat, 11 unsat) via a **custom model handler**, not
generic ONNX bounding. **Status: prototyped — VC replicates ABC's extract-backbone +
enumerate-patch-grid approach and matches it on a sat + an unsat case (`scratch/cctsdb_proto.py`);
the full custom handler is the remaining build.** (The raw ONNX still can't be loaded/attacked
directly — that's the wrong frame; see below.)

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

## vibecheck prototype — replicated, matches ABC

`scratch/cctsdb_proto.py` does exactly this with VC's stack (`onnx.utils.extract_model` +
onnx2torch) and reproduces ABC's verdicts on both a sat and an unsat case:

| instance | positions | min score | VC | ABC |
|---|---|---|---|---|
| `idx_00559_0` | 3844 | 0.813 (0 bad) | **unsat** | unsat |
| `idx_16972_2` | 3844 | 0.000 (2 bad) | **sat** | sat |

So this is **tractable, not out of reach** — my earlier "out of reach" was the wrong frame
(it assumed the raw ONNX must be bounded). The one wrinkle: the backbone has a fixed-batch
`Reshape` (`[48,2,16]`) that onnx2torch runs only at batch=1 (ABC uses `onnx2pytorch` quirks
`fix_batch_size`/`merge_batch_size_with_channel`); the prototype loops batch=1 (~10 s/instance).

## Remaining build (to solve all 39)

A `src/vibecheck/cctsdb_yolo.py` custom handler (mirroring `network_pair`/`sign_attack`):
extract backbone (patch-1 `364→461,463`; patch-3 `input→Gather_437,ArgMax_439`) → enumerate the
patch grid → batched (or batch=1) forward → `class_match · IoU` property → `sat`/`unsat`
verdict + cex; `main` hook + config + tests + AWS sweep. Complete verifier (proves the 11 unsat
by enumeration, finds the 28 sat). Prototyped + validated; full handler pending.
