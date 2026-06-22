# cctsdb_yolo_2023

Extended-track. 39 instances (1.0 ≡ 2.0). YOLO traffic-sign **patch** robustness on the
CCTSDB dataset. **Status: out of reach for vibecheck — no result.** α,β-CROWN solves all 39
(28 sat, 11 unsat); vibecheck cannot load or attack these nets, so it produces no verdict.

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

That leaves only black-box search (ORT forward + gradient-free optimization). For the 28 sat
cases a needle CE in the 12 288-dim patch is impractical to find gradient-free in the budget,
and the 11 unsat cases need sound bounds through the YOLO + post-processing that vibecheck
does not have.

## What it would take

1. **onnx2torch op support** for `ScatterND`, `Range`, `Where`, dynamic `Min`/`Max`/`Clip`
   (or a pre-pass that constant-folds the static placement/post-processing) → enables the
   generic `torch_attack` PGD on the 28 sat cases.
2. **Sound symbolic bounds** for the YOLO detector + post-processing for the 11 unsat cases —
   a substantial effort (how ABC certifies them).

Deferred: the cost (multiple new sound op handlers + a YOLO bound path) is out of scope for an
attack-only pass; documented here so it isn't re-attempted blindly.
