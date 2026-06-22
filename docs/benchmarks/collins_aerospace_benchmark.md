# collins_aerospace_benchmark

Extended-track. 6 instances, 3600 s budget. **YOLOv5-nano** (640×640) object-detection
robustness. All 6 instances are sat. Solved via vibecheck's **generic onnx2torch PGD attack**
(`src/vibecheck/torch_attack.py`) — vibecheck reports **6/6 sat**, each in ~30 s (vs ABC ~95 s),
a clean win on both coverage and speed.

## The benchmark

- **Model:** `yolov5nano_LRelu_640.onnx` — a full YOLOv5-nano detector: **60 Conv + 57
  LeakyRelu + Sigmoid/Split/Concat/Resize/Mul/Pow**, input `images [1,3,640,640]`, output
  `output0 [1,25200,11]` (25 200 anchor boxes × 11 = x,y,w,h,obj + 6 classes), flattened to
  277 200. No data-dependent control flow → fully differentiable through onnx2torch.
- **Spec:** L∞ robustness of one detection. Only a **small patch is perturbed** — ~405 of the
  1.23 M input dims (δ ∈ {0.001…0.1}); the rest are pinned. The output property is a **19-way
  disjunction** over specific anchor cells (objectness/class confidence up or down, or one cell
  ≥ another). A CE is any in-box image making one disjunct's unsafe region reachable.

## ABC reference

α,β-CROWN solves all 6 **sat** in ~95 s each (`vnncomp2026_results/alpha_beta_crown/
2026_collins_aerospace_benchmark`).

## vibecheck approach — generic torch-attack (`src/vibecheck/torch_attack.py`)

Incomplete / attack-only (never proves unsat). Engages on `torch_attack: true` (config). The
verdict is decided **only** by replaying the witness on the ORIGINAL model via CPU onnxruntime
(the scoring engine), so a torch/ORT mismatch can never yield a false sat:

1. Convert the ONNX to torch (autograd flows through the genuine LeakyRelu/Sigmoid/Conv ops —
   no surrogate needed).
2. PGD over the **perturbed** input dims (Adam, restart 0 = box center, restarts >0 = random
   box vertices), driving the spec disjunction's worst safe-margin < 0.
3. Validate each candidate on the original via ORT-CPU: clear CE (margin < 0) → `sat`;
   within-tolerance (0 ≤ margin ≤ `sat_validate_atol`) → stash + keep searching, emit if no
   clear CE found.

In practice the robustness property is already violated **at the box center** (the nominal
detection is fragile), so vibecheck returns a clear CE at restart 0 step 0; the ~30 s wall is
almost entirely the one-time onnx2torch convert + 127 MB vnnlib parse, not search.

## Results (vibecheck vs ABC)

Full sweep (6 instances, `configs/collins_aerospace_benchmark.yaml`, A10G GPU; verdicts from
`--results-file`):

| | vibecheck | α,β-CROWN |
|---|---|---|
| solved (sat) | **6 / 6** | 6 / 6 |
| wall per instance | **~28.7 s** | ~95 s |

vibecheck matches ABC on coverage (**6/6 sat**) and is ~3.3× faster — the ~28.7 s is almost
entirely the one-time onnx2torch convert + 127 MB vnnlib parse; the CE itself is found at
restart 0 step 0 (the nominal detection is already fragile inside the box). Every witness was
re-validated on the ORIGINAL model via CPU onnxruntime (clear CE, margin < 0).

## Reproduce

```bash
.venv/bin/python -m vibecheck.main \
  --net  .../collins_aerospace_benchmark/1.0/onnx/yolov5nano_LRelu_640.onnx \
  --spec .../vnnlib/img_14421_perturbed_bbox_3_delta_0.001.vnnlib \
  --timeout 3600 --results-file out.txt --config configs/collins_aerospace_benchmark.yaml
```

Integration pins: `tests/integration/test_collins_aerospace_benchmark.py` (2 sat cases; needs a
CUDA GPU + the 127 MB specs, skips otherwise). Unit: `tests/test_torch_attack.py` (100 % cov).

## Key unresolved issues

- The attack is **incomplete** — it never proves robustness (unsat). For this all-sat benchmark
  that's sufficient, but certifying a robust YOLOv5 detection would need sound bounds through
  60 convs + the detection head (intractable for the 1.23 M-dim input at full resolution).
