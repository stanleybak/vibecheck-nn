# smart_turn_multimodal_2026

Extended-track, **2.0/ (v2 vnnlib)**. 50 instances, one model. Solved via the
**surrogate-attack** mode (incomplete / attack-only): vibecheck reports **50/50 sat**.

## The benchmark

- **Model:** `smart-turn-multimodal-cpu.onnx` — a **41.6 M-param, INT8-quantized**
  multimodal (audio + video) transformer for conversational turn-detection. 692 nodes:
  22 Conv, 34 MatMul + 5 Gemm, 5 Softmax (attention), 9 Erf (GELU), ~11 LayerNorm
  (ReduceMean/Sqrt/Pow/Sub), Sigmoid head; **199 DequantizeLinear + 119 QuantizeLinear**.
  (The `com.microsoft.nchwc` domain is *declared* in opset_import but no node uses it —
  every node is standard `ai.onnx`; the only non-standard ops are the Q/DQ pair.)
- **Inputs:** `X1` audio `[1,80,800]` (64 000) + `X2` video `[1,3,32,112,112]` (1 204 224)
  → **≈ 1.27 M input dims**. **Output:** `Y [1,1]` (post-sigmoid logit).
- **Spec (all 50):** L∞ robustness — every input dim perturbed by a uniform per-modality
  radius (**audio ε=0.05, video ε=0.03**), output violation `(> Y[0,0] 0.5)` (the
  decision flips past 0.5). The vnnlib is ~124 MB (1.27 M box constraints).

## ABC reference

α,β-CROWN 2025 **cannot load it**: `onnx2pytorch` has no `DequantizeLinear` converter
(`NotImplementedError`). So ABC scores **`error` on all 50** — purely a loading wall, not
robustness or difficulty (de-quantizing would verify a *different* model, so the ABC-2025
baseline keeps `error`). For reference, even if it loaded, complete L∞ verification over
1.27 M dims through a 41 M-param transformer is intractable for any current tool.

## vibecheck approach — surrogate-attack mode (`src/vibecheck/surrogate_pgd.py`)

The model can't be soundly bounded (quantized ops), so this is **incomplete / attack-only
(never returns unsat)**:

1. **Fold Q/DQ → a continuous float surrogate ONNX** — weight `DequantizeLinear` → baked
   float constants (incl. per-axis scale); activation `Quantize`/`Dequantize` pairs →
   `Identity` (drops the rounding ⇒ differentiable). Built in `prepare_instance.sh`
   (`--build-surrogate`, untimed).
2. **onnx2torch → torch GPU** (handles Softmax/Erf/LayerNorm that vibecheck's own forward
   graph does not), used **only for the gradient** (STE: the true gradient is ~0 a.e.
   because of the `round`).
3. **PGD for the whole timeout** maximizing the output violation over the L∞ box.
4. **Every candidate is validated on the ORIGINAL quantized model via CPU onnxruntime**
   (the VNNCOMP scoring engine), witness in-box + strict `Y > 0.5`. The verdict is decided
   **only by the original model** — a mismatched surrogate can never yield a false sat.

Config: `configs/smart_turn_multimodal_2026.yaml` (`surrogate_attack: true`,
`surrogate_attack_restarts: 3`, `surrogate_attack_steps: 60`).

## Results — 50/50 sat

| | count | how |
|---|---|---|
| sat at center (trivial) | **20** | nominal already `Y=0.9176 > 0.5` (forward pass) |
| sat via STE-PGD | **30** | "boundary" nominal `Y=0.5`; one gradient step crosses a quantization cell to `Y>0.5` |
| robust | 0 | — |

The INT8 output is effectively discrete (`{0.5, 0.6425, 0.7067, 0.9176}`); the 30 boundary
nominals sit exactly on `Y=0.5` and a single gradient-aligned corner tips them past 0.5
(random corner-sampling misses this — it needs the surrogate gradient). **Cost:** ~0.5 s/
PGD-step on the local GPU, SAT typically at step 1 → ~4 s/instance (incl. onnx2torch
convert); the full 50 in ~2 min on one GPU.

## Reproduce

```bash
# one instance (full harness path):
.venv/bin/python -m vibecheck.main \
  --net  .../smart_turn_multimodal_2026/2.0/onnx/smart-turn-multimodal-cpu.onnx \
  --spec .../2.0/vnnlib/instance_0.vnnlib.gz \
  --config configs/smart_turn_multimodal_2026.yaml --timeout 100 \
  --results-file /tmp/out.txt        # -> sat + a ~1.27M-line counterexample (5.6MB gz)
```

## Key unresolved issues

- **Incomplete:** never returns `unsat`. This benchmark appears to be all-sat (empirically
  50/50), so that's not a practical limit here, but a genuinely-robust quantized instance
  would only ever come back `unknown`/`timeout`.
- **v2 multi-input counterexample format:** vibecheck emits the witness as flat
  `(X_0 …) … (Y_0 …)` over the concatenated inputs (X1 then X2, original-model order) +
  the ORT-CPU output. This is internally validated on the original model, but the exact
  format the official v2 multi-input scorer expects is unconfirmed.
- **onnx2torch dependency** (added to `pyproject.toml`): needed in the AWS vibecheck venv
  before this runs there.
