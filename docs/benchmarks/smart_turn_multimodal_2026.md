# smart_turn_multimodal_2026

Extended-track, **2.0/ (v2 vnnlib)**. 50 instances, one model. Solved via the
**surrogate-attack** mode (incomplete / attack-only): vibecheck reports **50/50 sat**
(clear or within-tolerance), each in ≤ ~70 s of the 100 s budget. **α,β-CROWN produces no
result** (can't load the quantized ops), so this is a clean win.

## The benchmark

- **Model:** `smart-turn-multimodal-cpu.onnx` — a **41.6 M-param, INT8-quantized**
  multimodal (audio + video) transformer (turn-detection). 692 nodes: Conv/MatMul/Gemm,
  Softmax, Erf (GELU), LayerNorm, Sigmoid head; **199 DequantizeLinear + 119
  QuantizeLinear** (activations are **uint8**, weights int8, biases int32).
- **Inputs:** `input_features` audio `[1,80,800]` (64 000) + `pixel_values` video
  `[1,3,32,112,112]` (1 204 224) → **≈ 1.27 M input dims**. **Output:** `Y [1,1]`
  (post-sigmoid). The vnnlib is ~124 MB (1.27 M box constraints).
- **Spec (all 50):** L∞ robustness, output violation `(> Y[0,0] 0.5)`.

**Key structural fact:** the quantized model is **piecewise-constant over each instance's
L∞ box** (the box sits inside one input quantization cell — center = lo-corner = hi-corner).
So PGD *inside* the box can't move the output: **the center value is the verdict.** Across
instances the center is either `Y=0.918` (a CLEAR CE, `> 0.5`) or `Y=0.5` (the quantized
logit pinned at 0 → within `sat_validate_atol`=1e-4 of violating → a WITHIN-TOLERANCE CE the
VNNCOMP scorer accepts as CORRECT_UP_TO_TOLERANCE).

## ABC reference

α,β-CROWN 2025 cannot load it (`onnx2pytorch` has no `DequantizeLinear` converter), so it
has **no smart_turn results** in our 2026 report — a loading wall, not difficulty. (Complete
L∞ verification over 1.27 M dims through a 41 M-param transformer is intractable anyway.)

## vibecheck approach — surrogate-attack mode (`src/vibecheck/surrogate_pgd.py`)

Incomplete / attack-only (never returns unsat). The verdict is decided **only** by replaying
the witness on the ORIGINAL quantized model via CPU onnxruntime (the scoring engine):

1. **Two folded surrogates** (built untimed in `--prepare-pkl`, which folds surrogates for
   quantized nets): the **float (STE)
   surrogate** (activation Q/DQ → Identity, differentiable — the gradient oracle) and the
   **fake-quant surrogate** (activation Q/DQ → `Round`+`Clip`, reproduces the INT8 rounding —
   a fast GPU eval oracle to rank candidates).
2. **Search** (`device: gpu`): ORT-confirm the **center**; then PGD restarts seeded from the
   center then seeded random in-box points, with **gradual clamped L∞ steps**
   (`surrogate_alphas=[0.05,0.1,0.2,0.02]`, no single jump-to-vertex). The fake-quant eval
   gates the (slower) ORT confirm. **No box-corner enumeration** (1.27 M dims).
3. **Within-tolerance disposition** (CLAUDE policy, `keep_searching_within_tol=True`): a
   CLEAR CE (`Y>0.5`) returns immediately; a within-tol CE (`Y≈0.5`) is stashed while the
   search keeps looking for a clear one, and emitted only if none is found.
4. **v2 counterexample format** (`counterexample_format: auto`): per-tensor
   `NAME float32 [shape]` + C-order values (the VNNLIB 2.0 cex format), matching the input
   spec's version.

Config: `configs/smart_turn_multimodal_2026.yaml` (`surrogate_attack: true`,
`surrogate_attack_restarts: 4`, `surrogate_attack_steps: 30`, `device: gpu`).

## Results — 50/50 sat (sweep `smart_turn_sweep2`, A10G, 2026-06-21)

All 50 `sat`, avg 69.6 s / max 70.5 s (well under the 100 s timeout). Witnesses
externally re-validated on the **official-env ORT (onnxruntime 1.26.0 / onnx 1.21.0)**:
every sampled witness scores **CORRECT** (`Y=0.918`) or **CORRECT_UP_TO_TOLERANCE**
(`Y=0.5`) — zero rejections.

## Platform-float note (important)

The quantized output is sensitive at the decision boundary: for the same model+input+ORT
version, a **different CPU gives a different output** (e.g. instance_3 center: AWS A10G host
→ `0.5`, a different dev CPU → `0.918`) — float non-associativity in the deep conv/matmul,
amplified by INT8 quantization into a ±1-code flip. The **within-tol center witness is
accepted on every platform** (Y ≥ 0.5−atol on both), so it's the robust choice, and the
attack runs the ORT confirm on the same host it will be scored on (self-consistent). The
fake-quant eval has the same sub-ULP sensitivity (≈3/61 boundary points vs ORT) — it's only
a *ranking* oracle; ORT-CPU is authoritative, and its gate is conservative (never skips a
near-boundary candidate), so the imprecision can't cause a missed CE or a false sat. This
divergence is irreducible: truncating/simplifying the graph to isolate it changes ORT's
operator fusion and the flip vanishes (the fold math itself matches ORT exactly on clean
ties: 0/12, and over 12 800 random accumulated points: 0/12 800, bit-identical local & AWS).

## Reproduce

```bash
# prepare (untimed): folds both surrogates for the quantized net
.venv/bin/python -m vibecheck.main --net  .../2.0/onnx/smart-turn-multimodal-cpu.onnx \
  --spec .../2.0/vnnlib/instance_0.vnnlib --prepare-pkl
# one instance
.venv/bin/python -m vibecheck.main \
  --net  .../2.0/onnx/smart-turn-multimodal-cpu.onnx \
  --spec .../2.0/vnnlib/instance_0.vnnlib \
  --config configs/smart_turn_multimodal_2026.yaml --timeout 100 \
  --results-file /tmp/out.txt        # -> sat + a v2 counterexample
```

## Key unresolved issues

- **Incomplete:** never returns `unsat`. Empirically all-sat here, so not a practical limit,
  but a genuinely-robust quantized instance would only come back `unknown`/`timeout`.
- **onnx2torch dependency** (in `pyproject.toml`): required in the AWS venv for the surrogate
  convert. The box's `onnx` package is 1.20.1 vs the rules' 1.21.0 — immaterial to ORT
  inference (verified: same result both versions on one host), but worth aligning.
