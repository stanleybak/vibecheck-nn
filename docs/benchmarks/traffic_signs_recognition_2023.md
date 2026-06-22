# traffic_signs_recognition_2023

Regular-track, **1.0/** (v1 vnnlib; v2 is byte-equivalent). 45 instances, 480 s budget.
GTSRB **binarized neural networks** (BNNs): the "QConv" layers use `Sign` activations
(±1 weights/acts), which neither vibecheck nor α,β-CROWN can bound soundly through
onnx2pytorch. Both tools therefore run an **incomplete attack** (never proves unsat).
Solved via vibecheck's **Sign-BNN STE-PGD attack** mode (`src/vibecheck/sign_attack.py`).

## The benchmark

- **Models:** several GTSRB BNNs of increasing depth, e.g.
  `3_30_30_QConv_16_3_QConv_32_2_Dense_43_ep_30.onnx` (2 binarized conv blocks → dense-43)
  up to `3_48_48_QConv_32_5_MP_2_BN_..._Dense_256_BN_Dense_43`. Input is an **NHWC** image
  (`[1,30,30,3]` etc.), pixel values in **[0,255]** (no normalization in-graph). Each
  binarized block is `Conv → Sign → Add(bias) → Sign` (onnx2pytorch surfaces two `Sign`
  modules per block, named `.../Sign` and `.../Sign_1`).
- **Spec:** L∞ robustness on a single image at radius `eps ∈ {1,3,5,10,15}` (pixel units),
  output is a 43-class score; the property is the usual "argmax stays the true class", so a
  CE is any in-box image whose argmax flips. 9 images × 5 eps = 45 instances.

**Key structural fact:** `Sign` is a hard step — its true gradient is 0 almost everywhere, so
a verifier must either bound it (impossible to do tightly through onnx2pytorch) or attack it
with a surrogate gradient. Both vibecheck and ABC attack with a **straight-through estimator
(STE)**.

## ABC reference

α,β-CROWN runs its own STE-PGD and gets **43/45 sat, 2 timeouts** — both timeouts are the
**eps_1** (smallest radius) cases `model_30_idx_11379` and `model_30_idx_12375` (487 s each;
likely truly robust at eps_1, or a needle CE neither tool finds). ABC's sat cases solve in
~8 s each. (Reference: `vnncomp2026_results/alpha_beta_crown/2026_traffic_signs_recognition_2023`.)

## vibecheck approach — Sign-BNN STE-PGD (`src/vibecheck/sign_attack.py`)

Incomplete / attack-only. Engages only when `sign_attack: true` (config) AND the ONNX has
`Sign` ops. The verdict is decided **only** by replaying the witness on the ORIGINAL
(true-`Sign`) model via CPU onnxruntime (the scoring engine), so a mismatched surrogate can
never yield a false sat. Recipe:

1. **Forward = true `Sign`; backward = clipped STE** `grad[|x|≥eps]=0; grad/=eps`. The
   `Sign→Add→Sign` merge STEs only the **first** sign (the real pre-activation); the second
   is identity on {−1,+1} so it gets a **pass-through** backward (STE-ing it would zero the
   gradient — its input is always ≈±1 ≥ eps).
2. **Per-layer ADAPTIVE clip** `eps = frac · median(|pre-act|)`, computed per forward.
   Binarized-conv pre-activations of 0–255 pixels are **O(thousands)** after the first conv
   (median ≈ 643, max ≈ 2797) but **O(8)** after the second — orders of magnitude apart. A
   *fixed* eps zeros a whole layer's gradient and the PGD stalls at margin −1.0 (this was the
   idx_11379 eps_3 failure). The adaptive eps tracks each layer's own scale, so only the
   (~`frac`-)nearest-zero pre-acts — the *flippable* ones — carry gradient. Small fracs work;
   restart `r` uses `sign_attack_clip_fracs[r % len]` = `[0.05,0.2,0.1,0.02]`.
3. **Optimize the pre-softmax logits** (a trailing softmax saturates to one-hot → 0 gradient).
4. Adam + ExponentialLR, restart 0 = box center, restarts >0 = random box **vertices** (the
   L∞ CEs here are vertex-like — most pixels pinned to lo/hi), a gentle off-zero pre-act
   penalty to escape the plateau.

### The idx_11379 fix (measured)

idx_11379 eps_3 stalled at margin −1.0 with the original fixed eps∈{0.1,5}. ABC's stored
witness, replayed through our STE forward, reproduced its CE **exactly** (|ΔY|=0) — proving
it was an *optimization-reachability* gap, not a loss/soundness issue. The probe
(`scratch/probe_idx11379_pgd.py`) showed a fixed eps of 0.1/5 zeroed ~99.4 % of the
first-conv gradient; eps≈50 (or the scale-invariant adaptive frac=0.05) cracks it **from the
center in 10 steps** (0.7 s). The adaptive eps generalizes across the benchmark's model sizes.

## Results (vibecheck vs ABC)

Full sweep (45 instances, `configs/traffic_signs_recognition_2023.yaml`, A10G GPU; verdicts
from `--results-file`):

| | vibecheck | α,β-CROWN |
|---|---|---|
| solved (sat) | **42 / 45** | **43 / 45** |
| both-solved | 42 | 42 |
| VC-only wins | 0 | — |
| ABC-only wins (VC misses) | **1** | — |
| both-miss (ties) | 2 | 2 |

- **42 both-solved sat**: VC cracks every model_30/48/64 case at eps ∈ {3,5,10,15} and the
  easier eps_1 cases, each in ~5 s (vs ABC ~8 s) — the per-layer adaptive STE makes the attack
  fast and scale-robust.
- **2 both-miss (ties):** `model_30_idx_11379_eps_1` and `model_30_idx_12375_eps_1` — ABC also
  **times out** (487 s) on these, so they don't count against VC.
- **1 loss:** `model_48_idx_10645_eps_1` — VC=unknown, ABC=sat (10 s). ABC's stored witness is
  a **within-tolerance margin-0 class tie** (eps_1 is the smallest radius — a strict flip
  isn't achievable, only a tie), and our STE forward reproduces it exactly (|ΔY|=0), so it's
  an **optimization-reachability** gap, not a loss/soundness issue. Measured: general-loss
  adaptive-STE PGD makes **zero** progress (best ORT margin 0.9999 ≈ one-hot after 52
  restarts) — the deeper model_48 (MaxPool+BN, 4 Sign blocks) is robust to first-layer sign
  flips at radius 1, so the gradient can't reach the rare tie vertex. ABC finds it via its
  within-trajectory loose/tight eps rounds. Closing it would need that recipe (a measured dead
  end for the current adaptive-STE method); deferred per "debug, else move on".

## Reproduce

```bash
# one instance
.venv/bin/python -m vibecheck.main \
  --net  .../traffic_signs_recognition_2023/1.0/onnx/3_30_30_QConv_16_3_QConv_32_2_Dense_43_ep_30.onnx \
  --spec .../vnnlib/model_30_idx_11379_eps_3.00000.vnnlib \
  --timeout 480 --results-file out.txt --config configs/traffic_signs_recognition_2023.yaml
```

Integration pins: `tests/integration/test_traffic_signs_recognition_2023.py` (3 sat cases incl.
the idx_11379 eps_3 regression). Unit: `tests/test_sign_attack.py` (100 % coverage).

## Key unresolved issues

- **eps_1 (smallest radius):** the 2 cases ABC also times out on (`idx_11379`, `idx_12375`)
  are the hardest — a CE, if one exists, is a needle inside a width-2 box. <!-- EPS1_NOTE -->
- The attack is **incomplete**: it never proves robustness (unsat). For the GTSRB BNNs that is
  acceptable (no tool proves unsat here), but a sound BNN bound (e.g. exact MILP over the
  ±1 activations) would be needed to *certify* the robust instances.
