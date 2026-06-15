# relusplitter_2026

VNNCOMP 2026 **regular-track** benchmark. MNIST `Gemm`+`ReLU` MLPs in a base
form plus 4 ReLU-split variants, 120 instances, 180 s per-instance budget.

## Final score

| verifier | solved | unsat | sat | timeout | crash/unknown | source |
|---|---|---|---|---|---|---|
| **vibecheck** | **120/120** | 108 | 12 | 0 | 0 | AWS A10G, 2.0 (v2) vnnlib, `--results-file`; `vc_sweep.jsonl` (2026-06-15) |
| AB-CROWN | **87 / 111 run** | 77 | 10 | 5 | **19 crash** | AWS A10G, `vnncomp25/relusplitter.yaml`, **1.0 (v1) vnnlib**; `abc_sweep.jsonl` |

**vibecheck beats AB-CROWN on this benchmark.** VC solves all 120 with default
settings (no `configs/relusplitter_2026.yaml`). AB-CROWN, run with its own
official relusplitter config, **crashes on 19 of the 111 instances it ran** —
all of them ReLU-split nets — and the sweep was stopped at 111/120. Where both
produce a definitive verdict, **they agree on all 87 (0 disagreements)**; VC
additionally solves the 19 ABC crashes, the 5 ABC timeouts, and the 9 unrun.

Notes: ABC can't parse the v2 vnnlib at all (`read_vnnlib.py` asserts on the
`(vnnlib-version <2.0>)` header), so ABC was run on the **1.0** (v1) vnnlib —
byte-identical onnx, semantically-identical spec (proven by the equivalence
oracle). VC unsat wall ≤ ~50 s; ABC unsat wall mean 36.8 s, max 183 s.

## Algorithmic win — VC's fold is robust where AB-CROWN's crashes

The benchmark ships each MLP as a *base* net plus 4 *ReLU-split* variants
(`pct0.2`..`pct1.0`): a split inserts a
`Gemm(C→C+S) → ReLU → Gemm(C+S→C) → ReLU` block whose `S` extra neurons are
±-paired rows (`w, −w` with biases `b, −b`) recombined by a ±1 selector.

Both tools fold this away via the `ReLU(z) − ReLU(−z) = z` identity:

- **vibecheck** — `fold_gemm` (`onnx_optimizer.py`, auto-applied via default
  `optimize_relu_relation=True`) collapses the block to a single
  `Gemm(C→C) → ReLU`. Confirmed: pct0.2/pct1.0 nets collapse to the **exact
  base layer sizes** `[(128,784),(256,128),(256,256)×3,(10,256)]`, so a split
  net verifies on the same (smaller) graph as its base.
- **AB-CROWN** — auto_LiRPA's `optimize_relu_relation`
  (`auto_LiRPA/optimize_graph.py:216`, called unconditionally from
  `bound_general.py:151`) is the same fold. **But it has a bug**: on certain
  split configurations the bias-fusion line
  `b_new[dst] = bm[pairs[src]] + bs[src]` (`optimize_graph.py:330`) indexes a
  size-128 array at index 128 →
  `IndexError: index 128 is out of bounds for dimension 0 with size 128`,
  thrown during `BoundedModule` construction (before any verification). It
  fires only on split nets (base nets have nothing to fold — 0/all-base
  crashes, 19/all-split crashes), so AB-CROWN aborts with no verdict on ~17 %
  of the split instances.

So `fold_gemm` is not just parity with auto_LiRPA's pass — it is a **more
robust re-implementation**: VC collapses the splits correctly where ABC's fold
throws. This is the difference between VC's 120/120 and ABC's 87/111.

## v2 (tensor-indexed) vnnlib

The 2026 set ships **v2** vnnlib: a `(declare-network ...)` header + tensor
atoms (`X[0,i]`, `Y[0,i]`). vibecheck auto-detects the version
(`vnnlib_loader.detect_version`) and routes v2 through a ported recursive-
descent s-expr parser + adapter to the same `VNNSpec` the v1 regex path
produces. The equivalence oracle confirms v1 and v2 yield an **identical**
`VNNSpec` for all 20 relusplitter_2026 specs (run on AWS: 20/20 match). The
spec is the classic output-OR: `(or (and (>= Y[0,k] Y[0,label])) ...)` over the
non-label classes. (AB-CROWN, by contrast, only parses v1 — hence the v1 run.)

## Benchmark-specific knobs

**None.** Default settings solve 120/120, so there is no
`configs/relusplitter_2026.yaml`. (If a future regression makes a case miss,
add overrides there with measured justification.)

## Reproduction

Single instance (base + v2 vnnlib), on AWS:
```bash
B=~/vnncomp2026_benchmarks/benchmarks/relusplitter_2026/2.0
~/vibe/bin/python -m vibecheck.main --net $B/onnx/model_3_3.onnx \
  --spec $B/vnnlib/d3_eps_0.01_sample_9_label_0.vnnlib \
  --device gpu --timeout 180 --results-file /tmp/r.txt
cat /tmp/r.txt   # -> unsat
```
A SAT case (counterexample through the fold):
```bash
--net $B/onnx/model_1_1~d1_eps_0.02_sample_2_label_9~pct1.0~cnt128~seed850851855.onnx \
--spec $B/vnnlib/d1_eps_0.02_sample_2_label_9.vnnlib     # -> sat (~1.5s)
```
Full sweeps (verdicts strictly from `--results-file` / ABC's own results file):
```bash
# VC  : scratch/relusplitter_2026_vc_sweep.py  (RS_BENCH=…/2.0)
# ABC : scratch/relusplitter_2026_abc_sweep.py (RS_BENCH=…/1.0, ABC can't parse v2)
```

## Soundness gate

Full sweep with `--pgd-restarts 0 --disable-sat-finding` (no PGD, no
counterexample search): **108 UNSAT cases re-verify `unsat`** (sound, not
PGD-masked), and **all 12 real-SAT cases come back `timeout`, none `unsat`** →
**0 false-verifies**. Cross-check vs AB-CROWN: of ABC's 10 SAT verdicts, all 10
are also VC SAT; **there is no instance where ABC found a counterexample VC
missed** (0 ABC-sat / VC-unsat). VC additionally finds 2 SAT (the `pct0.4`
splits of the two SAT samples) that ABC **crashed** on rather than disagreeing.

## Integration tests

`tests/integration/test_relusplitter_2026.py` (`relusplitter_2026/2.0`):

1. **SAT** — `model_1_1 pct1.0` / `d1_eps_0.02_sample_2_label_9`: real
   counterexample found ~1.5 s **through the fold**. Pins fold + SAT + v2 loader.
2. **UNSAT, hard, high-pct** — `model_2_2 pct1.0` / `d2_eps_0.04_sample_14_label_7`:
   slowest instance (~50 s); pins the fold + perf-regression catcher.
3. **UNSAT, base** — `model_3_3` / `d3_eps_0.01_sample_9_label_0` (~1-3 s):
   base net (no fold), validates the v2-loader + basic verify path.

## Known unsolved cases

None — vibecheck solves 120/120 within the per-instance budget.
