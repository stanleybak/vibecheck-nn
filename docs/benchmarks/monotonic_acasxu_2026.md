# monotonic_acasxu_2026

Extended-track, **2.0/** (v2 vnnlib). 50 instances, 100 s budget. **Monotonicity** of an ACAS Xu
controller: along a designated input coordinate, an output must move monotonically; a
counterexample is a pair of inputs that violates it. vibecheck **matches α,β-CROWN: 50/50 sat**
(all real violations), ~3 s each.

## The benchmark

- **Pair format:** the net field is `[('f', net), ('g', net)]` with **f = g (the same ACAS Xu
  net)**. `main._maybe_network_pair` merges it into one ONNX (`network_pair.py`) that evaluates
  the net at two correlated input points: `x_g = clamp(x_f − δ·e_k, LO, HI)` with `δ ≥ 0` on the
  monotone coordinate `k` (the other coordinates equal). The merge is onnxruntime-oracle-gated.
- **Spec:** input box on `x_f` + `x_g[k] ≤ x_f[k]`, output e.g. `Y_f[3] < Y_g[3]` — i.e.
  decreasing input `k` should not increase output 3. A `sat` witness is an `(x_f, x_g)` pair
  where it does (monotonicity broken). The authoritative spec is the **`.gz`** (the loose
  `instance_*.vnnlib` may be a stale sibling; `network_pair` reads `.gz` first).

## ABC reference

α,β-CROWN: **50/50 sat** (`vnncomp2026_results/alpha_beta_crown/2026_monotonic_acasxu_2026`).
These are shallow, real violations — found quickly by CE search; no per-benchmark config needed
(default `input_split_small` profile + Phase-0/leaf PGD).

## Results (vibecheck vs ABC)

| | vibecheck | α,β-CROWN |
|---|---|---|
| sat | **50 / 50** | 50 / 50 |
| wall | ~3 s each | — |

Per-instance: identical (all sat). 0 misses, 0 conflicts. Each witness is the merged net's
`(x_f, x_g)` input pair; the strict `<` boundary is handled by the sound non-strict-closure
encoding + PGD/ORT confirmation (`network_pair._parse_output_atoms`).

## Well-formed + CLEAR counterexamples

Two emit-side fixes make the sat verdicts robust:

- **Per-network cex** (`network_pair.reconstruct_pair_cex`): the merged net collapses I/O, so a
  naive cex is malformed for the v2 spec (which declares X_f,Y_f,X_g,Y_g interleaved and
  validates BY ORDER). The witness is mapped back to the original f,g tensors and emitted in
  declaration order (`main._cex_v2` order arg).
- **Strict CE-check on the ORIGINAL nets** (`verify_graph._try_clear_ce_upgrade` /
  `_orig_pair_or_self_margin`, `clear_ce_upgrade_budget`=8 s). The property is **strict**:
  `(< Y_f[3] Y_g[3])`. The verifier bound uses the non-strict CLOSURE `<=` (sound for UNSAT — no
  point in the closure ⟹ no strict CE), but a COUNTEREXAMPLE must violate the strict `<`. With
  `f = g` the trivial diagonal `δ = 0` gives `x_f == x_g` → `Y_diff` **exactly 0**, which the
  merged closure spec accepts but the strict scorer does NOT. So the CE-check is done on the
  **original** f,g (the nets the scorer replays — the merged net is only oracle-faithful to
  `ORACLE_TOL`=1e-3) and requires a STRICT violation (`Y_diff < −atol`):
  - leaf settles on the diagonal → near-boundary on the originals → bounded margin-minimizing PGD
    (`pgd_via_onnx(accept_margin=−atol)`) finds a CLEAR CE (`Y_diff` clearly < 0) → emit that;
  - if no strict CE exists for a strict pair, the witness is **not** a valid counterexample →
    downgrade to `unknown` (never emit a non-strict sat). For monotonic clear CEs exist
    everywhere, so this never fires — all 50 emit strict CEs.

  AWS check (the cex carries Y from the ORIGINAL nets, which is what the scorer replays):
  `Y_f[3]−Y_g[3]` ∈ [−0.14, −5.6e−4] across instances — **every one strictly < 0** (smallest
  margin −5.6e−4, still a genuine strict violation, no output tolerance). Branch-history: the
  original `monotonic_ce_fix` upgrade keyed off the now-removed `keep_searching_within_tol`;
  this is the 2026 output-strict re-port with the CE-check moved onto the original nets.

## Reproduce

```bash
.venv/bin/python -m vibecheck.main \
  --net "[('f','onnx/ACASXU_run2a_2_2_batch_2000.onnx'),('g','onnx/ACASXU_run2a_2_2_batch_2000.onnx')]" \
  --spec vnnlib/instance_0.vnnlib --timeout 100 --results-file out.txt
# (run with cwd = the benchmark 2.0/ dir so the relative onnx path resolves)
```

Integration pin: `tests/integration/test_monotonic_acasxu_2026.py`.

## Key unresolved issues

- None — VC matches ABC at 50/50.
