# CLAUDE.md

Guidance for Claude Code in this repository (branch `clean-slate`).

## Current goal

Build **vibecheck2** (`src/vibecheck2/`): a clean-slate reimplementation of the
verifier per `docs/clean_slate_design.md`. The v1 tree (`src/vibecheck/`) stays
untouched as the reference implementation, verdict oracle, and reused front end
(ONNX loader, VNNLIB parser, CE validation chokepoint).

**Priorities, in order: soundness > clean design > small codebase > speed.**

## Working style

- Work benchmark categories one by one against the parity matrix
  (`scratch/clean_slate/named_cases_2026.txt`); do not accept misses vs what v1
  solves. 60s dev timeouts; re-check near-misses at the official budget.
- When stuck, read how v1 solved it, then update the design and implement it
  cleanly. Never patch v1's solution in sideways.
- One implementation per concept: all affine ops through `LinMap`, all
  nonlinearities through `RelaxLib`, one forward and one backward propagator,
  one attack engine, one BaB, one memory service. Weird benchmarks get
  `handlers/` strategies on top of the core, never special cases inside it.
- Back causal claims with a measurement, a code reference (`file:line`), or an
  observed log line, not "probably".
- **Never live-edit `src/` while a sweep is running.** Sweep subprocesses
  import the tree per case; an edit window (even an uncommitted one) taints
  every verdict produced during it. Freeze a copy (`git archive` to a temp
  dir) for the sweep, or hold edits until it finishes. A mid-sweep unsound
  plane once silently invalidated ~200 sweep rows.

## What is vibecheck?

A neural network verifier: ONNX net + VNNLIB spec -> `unsat` (verified), `sat`
(counterexample), or `unknown`/`timeout`, via zonotope/CROWN abstract
interpretation, alpha/beta optimization, dual-ascent LP, PGD falsification,
and branch-and-bound.

## Hard rules (soundness / disaster prevention)

- **Use `~/repositories/vibecheck-nn/.venv/bin/python` or `.venv/bin/python`
  with `PYTHONPATH=src`** for all runs. **Never `git push`** without explicit
  per-push approval; committing locally as work lands is fine on this branch.
- **The verdict is the `--results-file` contents, never exit code or stdout.**
  Refuse to count a case whose file is missing or unexpected.
- **Nonlinear-op bounds must be closed-form / provably bracketing, never
  sampling-based.** Adversarial sampling VALIDATES planes in tests; it never
  defines them.
- **No silent exception swallowing.** Catch one specific type with a
  documented reason, or catch+log+re-raise. Op dispatches end in
  `else: raise NotImplementedError`, never a silent skip.
- **Never silently swallow OOM.** The only sanctioned CUDA-OOM catch lives in
  `vibecheck2/core/memory.py` (halve, log, re-raise at the floor).
- **Never broad-kill processes** (`pkill -f python` kills this session);
  match narrowly or kill specific PIDs.
- **Cap memory-uncertain runs:** `systemd-run --user --scope -p MemoryMax=8G`.
  The local GPU is an 8 GB laptop part on the machine hosting this session:
  small batches, one GPU job at a time.
- Every `sat` must survive the ORT-CPU replay chokepoint (input box within
  1e-4, output strictly violating) before it is emitted.

## Testing

```bash
# vibecheck2 unit tests (fast, no benchmark files)
PYTHONPATH=src taskset -c 0,1 ~/repositories/vibecheck-nn/.venv/bin/python -m pytest tests2/ -q
# v1 suite (regression safety when touching shared v1 code)
taskset -c 0,1 .venv/bin/python -m pytest tests/ -k "not vnncomp" -m "not integration" -q
# point-prop / verdict parity vs ORT and the results matrix: scripts in scratch/clean_slate/
```

Benchmarks live at `~/repositories/vnncomp2026_benchmarks` (see
`tests/paths.yaml`); parity targets are
`~/repositories/vnncomp2026_results_official/{vibecheck,alpha_beta_crown}/results.csv`.
