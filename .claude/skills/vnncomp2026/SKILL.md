---
name: vnncomp2026
description: Periodic campaign loop — beat α,β-CROWN on every VNNCOMP 2026 benchmark, one at a time (regular track then extended), with measurement-driven feature work, soundness gates, and per-benchmark merge deliverables. Invoke on each loop wakeup; checks running AWS work first and only continues when prior runs are done.
---

# VNNCOMP 2026 — beat α,β-CROWN on every benchmark

**Standing goal:** maximize benchmark instances solved within official timeouts on the VNNCOMP 2026 benchmark set, beating α,β-CROWN (ABC) on every instance. Work one benchmark at a time, regular track first (in the order below, skipping any already done — check `docs/benchmarks/*.md` and prior progress notes), then the extended track.

This is a periodic task: **on each wakeup, FIRST check whether sweeps/experiments from the previous iteration are still running on AWS (`ps aux | grep -E "vibe/bin/python|abc/bin/python" | grep -v grep` + `nvidia-smi`) — if anything is running, check its progress, report status to the user, and do NOT interfere or launch competing GPU work; only continue working toward the goal when the previous runs are done.** Never stall for days waiting: anything long-running gets launched detached with a results file and polled on later wakeups.

## Benchmarks (2026 determinations)

- **Regular (24):** acasxu_2023, cersyve, cgan2026 (updated), challenging_certified_training_2026 (new), cifar100_2024, collins_rul_cnn_2022, cora_2024, dist_shift_2023, linearizenn_2024, lsnc_relu, malbeware, metaroom_2023, ml4acopf_2024, nn4sys/1.0, relusplitter_2026 (updated), safenlp_2024, sat_relu, soundnessbench_2026 (updated), tinyimagenet_2024, tllverifybench_2023, traffic_signs_recognition_2023, vggnet16_2022/1.0, vit_2023, yolo_2023.
- **Extended (6):** adaptive_cruise_control_non_linear_2026/2.0, cctsdb_yolo_2023, collins_aerospace_benchmark, isomorphic_acasxu_2026/2.0, monotonic_acasxu_2026/2.0, smart_turn_multimodal_2026/2.0.
- Benchmarks marked "old" with an existing `docs/benchmarks/<name>.md` and matching config need only a **confirmation sweep** on the 2026 instance set; re-open them only if the instances changed or the sweep regresses. **Defer ALL confirmation sweeps of unchanged benchmarks until after every regular- and extended-track benchmark is solved** (user directive 2026-06-12; acasxu 186/186 already validated the 2026 harness) — verify file identity vs 2025 (md5) now, but spend GPU time only on updated/new benchmarks: cgan2026, challenging_certified_training_2026, lsnc_relu, ml4acopf_2024, relusplitter_2026, soundnessbench_2026, traffic_signs_recognition_2023, vggnet16_2022, then the extended six.

Benchmarks live at `~/repositories/vnncomp2026_benchmarks` locally. **AWS disk is tight (30 GB root EBS, no other usable storage — do NOT format/mount the NVMe instance store): copy ONE benchmark at a time** to `~/vnncomp2026_benchmarks/benchmarks/<name>/` on the box, and delete it there after the benchmark is merged to main. **Keep files UNCOMPRESSED on the box, no `.gz` duplicates** (user directive 2026-06-12: repeated gunzip is wasteful — disk is the cheaper resource): when copying a benchmark, send the uncompressed versions and skip `.gz` files that have an uncompressed sibling. The 2025 set on the box is deduped this way (uncompressed-only). If space runs out for a big benchmark, free the same benchmark's dir inside `~/vnncomp2025_benchmarks` first — its content is byte-identical to the 2026 `1.0/` dir (verified by checksum 2026-06-12 for all old benchmarks except nn4sys, whose mscn_2048d_dual.onnx + 17 cardinality vnnlib changed). Measurements run on AWS (see CLAUDE.md for hosts, idle_since discipline, keepalive, verdict-file rules).

## VNNLIB parsing

Use the common parser at `~/repositories/vnncomp2026_benchmarks/scripts/vnnlib_parser.py` (`parse_vnnlib(path)` auto-detects v1/v2, handles .gz, returns `VnnlibProperty → DisjunctiveSpec → ConjunctiveSpec → PolynomialConstraint` over flattened X_i/Y_i). v1 and v2 were cross-checked to parse identically on all 23,794 dual-version properties — **when both exist, use 2.0**. Wire vibecheck to consume this object model (or verify its existing loader agrees with it on v1).

Known upstream benchmark bugs — work around, note in the README, don't burn time: unconverted v1 files inside 2.0/ dirs (safenlp 20,544/21,624; malbeware 325/375; linearizenn 60/120; cora 39/99; cersyve 6/9), malformed s-exprs in linearizenn `*_io`, invalid infix assert in all monotonic_acasxu files, contradictory `and` (likely meant `or`) making all isomorphic_acasxu trivially UNSAT, broken instances.csv paths in monotonic_acasxu (nonexistent `onnx/original/`) and cgan small_transformer.

## Per-benchmark procedure

1. **Branch** `bench/<name>_2026` from main. Get the ABC reference: their config for this benchmark if one exists; otherwise find their config for the **closest model family** and adapt — same for vibecheck's starting config. Establish ABC's per-instance verdicts/times on AWS (published results if available, otherwise run ABC). **Probe before sweeping (user directive 2026-06-12): NEVER open a benchmark with a full multi-hour sweep.** Start with a handful of instances (e.g. one short-timeout case per model, ~5-10 min total), sanity-check per-case outputs (verdict files exist, timings plausible, no config/loader errors), and let probe results steer which instances to run next. Full sweeps come later — typically only the final clean sweep and the soundness gate; in between, run the smallest instance subset that answers the current question.
2. **Settings first:** try to beat ABC using existing vibecheck machinery via `configs/<name>.yaml` only. Sweep, compare per-instance (match by onnx+vnnlib pair, never basename).
3. **If layer types are unsupported:** implement soundly (symbolic/provably-sound bounds only, never sampling; every op dispatch `else: raise`). **Reuse existing code; prefer emitting canonical ops over new handler types. Simplify code where possible** — but scan `docs/benchmarks/*.md` first to see which benchmarks use the feature you're touching, and re-sweep those benchmarks if you change shared code. Every new feature is gated by tests (implemented = tested).
4. **When stuck — measure, don't speculate:** take the **fastest instance ABC solves that we miss**, and microbenchmark both tools on it: ABC phase-by-phase (build / initial bounds / α / BaB domains, instrumented if needed) vs our phase-by-phase, aligned on wall-time with bound values. Run ABC **feature ablations** (toggle one config knob per run) on the separating instances to find what's load-bearing; implement what flips verdicts and we lack. Localize bound divergence per-node (transplant/width-comparison) before writing code. Iterate. If 1–2 attack angles fail on a case, surface a diagnostic to the user rather than spinning.
5. **Soundness gate (mandatory before merging):** full sweep with all counterexample search disabled (`pgd_restarts: 0`, `parallel_pgd_enabled: false`, `--disable-sat-finding`). Every verdict must be unsat/timeout — any SAT case "proven" UNSAT is a stop-everything soundness bug. Also audit all sweeps for `sat` verdicts and re-validate each witness by point propagation.
6. **Finish:** one clean full sweep with the final code+config (re-run, not assembled from partial sweeps; verdicts ONLY from `--results-file`). Deliverables: `configs/<name>.yaml` (only overrides, commented why), `docs/benchmarks/<name>.md` (scores vc-vs-ABC with sweep id, per-instance wins/misses, key methods implemented, knobs and why, repro commands, **key unresolved issues explained**), integration tests (~3 pinned cases incl. 1 SAT if cracked), unit suite green locally and on AWS.
7. **Git:** commit and push to the feature branch as you make progress; when the benchmark is done, **merge to main keeping history** (no squash; fast-forward or merge commit), push, keep the branch. Before merging, verify the full prior integration suite still passes — don't regress other benchmarks.
8. Move to the next benchmark.

## Sweep economy

**Do NOT run full cross-benchmark sweeps after each benchmark.** Two campaign-wide full sweeps only: once after ALL regular-track benchmarks are done, and again after the extended track is added. In between, scope re-sweeps by impact: when you change shared code (or fix a soundness bug), scan `docs/benchmarks/*.md` to identify which benchmarks use the touched feature and re-sweep only those (plus their integration pins).

## Soundness bugs — highest priority

A soundness bug (any SAT case "proven" UNSAT, any silently dropped op/bias, any sampling-based bound) preempts everything: stop the current optimization work, root-cause it, fix it with a pinned test, then sweep the likely-affected benchmarks (identified via the READMEs as above) before resuming the campaign.

## When all benchmarks beat ABC

Switch to the **performance phase:** find instances where we solve but are clearly slower than ABC (compare per-instance times), profile, optimize, re-sweep to confirm no verdict regressions.

## Operational rules (hard-learned — follow them)

- Verdicts come from `--results-file` contents only; never exit codes, never stdout. Refuse to count cases with missing/odd verdict files. Suspicious uniform timings = investigate before reporting.
- **All smoke tests and experiments run on AWS, never local** (user directive 2026-06-12) — local has no usable GPU headroom and hosts the Claude session. If the AWS box is down/unreachable, tell the user and wait for them to bring it up rather than falling back to local.
- **One GPU job at a time.** Timing-sensitive verdicts (anything near the timeout) are invalid under contention — check `nvidia-smi` + `ps` before and after, and re-run contaminated probes. Running close to the timeout is fine and expected for exploration.
- Detached jobs: `nohup setsid`, results to `~/persistent_runs/`, done-flag files. Beware self-matching process greps (a `bash -c` wait-loop whose command string contains the pattern waits on itself — use `pgrep -x` or distinct flag files). Kill only specific PIDs you started.
- Keepalive the AWS box while work runs (bounded loops, e.g. ≤6h); kill it when done so idle-shutdown stops billing. Refresh `/tmp/idle_since` on every ssh.
- Causal claims about ABC require a measurement, source reference with line numbers, or an ablation — never inference. Measured dead ends get written down (memory + findings doc) so they aren't retried.
- **Report progress to the user periodically:** within the current benchmark (phase, current scores vs ABC, what's running) and across the campaign (benchmarks done / remaining). Report failures plainly.
- Never `git commit`/`merge`/`push` beyond the scope above; soundness regressions or destructive/irreversible actions stop the loop and ask the user.
