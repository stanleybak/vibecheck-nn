# Benchmark campaign — overview & operations

Campaign-level process and infrastructure for the per-benchmark optimization work.
Per-benchmark records (scores, wins, knobs, repro) live in the sibling
`docs/benchmarks/<name>.md` files; current 2026 benchmark lists, status, and known
upstream-benchmark bugs are in the `vnncomp2026` skill (`.claude/skills/vnncomp2026/SKILL.md`).

## Workflow

Each VNNCOMP benchmark is optimized on its own `bench/<name>` branch, then merged to `main`;
goal is to beat α,β-CROWN per instance.

Per benchmark:

1. Create `configs/<name>.yaml` — ONLY the overrides on top of `configs/default.yaml` (keys map
   1:1 to `Settings` attrs).
2. Optimize on a remote GPU side-by-side with α,β-CROWN; cross-check against published reference
   results. When stuck, **measure** — take the fastest instance ABC solves that we miss and
   microbenchmark both tools phase-by-phase / ablate ABC config knobs — don't speculate.
3. Add `tests/integration/test_<name>.py` — ~3 pinned cases (1 SAT if cracked + 2 hard UNSAT),
   each with a `max_wall_s` ~1.5× observed. Every merge re-runs all prior benchmarks' cases.
4. Write `docs/benchmarks/<name>.md` — the canonical per-benchmark record: scores vs ABC (with
   sweep id + timestamp), algorithmic wins, benchmark-specific knobs + *why*, repro commands
   (single case + full sweep), integration cases, known unsolved cases.
5. **Soundness gate before merging:** a full sweep with counterexample search off
   (`--disable-sat-finding`, `pgd_restarts: 0`, `parallel_pgd_enabled: false`). Every verdict
   must be unsat/timeout — any SAT case "proven" UNSAT is a stop-everything soundness bug.
   Also re-validate every `sat` witness by point propagation.
6. Present a pre-merge gap report (unsolved cases, visible ABC wins, score delta) before merging.

When implementing a multi-phase plan, push through every phase; only stop on a genuinely failed
gate (revert that one ablation, then continue) or a destructive action needing authorization.

Allowed reference: read α,β-CROWN / auto_LiRPA source or run them with debug prints (especially
for non-ReLU activations — tanh, sigmoid, GELU, MHA), then re-implement.

### Sweep economy

Don't run full cross-benchmark sweeps after each benchmark. When you change shared code (or fix
a soundness bug), scan `docs/benchmarks/*.md` to find which benchmarks use the touched feature
and re-sweep only those (plus their integration pins). Save campaign-wide full sweeps for major
milestones (e.g. after all regular-track benchmarks, after the extended track is added).

## Remote GPU (AWS only)

ALL pytest runs and any heavy/long/GPU-scaled work run on **AWS — never local**, since local
hosts the tmux session Claude runs in and an OOM there kills the session (lost context, mid-flight
work). Do not run test suites locally, not even quick smoke runs. **If AWS is unreachable (box
down, or `$AWS_GPU_HOST`/`$AWS_GPU_PEM` unset in the shell), ASK the user — never fall back to
running locally.**

- **AWS g5** — `ssh -i "$AWS_GPU_PEM" "$AWS_GPU_HOST"` (A10G / 24 GB). Connection details in env
  vars (kept out of git); runbook in `AWS_SETUP.txt`. The user starts/stops via the AWS console
  — Claude only SSHes in. Vibecheck checkout at `~/vibecheck` (`~/vibe/bin/python`); α,β-CROWN
  per `AWS_SETUP.txt`.
  - **Every ssh must `sudo rm -f /tmp/idle_since`** — batch `ssh host 'cmd'` calls don't register
    in `who`, so they accrue idle seconds even while Claude is actively working; the 5-min
    idle-shutdown (`/usr/local/bin/idle-shutdown.sh`, fires when GPU<5% AND no interactive ssh)
    would otherwise stop the box mid-task. Bake the `rm` into the command:
    `ssh -i "$AWS_GPU_PEM" "$AWS_GPU_HOST" 'sudo rm -f /tmp/idle_since; <real command>'`.
  - **Idle-shutdown protocol:** if the user's been idle 31+ min while an AWS sweep might be
    holding the box up, SSH in, check `nvidia-smi` + processes, and if nothing useful is running
    `sudo shutdown -h now` to stop the ~$1/hr billing. Don't shut down if a justifying long
    sweep is producing results.

The 2026 benchmark set lives locally and is rsync'd to AWS before first use (see the
`vnncomp2026` skill). Sync vibecheck with
`rsync -az --exclude '.venv' --exclude '__pycache__' <local-repo>/ -e "ssh -i \"$AWS_GPU_PEM\"" "$AWS_GPU_HOST":~/vibecheck/`
(re-run `pip install -e .` on AWS if `pyproject.toml` changed).

### Run discipline

- **One GPU job at a time.** Timing-sensitive verdicts (anything near the timeout) are invalid
  under contention — check `nvidia-smi` + `ps` before and after, re-run contaminated probes.
- **Detached runs:** `nohup setsid`, results to `~/persistent_runs/` (survives reboot, unlike
  `/tmp`), with a done-flag file. Beware self-matching process greps. Kill only specific PIDs
  you started.
- **Never keep anything in the remote `/tmp`.** A stop/start (idle-shutdown, manual stop) **wipes
  `/tmp` and changes the public IP**. Diagnostic scripts, ABC oracles, and results all vanish.
  Keep diagnostic scripts in the local repo (e.g. `scratch/`) and `rsync` them up so they can be
  re-pushed after any restart; write results/oracles to `~/persistent_runs/`. (Learned the hard
  way 2026-06: an idle-shutdown mid-session wiped the harness + ABC oracle and rotated the IP.)
- **Cache `details` to `~/persistent_runs/`.** When running an experiment for the user, pickle the
  returned `details` dict to `~/persistent_runs/vibecheck_runs/{slug}.pkl` (include instance id +
  config in the slug) so re-views of the same run ("Phase 7 timing?", "unstable count at L3?")
  don't re-run a 60 s benchmark. Say so when answering from cache; re-run + overwrite for new
  instance/settings/code.
