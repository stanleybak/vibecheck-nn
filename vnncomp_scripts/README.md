# vibecheck VNNCOMP scripts

The VNNCOMP toolkit submission interface for vibecheck. Submit this directory
(`vnncomp_scripts`) as the **scripts directory** in the submission form, at the
commit you want evaluated. The backend clones the repo, `cd`s here, and runs the
three required scripts (each takes a leading `v1` version string).

## AMI and GPU driver

Recommended AMI: **Ubuntu Server 24.04 LTS**. Its glibc is new enough for the
torch wheels (avoid 18.04, which is EOL and too old), and it is a clean, current
base image.

The 24.04 base image does **not** ship an NVIDIA driver, and `install_tool.sh`
does not install one. vibecheck's configs assume a CUDA GPU, so install the
driver via the post-install hook:

1. In the submission form, set **post-install script = `vnncomp_scripts/post_install.sh`**.
   It installs the newest headless `-server` NVIDIA driver (idempotent: a working
   driver short-circuits).
2. Enable **"restart after post-install"**. Installing the driver requires a
   reboot for the kernel module to load; without the restart, `nvidia-smi` (and
   the timed runs) will not see the GPU. After the reboot the driver is loaded
   and benchmark execution proceeds on GPU.
3. If the image lacks passwordless sudo for the apt steps, also enable **"run
   installation as root"**.

If your chosen AMI already has the driver (a Deep Learning image, or a VNNCOMP
GPU image), `post_install.sh` detects it and does nothing, and the restart toggle
is unnecessary.

## Submission form summary

| Field | Value |
| --- | --- |
| scripts directory | `vnncomp_scripts` |
| version string | `v1` |
| AMI | Ubuntu Server 24.04 LTS |
| GPU instance | yes |
| post-install script | `vnncomp_scripts/post_install.sh` (driver install on a base AMI) |
| restart after post-install | yes (so the driver module loads) |
| run install as root | only if the image lacks passwordless sudo |

## Scripts

- **`install_tool.sh v1`** - builds a self-contained `.venv` at the repo root via
  the [uv](https://docs.astral.sh/uv/) package manager (Python 3.12 + torch +
  gurobipy), then verifies the env (prints python/torch/gurobipy versions and
  imports vibecheck). Does not install GPU drivers.
- **`prepare_instance.sh v1 <category> <onnx> <vnnlib>`** - runs ONCE per
  instance, untimed: narrow zombie cleanup, a GPU sanity check that raises a loud
  alarm if no usable CUDA GPU is found (see below), a pre-parse `.pkl` cache, and
  a short warmup.
- **`run_instance.sh v1 <category> <onnx> <vnnlib> <results_file> <timeout>`** -
  the timed run. Writes the authoritative verdict to `<results_file>`: line 1 is
  `unsat` / `sat` / `unknown` / `timeout`; for `sat`, the counterexample
  s-expression follows. Always exits 0 so the harness reads the verdict file
  rather than vibecheck's exit code.
- **`post_install.sh`** - optional; installs the NVIDIA driver on a base AMI
  (see above).

Each script brackets its stdout with banner lines (visual dividers in the
captured log that also serve as parse anchors):

```
[vibecheck:<script>] BEGIN <key=val ...>
[vibecheck:<script>] END   verdict=... elapsed=...s ...
```

vibecheck runs `--verbose` by default (per-phase progress, line-buffered so a
timed-out/SIGKILL'd run still flushes its trace instead of leaving an empty log).
The verdict in the run END banner comes from `<results_file>` and is the
authoritative one.

### GPU alarm

`prepare_instance.sh` checks that `nvidia-smi` exists, actually runs (driver
loaded, not just installed-pending-reboot), and that `torch.cuda.is_available()`
is true. If any fails it prints a loud `GPU ALARM` block, since the configs
assume a GPU and CPU runs will time out. Set `VIBECHECK_REQUIRE_GPU=1` to turn
the alarm into a hard prepare failure instead of a warning.

## Environment variables

| Var | Effect |
| --- | --- |
| `VNNCOMP_PYTHON_PATH` | dir of the python to use (default: the repo `.venv/bin`) |
| `VIBECHECK_PKL_CACHE_DIR` | pre-parse cache location (default `/tmp/vibecheck_pkl`) |
| `VIBECHECK_QUIET=1` | drop `--verbose` (quiet runs) |
| `VIBECHECK_HEARTBEAT=N` | emit a `[heartbeat]` line every N seconds so a stalled phase is visible (localizes a hang before the timeout kill) |
| `VIBECHECK_GPU_RESET=1` | GPU power-mode reset in prepare (needs sudo) |
| `VIBECHECK_REQUIRE_GPU=1` | make the prepare GPU alarm a hard failure |

## Gurobi license

`gurobipy` is a normal dependency; the pip wheel ships a size-limited license
that suffices for the regular-track benchmarks. A full license is only needed for
very large models, and the right place to activate it on the instance is
`post_install.sh` (not `install_tool.sh`).

## Local helpers (not used by the competition harness)

- **`run_benchmarks.py <category|all|regular> [--bench-version 1.0]`** - replays
  prepare+run over a benchmark's instances.csv into a results.csv. Handles the
  2026 versioned layout (`benchmarks/<cat>/<version>/`); `--bench-version`
  selects 1.0 (v1 specs; default) or 2.0 (v2; not yet parseable). `--log-dir DIR`
  saves each instance's stdout for inspection.
- **`parse_log.py LOG|DIR`** (or pipe a log on stdin) - summarizes captured run
  logs: verdict, timing, net/spec shape, and (for a killed/timed-out run) the
  last heartbeat phase it hung in.
