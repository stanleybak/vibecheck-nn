# vibecheck VNNCOMP scripts

The VNNCOMP toolkit submission interface for vibecheck. Submit this directory
(`vnncomp_scripts`) as the **scripts directory** in the submission form, at the
commit you want evaluated. The backend clones the repo, `cd`s here, and runs the
three required scripts (each takes a leading `v1` version string).

## AMI and GPU driver

Recommended AMI: **Ubuntu Server 24.04 LTS** (its glibc is new enough for the
torch wheels; avoid 18.04, which is EOL and too old).

The 24.04 base image ships without an NVIDIA driver. `install_tool.sh` installs
one itself: a pinned `.run` driver (`580.159.03` by default, which supports both
the vibecheck torch-cu130 venv and abc's torch-cu128) built with dkms and loaded
in place via `rmmod`/`modprobe`, so **no reboot is required**. No CUDA
toolkit/nvcc is installed: torch bundles its own CUDA runtime.

The driver step is gated: by default it installs only when `nvidia-smi` is not
already working (so a Deep Learning / pre-driver image is left untouched).
`VIBECHECK_INSTALL_DRIVER=0` skips it, `=1` forces an attempt, and
`VIBECHECK_DRIVER_VERSION=X` overrides the version. A driver failure only WARNs:
the toolkit install still succeeds, and `prepare_instance.sh`'s GPU alarm flags a
missing GPU later. In the rare case the in-place module load does not take, reboot
the instance (the form's restart toggle) before benchmark execution.

## Submission form

| Field | Value |
| --- | --- |
| scripts directory | `vnncomp_scripts` |
| version string | `v1` |
| AMI | Ubuntu Server 24.04 LTS |
| GPU instance | yes |
| post-install script | none |
| pause after installation | ON for the first run (inspect that the driver loaded + the install log), OFF afterwards |
| run installation as root | OFF if the image has passwordless sudo (the script uses `sudo -n`). If the install log shows the apt/driver steps failing on sudo, the image lacks it: set this ON. |
| run post-installation as root | OFF (no post-install script) |
| run benchmark execution as root | OFF (vibecheck runs fine as the default user) |
| restart after post-install | OFF (the driver loads without a reboot); enable only as a fallback if the GPU alarm fires |

## Scripts

- **`install_tool.sh v1`** - apt hygiene (stop+purge the unattended-upgrades /
  snapd lock-holders), install the pinned NVIDIA driver (gated; loaded without a
  reboot), then build a self-contained `.venv` at the repo root via the
  [uv](https://docs.astral.sh/uv/) package manager (Python 3.12 + torch +
  gurobipy) and verify it (prints python/torch/gurobipy versions, imports
  vibecheck). Does not install a CUDA toolkit (torch ships its own).
- **`prepare_instance.sh v1 <category> <onnx> <vnnlib>`** - runs ONCE per
  instance, untimed: narrow zombie cleanup, a GPU sanity check that raises a loud
  alarm if no usable CUDA GPU is found (see below), a pre-parse `.pkl` cache
  (handles `.gz` inputs), and a short warmup.
- **`run_instance.sh v1 <category> <onnx> <vnnlib> <results_file> <timeout>`** -
  the timed run. Writes the authoritative verdict to `<results_file>`: line 1 is
  `unsat` / `sat` / `unknown` / `timeout`; for `sat`, the counterexample
  s-expression follows. Always exits 0 so the harness reads the verdict file
  rather than vibecheck's exit code.

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
loaded), and that `torch.cuda.is_available()` is true. If any fails it prints a
loud `GPU ALARM` block, since the configs assume a GPU and CPU runs will time
out. Set `VIBECHECK_REQUIRE_GPU=1` to turn the alarm into a hard prepare failure
instead of a warning.

## Environment variables

| Var | Effect |
| --- | --- |
| `VNNCOMP_PYTHON_PATH` | dir of the python to use (default: the repo `.venv/bin`) |
| `VIBECHECK_PKL_CACHE_DIR` | pre-parse cache location (default `/tmp/vibecheck_pkl`) |
| `VIBECHECK_SKIP_USER_CHECK=1` | bypass install_tool.sh's "must run as ubuntu/root" guard (for a dev box) |
| `VIBECHECK_INSTALL_DRIVER` | `auto` (default; install only if no working driver), `0` (skip), `1` (force) |
| `VIBECHECK_DRIVER_VERSION` | NVIDIA `.run` driver version to install (default `580.159.03`) |
| `VIBECHECK_QUIET=1` | drop `--verbose` (quiet runs) |
| `VIBECHECK_HEARTBEAT=N` | emit a `[heartbeat]` line every N seconds so a stalled phase is visible (localizes a hang before the timeout kill) |
| `VIBECHECK_GPU_RESET=1` | GPU power-mode reset in prepare (needs sudo) |
| `VIBECHECK_REQUIRE_GPU=1` | make the prepare GPU alarm a hard failure |

## Gurobi license

`gurobipy` is a normal dependency; the pip wheel ships a size-limited license
(~2000 vars/constraints) that suffices for the small/medium regular-track models.
vibecheck wraps its Gurobi calls in `except GurobiError`, so a model that exceeds
the limit degrades to the GPU/zonotope path rather than crashing. No license
setup is needed to start.

A full license is only needed for large models where vibecheck builds big Gurobi
LPs/MILPs. The simplest option then is a Gurobi **academic WLS (web) license**: it
is machine-independent, so a short post-install script can write the three-line
`gurobi.lic` (`WLSACCESSID` / `WLSSECRET` / `LICENSEID`) and verify with
`gurobi_cl`, with no per-instance regeneration or campus-network step. Provide
that script through the form's post-install upload so the credentials stay out of
git.

## Local helpers (not used by the competition harness)

- **`run_benchmarks.py <category|all|regular> [--bench-version 1.0]`** - replays
  prepare+run over a benchmark's instances.csv into a results.csv. Handles the
  2026 versioned layout (`benchmarks/<cat>/<version>/`); `--bench-version`
  selects 1.0 (v1 specs; default) or 2.0 (v2; not yet parseable). `--log-dir DIR`
  saves each instance's stdout for inspection.
- **`parse_log.py LOG|DIR`** (or pipe a log on stdin) - summarizes captured run
  logs: verdict, timing, net/spec shape, and (for a killed/timed-out run) the
  last heartbeat phase it hung in.
