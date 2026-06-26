#!/bin/bash
# VNNCOMP prepare_instance.sh for vibecheck. Runs ONCE per instance, before the
# timed run, and is not counted against the timeout.
#   args: <version> <category> <onnx> <vnnlib>
# Does: zombie cleanup, GPU sanity check, pre-parse cache (skips the ONNX parse
# on the timed run), and a short warmup to JIT/compile-warm the CUDA kernels.
set -u

TOOL_NAME=vibecheck
VERSION_STRING=v1

if [ "$1" != "${VERSION_STRING}" ]; then
	echo "Expected first argument (version string) '${VERSION_STRING}', got '$1'"
	exit 1
fi

CATEGORY=$2
ONNX_FILE=$3
VNNLIB_FILE=$4

TOOL_DIR=$(dirname "$(dirname "$(realpath "$0")")")
PY="${VNNCOMP_PYTHON_PATH:-$TOOL_DIR/.venv/bin}/python"

# BEGIN banner (visual divider + parse anchor; see parse_log.py).
echo "================================================================"
echo "[vibecheck:prepare_instance] BEGIN category=$CATEGORY"
echo "    onnx=$ONNX_FILE"
echo "    vnnlib=$VNNLIB_FILE"
echo "================================================================"
T_START=$(date +%s.%N)

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# 1. Kill stale vibecheck verifier processes from a previous (possibly killed)
#    instance, then VERIFY they are actually gone before proceeding. A SIGKILL'd
#    GPU process can take several seconds to release its CUDA context (a busy
#    run_instance that got sat-committed / deadline-killed lingers a moment), and
#    if it is still holding GPU memory when this instance starts it can OOM the
#    timed run. prepare is UNTIMED, so we poll-and-wait (up to ~30s), escalating
#    to SIGKILL, rather than a single fire-and-forget pkill + sleep. NARROW match
#    on purpose: a broad `killall python` would also kill a batch orchestrator
#    (run_benchmarks.py) and, on a dev box, the tmux/agent session — pkill -f
#    matches only the verifier cmdline (`-m vibecheck.main`).
stale_running() { pgrep -f 'vibecheck\.main' >/dev/null 2>&1; }
pkill -f 'vibecheck\.main' 2>/dev/null || true          # polite SIGTERM first
waited=0
while stale_running; do
	pkill -9 -f 'vibecheck\.main' 2>/dev/null || true   # then force, once per second
	sleep 1; waited=$((waited + 1))
	if [ "$waited" -eq 10 ]; then                        # taking unusually long: surface it
		echo "[prepare] stale vibecheck.main still alive after ${waited}s; still killing. Procs:"
		pgrep -af 'vibecheck\.main' 2>/dev/null || true
	fi
	[ "$waited" -ge 20 ] && break                        # stop the main poll at 20s
done
# Last resort: one more SIGKILL and a 5s grace (a busy GPU process can sit in
# uninterruptible CUDA-context teardown for a few seconds after the signal).
if stale_running; then
	echo "[prepare] re-issuing SIGKILL to stubborn stale proc(s) + waiting 5s..."
	pkill -9 -f 'vibecheck\.main' 2>/dev/null || true
	sleep 5
fi
# If it STILL won't die the environment is wedged (zombie / uninterruptible proc
# holding the GPU): make it LOUD with proc + GPU detail. We do NOT hard-fail
# prepare — the run still gets a chance (the proc may clear during its startup),
# and the GPU sanity check below + the timed run will surface any real fallout —
# but this banner is the unmistakable diagnosis if the next run OOMs/times out.
if stale_running; then
	echo "################################################################"
	echo "## ERROR: could not kill stale vibecheck.main process(es) after 25s;"
	echo "## they may hold GPU memory and OOM/timeout this instance:"
	pgrep -af 'vibecheck\.main' 2>/dev/null || true
	command -v nvidia-smi >/dev/null 2>&1 && \
		nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true
	echo "################################################################"
fi

# 2. GPU sanity check + alarm. The benchmark configs assume a CUDA GPU; without
#    one every timed run falls back to CPU and will almost certainly TIME OUT,
#    so make a missing/broken GPU LOUD in the prepare log instead of a silent
#    fallback. We check three things: nvidia-smi exists, it actually runs (the
#    driver is loaded, not just installed-pending-reboot), and torch can see
#    CUDA (nvidia-smi can work while a torch/CUDA mismatch leaves it blind).
#    VIBECHECK_GPU_RESET=1 does an optional power-mode reset (needs sudo);
#    VIBECHECK_REQUIRE_GPU=1 turns the alarm into a hard prepare failure.
gpu_alarm=""
if ! command -v nvidia-smi >/dev/null 2>&1; then
	gpu_alarm="nvidia-smi not found (no NVIDIA driver installed)"
elif ! nvidia-smi >/dev/null 2>&1; then
	gpu_alarm="nvidia-smi present but failing (driver not loaded - reboot needed after a driver install?)"
else
	[ "${VIBECHECK_GPU_RESET:-0}" = "1" ] && (sudo -n nvidia-smi -pm 1) >/dev/null 2>&1
	nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
	if ! "$PY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
		gpu_alarm="nvidia-smi works but torch.cuda.is_available() is False (torch/CUDA mismatch)"
	fi
fi

if [ -n "$gpu_alarm" ]; then
	echo "################################################################"
	echo "## GPU ALARM: $gpu_alarm"
	echo "## Configs assume a CUDA GPU; on CPU the timed runs will TIME OUT."
	echo "## On a base Ubuntu AMI, install_tool.sh installs the driver (gated on"
	echo "## VIBECHECK_INSTALL_DRIVER) and loads it without a reboot. If it could"
	echo "## not load in place, reboot the instance so the kernel module loads."
	echo "################################################################"
	if [ "${VIBECHECK_REQUIRE_GPU:-0}" = "1" ]; then
		echo "[vibecheck:prepare_instance] END status=fail ($gpu_alarm)"
		exit 1
	fi
fi

# 3. Prepare cache (--prepare-pkl-unsafe): for a normal model, parse onnx+vnnlib (handles
#    .gz) into a .pkl the timed run loads via --allow-unsafe-pkl-loading, skipping
#    the ONNX parse / shape inference / folding (subsumes a plain gunzip — caches
#    the fully parsed graph, not just decompressed bytes). For a QUANTIZED model
#    (e.g. smart_turn) it instead folds the float (STE) + fake-quant surrogates
#    (untimed) so the timed surrogate-attack run only pays the cheap onnx2torch
#    convert. One step handles both; non-fatal on failure (the timed run copes).
# Verbose by default (VIBECHECK_QUIET=1 to silence): surfaces the ONNX/VNNLIB
# load + shape inference here, which is where most parse/compat issues (e.g. an
# unparseable spec) show up - and prepare's stdout is captured by VNNCOMP too.
PREP_DEBUG="--verbose"
[ "${VIBECHECK_QUIET:-0}" = "1" ] && PREP_DEBUG=""
case "$ONNX_FILE" in *.gz) echo "Note: onnx is gzipped (.gz); decompressed and parsed transparently." ;; esac
case "$VNNLIB_FILE" in *.gz) echo "Note: vnnlib is gzipped (.gz); decompressed and parsed transparently." ;; esac
echo "Preparing cache (--prepare-pkl-unsafe)..."
"$PY" -m vibecheck.main --net "$ONNX_FILE" --spec "$VNNLIB_FILE" --prepare-pkl-unsafe $PREP_DEBUG \
	|| echo "WARNING: prepare-pkl failed; timed run will parse / fold normally"

# 4. Warmup: a short real run (5s) to trigger torch.compile / CUDA kernel
#    compilation for this graph so the timed run doesn't pay first-call JIT.
#    Result is discarded.
echo "Warmup (5s)..."
CONFIG_ARG=""
CFG="$TOOL_DIR/configs/${CATEGORY}.yaml"
[ -f "$CFG" ] && CONFIG_ARG="--config $CFG"
TMP_RES=$(mktemp)
timeout -k 5 60 "$PY" -m vibecheck.main \
	--net "$ONNX_FILE" --spec "$VNNLIB_FILE" --timeout 5 \
	$CONFIG_ARG --allow-unsafe-pkl-loading --results-file "$TMP_RES" \
	>/dev/null 2>&1 || true
rm -f "$TMP_RES"
echo "Warmup done."

# Clean up the warmup verifier (narrow; see note above).
pkill -f 'vibecheck\.main' 2>/dev/null || true
sleep 1

ELAPSED=$(awk "BEGIN{printf \"%.2f\", $(date +%s.%N) - $T_START}")
echo "================================================================"
echo "[vibecheck:prepare_instance] END status=ok elapsed=${ELAPSED}s category=$CATEGORY"
echo "================================================================"
exit 0
