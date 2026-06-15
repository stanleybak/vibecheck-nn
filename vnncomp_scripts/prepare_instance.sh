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
#    instance. NARROW on purpose: a broad `killall python` would also kill a
#    batch orchestrator (run_benchmarks.py) and, on a dev box, the tmux/agent
#    session. pkill -f matches only the verifier cmdline (`-m vibecheck.main`).
pkill -f 'vibecheck\.main' 2>/dev/null || true
sleep 1

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

# 3. Pre-parse cache: parse onnx+vnnlib (handles .gz) into a .pkl the timed run
#    loads via --allow-unsafe-pkl-loading, skipping the ONNX parse / shape
#    inference / folding. This subsumes a plain gunzip (it caches the fully
#    parsed graph, not just the decompressed bytes). Non-fatal on failure -
#    the timed run just parses normally.
# Verbose by default (VIBECHECK_QUIET=1 to silence): surfaces the ONNX/VNNLIB
# load + shape inference here, which is where most parse/compat issues (e.g. an
# unparseable spec) show up - and prepare's stdout is captured by VNNCOMP too.
PREP_DEBUG="--verbose"
[ "${VIBECHECK_QUIET:-0}" = "1" ] && PREP_DEBUG=""
case "$ONNX_FILE" in *.gz) echo "Note: onnx is gzipped (.gz); decompressed and parsed transparently." ;; esac
case "$VNNLIB_FILE" in *.gz) echo "Note: vnnlib is gzipped (.gz); decompressed and parsed transparently." ;; esac
echo "Building pre-parse cache..."
"$PY" -m vibecheck.main --net "$ONNX_FILE" --spec "$VNNLIB_FILE" --write-pkl $PREP_DEBUG \
	|| echo "WARNING: pre-parse cache failed; timed run will parse normally"

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
