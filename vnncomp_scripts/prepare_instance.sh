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

echo "Preparing $TOOL_NAME for '$CATEGORY': onnx=$ONNX_FILE vnnlib=$VNNLIB_FILE"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# 1. Kill stale vibecheck verifier processes from a previous (possibly killed)
#    instance. NARROW on purpose: a broad `killall python` would also kill a
#    batch orchestrator (run_benchmarks.py) and, on a dev box, the tmux/agent
#    session. pkill -f matches only the verifier cmdline (`-m vibecheck.main`).
pkill -f 'vibecheck\.main' 2>/dev/null || true
sleep 1

# 2. GPU sanity check. vibecheck runs on CPU if CUDA is absent, but the
#    benchmark configs assume GPU; warn loudly if it's missing. Optional hard
#    reset (needs passwordless sudo on the competition image) gated behind
#    VIBECHECK_GPU_RESET=1 — off by default so this is harmless on dev boxes.
if command -v nvidia-smi >/dev/null 2>&1; then
	if [ "${VIBECHECK_GPU_RESET:-0}" = "1" ]; then
		(sudo -n nvidia-smi -pm 1) >/dev/null 2>&1
	fi
	nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
else
	echo "WARNING: nvidia-smi not found — vibecheck will fall back to CPU"
fi

# 3. Pre-parse cache: parse onnx+vnnlib (handles .gz) into a .pkl the timed run
#    loads via --allow-unsafe-pkl-loading, skipping the ONNX parse / shape
#    inference / folding. This subsumes a plain gunzip (it caches the fully
#    parsed graph, not just the decompressed bytes). Non-fatal on failure —
#    the timed run just parses normally.
echo "Building pre-parse cache..."
"$PY" -m vibecheck.main --net "$ONNX_FILE" --spec "$VNNLIB_FILE" --write-pkl \
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

# Clean up the warmup verifier (narrow; see note above).
pkill -f 'vibecheck\.main' 2>/dev/null || true
sleep 1

echo "Preparation finished."
exit 0
