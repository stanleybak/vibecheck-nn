#!/bin/bash
# VNNCOMP run_instance.sh for vibecheck.
#   args: <version> <category> <onnx> <vnnlib> <results_file> <timeout>
# Writes the authoritative verdict to <results_file>: first line is one of
# unsat / sat / unknown / timeout; for `sat` the counterexample s-expression
# follows (the harness splits it into <instance>.counterexample.gz).
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
RESULTS_FILE=$5
TIMEOUT=$6

TOOL_DIR=$(dirname "$(dirname "$(realpath "$0")")")
# Python: repo venv by default; override with VNNCOMP_PYTHON_PATH (dir of python).
PY="${VNNCOMP_PYTHON_PATH:-$TOOL_DIR/.venv/bin}/python"

# Map category -> per-benchmark config if one exists; otherwise vibecheck's
# instance-based default profile is used (omit --config).
CONFIG_ARG=""
CFG="$TOOL_DIR/configs/${CATEGORY}.yaml"
CFG_DISP="<instance-default profile>"
if [ -f "$CFG" ]; then
	CONFIG_ARG="--config $CFG"
	CFG_DISP="$CFG"
fi

# Old results must not linger if the run dies before writing (main.py also
# writes 'unknown' on any crash, so the file is always populated).
rm -f "$RESULTS_FILE"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# VNNCOMP captures this script's stdout/stderr as the per-instance log shown on
# the toolkit details page, so run vibecheck --verbose by DEFAULT: per-phase
# progress, with stdout line-buffered so a timed-out run that gets SIGKILL'd
# still flushes its trace instead of leaving an empty log (exactly the case you
# most need to debug). VIBECHECK_QUIET=1 silences it; VIBECHECK_HEARTBEAT=N adds
# periodic in-phase "[heartbeat]" lines so a STALLED phase is visible in the log
# (plain --verbose only prints at phase boundaries, so a phase that never returns
# never prints — the heartbeat is what surfaces a hang before the timeout-kill).
DEBUG_ARGS="--verbose"
[ "${VIBECHECK_QUIET:-0}" = "1" ] && DEBUG_ARGS=""
[ -n "${VIBECHECK_HEARTBEAT:-}" ] && DEBUG_ARGS="$DEBUG_ARGS --heartbeat $VIBECHECK_HEARTBEAT"

# BEGIN banner — a visual divider in the captured stdout that also serves as the
# stable parse anchor for parse_log.py (`[vibecheck:run_instance] BEGIN key=val`).
echo "================================================================"
echo "[vibecheck:run_instance] BEGIN category=$CATEGORY timeout=${TIMEOUT}s"
echo "    onnx=$ONNX_FILE"
echo "    vnnlib=$VNNLIB_FILE"
echo "    config=$CFG_DISP"
echo "================================================================"

T_START=$(date +%s.%N)

# mode=graph, bits=32, device=gpu are vibecheck's CLI defaults, so omitted.
# --allow-unsafe-pkl-loading reuses prepare_instance.sh's pre-parse cache.
"$PY" -m vibecheck.main \
	--net "$ONNX_FILE" \
	--spec "$VNNLIB_FILE" \
	--timeout "$TIMEOUT" \
	$CONFIG_ARG \
	$DEBUG_ARGS \
	--allow-unsafe-pkl-loading \
	--results-file "$RESULTS_FILE"
RUN_RC=$?

ELAPSED=$(awk "BEGIN{printf \"%.2f\", $(date +%s.%N) - $T_START}")
VERDICT=$(head -n1 "$RESULTS_FILE" 2>/dev/null | tr -d '[:space:]')
[ -z "$VERDICT" ] && VERDICT=unknown

# END banner — closing divider + parse anchor with the authoritative verdict.
echo "================================================================"
echo "[vibecheck:run_instance] END verdict=$VERDICT elapsed=${ELAPSED}s rc=$RUN_RC category=$CATEGORY"
echo "================================================================"

# The verdict file is authoritative; return 0 so the harness reads the file
# rather than treating vibecheck's verdict exit code (1=unknown) as a failure.
exit 0
