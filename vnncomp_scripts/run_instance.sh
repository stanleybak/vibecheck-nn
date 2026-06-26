#!/bin/bash
# VNNCOMP run_instance.sh for vibecheck.
#   args: <version> <category> <onnx> <vnnlib> <results_file> <timeout>
# Writes the authoritative verdict to <results_file>: first line is one of
# unsat / sat / unknown / timeout; for `sat` the counterexample s-expression
# follows (the harness splits it into <instance>.counterexample.gz).
set -u

VERSION_STRING=v1
if [ "$1" != "$VERSION_STRING" ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
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

# Per-benchmark config if one exists; otherwise vibecheck's instance-based default profile.
CONFIG_ARG=""
CFG="$TOOL_DIR/configs/${CATEGORY}.yaml"
CFG_DISP="<instance-default profile>"
if [ -f "$CFG" ]; then
	CONFIG_ARG="--config $CFG"
	CFG_DISP="$CFG"
fi

rm -f "$RESULTS_FILE"   # don't let a stale verdict linger if the run dies early
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# BEGIN banner — visual divider + stable parse anchor for parse_log.py.
echo "================================================================"
echo "[vibecheck:run_instance] BEGIN category=$CATEGORY timeout=${TIMEOUT}s"
echo "    onnx=$ONNX_FILE"
echo "    vnnlib=$VNNLIB_FILE"
echo "    config=$CFG_DISP"
echo "================================================================"

T_START=$(date +%s.%N)

# Run vibecheck in the foreground. The COMPETITION harness enforces the hard
# wall-clock kill (it runs this whole script under `timeout`), so we don't manage
# process lifetime here. main.py respects --timeout cooperatively, pre-seeds the
# results file with 'timeout', and writes the final verdict (+ any counterexample)
# ATOMICALLY (temp + os.replace), so the file always holds a complete verdict no
# matter how the run stops — and that string is all the scorer reads.
#   --verbose                  per-phase progress for the toolkit details log
#   --allow-unsafe-pkl-loading reuse prepare_instance.sh's pre-parse cache
#   (mode=graph, bits=32, device=gpu are CLI defaults, so omitted)
"$PY" -m vibecheck.main \
	--net "$ONNX_FILE" \
	--spec "$VNNLIB_FILE" \
	--timeout "$TIMEOUT" \
	$CONFIG_ARG \
	--verbose \
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

# The verdict file is authoritative; exit 0 so the harness reads the file rather
# than treating vibecheck's verdict exit code (1=unknown) as a tool failure.
exit 0
