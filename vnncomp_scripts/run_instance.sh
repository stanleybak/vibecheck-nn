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

echo "Running $TOOL_NAME on '$CATEGORY': onnx=$ONNX_FILE vnnlib=$VNNLIB_FILE timeout=${TIMEOUT}s"

# Map category -> per-benchmark config if one exists; otherwise vibecheck's
# instance-based default profile is used (omit --config).
CONFIG_ARG=""
CFG="$TOOL_DIR/configs/${CATEGORY}.yaml"
if [ -f "$CFG" ]; then
	CONFIG_ARG="--config $CFG"
	echo "Using config: $CFG"
else
	echo "No configs/${CATEGORY}.yaml — using instance-based default profile"
fi

# Old results must not linger if the run dies before writing (main.py also
# writes 'unknown' on any crash, so the file is always populated).
rm -f "$RESULTS_FILE"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# mode=graph, bits=32, device=gpu are vibecheck's CLI defaults, so omitted.
# --allow-unsafe-pkl-loading reuses prepare_instance.sh's pre-parse cache.
"$PY" -m vibecheck.main \
	--net "$ONNX_FILE" \
	--spec "$VNNLIB_FILE" \
	--timeout "$TIMEOUT" \
	$CONFIG_ARG \
	--allow-unsafe-pkl-loading \
	--results-file "$RESULTS_FILE"

# The verdict file is authoritative; return 0 so the harness reads the file
# rather than treating vibecheck's verdict exit code (1=unknown) as a failure.
exit 0
