#!/bin/bash
# VNNCOMP run_instance.sh for vibecheck.
#   args: <version> <category> <onnx> <vnnlib> <results_file> <timeout>
# Writes the authoritative verdict to <results_file>: first line is one of
# unsat / sat / unknown / timeout; for `sat` the counterexample s-expression
# follows (the harness splits it into <instance>.counterexample.gz).
set -u
# Opt out of any xtrace inherited from the harness (`bash -x` / exported SHELLOPTS):
# the 0.5s sat-commit poll loop below would otherwise flood the captured per-instance
# log with `+ kill -0 / + head -c3 / + sleep 0.5` lines every tick. The intended log
# is vibecheck's own --verbose phase trace (its stdout), not this script's plumbing.
{ set +x; } 2>/dev/null

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
# never prints - the heartbeat is what surfaces a hang before the timeout-kill).
DEBUG_ARGS="--verbose"
[ "${VIBECHECK_QUIET:-0}" = "1" ] && DEBUG_ARGS=""
[ -n "${VIBECHECK_HEARTBEAT:-}" ] && DEBUG_ARGS="$DEBUG_ARGS --heartbeat $VIBECHECK_HEARTBEAT"

# BEGIN banner - a visual divider in the captured stdout that also serves as the
# stable parse anchor for parse_log.py (`[vibecheck:run_instance] BEGIN key=val`).
echo "================================================================"
echo "[vibecheck:run_instance] BEGIN category=$CATEGORY timeout=${TIMEOUT}s"
echo "    onnx=$ONNX_FILE"
echo "    vnnlib=$VNNLIB_FILE"
echo "    config=$CFG_DISP"
echo "================================================================"

T_START=$(date +%s.%N)

# TIMEOUT HANDLING. VNNCOMP times THIS script and kills it at $TIMEOUT, so the
# hard deadline is the harness's job — we don't re-implement it. main.py pre-seeds
# $RESULTS_FILE with 'timeout' and writes the verdict (incl. any counterexample)
# ATOMICALLY (temp + os.replace, which a kill cannot tear), and that file's result
# string is all the scorer reads (no wall-time check, time bonus off). So whenever
# the tool stops, the file holds a correct, complete verdict.
#
# VC emits a `sat` ONLY for a genuine counterexample and returns IMMEDIATELY (there
# is no within-tolerance "write a sat early then keep searching" path any more), so
# there is nothing to gain by polling/early-killing a running tool — we just let it
# run and `wait` for it. main.py gets a small startup reserve (TIMEOUT-RESERVE) so a
# no-CE run self-stops and WRITES its verdict before the harness's wall-clock kill.
RESERVE=2
MAIN_TIMEOUT=$(awk "BEGIN{t=$TIMEOUT-$RESERVE; h=$TIMEOUT/2; if(t<h)t=h; if(t<1)t=1; print t}")

# mode=graph, bits=32, device=gpu are vibecheck's CLI defaults, so omitted.
# --allow-unsafe-pkl-loading reuses prepare_instance.sh's pre-parse cache.
# `setsid` puts the tool in its OWN process group so we can reap it AND any helper
# processes (torch/CUDA) with one group-kill, leaving nothing on the GPU. In a
# script (no job control) setsid exec's, so $! is the tool itself == its own pgid.
SETSID=""; command -v setsid >/dev/null 2>&1 && SETSID=setsid
$SETSID "$PY" -m vibecheck.main \
	--net "$ONNX_FILE" \
	--spec "$VNNLIB_FILE" \
	--timeout "$MAIN_TIMEOUT" \
	$CONFIG_ARG \
	$DEBUG_ARGS \
	--allow-unsafe-pkl-loading \
	--results-file "$RESULTS_FILE" &
TOOL_PID=$!
# Reap the tool on ANY exit of this script, INCLUDING the harness terminating it
# (TERM/INT — bash does NOT run an EXIT trap for an untrapped signal, so trap them
# explicitly). Group-kill only when setsid gave the tool its own group; otherwise
# a negative-pid kill would hit THIS script's group.
reap() { [ -n "$SETSID" ] && kill -9 "-$TOOL_PID" 2>/dev/null; kill -9 "$TOOL_PID" 2>/dev/null; }
trap 'reap; exit 143' TERM INT
trap reap EXIT

# Just wait for the tool (it self-stops at MAIN_TIMEOUT and writes its verdict; the
# harness kills THIS script at $TIMEOUT, and the trap reaps the tool if so).
wait "$TOOL_PID" 2>/dev/null
RUN_RC=$?
trap - TERM INT EXIT
# The pre-seeded/early-emitted verdict in the results file is authoritative
# regardless of how the tool stopped, so we just read it below.

ELAPSED=$(awk "BEGIN{printf \"%.2f\", $(date +%s.%N) - $T_START}")
VERDICT=$(head -n1 "$RESULTS_FILE" 2>/dev/null | tr -d '[:space:]')
[ -z "$VERDICT" ] && VERDICT=unknown

# END banner - closing divider + parse anchor with the authoritative verdict.
echo "================================================================"
echo "[vibecheck:run_instance] END verdict=$VERDICT elapsed=${ELAPSED}s rc=$RUN_RC category=$CATEGORY"
echo "================================================================"

# The verdict file is authoritative; return 0 so the harness reads the file
# rather than treating vibecheck's verdict exit code (1=unknown) as a failure.
exit 0
