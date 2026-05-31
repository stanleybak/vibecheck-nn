#!/bin/bash
# Local baseline sweep for sat_relu (tiny nets, safe locally).
# Verdict from --results-file ONLY. Expected verdict from filename prefix.
cd /home/stan/repositories/vibecheck
B=/home/stan/repositories/vnncomp2025_benchmarks/benchmarks/sat_relu
CFG=$1                      # optional --config path
TMO=${2:-20}
PY=.venv/bin/python
OUT=/tmp/sat_relu_sweep.csv
echo "onnx,expected,verdict,wall,ok" > $OUT
ok=0; tot=0; sat_ok=0; sat_tot=0; uns_ok=0; uns_tot=0
while IFS=, read -r onnx vnn tmo_inst; do
  net="$B/$onnx"; spec="$B/$vnn"
  [ -f "$net" ] || net="${net}.gz"; [ -f "$spec" ] || spec="${spec}.gz"
  base=$(basename "$onnx")
  exp=$([[ "$base" == unsat_* ]] && echo unsat || echo sat)
  tot=$((tot+1)); [ "$exp" == sat ] && sat_tot=$((sat_tot+1)) || uns_tot=$((uns_tot+1))
  rm -f /tmp/srx.txt
  t0=$(date +%s.%N)
  timeout $((TMO+30)) $PY -m vibecheck.main --net "$net" --spec "$spec" \
    ${CFG:+--config $CFG} --timeout $TMO --results-file /tmp/srx.txt >/dev/null 2>&1
  t1=$(date +%s.%N); wall=$(echo "$t1-$t0"|bc)
  v=$(cat /tmp/srx.txt 2>/dev/null || echo MISSING)
  good=$([ "$v" == "$exp" ] && echo 1 || echo 0)
  ok=$((ok+good))
  if [ "$exp" == sat ]; then sat_ok=$((sat_ok+good)); else uns_ok=$((uns_ok+good)); fi
  printf "%s,%s,%s,%.1f,%s\n" "$base" "$exp" "$v" "$wall" "$good" >> $OUT
done < $B/instances.csv
echo "=== sat_relu baseline: $ok/$tot correct (sat $sat_ok/$sat_tot, unsat $uns_ok/$uns_tot) ==="
echo "wrong verdicts (non-matching):"
awk -F, 'NR>1 && $5==0{print "  "$1" expected="$2" got="$3}' $OUT | head -30
