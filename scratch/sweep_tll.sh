#!/bin/bash
# Baseline sweep for tllverifybench_2023. Verdict from --results-file ONLY;
# expected verdict from the ABC reference results.csv (sat/unsat).
cd /home/stan/repositories/vibecheck
B=/home/stan/repositories/vnncomp2025_benchmarks/benchmarks/tllverifybench_2023
ABC=~/repositories/vnncomp2025_results/alpha_beta_crown/2025_tllverifybench_2023/results.csv
CFG=$1; TMO=${2:-60}; PY=.venv/bin/python
OUT=/tmp/tll_sweep.csv; echo "onnx,expected,verdict,wall,ok" > $OUT
ok=0; tot=0; sok=0; stot=0; uok=0; utot=0
while IFS=, read -r onnx vnn tmo_inst; do
  net="$B/$onnx"; spec="$B/$vnn"
  [ -f "$net" ] || net="${net}.gz"; [ -f "$spec" ] || spec="${spec}.gz"
  exp=$(awk -F, -v o="$onnx" -v v="$vnn" 'index($2,o) && index($3,v){print $5; exit}' $ABC)
  [ -z "$exp" ] && exp="?"
  tot=$((tot+1)); [ "$exp" == sat ] && stot=$((stot+1)); [ "$exp" == unsat ] && utot=$((utot+1))
  rm -f /tmp/tllx.txt; t0=$(date +%s.%N)
  timeout $((TMO+30)) $PY -m vibecheck.main --net "$net" --spec "$spec" \
    ${CFG:+--config $CFG} --timeout $TMO --results-file /tmp/tllx.txt >/dev/null 2>&1
  t1=$(date +%s.%N); wall=$(echo "$t1-$t0"|bc); v=$(cat /tmp/tllx.txt 2>/dev/null || echo MISSING)
  good=$([ "$v" == "$exp" ] && echo 1 || echo 0); ok=$((ok+good))
  [ "$exp" == sat ] && sok=$((sok+good)); [ "$exp" == unsat ] && uok=$((uok+good))
  printf "%s,%s,%s,%.1f,%s\n" "$(basename $onnx)" "$exp" "$v" "$wall" "$good" >> $OUT
done < $B/instances.csv
echo "=== tll baseline: $ok/$tot (sat $sok/$stot, unsat $uok/$utot) ==="
awk -F, 'NR>1 && $5==0{print "  MISS: "$1" exp="$2" got="$3" ("$4"s)"}' $OUT
