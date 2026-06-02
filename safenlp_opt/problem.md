# safenlp_2024 optimization — problem statement (frozen)

Goal: vibecheck should match AB-CROWN on safenlp_2024 (1080 instances) within
the VNN-COMP 20s/case budget. ABC published: 646 sat, 434 unsat (solves all).

Nets (both): 30 -> MatMul(30x128)+Add -> ReLU -> MatMul(128x2)+Add -> 2.
Spec: 30-dim input hyperrectangle; unsafe iff Y_0 <= Y_1 reachable.
  SAT = violating input exists (counterexample); UNSAT = prove Y_0 > Y_1.

Recon (2026-05-30): SAT-finding solid (PGD 12/12). Gap = hard UNSAT cases:
CROWN bound on Y_0-Y_1 very loose (worst ~ -10.7 on a 1-ReLU net); default
pipeline gives up at ~2.5s; input-split (cap 35) cracks ~half but explodes on
the hardest (30-dim split = wrong axis for 1-ReLU net). `--mode milp` gen-LP
racing reports "feasibility SAT" up to bins=118 then unknown — NOT exact,
which is suspicious for a 128-ReLU net.

Out of scope unless needed: implementing full beta-CROWN per-neuron BaB.

Tools: AB-CROWN on server1 (stan@100.107.254.48, ~/Desktop/temp/abcrown or
~/repositories) for ground-truth verdicts, runtime, settings, dynamic tracing.
Benchmarks on server1 ~/repositories/vnncomp2025_benchmarks/benchmarks/safenlp_2024.
