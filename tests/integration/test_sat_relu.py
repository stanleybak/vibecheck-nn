"""Integration tests for sat_relu.

VNNCOMP track. 100 instances: random 3-SAT problems compiled into
integer-weighted ReLU nets `30..100 vars → Gemm → ReLU → Gemm → 2`. The
unsafe spec is a CONJUNCTION `Y_0 >= 1 AND Y_1 <= 0` that holds iff the SAT
formula is satisfiable. `sat_*` instances are satisfiable (a witness exists),
`unsat_*` are not (the property is verified).

vibecheck path (see configs/sat_relu.yaml):
  - `sat_*`  → PGD finds the witness (root attack), ~1 s.
  - `unsat_*` → the exact MILP proves the JOINT feasibility of both conjuncts
    infeasible. Four pieces make this work:
      1. conjunctive-disjunct feasibility (verify_milp): a multi-conjunct
         unsafe region is the INTERSECTION of its halfspaces — proven empty by
         a single MILP enforcing ALL of them, not one-at-a-time.
      2. racing fix: wait for BOTH feas+opt workers before escalating (opt
         minimises only the first halfspace, which is reachable, so it must not
         abort the feas worker that proves the joint MILP infeasible).
      3. classification-preserving inflation clamp: the FP bound inflation must
         not flip the many lo==0 always-active neurons to unstable.
      4. config: FP inflation OFF (integer weights ⇒ non-integer big-M would
         defeat Gurobi's integer presolve, 90 s vs 0.1 s) and
         input_split_max_dims=0 (route every case to the MILP, not the
         input-split graph pipeline that can't prove the conjunction).

vibecheck 100/100 vs AB-CROWN 100/100; ~0.2 s/MILP.

Representative cases: one SAT, two UNSAT (small + large), plus a soundness
probe that must NOT false-verify the SAT case with PGD disabled.
"""
import pytest
from ._runner import run_case


BENCHMARK_DIR = 'sat_relu'
CONFIG_YAML = 'sat_relu.yaml'

CASES = [
    dict(
        desc='sat_relu sat_v30_c38 (SAT, PGD root attack)',
        net='onnx/sat_v30_c38.onnx',
        vnnlib='vnnlib/sat_v30_c38.vnnlib',
        expected='sat', timeout=40, max_wall_s=10.0,
    ),
    dict(
        desc='sat_relu unsat_v30_c38 (UNSAT, conjunctive MILP)',
        net='onnx/unsat_v30_c38.onnx',
        vnnlib='vnnlib/unsat_v30_c38.vnnlib',
        expected='verified', timeout=40, max_wall_s=12.0,
    ),
    dict(
        desc='sat_relu unsat_v100_c373 (UNSAT, large; integer-bound + routing)',
        net='onnx/unsat_v100_c373.onnx',
        vnnlib='vnnlib/unsat_v100_c373.vnnlib',
        expected='verified', timeout=40, max_wall_s=15.0,
    ),
    # SOUNDNESS PROBE — a genuinely-SAT case with SAT-finding disabled must NOT
    # verify (the conjunctive MILP is exact only because inflation is off and
    # the bounds are exact integers; a false `verified` here would mean the
    # joint-feasibility encoding is unsound).
    dict(
        desc='sat_relu sat_v30_c38 SOUNDNESS (SAT, sat-finding off → NOT verify)',
        net='onnx/sat_v30_c38.onnx',
        vnnlib='vnnlib/sat_v30_c38.vnnlib',
        expected='unknown', timeout=40, max_wall_s=12.0,
        extra_settings=dict(disable_sat_finding=True),
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize('case', CASES, ids=[c['desc'] for c in CASES])
def test_sat_relu(case, vnncomp_benchmarks):
    run_case(case, CONFIG_YAML, vnncomp_benchmarks, BENCHMARK_DIR)
