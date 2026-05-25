"""Integration tests for nn4sys (partial — 121/194 covered).

VNNCOMP regular track. 12 ML-for-systems models. 121 verified via:
  - Per-disjunct sub-verification for `lindex` family (10000 X subboxes
    decomposed into batched-CROWN fast path)
  - Gather op support for `pensieve_*_simple` family
  - ONNX input-shape loader fix (keep all-concrete dims) for
    `pensieve_big_parallel` family

Two critical soundness bugs caught + fixed during this benchmark:
  - milp_verify Gurobi FeasibilityTol vs float32-zono 1 ulp gap
  - vnnlib parser dropped X constraints from mixed X/Y conjuncts

8 pensieve_small_parallel cases blocked on Gather/Slice topology
quirk; 65 mscn cases blocked on Div + ReduceSum dispatchers (each
Div is masked-mean `Div(ReduceSum(feat*mask), ReduceSum(mask))`).

Three representative cases:
  - lindex_10000 (UNSAT, ~0.5 s) — fast batched CROWN over 10000
    unique X subboxes. Regression for both soundness fixes.
  - pensieve_simple_0 on pensieve_small_simple (UNSAT, ~2 s) — Gather
    op + per-disjunct.
  - pensieve_parallel_1 on pensieve_big_parallel (UNSAT, ~2 s) —
    fixed-shape `[12, 8]` input loader regression.
"""
import pytest
from ._runner import run_case


BENCHMARK_DIR = 'nn4sys'
CONFIG_YAML = 'nn4sys.yaml'

CASES = [
    dict(
        desc='nn4sys lindex_10000 (UNSAT, per-disjunct fast CROWN)',
        net='onnx/lindex.onnx',
        vnnlib='vnnlib/lindex_10000.vnnlib',
        expected='verified', timeout=30, max_wall_s=5.0,
    ),
    dict(
        desc='nn4sys pensieve_simple_0 (UNSAT, Gather op + per-disjunct)',
        net='onnx/pensieve_small_simple.onnx',
        vnnlib='vnnlib/pensieve_simple_0.vnnlib',
        expected='verified', timeout=30, max_wall_s=10.0,
    ),
    dict(
        desc='nn4sys pensieve_parallel_1 (UNSAT, fixed-shape input loader)',
        net='onnx/pensieve_big_parallel.onnx',
        vnnlib='vnnlib/pensieve_parallel_1.vnnlib',
        expected='verified', timeout=30, max_wall_s=10.0,
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize('case', CASES, ids=[c['desc'] for c in CASES])
def test_nn4sys(case, vnncomp_benchmarks):
    run_case(case, CONFIG_YAML, vnncomp_benchmarks, BENCHMARK_DIR)
