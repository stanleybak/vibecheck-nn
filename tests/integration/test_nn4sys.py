"""Integration tests for nn4sys.

VNNCOMP regular track. 12 ML-for-systems models verified via:
  - Per-disjunct sub-verification for `lindex` family (10000 X subboxes
    decomposed into batched-CROWN fast path)
  - Gather op support for `pensieve_*_simple` family
  - ONNX input-shape loader fix (keep all-concrete dims) for
    `pensieve_*_parallel` family
  - Pow / Div / ReduceSum / MulBilinear handlers (chord-tangent Pow,
    box-fallback Div) for `mscn_*` and `pensieve_*_parallel` families
  - Auto-routing: graphs with non-LP ops force
    `tighten_formulation=skip`, `phase2_crown_enabled=False`, and lift
    `input_split_max_dims` based on varying-dim count

Soundness bugs caught + fixed during this benchmark:
  - milp_verify Gurobi FeasibilityTol vs float32-zono 1 ulp gap
  - vnnlib parser dropped X constraints from mixed X/Y conjuncts
  - **Silent op-skip in gg builder for Pow/Div/ReduceSum/MulBilinear**:
    pensieve_*_parallel verification was passing pre-fix because the
    gg op-emit fell through to `alias[name] = src` on these ops,
    effectively computing on a network with Pow/Div stripped — the
    "verified" lb of ~167069 didn't reflect the real network (whose
    Y_0 ranges ~5.55-5.59). The Pow/Div/ReduceSum handlers added in
    this branch produce correct (but looser) bounds.

Cases:
  - lindex_10000 — fast batched CROWN over 10000 unique X subboxes
  - pensieve_simple_0 — Gather op + per-disjunct, no Pow/Div
  - mscn_128d cardinality_0_1 — per-disjunct + Pow/Div/ReduceSum
    correct bound (chord-tangent Pow + box-fallback Div), verifies via
    Phase 0.5 α-CROWN since `phase2_crown_enabled=False` is auto-set
  - mscn_2048d cardinality_0_1 — same path on 2048-dim variant,
    verifies through input-split fast-leaf (lifted from 20 → 154 dims
    because n_varying=1)
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
        desc='nn4sys mscn_128d cardinality_0_1 '
             '(UNSAT, Pow/Div/ReduceSum + phase2_crown auto-off)',
        net='onnx/mscn_128d.onnx',
        vnnlib='vnnlib/cardinality_0_1_128.vnnlib',
        expected='verified', timeout=30, max_wall_s=5.0,
    ),
    dict(
        desc='nn4sys mscn_2048d cardinality_0_1 '
             '(UNSAT, input-split max_dims auto-lifted to 154)',
        net='onnx/mscn_2048d.onnx',
        vnnlib='vnnlib/cardinality_0_1_2048.vnnlib',
        expected='verified', timeout=30, max_wall_s=5.0,
    ),
    dict(
        desc='nn4sys pensieve_parallel_1 '
             '(UNSAT, Sub-bilinear + sound Div Lagrange-remainder)',
        net='onnx/pensieve_big_parallel.onnx',
        vnnlib='vnnlib/pensieve_parallel_1.vnnlib',
        expected='verified', timeout=30, max_wall_s=5.0,
    ),
    dict(
        desc='nn4sys pensieve_small_parallel_1 '
             '(UNSAT, Gather shape-inference fix + Sub-bilinear)',
        net='onnx/pensieve_small_parallel.onnx',
        vnnlib='vnnlib/pensieve_parallel_1.vnnlib',
        expected='verified', timeout=30, max_wall_s=5.0,
    ),
    dict(
        desc='nn4sys pensieve_big_parallel_32 '
             '(UNSAT, auto-routed input-split + α-Pow/Div + shared-gen Div)',
        net='onnx/pensieve_big_parallel.onnx',
        vnnlib='vnnlib/pensieve_parallel_32.vnnlib',
        expected='verified', timeout=30, max_wall_s=5.0,
    ),
    dict(
        desc='nn4sys pensieve_big_parallel_83 '
             '(UNSAT, hard case: needs ABC-style Recip+McCormick Div + '
             'softmax-clamp + input-split)',
        net='onnx/pensieve_big_parallel.onnx',
        vnnlib='vnnlib/pensieve_parallel_83.vnnlib',
        expected='verified', timeout=140, max_wall_s=140.0,
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize('case', CASES, ids=[c['desc'] for c in CASES])
def test_nn4sys(case, vnncomp_benchmarks):
    run_case(case, CONFIG_YAML, vnncomp_benchmarks, BENCHMARK_DIR)
