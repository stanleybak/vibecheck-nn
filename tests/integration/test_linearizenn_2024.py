"""Integration tests for linearizenn_2024.

VNNCOMP regular track. 60 instances across 11 AllInOne MLPs (4-input,
deep ReLU + Slice/MatMul skip branch + Concat → output). The Slice and
Concat ops shipped a real soundness bug pre-fix: `_spec_backward_graph`
silently skipped unhandled ops, so `ew` died mid-chain, `ew_at[input]`
stayed zero, and `spec_lb = acc + 0 = acc` was vacuously positive —
declared `verified` on a real SAT case (`prop_10_10`).

vibecheck 60/60 vs AB-CROWN 60/60. Wall: ~19 s vs AB-CROWN's 473 s.

Three representative cases:
  - prop_10_10 (SAT, ~0.5 s) — regression for the slice/concat
    soundness bug. ABC's counterexample produces Y_0 ≈ 40.27 ≥ 40.204;
    pre-fix vibecheck declared `verified` (unsound).
  - prop_120_120 (UNSAT, ~0.3 s) — needs batched input split BaB
    (cersyve-style) to close in time; unbatched fast_leaf timed out.
  - prop_120_120_4 (UNSAT, ~0.3 s) — extra batched-BaB regression
    on the biggest model.
"""
import pytest
from ._runner import run_case


BENCHMARK_DIR = 'linearizenn_2024'
CONFIG_YAML = 'linearizenn_2024.yaml'

CASES = [
    dict(
        desc='linearizenn prop_10_10 (SAT, slice/concat soundness regression)',
        net='onnx/AllInOne_10_10.onnx',
        vnnlib='vnnlib/prop_10_10.vnnlib',
        expected='sat', timeout=30, max_wall_s=5.0,
    ),
    dict(
        desc='linearizenn prop_120_120 (UNSAT, batched input split)',
        net='onnx/AllInOne_120_120.onnx',
        vnnlib='vnnlib/prop_120_120.vnnlib',
        expected='verified', timeout=30, max_wall_s=3.0,
    ),
    dict(
        desc='linearizenn prop_120_120_4 (UNSAT, batched input split)',
        net='onnx/AllInOne_120_120.onnx',
        vnnlib='vnnlib/prop_120_120_4.vnnlib',
        expected='verified', timeout=30, max_wall_s=3.0,
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize('case', CASES, ids=[c['desc'] for c in CASES])
def test_linearizenn_2024(case, vnncomp_benchmarks):
    run_case(case, CONFIG_YAML, vnncomp_benchmarks, BENCHMARK_DIR)
