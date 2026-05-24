"""Integration tests for metaroom_2023.

VNNCOMP 2023 regular track. 100 instances across 4cnn/6cnn CNN models
(3×32×56 input → 20 classes). 44 of 100 cases crashed in the
historical `milp_verify` pipeline (zonotope shape bug in
`_evaluate_region`'s `apply_relu` on `_tz_` models). Fix: route to the
graph pipeline (`auto_route_milp_for_conv: false`).

vibecheck 99/100 vs AB-CROWN 99/100 (1 shared unsolved — both timeout
on `6cnn_ry_39_6 / spec_119`). Wall: ~93 s vs AB-CROWN's 892 s —
10× faster at parity.

Three representative cases:
  - 4cnn_ry_0_0 spec_100 (SAT, ~0.6 s) — root-PGD on 2-Conv net.
  - 6cnn_tz_35_5 spec_176 (UNSAT, ~2 s) — `_tz_` regression for the
    auto-route fix (pre-fix crashed with shape mismatch).
  - 6cnn_ry_5_0 spec_130 (UNSAT, ~0.3 s) — `_ry_` 4-Conv UNSAT.
"""
import pytest
from ._runner import run_case


BENCHMARK_DIR = 'metaroom_2023'
CONFIG_YAML = 'metaroom_2023.yaml'

CASES = [
    dict(
        desc='metaroom 4cnn_ry_0_0 spec_100 (SAT)',
        net='onnx/4cnn_ry_0_0_no_custom_OP.onnx',
        vnnlib='vnnlib/spec_idx_100_eps_0.00000436.vnnlib',
        expected='sat', timeout=60, max_wall_s=5.0,
    ),
    dict(
        desc='metaroom 6cnn_tz_35_5 spec_176 (UNSAT, _tz_ auto-route fix)',
        net='onnx/6cnn_tz_35_5_no_custom_OP.onnx',
        vnnlib='vnnlib/spec_idx_176_eps_0.00001000.vnnlib',
        expected='verified', timeout=60, max_wall_s=10.0,
    ),
    dict(
        desc='metaroom 6cnn_ry_5_0 spec_130 (UNSAT)',
        net='onnx/6cnn_ry_5_0_no_custom_OP.onnx',
        vnnlib='vnnlib/spec_idx_130_eps_0.00000436.vnnlib',
        expected='verified', timeout=60, max_wall_s=5.0,
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize('case', CASES, ids=[c['desc'] for c in CASES])
def test_metaroom_2023(case, vnncomp_benchmarks):
    run_case(case, CONFIG_YAML, vnncomp_benchmarks, BENCHMARK_DIR)
