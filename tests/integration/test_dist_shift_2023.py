"""Integration tests for dist_shift_2023.

VNNCOMP regular track. 72 instances on mnist_concat (encoder MLP +
classifier MLP, with Sigmoid between the encoder and classifier). The
Sigmoid in the middle of the active path forced a chain of fixes to
gen-LP / α-zono state alignment.

vibecheck 72/72 vs AB-CROWN 72/72. Wall: ~155 s vs AB-CROWN's 515 s.

Three representative cases:
  - index4739 (SAT, ~0.05 s) — PGD root attack on plain encoder.
  - index7901 (UNSAT, ~2 s) — Sigmoid forward + α-zono state alignment.
  - index2204 (UNSAT, ~5 s) — Phase 2.5 zono-lift with parallelogram
    sigmoid (pre-fix this case plateaued at lb=-0.22 in dual-ascent).
"""
import pytest
from ._runner import run_case


BENCHMARK_DIR = 'dist_shift_2023'
CONFIG_YAML = 'dist_shift_2023.yaml'

CASES = [
    dict(
        desc='dist_shift index4739 (SAT)',
        net='onnx/mnist_concat.onnx',
        vnnlib='vnnlib/index4739_delta0.13.vnnlib',
        expected='sat', timeout=30, max_wall_s=5.0,
    ),
    dict(
        desc='dist_shift index7901 (UNSAT, sigmoid state alignment)',
        net='onnx/mnist_concat.onnx',
        vnnlib='vnnlib/index7901_delta0.13.vnnlib',
        expected='verified', timeout=30, max_wall_s=10.0,
    ),
    dict(
        desc='dist_shift index2204 (UNSAT, Phase 2.5 + parallelogram)',
        net='onnx/mnist_concat.onnx',
        vnnlib='vnnlib/index2204_delta0.13.vnnlib',
        expected='verified', timeout=30, max_wall_s=10.0,
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize('case', CASES, ids=[c['desc'] for c in CASES])
def test_dist_shift_2023(case, vnncomp_benchmarks):
    run_case(case, CONFIG_YAML, vnncomp_benchmarks, BENCHMARK_DIR)
