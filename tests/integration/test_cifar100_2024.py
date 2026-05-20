"""Integration tests for cifar100_2024.

Cases probed on remote with the v6 hybrid config:
  - 1 SAT (the case 22 AB-WIN regression that the pre-α-CROWN PGD hook closed)
  - 2 hard UNSAT (~30-45s) verified via the Phase 8 dual-ascent BaB.
"""
import pytest
from ._runner import run_case


BENCHMARK_DIR = 'cifar100_2024'
CONFIG_YAML = 'cifar100_2024.yaml'

CASES = [
    dict(
        desc='cifar100 large prop_6583 (SAT, case 22 regression catcher)',
        net='onnx/CIFAR100_resnet_large.onnx',
        vnnlib='vnnlib/CIFAR100_resnet_large_prop_idx_6583_sidx_8860_eps_0.0039.vnnlib',
        expected='sat', timeout=100, max_wall_s=20.0,
    ),
    dict(
        desc='cifar100 medium prop_2925 (hard UNSAT, ~33s)',
        net='onnx/CIFAR100_resnet_medium.onnx',
        vnnlib='vnnlib/CIFAR100_resnet_medium_prop_idx_2925_sidx_8815_eps_0.0039.vnnlib',
        expected='verified', timeout=100, max_wall_s=70.0,
    ),
    dict(
        desc='cifar100 medium prop_230 (hard UNSAT, ~42s)',
        net='onnx/CIFAR100_resnet_medium.onnx',
        vnnlib='vnnlib/CIFAR100_resnet_medium_prop_idx_230_sidx_1968_eps_0.0039.vnnlib',
        expected='verified', timeout=100, max_wall_s=80.0,
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize('case', CASES, ids=[c['desc'] for c in CASES])
def test_cifar100_2024(case, vnncomp_benchmarks):
    run_case(case, CONFIG_YAML, vnncomp_benchmarks, BENCHMARK_DIR)
