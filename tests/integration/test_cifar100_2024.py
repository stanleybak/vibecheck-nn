"""Integration tests for cifar100_2024.

Cases probed on AWS A10G with the lean config (cascade-skip + reverse_g:true +
pre-cascade PGD SAT catcher; see configs/cifar100_2024.yaml):
  - 1 SAT (resnet_large; caught by the pre-cascade PGD — also the reverse-g
    soundness guard: reverse_g once false-verified resnet_large SAT cases via a
    formulation-tag dispatch bug, FIXED 2026-06-07; this case MUST verdict `sat`,
    never `unsat`, with reverse_g enabled. See tests/test_reverse_g.py).
  - 2 hard UNSAT verified via the Phase 8 dual-ascent BaB (now ~21s/30s, faster
    than the old hybrid config's ~33s/42s).
"""
import pytest
from ._runner import run_case


BENCHMARK_DIR = 'cifar100_2024'
CONFIG_YAML = 'cifar100_2024.yaml'

CASES = [
    dict(
        desc='cifar100 large prop_6583 (SAT; reverse_g-unsound soundness guard)',
        net='onnx/CIFAR100_resnet_large.onnx',
        vnnlib='vnnlib/CIFAR100_resnet_large_prop_idx_6583_sidx_8860_eps_0.0039.vnnlib',
        expected='sat', timeout=100, max_wall_s=22.0,
    ),
    dict(
        desc='cifar100 medium prop_2925 (hard UNSAT, ~21s)',
        net='onnx/CIFAR100_resnet_medium.onnx',
        vnnlib='vnnlib/CIFAR100_resnet_medium_prop_idx_2925_sidx_8815_eps_0.0039.vnnlib',
        expected='verified', timeout=100, max_wall_s=38.0,
    ),
    dict(
        desc='cifar100 medium prop_230 (hard UNSAT, ~30s)',
        net='onnx/CIFAR100_resnet_medium.onnx',
        vnnlib='vnnlib/CIFAR100_resnet_medium_prop_idx_230_sidx_1968_eps_0.0039.vnnlib',
        expected='verified', timeout=100, max_wall_s=48.0,
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize('case', CASES, ids=[c['desc'] for c in CASES])
def test_cifar100_2024(case, vnncomp_benchmarks):
    run_case(case, CONFIG_YAML, vnncomp_benchmarks, BENCHMARK_DIR)
