"""Integration tests for tinyimagenet_2024.

Cases probed on remote with the v6 hybrid config:
  - 1 SAT (easy, exercises pre-α-CROWN PGD short-circuit on patches-zono)
  - 2 hard UNSAT (~36-50s) verified via Phase 8 dual-ascent BaB.
"""
import pytest
from ._runner import run_case


BENCHMARK_DIR = 'tinyimagenet_2024'
CONFIG_YAML = 'tinyimagenet_2024.yaml'

CASES = [
    dict(
        desc='tinyimagenet medium prop_9262 (SAT, easy)',
        net='onnx/TinyImageNet_resnet_medium.onnx',
        vnnlib='vnnlib/TinyImageNet_resnet_medium_prop_idx_9262_sidx_880_eps_0.0039.vnnlib',
        expected='sat', timeout=100, max_wall_s=15.0,
    ),
    dict(
        desc='tinyimagenet medium prop_1175 (hard UNSAT, ~36s)',
        net='onnx/TinyImageNet_resnet_medium.onnx',
        vnnlib='vnnlib/TinyImageNet_resnet_medium_prop_idx_1175_sidx_7775_eps_0.0039.vnnlib',
        expected='verified', timeout=100, max_wall_s=70.0,
    ),
    dict(
        desc='tinyimagenet medium prop_9215 (hard UNSAT, ~50s)',
        net='onnx/TinyImageNet_resnet_medium.onnx',
        vnnlib='vnnlib/TinyImageNet_resnet_medium_prop_idx_9215_sidx_4878_eps_0.0039.vnnlib',
        expected='verified', timeout=100, max_wall_s=85.0,
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize('case', CASES, ids=[c['desc'] for c in CASES])
def test_tinyimagenet_2024(case, vnncomp_benchmarks):
    run_case(case, CONFIG_YAML, vnncomp_benchmarks, BENCHMARK_DIR)
