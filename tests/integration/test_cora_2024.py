"""Integration tests for cora_2024.

VNNCOMP regular track. 9 MLPs (mnist/cifar10/svhn × point/set/trades),
8 FC layers (250-wide hidden, 10 output), 7 ReLUs. Multi-disjunct DNF
specs (9 disjuncts: y_target >= y_j for j != target). 180 instances.

vibecheck 151/180 vs AB-CROWN 153/180 — tied on 151, miss 2 cifar10-set
UNSAT cases (img96, img100) that AB-CROWN cracks in ~8 s but our
bab_refine cascade takes >30 s. Wall: ~16 min total vs AB-CROWN ~34 min.

Three representative cases, one per model family:
  - mnist-point img0 (SAT, ~0.5 s) — PGD root attack on plain MLP.
  - cifar10-set img439 (UNSAT, ~3 s) — exercises the input-normalization
    Mul (scalar) + Add (scalar) path that triggered the scalar-bias
    broadcast fix added in this branch.
  - mnist-set img7 (UNSAT, ~0.5 s) — `n_output` fix (was using last
    ReLU's width instead of the actual 10-class output Add's width
    when the gpu_graph op-list serializer dropped shape metadata).
"""
import pytest
from ._runner import run_case


BENCHMARK_DIR = 'cora_2024'
CONFIG_YAML = 'cora_2024.yaml'

CASES = [
    dict(
        desc='cora mnist-point img0 (SAT, ~0.5s)',
        net='onnx/mnist-point.onnx',
        vnnlib='vnnlib/mnist-img0.vnnlib',
        expected='sat', timeout=30, max_wall_s=8.0,
    ),
    dict(
        desc='cora cifar10-set img398 (UNSAT, scalar-bias normalization)',
        net='onnx/cifar10-set.onnx',
        vnnlib='vnnlib/cifar10-img398.vnnlib',
        expected='verified', timeout=30, max_wall_s=10.0,
    ),
    dict(
        desc='cora mnist-set img7 (UNSAT, n_output fix)',
        net='onnx/mnist-set.onnx',
        vnnlib='vnnlib/mnist-img7.vnnlib',
        expected='verified', timeout=30, max_wall_s=5.0,
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize('case', CASES, ids=[c['desc'] for c in CASES])
def test_cora_2024(case, vnncomp_benchmarks):
    run_case(case, CONFIG_YAML, vnncomp_benchmarks, BENCHMARK_DIR)
