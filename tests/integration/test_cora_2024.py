"""Integration tests for cora_2024.

VNNCOMP regular track. 9 MLPs (mnist/cifar10/svhn × point/set/trades),
8 FC layers (250-wide hidden, 10 output), 7 ReLUs. Multi-disjunct DNF
specs (9 disjuncts: y_target >= y_j for j != target). 180 instances.

Lean config (2026-06-07): skip the MILP-tighten cascade. The fast GPU Phase-8
BaB + input-split closes the cifar10-set UNSAT cases (img96/img100) in ~12 s
from the phase-0.5 α-CROWN bounds, where the old bab_refine cascade took >30 s
and missed them. So vibecheck 153/180 = AB-CROWN (+2 vs the old 151), 0
false-verify, 0 MISS, all-GPU (no Gurobi). See configs/cora_2024.yaml.

Three representative cases, one per model family:
  - mnist-point img0 (SAT, ~3 s) — PGD root attack on plain MLP.
  - cifar10-set img96 (UNSAT, ~12 s) — was an old-config MISS (cascade >30 s),
    now closed by the lean Phase-8 BaB + input-split.
  - mnist-set img7 (UNSAT, ~5 s) — `n_output` fix (was using last ReLU's width
    instead of the actual 10-class output Add's width when the gpu_graph op-list
    serializer dropped shape metadata).
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
        desc='cora cifar10-set img96 (UNSAT, phase8 dual-ascent BaB)',
        net='onnx/cifar10-set.onnx',
        vnnlib='vnnlib/cifar10-img96.vnnlib',
        expected='verified', timeout=30, max_wall_s=20.0,
    ),
    dict(
        desc='cora mnist-set img7 (UNSAT, n_output fix)',
        net='onnx/mnist-set.onnx',
        vnnlib='vnnlib/mnist-img7.vnnlib',
        expected='verified', timeout=30, max_wall_s=12.0,
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize('case', CASES, ids=[c['desc'] for c in CASES])
def test_cora_2024(case, vnncomp_benchmarks):
    run_case(case, CONFIG_YAML, vnncomp_benchmarks, BENCHMARK_DIR)
