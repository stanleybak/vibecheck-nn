"""Integration tests for collins_rul_cnn_2022.

VNNCOMP regular track. Small CNNs (Conv×6 + ReLU×5 + Dropout + Flatten)
predicting remaining useful life of jet engines. 62 instances across 3
models: NN_rul_small_window_20, NN_rul_full_window_20,
NN_rul_full_window_40 × {robustness, monotonicity, if_then} specs.

Defaults work — every case is closed by CROWN's first pass (UNSAT) or
PGD's root attack (SAT). 62/62 in ~3 s vs AB-CROWN's 430 s (~140×).

The Dropout op in the model exercises the gpu_graph passthrough alias
introduced in this branch (Dropout/Identity/Cast are skipped at emit
but downstream consumers are aliased to the upstream producer).

Three representative cases, one per model family:
  - small_window_20 robustness_2perturbations_delta5 (SAT, ~0.5 s) —
    SAT path via root PGD.
  - full_window_20 robustness_8perturbations_delta40 (UNSAT, ~0.01 s) —
    CROWN closes immediately.
  - full_window_40 monotonicity_CI_shift20 (UNSAT, ~0.01 s) —
    monotonicity spec on the largest model.
"""
import pytest
from ._runner import run_case


BENCHMARK_DIR = 'collins_rul_cnn_2022'
CONFIG_YAML = 'collins_rul_cnn_2022.yaml'

CASES = [
    dict(
        desc='collins small_window_20 robustness_2pert_delta5 (SAT, ~0.5s)',
        net='onnx/NN_rul_small_window_20.onnx',
        vnnlib='vnnlib/robustness_2perturbations_delta5_epsilon10_w20.vnnlib',
        expected='sat', timeout=30, max_wall_s=5.0,
    ),
    dict(
        desc='collins full_window_20 robustness_8pert_delta40 (UNSAT, ~0.01s)',
        net='onnx/NN_rul_full_window_20.onnx',
        vnnlib='vnnlib/robustness_8perturbations_delta40_epsilon10_w20.vnnlib',
        expected='verified', timeout=30, max_wall_s=5.0,
    ),
    dict(
        desc='collins full_window_40 monotonicity_CI_shift20 (UNSAT, ~0.01s)',
        net='onnx/NN_rul_full_window_40.onnx',
        vnnlib='vnnlib/monotonicity_CI_shift20_w40.vnnlib',
        expected='verified', timeout=30, max_wall_s=5.0,
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize('case', CASES, ids=[c['desc'] for c in CASES])
def test_collins_rul_cnn_2022(case, vnncomp_benchmarks):
    run_case(case, CONFIG_YAML, vnncomp_benchmarks, BENCHMARK_DIR)
