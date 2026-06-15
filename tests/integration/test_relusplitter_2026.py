"""Integration tests for relusplitter_2026 (VNNCOMP 2026, regular track).

MNIST Gemm+ReLU MLPs in a base form plus 4 ReLU-split variants (pct0.2..1.0):
each split inserts a `Gemm(C->C+S) -> ReLU -> Gemm(C+S->C) -> ReLU` block whose
S extra neurons are ±-paired. `fold_gemm` (auto-applied, default
optimize_relu_relation=True) collapses every split back to the exact base layer
sizes via ReLU(z) - ReLU(-z) = z, so a split net verifies on the same (smaller)
graph as its base. AB-CROWN's auto_LiRPA backend has the same fold
(optimize_relu_relation), but it CRASHES (IndexError in optimize_graph.py:330)
on certain split nets — so VC's fold_gemm is the more robust re-implementation
and VC beats ABC here (VC 120/120 vs ABC 87/111, agreeing on all 87 it solved).

VC solves 120/120 on the 2.0 (v2 vnnlib) set with DEFAULT settings (no custom
config): 108 unsat (verified) + 12 sat (counterexamples), zero unknown/timeout.
The v2 vnnlib (tensor-indexed `X[0,i]` / `Y[0,i]`) is loaded by the ported v2
parser; v1 and v2 produce an identical VNNSpec (equivalence oracle).

Cases pin:
  - 1 SAT: model_1_1 pct1.0 split — a real counterexample found in ~1.5s
    THROUGH the fold (the split is collapsed, then the base net misclassifies
    inside the eps ball). Exercises fold + SAT + v2 loader.
  - 1 hard UNSAT, high-pct split (model_2_2 pct1.0, ~49 s on RTX-3080) — the
    slowest instance; pins the fold AND acts as a perf-regression catcher.
  - 1 base UNSAT (model_3_3, ~1-2 s) — base net (no fold), validates the v2
    loader + basic verify path.

Run on a GPU (server1 RTX-3080 / AWS A10G); wall bounds are loose
(regression catchers, ~1.5-2x observed), not perf benchmarks.
"""
import pytest
from ._runner import run_case


BENCHMARK_DIR = 'relusplitter_2026/2.0'
CONFIG_YAML = 'relusplitter_2026.yaml'  # absent -> default settings (solve 120/120)

CASES = [
    dict(
        desc='model_1_1 pct1.0 (SAT, counterexample through the fold, ~1.5s)',
        net=('onnx/model_1_1~d1_eps_0.02_sample_2_label_9~pct1.0~cnt128~'
             'seed850851855.onnx'),
        vnnlib='vnnlib/d1_eps_0.02_sample_2_label_9.vnnlib',
        expected='sat', timeout=180, max_wall_s=15.0,
    ),
    dict(
        desc='model_2_2 pct1.0 (UNSAT, hardest; pins fold + perf, ~49s RTX-3080)',
        net=('onnx/model_2_2~d2_eps_0.04_sample_14_label_7~pct1.0~cnt128~'
             'seed850851855.onnx'),
        vnnlib='vnnlib/d2_eps_0.04_sample_14_label_7.vnnlib',
        expected='verified', timeout=180, max_wall_s=95.0,
    ),
    dict(
        desc='model_3_3 base (UNSAT, base net no fold, v2 loader path, ~1-2s)',
        net='onnx/model_3_3.onnx',
        vnnlib='vnnlib/d3_eps_0.01_sample_9_label_0.vnnlib',
        expected='verified', timeout=180, max_wall_s=15.0,
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize('case', CASES, ids=[c['desc'] for c in CASES])
def test_relusplitter_2026(case, vnncomp_benchmarks):
    run_case(case, CONFIG_YAML, vnncomp_benchmarks, BENCHMARK_DIR)
