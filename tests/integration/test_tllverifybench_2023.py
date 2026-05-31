"""Integration tests for tllverifybench_2023.

VNNCOMP regular track. 32 instances of Two-Level-Lattice (TLL) networks:
2-D input → deep `MatMul+Add` chains (3 consecutive linear layers per ReLU) →
8 ReLU layers → 1-D output. The spec is a single halfspace `Y_0 <= c`.

AB-CROWN verifies these with **batched input-split BaB** (no Gurobi): ~369
domains, ~2 s. vibecheck's default routed the pure-ReLU net to the SINGLE-LEAF
fast-leaf path (one 3-iter α-CROWN per leaf, ~3 s/leaf) and timed out. The
batched input-split path (`_input_split_batched`, GPU-parallel, AB-CROWN-style)
already existed but was gated to bilinear (Pow/Div/Mul) nets;
`configs/tllverifybench_2023.yaml` enables it for these → ~1.5 s/case.

vibecheck 29/32 vs AB-CROWN 32/32 (3 harder unsat instances — N=M=24×2,
N=M=40×1 — still time out; see docs/benchmarks/tllverifybench_2023.md).

Three representative cases (all pass): one SAT, two UNSAT (small + cracked-by-
batched).
"""
import pytest
from ._runner import run_case


BENCHMARK_DIR = 'tllverifybench_2023'
CONFIG_YAML = 'tllverifybench_2023.yaml'

CASES = [
    dict(
        desc='tll N=M=8 instance_0_3 (SAT, PGD root attack)',
        net='onnx/tllBench_n=2_N=M=8_m=1_instance_0_3.onnx',
        vnnlib='vnnlib/property_N=8_3.vnnlib',
        expected='sat', timeout=60, max_wall_s=10.0,
    ),
    dict(
        desc='tll N=M=8 instance_0_0 (UNSAT, batched input-split)',
        net='onnx/tllBench_n=2_N=M=8_m=1_instance_0_0.onnx',
        vnnlib='vnnlib/property_N=8_0.vnnlib',
        expected='verified', timeout=60, max_wall_s=12.0,
    ),
    dict(
        desc='tll N=M=16 instance_1_2 (UNSAT, deep net cracked by batched path)',
        net='onnx/tllBench_n=2_N=M=16_m=1_instance_1_2.onnx',
        vnnlib='vnnlib/property_N=16_2.vnnlib',
        expected='verified', timeout=60, max_wall_s=15.0,
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize('case', CASES, ids=[c['desc'] for c in CASES])
def test_tllverifybench_2023(case, vnncomp_benchmarks):
    run_case(case, CONFIG_YAML, vnncomp_benchmarks, BENCHMARK_DIR)
