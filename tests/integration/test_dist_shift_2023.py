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
    # SOUNDNESS PROBE — a genuinely-SAT case run with SAT-finding disabled.
    # `_build_alpha_zono_lp` fixed every z_alpha column that wasn't an input
    # symbol or a *listed* unstable e_new to 0. But `state_from_alpha_zono`
    # reserves e_new columns for unstable neurons it skips (no pre-ReLU
    # snapshot — mnist_concat's encoder/generator ReLUs); fixing those at 0
    # collapses each parallelogram to its center line, an UNSOUND ReLU
    # enclosure that excludes reachable outputs. Binarising the classifier
    # ReLUs then cut off a real CEX → the α-zono fallback false-verified this
    # SAT case (ObjBound +0.044 ≥ tol on AWS; the bin-0 LP also disagreed with
    # α-CROWN's LB, −1.19 vs −1.36). With every column free in [-1,1] the
    # relaxation is sound (fallback lb=−0.40 ≤ true margin −0.25); with PGD off
    # the verdict must be `unknown`, NEVER `verified`.
    dict(
        desc='dist_shift index4312 SOUNDNESS (SAT, sat-finding off → must NOT '
             'verify; α-zono orphan-column fix)',
        net='onnx/mnist_concat.onnx',
        vnnlib='vnnlib/index4312_delta0.13.vnnlib',
        expected='unknown', timeout=30, max_wall_s=15.0,
        extra_settings=dict(disable_sat_finding=True),
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize('case', CASES, ids=[c['desc'] for c in CASES])
def test_dist_shift_2023(case, vnncomp_benchmarks):
    run_case(case, CONFIG_YAML, vnncomp_benchmarks, BENCHMARK_DIR)
