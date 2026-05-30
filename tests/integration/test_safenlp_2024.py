"""Integration tests for safenlp_2024.

Tiny pure-ReLU FC nets (30 -> 128 ReLU -> 2) where the bias is exported as a
SEPARATE Add node. vibecheck auto-routes these to the exact per-neuron MILP
(`milp_verify`), the same approach AB-CROWN uses here (complete_verifier: mip).
One SAT (PGD) + two of the hardest UNSAT cases:

  - ruarobot hyperrectangle_132 (SAT, fast PGD) — counterexample exists; the
    PGD path finds it. Regression catcher for the SAT branch.
  - ruarobot hyperrectangle_992 (hard UNSAT) — 30-dim input-split BaB explodes
    here (it timed out at 20s before the exact-MILP route landed). The exact
    big-M MILP over the 128 ReLU binaries proves Y_0 > Y_1. Regression catcher
    for the gpu_layers Add-bias fold (without it the flattened net had all-zero
    biases and the MILP reported a spurious counterexample) and the FC-only
    direct-exact racing schedule.
  - ruarobot hyperrectangle_3558 (hard UNSAT) — second input-split-explode case;
    same exact-MILP route.

The VERDICT is the primary regression signal: all three fixes this benchmark
relies on manifest as a verdict flip if reverted — losing the gpu_layers Add-bias
fold makes the UNSAT cases spuriously 'sat'; losing the FC-only direct-exact
racing schedule or the auto-route makes them time out to 'unknown'. The
max_wall_s bounds are deliberately LOOSE: these exact MILPs run ~1.5s locally but
~17s on slower Gurobi hardware (server1), a documented ~15x spread, so a tight
wall bound would false-fail rather than catch a real regression.
"""
import pytest
from ._runner import run_case


BENCHMARK_DIR = 'safenlp_2024'
CONFIG_YAML = 'safenlp_2024.yaml'

CASES = [
    dict(
        desc='safenlp ruarobot hyperrectangle_132 (SAT, fast PGD)',
        net='onnx/ruarobot/perturbations_0.onnx',
        vnnlib='vnnlib/ruarobot/hyperrectangle_132.vnnlib',
        expected='sat', timeout=20, max_wall_s=10.0,
    ),
    dict(
        desc='safenlp ruarobot hyperrectangle_992 (hard UNSAT, exact MILP)',
        net='onnx/ruarobot/perturbations_0.onnx',
        vnnlib='vnnlib/ruarobot/hyperrectangle_992.vnnlib',
        expected='verified', timeout=20, max_wall_s=20.0,
    ),
    dict(
        desc='safenlp ruarobot hyperrectangle_3558 (hard UNSAT, exact MILP)',
        net='onnx/ruarobot/perturbations_0.onnx',
        vnnlib='vnnlib/ruarobot/hyperrectangle_3558.vnnlib',
        expected='verified', timeout=20, max_wall_s=20.0,
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize('case', CASES, ids=[c['desc'] for c in CASES])
def test_safenlp_2024(case, vnncomp_benchmarks):
    run_case(case, CONFIG_YAML, vnncomp_benchmarks, BENCHMARK_DIR)
