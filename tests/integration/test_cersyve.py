"""Integration tests for cersyve.

Cases probed locally with the batched + clipping config (12/12 in
60s/case). One SAT for the PGD path + two of the hardest UNSAT cases
(previously stuck even at 600s/case before clipping landed):

  - pendulum_pretrain_inv (SAT, ~0.3s) — PGD finds a witness on the AND-
    conjunct spec. Regression catcher for the PGD sign-flip + conjunct
    margin bugs that made all 6 SAT cases return 'unknown' before this
    benchmark.
  - point_mass_finetune_inv (UNSAT, ~17s, 4-D input) — without domain
    clipping this case is unverified even at 600s. Regression catcher
    for the batched BaB + clipping path.
  - lane_keep_finetune_inv (UNSAT, ~15s, 4-D input) — same as above;
    the second 4-D `_finetune_inv` boundary case.
"""
import pytest
from ._runner import run_case


BENCHMARK_DIR = 'cersyve'
CONFIG_YAML = 'cersyve.yaml'

CASES = [
    dict(
        desc='cersyve pendulum_pretrain_inv (SAT, fast PGD)',
        net='onnx/pendulum_pretrain_inv.onnx.gz',
        vnnlib='vnnlib/prop_pendulum.vnnlib.gz',
        expected='sat', timeout=30, max_wall_s=10.0,
    ),
    dict(
        desc='cersyve point_mass_finetune_inv (hard UNSAT, 4-D input)',
        net='onnx/point_mass_finetune_inv.onnx.gz',
        vnnlib='vnnlib/prop_point_mass.vnnlib.gz',
        expected='verified', timeout=60, max_wall_s=40.0,
    ),
    dict(
        desc='cersyve lane_keep_finetune_inv (hard UNSAT, 4-D input)',
        net='onnx/lane_keep_finetune_inv.onnx.gz',
        vnnlib='vnnlib/prop_lane_keep.vnnlib.gz',
        expected='verified', timeout=60, max_wall_s=40.0,
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize('case', CASES, ids=[c['desc'] for c in CASES])
def test_cersyve(case, vnncomp_benchmarks):
    run_case(case, CONFIG_YAML, vnncomp_benchmarks, BENCHMARK_DIR)
