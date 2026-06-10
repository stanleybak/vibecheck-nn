"""Integration tests for yolo_2023 (extended track).

TinyYOLO: ResNet-style ReLU CNN with identity Pads (dropped by
drop_identity_pads), AveragePools (emitted as equivalent depthwise-uniform
convs), and CONJUNCTIVE disjuncts — each spec disjunct is an AND of 5
objectness constraints (Y_i <= -1), so the suite pins the ANY-closure
semantics (refuting one conjunct closes the disjunct). The benchmark has no
SAT cases (ABC: 62 unsat / 10 timeout of 72), so all pins are UNSAT:
  - 1 fast unsat (zono/joint-alpha closes a conjunct early)
  - 1 ANY-closure regression guard (individually-unrefutable conjunct present)
  - 1 mid-depth unsat (per-query zono-lift alpha-CROWN does the closing)
Timings observed on the AWS g5 A10G full sweep 2026-06-09 (archived in
scratch/yolo_sweep_2026-06-09/); max_wall_s ~1.5x observed.
"""
import pytest
from ._runner import run_case


BENCHMARK_DIR = 'yolo_2023'
CONFIG_YAML = 'yolo_2023.yaml'

CASES = [
    dict(
        desc='TinyYOLO prop_000306 (fast UNSAT, ~29s)',
        net='onnx/TinyYOLO.onnx',
        vnnlib='vnnlib/TinyYOLO_prop_000306_eps_1_255.vnnlib',
        expected='verified', timeout=300, max_wall_s=45.0,
    ),
    dict(
        # ANY-closure regression guard: this spec has a conjunct at
        # lo ~ -8 (individually unrefutable) while plain zono already
        # refutes sibling conjunct Y_760 (margin +0.49). With the old
        # all()-closure this case can NEVER verify; with any() it closes
        # in ~37s. Revert f303be8 -> this times out.
        desc='TinyYOLO prop_000299 (UNSAT, ANY-closure guard, ~37s)',
        net='onnx/TinyYOLO.onnx',
        vnnlib='vnnlib/TinyYOLO_prop_000299_eps_1_255.vnnlib',
        expected='verified', timeout=300, max_wall_s=60.0,
    ),
    dict(
        desc='TinyYOLO prop_000459 (UNSAT, zono-lift closes, ~55s)',
        net='onnx/TinyYOLO.onnx',
        vnnlib='vnnlib/TinyYOLO_prop_000459_eps_1_255.vnnlib',
        expected='verified', timeout=300, max_wall_s=85.0,
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize('case', CASES, ids=[c['desc'] for c in CASES])
def test_yolo_2023(case, vnncomp_benchmarks):
    run_case(case, CONFIG_YAML, vnncomp_benchmarks, BENCHMARK_DIR)
