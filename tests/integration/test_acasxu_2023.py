"""Integration tests for acasxu_2023 — input-split BaB with backward-CROWN
intermediate bounds + leaf-PGD + vectorized GPU split.

Runs through the PRODUCTION path (`verify_graph` + `configs/acasxu_2023.yaml`).
The config routes acasxu to the batched input-split BaB with AB-CROWN's
`bound_prop_method: crown` intermediate bounds (`input_split_crown_intermediate`),
NOT the old freeze-replay hybrid (which had no fast deadline and timed out 13/32
on prop_1's wide box). Forward-zono intermediate bounds were ~2x too loose for
ACAS Xu's amplifying weights (root margin fwd-zono -2570 vs crown -1101 on 3_3
prop_2) and diverged; backward CROWN is tighter AND cheaper and converges.

Three mechanisms this pins:
  - SAT-finding: narrow-witness SAT cases (1_5/1_9 prop_2/prop_7) are missed by
    root-box PGD; `input_split_leaf_pgd_*` attacks the WORST-margin leaves (the
    witness leaf can never close) and catches them in ~1s.
  - vectorized on-GPU 2-way split: the per-child CPU<->GPU loop was ~80% of wall;
    vectorizing it cut 3_3 prop_2 from ~126s to ~73s (clears the 116s timeout).
  - soundness: bounds are zono ∩ backward-CROWN (two over-approximations).

AB-CROWN does these in 8-18s; we're slower but solve all within timeout.
"""
import pytest
from ._runner import run_case


BENCHMARK_DIR = 'acasxu_2023'
CONFIG_YAML = 'acasxu_2023.yaml'

CASES = [
    dict(
        desc='acasxu 1_5 prop_2 (SAT, narrow witness via worst-margin leaf-PGD)',
        net='onnx/ACASXU_run2a_1_5_batch_2000.onnx',
        vnnlib='vnnlib/prop_2.vnnlib',
        expected='sat', timeout=60, max_wall_s=20.0,
    ),
    dict(
        # Multi-disjunct SAT (prop_7 = 2 disjuncts). Root-box PGD can't find it
        # (200k restarts fail); pins that worst-margin leaf-PGD catches it.
        desc='acasxu 1_9 prop_7 (SAT, multi-disjunct, leaf-PGD)',
        net='onnx/ACASXU_run2a_1_9_batch_2000.onnx',
        vnnlib='vnnlib/prop_7.vnnlib',
        expected='sat', timeout=120, max_wall_s=35.0,
    ),
    dict(
        # Hardest UNSAT; drove the vectorized-split work (~73s, was ~126s).
        desc='acasxu 3_3 prop_2 (UNSAT, hardest — vectorized split)',
        net='onnx/ACASXU_run2a_3_3_batch_2000.onnx',
        vnnlib='vnnlib/prop_2.vnnlib',
        expected='verified', timeout=120, max_wall_s=130.0,
    ),
    dict(
        # 2nd-hardest UNSAT (also timed out before the vectorized split).
        desc='acasxu 4_2 prop_2 (UNSAT, hard — vectorized split)',
        net='onnx/ACASXU_run2a_4_2_batch_2000.onnx',
        vnnlib='vnnlib/prop_2.vnnlib',
        expected='verified', timeout=120, max_wall_s=130.0,
    ),
    dict(
        desc='acasxu 1_1 prop_3 (UNSAT, fast)',
        net='onnx/ACASXU_run2a_1_1_batch_2000.onnx',
        vnnlib='vnnlib/prop_3.vnnlib',
        expected='verified', timeout=120, max_wall_s=70.0,
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize('case', CASES, ids=[c['desc'] for c in CASES])
def test_acasxu_2023(case, vnncomp_benchmarks):
    run_case(case, CONFIG_YAML, vnncomp_benchmarks, BENCHMARK_DIR)
