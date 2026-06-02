"""Integration tests for acasxu_2023 — input-split BaB with backward-CROWN
intermediate bounds + leaf-PGD + vectorized GPU split.

Runs through the PRODUCTION path (`verify_graph` + `configs/acasxu_2023.yaml`).
The config routes acasxu to the batched input-split BaB with AB-CROWN's
`bound_prop_method: crown` intermediate bounds (`input_split_crown_intermediate`),
NOT the old freeze-replay hybrid (which had no fast deadline and timed out 13/32
on prop_1's wide box). Forward-zono intermediate bounds were ~2x too loose for
ACAS Xu's amplifying weights (root margin fwd-zono -2570 vs crown -1101 on 3_3
prop_2) and diverged; backward CROWN is tighter AND cheaper and converges.

Mechanisms this pins:
  - SAT-finding: narrow-witness SAT cases (1_5/1_9 prop_2/prop_7) are missed by
    root-box PGD; `input_split_leaf_pgd_*` attacks the WORST-margin leaves (the
    witness leaf can never close) and catches them in ~1s.
  - vectorized on-GPU 2-way split: the per-child CPU<->GPU loop was ~80% of wall;
    vectorizing it cut 3_3 prop_2 from ~126s to ~73s (clears the 116s timeout).
  - wide-band α-CROWN boundary closing (`input_split_batched_alpha_boundary_eps:
    10`): plain CROWN leaves a fp32-floor tail of unclosed leaves that get dropped
    as degenerate → unknown on slower GPUs (3_3 prop_2 was 1 degenerate leaf of
    1.88M on the A10G). Wide-band α-CROWN closes them early: 1.88M→431k leaves,
    unknown@63s→verified@22s. (2026-06-02; see docs/benchmarks/acasxu_2023.md.)
  - serial-disjunct routing (`input_split_serial_disjuncts: true`): prop_6's 2
    input sub-boxes are routed through the α-CROWN single-box driver one at a time
    (each gets the full remaining budget) instead of the α-less multi-sub-batched
    path that timed out. prop_6 unknown@116s→verified@7.6s.
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
        # Hardest UNSAT. Plain CROWN left 1 degenerate leaf of 1.88M (unknown@63s
        # on the A10G); wide-band α-CROWN boundary closing fixes it. ~22s A10G /
        # ~24s laptop. The `verified` verdict is the primary guard — a degenerate-
        # leaf regression reverts to `unknown` regardless of wall.
        desc='acasxu 3_3 prop_2 (UNSAT, hardest — α-CROWN boundary closing)',
        net='onnx/ACASXU_run2a_3_3_batch_2000.onnx',
        vnnlib='vnnlib/prop_2.vnnlib',
        expected='verified', timeout=120, max_wall_s=60.0,
    ),
    dict(
        # 2nd-hardest UNSAT (timed out before the vectorized split). ~11s A10G.
        desc='acasxu 4_2 prop_2 (UNSAT, hard — vectorized split)',
        net='onnx/ACASXU_run2a_4_2_batch_2000.onnx',
        vnnlib='vnnlib/prop_2.vnnlib',
        expected='verified', timeout=120, max_wall_s=50.0,
    ),
    dict(
        # Disjunctive-input UNSAT — the regression guard for
        # `input_split_serial_disjuncts`. Without it, prop_6's 2 sub-boxes go to
        # the α-less multi-sub-batched path and time out@116s; with it they each
        # route through the α-CROWN single-box driver. ~6s laptop / ~10s A10G.
        desc='acasxu 1_1 prop_6 (UNSAT, disjunctive-input — serial disjuncts)',
        net='onnx/ACASXU_run2a_1_1_batch_2000.onnx',
        vnnlib='vnnlib/prop_6.vnnlib',
        expected='verified', timeout=120, max_wall_s=40.0,
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
