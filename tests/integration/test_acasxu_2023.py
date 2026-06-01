"""Integration tests for acasxu_2023 — the freeze-replay α-CROWN BaB.

Runs through the PRODUCTION path (`verify_graph` + `configs/acasxu_2023.yaml`),
which now routes acasxu to `verify_hybrid` via `use_hybrid_acasxu: true`. This is
the fix for a real disconnect: the batched input-split BaB that verify_graph used
for acasxu propagates FORWARD-zonotope intermediate bounds, which are ~1000x too
loose for ACAS Xu's amplifying weights (root spec margin -1597 vs true >0) — so it
DIVERGED (6.8M leaves) and timed out on 3_3 prop_2. The freeze-replay path
(`_full_freeze` / `_replay_batched`) tightens per-layer pre-ReLU bounds with
backward α-CROWN (intersected with forward-zono, so still sound) and converges.

(The test previously called `verify_hybrid` DIRECTLY, so it passed while the
production pipeline silently diverged — exactly the gap this rewrite closes.)

Three cases (120s budget; AB-CROWN does them in 8-18s, we're slower but solve):
  - 1_5 prop_2 (SAT, narrow witness) — between-rounds PGD on the worst leaf.
  - 3_3 prop_2 (UNSAT, ~46s) — the hardest case; drove the freeze design.
  - 1_1 prop_3 (UNSAT, ~30s) — second-hardest.
"""
import pytest
from ._runner import run_case


BENCHMARK_DIR = 'acasxu_2023'
CONFIG_YAML = 'acasxu_2023.yaml'

CASES = [
    dict(
        desc='acasxu 1_5 prop_2 (SAT, between-rounds PGD)',
        net='onnx/ACASXU_run2a_1_5_batch_2000.onnx',
        vnnlib='vnnlib/prop_2.vnnlib',
        expected='sat', timeout=60, max_wall_s=20.0,
    ),
    dict(
        desc='acasxu 3_3 prop_2 (UNSAT, hardest ~46s — diverged before the fix)',
        net='onnx/ACASXU_run2a_3_3_batch_2000.onnx',
        vnnlib='vnnlib/prop_2.vnnlib',
        expected='verified', timeout=120, max_wall_s=90.0,
    ),
    dict(
        desc='acasxu 1_1 prop_3 (UNSAT, ~30s)',
        net='onnx/ACASXU_run2a_1_1_batch_2000.onnx',
        vnnlib='vnnlib/prop_3.vnnlib',
        expected='verified', timeout=120, max_wall_s=70.0,
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize('case', CASES, ids=[c['desc'] for c in CASES])
def test_acasxu_2023(case, vnncomp_benchmarks):
    run_case(case, CONFIG_YAML, vnncomp_benchmarks, BENCHMARK_DIR)
