"""Integration tests for acasxu_2023 — the hybrid α-CROWN BaB pipeline.

Cases probed locally with 120s/case budget (186/186 in full sweep):

  - 1_5 prop_2 (SAT, narrow witness) — root PGD misses it; between-rounds
    PGD on the worst BaB leaf catches it. Regression catcher for the
    SAT-finding path + the multi-disjunct fix.
  - 3_3 prop_2 (UNSAT, hardest case ~47s) — the case that drove the entire
    32-split α-CROWN freeze + per-leaf spec α-opt design. AB-CROWN solves
    in 18s on the same hardware; we're slower but solve it.
  - 1_1 prop_3 (UNSAT, ~33s) — second-hardest UNSAT. Catches regressions
    in the per-query α machinery.

Calls verify_hybrid directly (not verify_graph) — the hybrid runner is
the canonical acasxu pipeline; verify_graph's selective-α path is now
considered legacy for this benchmark.
"""
import time
from pathlib import Path

import numpy as np
import pytest


CASES = [
    dict(
        desc='acasxu 1_5 prop_2 (SAT via between-rounds PGD)',
        net='onnx/ACASXU_run2a_1_5_batch_2000.onnx.gz',
        vnnlib='vnnlib/prop_2.vnnlib.gz',
        expected='sat', timeout=60, max_wall_s=15.0,
    ),
    dict(
        desc='acasxu 3_3 prop_2 (hardest UNSAT, ~47s)',
        net='onnx/ACASXU_run2a_3_3_batch_2000.onnx.gz',
        vnnlib='vnnlib/prop_2.vnnlib.gz',
        expected='unsat', timeout=120, max_wall_s=75.0,
    ),
    dict(
        desc='acasxu 1_1 prop_3 (second-hardest UNSAT, ~33s)',
        net='onnx/ACASXU_run2a_1_1_batch_2000.onnx.gz',
        vnnlib='vnnlib/prop_3.vnnlib.gz',
        expected='unsat', timeout=120, max_wall_s=50.0,
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize('case', CASES, ids=[c['desc'] for c in CASES])
def test_acasxu_2023(case, vnncomp_benchmarks):
    from vibecheck.network import ComputeGraph
    from vibecheck.vnnlib_loader import load_vnnlib
    from vibecheck.verify_hybrid_acasxu import verify_hybrid

    net_path = (Path(vnncomp_benchmarks) / 'acasxu_2023' / case['net'])
    vnn_path = (Path(vnncomp_benchmarks) / 'acasxu_2023' / case['vnnlib'])
    if not net_path.exists():
        # try without .gz suffix
        alt = Path(str(net_path)[:-3])
        if alt.exists(): net_path = alt
    if not vnn_path.exists():
        alt = Path(str(vnn_path)[:-3])
        if alt.exists(): vnn_path = alt
    assert net_path.exists(), f'missing net: {net_path}'
    assert vnn_path.exists(), f'missing vnnlib: {vnn_path}'

    graph = ComputeGraph.from_onnx(str(net_path), dtype=np.float32)
    spec = load_vnnlib(str(vnn_path))

    t0 = time.perf_counter()
    res = verify_hybrid(graph, spec, timeout=int(case['timeout']))
    wall = time.perf_counter() - t0

    assert res['verdict'] == case['expected'], (
        f"{case['desc']}: got {res['verdict']!r} (wall={wall:.1f}s, "
        f"phase={res.get('phase', '?')}); expected {case['expected']!r}")
    assert wall <= case['max_wall_s'], (
        f"{case['desc']}: verdict ok ({res['verdict']}) but wall={wall:.1f}s "
        f"exceeded budget {case['max_wall_s']:.1f}s — perf regression?")
