"""Integration pin for isomorphic_acasxu_2026 — the network-pair equivalence path.

Each instance's net field is a pair [('f',original),('g',perturbed)]; network_pair merges them
into ONE ONNX whose outputs are the per-output diffs Y_g[j]-Y_f[j] (oracle-gated), verified with
the default input-split profile (the merged net is ACAS Xu-shaped). Pins VC=ABC parity on an
unsat (equivalent) and a sat (not equivalent) case. Needs the 2026 2.0 clone + a CUDA GPU;
skips otherwise. See docs/benchmarks/isomorphic_acasxu_2026.md.
"""
import csv
import os

import pytest

_CAND = [
    os.path.expanduser('~/repositories/vnncomp2026_benchmarks/benchmarks/'
                       'isomorphic_acasxu_2026/2.0'),
    os.path.expanduser('~/Desktop/repositories/vnncomp2026_benchmarks/benchmarks/'
                       'isomorphic_acasxu_2026/2.0'),
]
# 'verified' = UNSAT proved (equivalent); 'sat' = a >0.05 output difference found.
CASES = [('instance_0.vnnlib', 'verified'), ('instance_29.vnnlib', 'sat')]


def _bench():
    return next((d for d in _CAND if os.path.isdir(d)), None)


def _field(bench, stem):
    for row in csv.reader(open(f'{bench}/instances.csv')):
        if row and row[0].strip() and os.path.basename(row[1].strip()) == stem:
            return row[0].strip()
    return None


@pytest.mark.integration
@pytest.mark.parametrize('stem,expected', CASES)
def test_iso_pair(stem, expected):
    bench = _bench()
    if bench is None:
        pytest.skip('isomorphic_acasxu_2026 2.0 not found locally')
    import torch
    if not torch.cuda.is_available():
        pytest.skip('merged-pair input-split BaB needs a CUDA GPU for the pinned timings')
    field = _field(bench, stem)
    if field is None or not os.path.exists(f'{bench}/vnnlib/{stem}'):
        pytest.skip('iso instance missing')

    import numpy as np
    from vibecheck import network_pair as npair
    from vibecheck.network import ComputeGraph
    from vibecheck.vnnlib_loader import load_vnnlib
    from vibecheck.config_profiles import default_settings_for
    from vibecheck.verify_graph import verify_graph

    merged_onnx, merged_spec = npair.build_merged_instance(field, f'{bench}/vnnlib/{stem}',
                                                           base_dir=bench)
    graph = ComputeGraph.from_onnx(merged_onnx, dtype=np.float32)
    spec = load_vnnlib(merged_spec)
    settings = default_settings_for(graph, spec, device='gpu', bits=32,
                                    total_timeout=100, pgd_restarts=100)
    settings.print_progress = False
    graph.optimize(settings)
    result, _ = verify_graph(graph, spec, settings)
    assert result == expected, f'{stem}: got {result!r}, expected {expected!r}'
