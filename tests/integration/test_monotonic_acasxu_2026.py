"""Integration pin for monotonic_acasxu_2026 — the network-pair monotonicity path.

The net field is [('f',net),('g',net)] (f=g); network_pair merges it into one ONNX evaluating
the net at two correlated input points (x_g = clamp(x_f - delta*e_k, ...)), verified with the
default input-split profile. All 50 instances are sat (real monotonicity violations); pins two.
Needs the 2026 2.0 clone + a CUDA GPU; skips otherwise. See
docs/benchmarks/monotonic_acasxu_2026.md.
"""
import csv
import os

import pytest

_CAND = [
    os.path.expanduser('~/repositories/vnncomp2026_benchmarks/benchmarks/'
                       'monotonic_acasxu_2026/2.0'),
    os.path.expanduser('~/Desktop/repositories/vnncomp2026_benchmarks/benchmarks/'
                       'monotonic_acasxu_2026/2.0'),
]
CASES = ['instance_0.vnnlib', 'instance_1.vnnlib']   # both sat


def _bench():
    return next((d for d in _CAND if os.path.isdir(d)), None)


def _field(bench, stem):
    for row in csv.reader(open(f'{bench}/instances.csv')):
        if row and row[0].strip() and os.path.basename(row[1].strip()) == stem:
            return row[0].strip()
    return None


@pytest.mark.integration
@pytest.mark.parametrize('stem', CASES)
def test_mono_pair(stem):
    bench = _bench()
    if bench is None:
        pytest.skip('monotonic_acasxu_2026 2.0 not found locally')
    import torch
    if not torch.cuda.is_available():
        pytest.skip('merged-pair verification needs a CUDA GPU for the pinned timings')
    field = _field(bench, stem)
    if field is None or not os.path.exists(f'{bench}/vnnlib/{stem}'):
        pytest.skip('mono instance missing')

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
    assert result == 'sat', f'{stem}: got {result!r}, expected sat'
