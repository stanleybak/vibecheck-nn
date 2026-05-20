"""Shared integration-test harness.

Each per-benchmark file declares `CASES` (list of dicts) and calls
`run_case(case, config_yaml, vnncomp_root, benchmark_dir)`. The harness:
  - loads net + spec
  - applies the benchmark's `configs/<benchmark>.yaml` overrides
  - calls `verify_graph` in-process (subprocess isolation is left to the
    sweep harness; integration tests run in pytest and accept process
    sharing)
  - asserts the verdict matches `case['expected']`
  - asserts wall time <= `case['max_wall_s']` (loose bound — regression
    catcher, not a perf benchmark)

Case schema:
  {
    'net': 'onnx/SomeNet.onnx',           # relative to <benchmark>/
    'vnnlib': 'vnnlib/prop_5_0.05.vnnlib',
    'expected': 'sat' | 'verified',       # 'verified' = UNSAT proved
    'max_wall_s': 30.0,
    'timeout': 100,                       # vnncomp per-instance budget
    'desc': 'one-line human description',  # for failure messages
  }
"""
import time
from pathlib import Path

import numpy as np


def run_case(case, config_yaml, vnncomp_root, benchmark_dir):
    """Run one integration case and assert verdict + wall time."""
    from vibecheck.network import ComputeGraph
    from vibecheck.vnnlib_loader import load_vnnlib
    from vibecheck.settings import default_settings
    from vibecheck.config_loader import load_config
    from vibecheck.verify_graph import verify_graph

    net_path = Path(vnncomp_root) / benchmark_dir / case['net']
    vnn_path = Path(vnncomp_root) / benchmark_dir / case['vnnlib']
    assert net_path.exists(), f'missing net: {net_path}'
    assert vnn_path.exists(), f'missing vnnlib: {vnn_path}'

    graph = ComputeGraph.from_onnx(str(net_path), dtype=np.float32)
    spec = load_vnnlib(str(vnn_path))

    cfg_path = Path(__file__).resolve().parents[2] / 'configs' / config_yaml
    yaml_overrides = load_config(str(cfg_path)) if cfg_path.exists() else {}
    overrides = dict(
        device='gpu', bits=32,
        total_timeout=int(case['timeout']),
        pgd_restarts=int(case.get('pgd_restarts', 100)))
    overrides.update(yaml_overrides)
    settings = default_settings(**overrides)
    settings.print_progress = False
    graph.optimize(settings)

    t0 = time.perf_counter()
    result, _det = verify_graph(graph, spec, settings)
    wall = time.perf_counter() - t0

    assert result == case['expected'], (
        f"{case['desc']}: got {result!r} (wall={wall:.1f}s); "
        f"expected {case['expected']!r}")
    assert wall <= case['max_wall_s'], (
        f"{case['desc']}: verdict ok ({result}) but wall={wall:.1f}s "
        f"exceeded budget {case['max_wall_s']:.1f}s — perf regression?")
