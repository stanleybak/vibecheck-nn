"""Integration pin for adaptive_cruise_control_non_linear_2026 — the nonlinear-v2
augment path. The spec is NONLINEAR (degree-2 atoms in X and Y + a nonlinear input
constraint), so the standard _runner path (load_vnnlib on the v2 spec) doesn't
apply: nonlinear_augment transpiles each instance to an augmented ONNX (runs f,
emits each constraint polynomial as an extra output) + a linear v1 DNF spec, which
routes through the acopf/trig verifier (CPU+f64 bounds, deterministic grid CE finder
+ CPU-f32 PGD).

These are tolerance-BOUNDARY instances (true f64 min margin = 0); the scorer runs
the net in float32 where rounding yields a clear CE, so the verdict is `sat`. Pins a
few ABC-sat instances that the deterministic grid (configs/adaptive_cruise_control_
non_linear_2026.yaml: pgd_sat_grid_*) cracks. Needs the 2026 2.0 clone; skips
otherwise. See docs/benchmarks/adaptive_cruise_control_2026.md.
"""
import os

import pytest

_CANDIDATES = [
    os.path.expanduser('~/repositories/vnncomp2026_benchmarks/benchmarks/'
                       'adaptive_cruise_control_non_linear_2026/2.0'),
    os.path.expanduser('~/Desktop/repositories/vnncomp2026_benchmarks/benchmarks/'
                       'adaptive_cruise_control_non_linear_2026/2.0'),
]

# ABC-sat instances the grid finder cracks (clear float32 CEs in the e2e run).
CASES = [3, 8, 33]


def _bench_dir():
    return next((d for d in _CANDIDATES if os.path.isdir(d)), None)


@pytest.mark.integration
@pytest.mark.parametrize('inst', CASES)
def test_adaptive_nonlinear_augment_sat(inst):
    bench = _bench_dir()
    if bench is None:
        pytest.skip('adaptive_cruise_control_non_linear_2026 2.0 not found locally')
    import glob
    nets = glob.glob(f'{bench}/onnx/*.onnx')
    vnnlib = f'{bench}/vnnlib/instance_{inst}.vnnlib'
    if not nets or not os.path.exists(vnnlib):
        pytest.skip('adaptive files missing')
    net = nets[0]

    from vibecheck.nonlinear_augment import build_augmented_instance
    from vibecheck.network import ComputeGraph
    from vibecheck.vnnlib_loader import load_vnnlib
    from vibecheck.settings import default_settings
    from vibecheck.config_loader import load_config
    from vibecheck.verify_graph import verify_graph, _validate_sat_witness
    import numpy as np

    aug_onnx, aug_vnnlib = build_augmented_instance(net, vnnlib)
    graph = ComputeGraph.from_onnx(aug_onnx, dtype=np.float64)
    graph.onnx_path = aug_onnx
    spec = load_vnnlib(aug_vnnlib)

    cfg = os.path.join(os.path.dirname(__file__), '..', '..', 'configs',
                       'adaptive_cruise_control_non_linear_2026.yaml')
    # device defaults to GPU (the acopf/trig path follows settings.device);
    # pin pgd_seed=0 so the CE search is reproducible (seeds 0..9 all crack
    # these — see docs). _resolve_device falls back to CPU if no CUDA.
    overrides = dict(device='gpu', bits=32, total_timeout=100, pgd_seed=0)
    overrides.update(load_config(cfg))
    settings = default_settings(**overrides)
    settings.print_progress = False
    graph.optimize(settings)

    result, det = verify_graph(graph, spec, settings)
    assert result == 'sat', f'instance_{inst}: expected sat, got {result!r}'
    # the witness must violate the (augmented) spec on ORT within tolerance
    w = np.asarray(det['witness']).flatten()
    ok, _info = _validate_sat_witness(aug_onnx, spec, w)
    assert ok, f'instance_{inst}: emitted witness rejected by ORT'
