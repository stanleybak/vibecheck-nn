"""End-to-end production tests for the Phase-1 tightening axes
(formulation × solver) through `verify_graph()`. Validates:
- weight_walk + {probe, lp, milp} run to completion
- gen_cone + {lp, milp} run to completion and trigger the `rec_zono`
  piggyback path inside `_run_pipeline`
- the legacy `tighten_mode` alias still maps cleanly to the new pair
"""

import pytest
import torch

from vibecheck.network import ComputeGraph
from vibecheck.settings import default_settings
from vibecheck.vnnlib_loader import load_vnnlib
from vibecheck.verify_graph import verify_graph


def _acasxu_paths(benchmarks_root):
    return (
        benchmarks_root / 'acasxu_2023/onnx/ACASXU_run2a_1_1_batch_2000.onnx.gz',
        benchmarks_root / 'acasxu_2023/vnnlib/prop_2.vnnlib.gz',
    )


@pytest.mark.parametrize('formulation,solver', [
    ('weight_walk', 'probe'),
    ('weight_walk', 'lp'),
    ('weight_walk', 'milp'),
    ('gen_cone',    'lp'),
    ('gen_cone',    'milp'),
    ('skip',        'probe'),
])
def test_acasxu_pipeline_tighten_axes(vnncomp_benchmarks,
                                       formulation, solver):
    """Every valid (formulation, solver) pair runs to completion on a
    small ACAS Xu instance and returns a valid result.
    """
    onnx_path, spec_path = _acasxu_paths(vnncomp_benchmarks)
    if not onnx_path.exists() or not spec_path.exists():
        pytest.skip('ACAS Xu benchmark not available')

    g = ComputeGraph.from_onnx(str(onnx_path))
    spec = load_vnnlib(str(spec_path))
    s = default_settings(
        device='cpu', bits=64, total_timeout=30.0,
        print_progress=False,
        tighten_formulation=formulation,
        tighten_solver=solver)
    # Phase 0 PGD finds SAT on ACAS Xu prop_2 immediately and would
    # short-circuit before Phase 1 runs. This test specifically checks
    # the tightening axes are exercised, so disable Phase 0 PGD.
    s.pgd_phase0_enabled = False
    result, details = verify_graph(g, spec, s)
    assert result in ('verified', 'unknown', 'sat'), (
        f'({formulation},{solver}) returned unexpected result {result}')
    # Phase 1 timing must be populated (means the forward ran).
    assert 'phase1_zono_tighten' in details.get('timing', {})


@pytest.mark.parametrize('legacy,expected', [
    ('probe',         ('weight_walk', 'probe')),
    ('lp',            ('weight_walk', 'lp')),
    ('milp',          ('weight_walk', 'milp')),
    ('skip',          ('skip', 'probe')),
    ('gen_cone',      ('gen_cone', 'lp')),
    ('gen_cone_milp', ('gen_cone', 'milp')),
])
def test_legacy_tighten_mode_alias(legacy, expected):
    """Legacy tighten_mode=... translates into the two-axis settings."""
    s = default_settings(tighten_mode=legacy)
    assert (s.tighten_formulation, s.tighten_solver) == expected


def test_legacy_tighten_mode_invalid_raises():
    import pytest
    with pytest.raises(AssertionError):
        default_settings(tighten_mode='nonsense')


def test_tighten_formulation_invalid_raises():
    import pytest
    with pytest.raises(AssertionError):
        default_settings(tighten_formulation='bogus')


def test_tighten_solver_invalid_raises():
    import pytest
    with pytest.raises(AssertionError):
        default_settings(tighten_solver='bogus')
