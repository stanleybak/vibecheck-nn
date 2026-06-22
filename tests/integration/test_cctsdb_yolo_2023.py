"""Integration pin for cctsdb_yolo_2023 — the complete patch-enumeration handler.

The YOLO patch ONNX bakes in control-flow ops (ScatterND/Range/Where/ArgMax/dynamic-Min-Max)
that vibecheck's graph loader and onnx2torch can't ingest; the verification is instead a finite
enumeration of integer patch positions through the ORIGINAL net on ORT-CPU (src/vibecheck/
cctsdb_yolo.py). Complete: pins an unsat (every placement safe) and a sat (some placement breaks
the detection). ORT-CPU, no GPU needed. Needs the 2026 clone; skips otherwise. See
docs/benchmarks/cctsdb_yolo_2023.md.
"""
import os

import pytest

_CAND = [
    os.path.expanduser('~/repositories/vnncomp2026_benchmarks/benchmarks/cctsdb_yolo_2023/1.0'),
    os.path.expanduser('~/Desktop/repositories/vnncomp2026_benchmarks/benchmarks/'
                       'cctsdb_yolo_2023/1.0'),
]
CASES = [
    ('patch-1.onnx', 'spec_onnx_patch-1_idx_00559_0.vnnlib', 'unsat'),
    ('patch-1.onnx', 'spec_onnx_patch-1_idx_16972_2.vnnlib', 'sat'),
]


def _bench():
    return next((d for d in _CAND if os.path.isdir(d)), None)


@pytest.mark.integration
@pytest.mark.parametrize('onnx_name,stem,expected', CASES)
def test_cctsdb_enumeration(onnx_name, stem, expected):
    bench = _bench()
    if bench is None:
        pytest.skip('cctsdb_yolo_2023 1.0 not found locally')
    onnx = f'{bench}/onnx/{onnx_name}'
    vnnlib = f'{bench}/vnnlib/{stem}'
    if not (os.path.exists(onnx) and os.path.exists(vnnlib)):
        pytest.skip('cctsdb files missing')

    from vibecheck import cctsdb_yolo as cy
    from vibecheck.settings import default_settings
    assert cy.has_cctsdb_structure(onnx, vnnlib)
    settings = default_settings(cctsdb_yolo=True)
    verdict, wit = cy.cctsdb_yolo_verify(onnx, vnnlib, settings, timeout=350, log=lambda _m: None)
    assert verdict == expected, f'{stem}: got {verdict!r}, expected {expected!r}'
    if expected == 'sat':
        # the witness is a real placement that breaks the detection on the ORIGINAL net (ORT)
        import numpy as np
        from vibecheck.surrogate_pgd import _ort_eval
        from vibecheck.vnnlib_loader import load_vnnlib
        y = _ort_eval(onnx, wit)
        margin = cy._worst_margin_np(y, load_vnnlib(vnnlib).disjuncts)
        assert margin <= float(settings.sat_validate_atol), f'witness ORT margin {margin} not a CE'
