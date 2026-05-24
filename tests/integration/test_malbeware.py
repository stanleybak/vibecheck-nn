"""Integration tests for malbeware.

VNNCOMP regular track. 150 instances across 3 malware-classifier
models (64×64 image → 25 classes). Pre-fix: 50/50 linear cases all
crashed with `ValueError: max() arg is an empty sequence` in
`_phase1_bab_refine` (no ReLU layers). Post-fix: 148/150.

vibecheck 150/150 vs AB-CROWN 150/150. Wall: ~66 s vs AB-CROWN's
1026 s — 15× faster at full coverage.

Three representative cases:
  - linear-25 Adialer.C idx-0 (UNSAT, ~0.5 s) — pure-linear-no-ReLU
    fast path. Pre-fix this crashed.
  - 4-25 Allaple.A idx-11 (UNSAT, ~6 s) — full-unstable MIP fallback
    (ABC's `complete_verifier: mip` strategy). 4917 binarized neurons
    return INFEASIBLE; default 200-cap plateaued at lb=-0.59.
  - 16-25 Allaple.A idx-11 (SAT, ~3 s) — root-PGD on conv net.
"""
import pytest
from ._runner import run_case


BENCHMARK_DIR = 'malbeware'
CONFIG_YAML = 'malbeware.yaml'

CASES = [
    dict(
        desc='malbeware linear-25 Adialer.C idx-0 (UNSAT, no-ReLU fast path)',
        net='onnx/malware_malimg_family_scaled_linear-25.onnx',
        vnnlib='vnnlib/malbeware_family-Adialer.C_label-0_eps-1_idx-0.vnnlib',
        expected='verified', timeout=30, max_wall_s=5.0,
    ),
    dict(
        desc='malbeware 16-25 Allaple.A idx-11 (SAT, root-PGD)',
        net='onnx/malware_malimg_family_scaled_16-25.onnx',
        vnnlib='vnnlib/malbeware_family-Allaple.A_label-2_eps-3_idx-11.vnnlib',
        expected='sat', timeout=30, max_wall_s=10.0,
    ),
    dict(
        desc='malbeware 4-25 Allaple.A idx-11 (UNSAT, full-MIP fallback)',
        net='onnx/malware_malimg_family_scaled_4-25.onnx',
        vnnlib='vnnlib/malbeware_family-Allaple.A_label-2_eps-3_idx-11.vnnlib',
        expected='verified', timeout=30, max_wall_s=15.0,
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize('case', CASES, ids=[c['desc'] for c in CASES])
def test_malbeware(case, vnncomp_benchmarks):
    run_case(case, CONFIG_YAML, vnncomp_benchmarks, BENCHMARK_DIR)
