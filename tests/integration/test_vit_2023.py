"""Integration tests for vit_2023 (extended track).

Two ViT attention nets (ibp_3_3_8: 3 blocks; pgd_2_3_16: 2 blocks),
eps-robustness specs of 9 single-constraint disjuncts. The pipeline under
test is the attention stack: softmax decomposition with exact max-shift,
backward CROWN with exp/recip planes + McCormick bilinear, joint
differentiable-intermediate alpha, and the batched beta-CROWN BaB with
relu+bilinear splits (attn_crown.py).

The benchmark has NO SAT cases (ABC published: 0 sat / 125 unsat of 200),
so all pins are UNSAT, chosen to cover the three closing mechanisms:
  - pgd_2043: root close by max-shift + joint intermediate-alpha (no BaB)
  - ibp_2759: beta-CROWN BaB close on the 3-block net (~39 domains)
  - pgd_8836: deep-gap long-iteration joint-alpha close
Timings observed on the AWS g5 A10G sweeps 2026-06-11 (see
docs/benchmarks/vit_2023.md / scratch/VIT_FINDINGS.md); max_wall_s ~1.5x
observed.
"""
import pytest
from ._runner import run_case


BENCHMARK_DIR = 'vit_2023'
CONFIG_YAML = 'vit_2023.yaml'

CASES = [
    dict(
        desc='pgd_2_3_16_2043 (UNSAT, joint-alpha root close, ~12s)',
        net='onnx/pgd_2_3_16.onnx',
        vnnlib='vnnlib/pgd_2_3_16_2043.vnnlib',
        expected='verified', timeout=100, max_wall_s=20.0,
    ),
    dict(
        desc='ibp_3_3_8_2759 (UNSAT, beta-BaB close, ~21s)',
        net='onnx/ibp_3_3_8.onnx',
        vnnlib='vnnlib/ibp_3_3_8_2759.vnnlib',
        expected='verified', timeout=100, max_wall_s=35.0,
    ),
    dict(
        # Observed close ~99s is at the edge of the official 100s budget;
        # the pin is a regression detector, so give the run headroom
        # (timeout=150) and gate on max_wall_s=150 (~1.5x observed).
        desc='pgd_2_3_16_8836 (UNSAT, deep-gap alpha close, ~99s)',
        net='onnx/pgd_2_3_16.onnx',
        vnnlib='vnnlib/pgd_2_3_16_8836.vnnlib',
        expected='verified', timeout=150, max_wall_s=150.0,
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize('case', CASES, ids=[c['desc'] for c in CASES])
def test_vit_2023(case, vnncomp_benchmarks):
    run_case(case, CONFIG_YAML, vnncomp_benchmarks, BENCHMARK_DIR)
